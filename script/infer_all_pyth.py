#!/usr/bin/env python3
"""Infer all .pyth datasets with top-down trial batching for large datasets."""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def parse_vehicle_count(name: str) -> int:
    part = name.split("_")[1]
    return int(part.split("m")[1])


def _fmt(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def add_summary_stats(data: dict) -> dict:
    costs = data["normalized_costs"]
    summary = {
        "mean": round(sum(costs) / len(costs), 4),
        "std": round(statistics.stdev(costs), 4) if len(costs) > 1 else 0.0,
        "min": round(min(costs), 4),
        "max": round(max(costs), 4),
        "count": len(costs),
    }
    data["summary_stats"] = summary
    return summary


def merge_results(partial_paths: list[Path], output_path: Path):
    merged = None
    total_inference_time = 0.0
    for p in partial_paths:
        with open(p) as f:
            chunk = json.load(f)
        # Accumulate inference time from partial results
        if "inference_time" in chunk and chunk["inference_time"] is not None:
            total_inference_time += chunk["inference_time"]
        if merged is None:
            merged = {k: v for k, v in chunk.items() if k not in ("summary_stats", "inference_time")}
        else:
            for key in ("costs", "normalized_costs", "raw_replay_costs",
                        "skipped_customers_count", "route_diagnostics",
                        "tw_violations_count", "appearance_violations_count",
                        "constraint_diagnostics", "raw_cost_components",
                        "normalized_cost_components", "routes"):
                if key in chunk:
                    merged[key].extend(chunk[key])
            merged["total_skipped_customers"] = sum(
                merged.get("skipped_customers_count", []))
            merged["total_tw_violations"] = sum(
                merged.get("tw_violations_count", []))
            merged["total_appearance_violations"] = sum(
                merged.get("appearance_violations_count", []))

    merged["inference_time"] = round(total_inference_time, 2)
    add_summary_stats(merged)
    merged = {"summary_stats": merged.pop("summary_stats"), **merged}
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)


def _make_subset(data, start, end):
    return data.__class__(
        data.veh_count, data.veh_capa, data.veh_speed,
        data.nodes[start:end].clone(),
        data.cust_mask[start:end].clone() if data.cust_mask is not None else None,
    )


def _build_infer_cmd(args, infer_script, data_file, result_json, veh_count):
    cmd = [
        sys.executable, str(infer_script),
        "--data-file", str(data_file),
        "--problem-type", "dvrptw",
        "--config-file", str(args.config_file),
        "--model-weight", str(args.model_weight),
        "--vehicles-count", str(veh_count),
        "--veh-capa", str(args.veh_capa),
        "--veh-speed", _fmt(args.veh_speed),
        "--max-print-instances", "0",
        "--verify-rollouts", "0",
        "--save-json", str(result_json),
    ]
    if args.sample:
        cmd.append("--sample")
    elif args.greedy:
        cmd.append("--greedy")
    cmd.append("--no-verify-routes")
    return cmd


def estimate_batch_size(data, config_path, model_weight, max_batch_size, min_batch_size, is_sample):
    """Estimate max safe batch size via one forward pass (no subprocess)."""
    from utils import parse_args
    from MODEL.infer import (
        _dataset_cls, _environment_cls, _build_env_params,
        _init_model, _load_model_weights_or_raise,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  No CUDA available, using min batch size")
        return max(1, min_batch_size)

    MEMORY_MARGIN = 0.80

    argv = ["--config-file", str(config_path), "--model-weight", str(model_weight)]
    args = parse_args(argv)
    args.greedy = not is_sample
    args.sample = is_sample

    dataset_cls = _dataset_cls("dvrptw")
    env_cls = _environment_cls("dvrptw")

    n_test = min(2, data.nodes.size(0))
    subset = _make_subset(data, 0, n_test)
    if not getattr(args, "no_normalize", False):
        subset.normalize()

    env_params = _build_env_params(args)
    env = env_cls(subset, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)

    learner = _init_model(args, dataset_cls, env_cls, device)
    learner.eval()
    _load_model_weights_or_raise(model_weight, learner)
    learner.eval()

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        actions, _, rewards = learner(env)

    peak = torch.cuda.max_memory_allocated(device)
    free, total = torch.cuda.mem_get_info(device)

    per_instance = peak / n_test
    available = free * MEMORY_MARGIN
    estimated = max(min_batch_size, min(int(available / per_instance), max_batch_size, data.nodes.size(0)))

    print(f"  VRAM: total={total//1024**2}MB, free={free//1024**2}MB, "
          f"peak_{n_test}inst={peak//1024**2}MB, per_inst={per_instance//1024**2}MB, "
          f"batch_size={estimated}")

    del learner, env, subset, args
    torch.cuda.empty_cache()
    return estimated


def main():
    parser = argparse.ArgumentParser(
        description="Infer all .pyth datasets with top-down trial batching."
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--infer-script", type=Path, default=Path("MODEL/infer.py"))
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--veh-capa", type=int, default=200)
    parser.add_argument("--veh-speed", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("infer"))
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override trial (disables top-down search)")
    parser.add_argument("--max-batch-size", type=int, default=4096,
                        help="Upper bound for trial (default: 4096)")
    parser.add_argument("--min-batch-size", type=int, default=1,
                        help="Lowest batch size to try (default: 1)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temp batch files for debugging")
    args = parser.parse_args()

    datasets = sorted(args.datasets_root.glob("*.pyth"))
    if not datasets:
        print(f"No .pyth files found in {args.datasets_root}")
        return 1

    datasets_root = args.datasets_root.resolve()
    infer_script = args.infer_script.resolve()
    config_file = args.config_file.resolve()
    model_weight = args.model_weight.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(datasets)} dataset(s):")
    for d in datasets:
        info = torch.load(d, map_location="cpu", weights_only=False)
        print(f"  {d.name}  shape={info.nodes.shape}")
        del info

    all_results = []

    for ds_path in datasets:
        print(f"\n{'='*70}")
        print(f"Processing: {ds_path.name}")
        print(f"{'='*70}")

        data = torch.load(ds_path, map_location="cpu", weights_only=False)
        B = data.nodes.size(0)
        print(f"  Instances: {B}")
        veh_count = parse_vehicle_count(ds_path.stem)

        result_json = args.output_dir / f"{ds_path.stem}.json"

        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            print(f"  Estimating batch size...")
            batch_size = estimate_batch_size(
                data, config_file, model_weight,
                args.max_batch_size, args.min_batch_size, args.sample)

        n_batches = math.ceil(B / batch_size)

        if n_batches <= 1:
            print(f"  Instances: {B} (single batch)")
            t_start = time.perf_counter()
            cmd = _build_infer_cmd(args, infer_script, ds_path, result_json, veh_count)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            t_elapsed = time.perf_counter() - t_start
            print(proc.stdout)
            if proc.returncode != 0:
                print(f"  FAILED: {proc.stderr.strip()}")
                all_results.append({"dataset": ds_path.name, "status": "failed"})
                del data
                continue

            with open(result_json) as f:
                result_data = json.load(f)
            summary = add_summary_stats(result_data)
            # Lấy inference_time từ JSON nếu có, nếu không thì dùng wall time
            inf_time = result_data.get("inference_time", t_elapsed)
            summary["inference_time"] = round(inf_time, 2)
            result_data = {"summary_stats": summary, **result_data}
            with open(result_json, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"  Result: mean={summary['mean']:.4f} std={summary['std']:.4f} time={summary['inference_time']:.2f}s")
            all_results.append({"dataset": ds_path.name, "status": "ok", **summary})
        else:
            print(f"  Instances: {B}, batch_size={batch_size}, batches={n_batches}")
            partials = []
            tmpdir = Path(tempfile.mkdtemp(prefix=f"batch_{ds_path.stem}_"))
            t_start = time.perf_counter()
            batch_times = []

            for bi in range(n_batches):
                start = bi * batch_size
                end = min((bi + 1) * batch_size, B)
                subset = _make_subset(data, start, end)
                tmp_pyth = tmpdir / f"batch_{bi:03d}.pyth"
                torch.save(subset, tmp_pyth)
                partial_json = tmpdir / f"batch_{bi:03d}.json"

                print(f"    Batch {bi+1}/{n_batches} [{start}:{end}]...")
                t_batch_start = time.perf_counter()
                cmd = _build_infer_cmd(args, infer_script, tmp_pyth, partial_json, veh_count)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                t_batch_elapsed = time.perf_counter() - t_batch_start
                if proc.returncode != 0:
                    print(f"      FAILED: {proc.stderr.strip()}")
                    break
                partials.append(partial_json)
                batch_times.append(t_batch_elapsed)
                print(f"      OK ({t_batch_elapsed:.2f}s)")

            t_elapsed = time.perf_counter() - t_start

            if len(partials) == n_batches:
                print(f"  Merging {len(partials)} partial results...")
                merge_results(partials, result_json)
                with open(result_json) as f:
                    result_data = json.load(f)
                summary = result_data["summary_stats"]
                # Ưu tiên inference_time từ merge_results (tổng từ các batch JSON),
                # fallback về wall time
                inf_time = result_data.get("inference_time", t_elapsed)
                summary["inference_time"] = round(inf_time, 2)
                print(f"  Result: mean={summary['mean']:.4f} std={summary['std']:.4f} time={summary['inference_time']:.2f}s")
                all_results.append({"dataset": ds_path.name, "status": "ok", **summary})
            else:
                all_results.append({"dataset": ds_path.name, "status": "failed"})

            if not args.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)
            del subset, tmp_pyth, partial_json

        del data

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        if r["status"] == "ok":
            print(f"  {r['dataset']}: mean={r['mean']:.4f} std={r['std']:.4f} [{r['count']} instances]")
        else:
            print(f"  {r['dataset']}: FAILED")

    # Load config file contents
    config_content = {}
    if config_file.exists():
        try:
            config_content = json.loads(config_file.read_text())
        except Exception:
            config_content = {"error": "failed to parse config file"}

    config_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": str(config_file),
        "model_weight": str(model_weight),
        "config_content": config_content,
        "infer_script": str(infer_script),
        "datasets_root": str(datasets_root),
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "decode_mode": "sample" if args.sample else "greedy",
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "min_batch_size": args.min_batch_size,
    }

    summary_file = args.output_dir / "summary.json"
    summary_file.write_text(json.dumps({"config": config_info, "results": all_results}, indent=2))
    print(f"\n  Aggregate summary saved to: {summary_file}")


if __name__ == "__main__":
    raise SystemExit(main())
