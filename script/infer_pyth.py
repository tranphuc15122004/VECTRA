#!/usr/bin/env python3
"""Infer .pyth datasets in batch units. Default batch size is 256."""

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


def fmt_number(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def add_summary_stats(data: dict, duration_sec: float = 0.0) -> dict:
    costs = data["normalized_costs"]
    count = len(costs)
    summary = {
        "mean": round(sum(costs) / count, 4),
        "std": round(statistics.stdev(costs), 4) if count > 1 else 0.0,
        "min": round(min(costs), 4),
        "max": round(max(costs), 4),
        "count": count,
        "total_duration_sec": round(duration_sec, 2),
        "mean_time_per_instance_sec": round(duration_sec / count, 4) if count > 0 else 0.0,
    }
    if "raw_replay_costs" in data and data["raw_replay_costs"]:
        rc = data["raw_replay_costs"]
        summary["raw_replay_mean"] = round(sum(rc) / len(rc), 4)
    data["summary_stats"] = summary
    return summary


def merge_results(partial_paths):
    merged = None
    for p in partial_paths:
        with open(p) as f:
            chunk = json.load(f)
        if merged is None:
            merged = {k: v for k, v in chunk.items() if k != "summary_stats"}
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
    return merged


def build_cmd(args, infer_script, data_file, result_json, veh_count):
    cmd = [
        sys.executable, str(infer_script),
        "--data-file", str(data_file),
        "--problem-type", "dvrptw",
        "--config-file", str(args.config_file),
        "--model-weight", str(args.model_weight),
        "--vehicles-count", str(veh_count),
        "--veh-capa", str(args.veh_capa),
        "--veh-speed", fmt_number(args.veh_speed),
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


def infer_single(args, infer_script, ds_path, result_json):
    veh_count = parse_vehicle_count(ds_path.stem)
    cmd = build_cmd(args, infer_script, ds_path, result_json, veh_count)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    print(proc.stdout, end="")
    return proc, elapsed


def infer_batched(args, infer_script, ds_path, batch_size, result_json):
    data = torch.load(ds_path, map_location="cpu", weights_only=False)
    B = data.nodes.size(0)
    n_batches = math.ceil(B / batch_size)
    veh_count = parse_vehicle_count(ds_path.stem)

    partials = []
    tmpdir = Path(tempfile.mkdtemp(prefix=f"batch_{ds_path.stem}_"))
    ok = True
    total_elapsed = 0.0

    for bi in range(n_batches):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, B)
        subset = data.__class__(
            data.veh_count, data.veh_capa, data.veh_speed,
            data.nodes[start:end].clone(),
            data.cust_mask[start:end].clone() if data.cust_mask is not None else None,
        )
        tmp_pyth = tmpdir / f"batch_{bi:03d}.pyth"
        torch.save(subset, tmp_pyth)
        partial_json = tmpdir / f"batch_{bi:03d}.json"

        print(f"    Batch {bi+1}/{n_batches} [{start}:{end}]...", end=" ")
        t0 = time.perf_counter()
        cmd = build_cmd(args, infer_script, tmp_pyth, partial_json, veh_count)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"FAILED ({elapsed:.1f}s): {proc.stderr.strip()}")
            ok = False
            break
        partials.append(partial_json)
        total_elapsed += elapsed
        print(f"OK ({elapsed:.1f}s)")

    merged = None
    if ok and len(partials) == n_batches:
        print(f"  Merging {len(partials)} partial results...")
        merged = merge_results(partials)
        if merged is not None:
            add_summary_stats(merged, duration_sec=total_elapsed)
            merged = {"summary_stats": merged.pop("summary_stats"), **merged}
            with open(result_json, "w") as f:
                json.dump(merged, f, indent=2)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return ok and merged is not None, total_elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Infer .pyth datasets in batch units (default batch size 256)."
    )
    parser.add_argument("--data-file", type=Path, default=None,
                        help="Single .pyth file to infer (default: all .pyth in datasets/)")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--infer-script", type=Path, default=Path("MODEL/infer.py"))
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--veh-capa", type=int, default=200)
    parser.add_argument("--veh-speed", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: infer/<timestamp>)")
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of instances per batch (default: 256)")
    args = parser.parse_args()

    infer_script = args.infer_script.resolve()
    config_file = args.config_file.resolve()
    model_weight = args.model_weight.resolve()

    if args.data_file is not None:
        datasets = [args.data_file.resolve()]
    else:
        datasets_root = args.datasets_root.resolve()
        datasets = sorted(datasets_root.glob("*.pyth"))
        if not datasets:
            print(f"No .pyth files found in {datasets_root}")
            return 1
        print(f"Found {len(datasets)} dataset(s) in {datasets_root}:")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("infer") / timestamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "infer_script": str(infer_script),
        "config_file": str(config_file),
        "model_weight": str(model_weight),
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "batch_size": args.batch_size,
        "decode_mode": "sample" if args.sample else "greedy",
        "data_file": str(args.data_file) if args.data_file else None,
        "datasets_root": str(args.datasets_root.resolve()) if not args.data_file else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config_info, indent=2))
    print(f"Config saved to: {output_dir / 'config.json'}")

    all_results = []

    for ds_path in datasets:
        ds_path = Path(ds_path)
        print(f"\n{'='*70}")
        print(f"Processing: {ds_path.name}")
        print(f"{'='*70}")

        data = torch.load(ds_path, map_location="cpu", weights_only=False)
        B = data.nodes.size(0)
        print(f"  Instances: {B}")

        do_batching = B > args.batch_size
        result_json = output_dir / f"{ds_path.stem}.json"

        if not do_batching:
            print(f"  Single batch (batch_size={B})")
            proc, elapsed = infer_single(args, infer_script, ds_path, result_json)
            print(flush=True)
            if proc.returncode != 0:
                all_results.append({"dataset": ds_path.name, "status": "failed", "duration_sec": round(elapsed, 2)})
                continue
        else:
            n_batches = math.ceil(B / args.batch_size)
            print(f"  Splitting into {n_batches} batches of size {args.batch_size}")
            ok, elapsed = infer_batched(args, infer_script, ds_path, args.batch_size, result_json)
            if not ok:
                all_results.append({"dataset": ds_path.name, "status": "failed", "duration_sec": round(elapsed, 2)})
                continue

        with open(result_json) as f:
            result_data = json.load(f)
        summary = add_summary_stats(result_data, duration_sec=elapsed)
        result_data = {"summary_stats": summary, **result_data}
        with open(result_json, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"  Result: mean={summary['mean']:.4f} std={summary['std']:.4f}  "
              f"time={elapsed:.1f}s  ({summary['mean_time_per_instance_sec']:.4f}s/inst)")
        all_results.append({"dataset": ds_path.name, "status": "ok", **summary})

        del data

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        if r["status"] == "ok":
            print(f"  {r['dataset']}: mean={r['mean']:.4f} std={r['std']:.4f}  "
                  f"time={r['total_duration_sec']:.1f}s  [{r['count']} instances]")
        else:
            print(f"  {r['dataset']}: FAILED  (time={r.get('duration_sec', 0):.1f}s)")

    total_instances = sum(r.get("count", 0) for r in all_results if r["status"] == "ok")
    print(f"\n  Total instances inferred: {total_instances}")

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps({"results": all_results, "config": config_info}, indent=2))
    print(f"  Aggregate summary saved to: {summary_file}")


if __name__ == "__main__":
    raise SystemExit(main())
