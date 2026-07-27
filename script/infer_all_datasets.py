#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from openpyxl import Workbook

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODEL.infer import (
    _build_env_params,
    _check_route_constraints,
    _clone_dataset,
    _compute_cost_components,
    _dataset_cls,
    _environment_cls,
    _init_model,
    _load_model_weights_or_raise,
    _replay_routes_cost,
    _route_diag_for_instance,
    _run_inference,
    parse_infer_args,
)
from utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MODEL/infer.py over every CSV under a datasets directory and "
            "save per-case outputs plus aggregate summaries."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--infer-script", type=Path, default=Path("MODEL/infer.py"))
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch infer.py",
    )
    parser.add_argument("--problem-type", type=str, default="dvrptw")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--vehicles-count", type=int, required=True)
    parser.add_argument("--veh-capa", type=int, required=True)
    parser.add_argument("--veh-speed", type=float, required=True)
    parser.add_argument("--max-print-instances", type=int, default=1)
    parser.add_argument("--verify-rollouts", type=int, default=1)
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling decode in MODEL/infer.py",
    )
    decode_group.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decode in MODEL/infer.py (default)",
    )
    parser.add_argument(
        "--no-verify-routes",
        action="store_true",
        help="Disable route replay verification in infer.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: output/batch_infer_YYYYmmdd-HHMMSS",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="**/*.csv",
        help="Glob pattern relative to datasets root",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on matched CSV files; 0 means all files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failed dataset",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to infer.py (can be repeated)",
    )
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_csv_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _safe_get(seq: Any, idx: int, default: Any = None) -> Any:
    if isinstance(seq, list) and len(seq) > idx:
        return seq[idx]
    return default


def _safe_first(seq: Any, default: Any = None) -> Any:
    return _safe_get(seq, 0, default)


def _format_cli_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_infer_json(payload: dict[str, Any]) -> dict[str, Any]:
    route_diag = _safe_first(payload.get("route_diagnostics", []), {}) or {}
    raw_comp = _safe_first(payload.get("raw_cost_components", []), {}) or {}
    norm_comp = _safe_first(payload.get("normalized_cost_components", []), {}) or {}

    return {
        "normalized_cost": _safe_first(payload.get("normalized_costs", [])),
        "raw_replay_cost": _safe_first(payload.get("raw_replay_costs", [])),
        "total_skipped_customers": payload.get("total_skipped_customers"),
        "total_tw_violations": payload.get("total_tw_violations"),
        "total_appearance_violations": payload.get("total_appearance_violations"),
        "active_customers": route_diag.get("active_customers"),
        "visited_customers": route_diag.get("visited_customers"),
        "visit_steps": route_diag.get("visit_steps"),
        "missing_count": route_diag.get("missing_count"),
        "duplicate_count": route_diag.get("duplicate_count"),
        "extra_count": route_diag.get("extra_count"),
        "raw_total_cost": raw_comp.get("total_cost"),
        "raw_distance": raw_comp.get("distance"),
        "raw_late_time": raw_comp.get("late_time"),
        "raw_late_penalty": raw_comp.get("late_penalty"),
        "raw_skipped_orders": raw_comp.get("skipped_orders"),
        "raw_skipped_penalty": raw_comp.get("skipped_penalty"),
        "normalized_total_cost": norm_comp.get("total_cost"),
        "normalized_distance": norm_comp.get("distance"),
        "normalized_late_time": norm_comp.get("late_time"),
        "normalized_late_penalty": norm_comp.get("late_penalty"),
        "normalized_skipped_orders": norm_comp.get("skipped_orders"),
        "normalized_skipped_penalty": norm_comp.get("skipped_penalty"),
        "step_diagnostics_count": payload.get("step_diagnostics_count"),
    }


def _save_infer_payload(
    path: Path,
    routes,
    costs,
    raw_replay_costs,
    route_diagnostics,
    constraint_diagnostics,
    raw_cost_components,
    normalized_cost_components,
    inference_time,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inference_time": inference_time,
        "costs": [float(v) for v in costs.cpu().tolist()],
        "normalized_costs": [float(v) for v in costs.cpu().tolist()],
        "raw_replay_costs": [float(v) for v in raw_replay_costs.cpu().tolist()],
        "skipped_customers_count": [int(d.get("missing_count", 0)) for d in route_diagnostics],
        "total_skipped_customers": int(sum(int(d.get("missing_count", 0)) for d in route_diagnostics)),
        "route_diagnostics": route_diagnostics,
        "tw_violations_count": [int(d.get("tw_violation_count", 0)) for d in constraint_diagnostics],
        "appearance_violations_count": [int(d.get("appearance_violation_count", 0)) for d in constraint_diagnostics],
        "total_tw_violations": int(sum(int(d.get("tw_violation_count", 0)) for d in constraint_diagnostics)),
        "total_appearance_violations": int(sum(int(d.get("appearance_violation_count", 0)) for d in constraint_diagnostics)),
        "constraint_diagnostics": constraint_diagnostics,
        "raw_cost_components": raw_cost_components,
        "normalized_cost_components": normalized_cost_components,
        "routes": routes,
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_one_inproc(
    infer_args: argparse.Namespace,
    learner: torch.nn.Module,
    dataset_cls: type,
    env_cls: type,
    device: torch.device,
    csv_path: Path,
    datasets_root: Path,
    per_case_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    rel = csv_path.relative_to(datasets_root)
    result_json = (per_case_root / rel).with_suffix(".infer.json")
    log_path = (logs_root / rel).with_suffix(".log")
    ensure_dir(result_json.parent)
    ensure_dir(log_path.parent)

    start = time.perf_counter()
    try:
        data = dataset_cls.from_csv(
            str(csv_path),
            veh_count=infer_args.vehicles_count,
            veh_capa=infer_args.veh_capa,
            veh_speed=infer_args.veh_speed,
        )
        raw_data = _clone_dataset(data)
        if not infer_args.no_normalize:
            data.normalize()

        env_params = _build_env_params(infer_args)
        env = env_cls(data, None, None, *env_params)
        env.nodes = env.nodes.to(device)
        if env.init_cust_mask is not None:
            env.init_cust_mask = env.init_cust_mask.to(device)

        routes, costs = _run_inference(infer_args, env, learner)

        raw_cost_components = _compute_cost_components(
            raw_data, routes, infer_args.pending_cost, infer_args.late_cost,
        )
        normalized_cost_components = _compute_cost_components(
            data, routes, infer_args.pending_cost, infer_args.late_cost,
        )
        raw_replay_costs = _replay_routes_cost(
            raw_data, env_cls, env_params, routes, rollouts=infer_args.verify_rollouts,
        )
        route_diagnostics = [
            _route_diag_for_instance(data, routes, idx) for idx in range(len(routes))
        ]
        constraint_diagnostics = _check_route_constraints(raw_data, routes)
        inference_time = time.perf_counter() - start

        payload = _save_infer_payload(
            result_json, routes, costs, raw_replay_costs,
            route_diagnostics, constraint_diagnostics,
            raw_cost_components, normalized_cost_components,
            inference_time,
        )

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"Inference (in-process) on {csv_path}\n")
            f.write(f"Duration: {inference_time:.6f}s\n")
            f.write(f"Status: ok\n")

        row: dict[str, Any] = {
            "dataset_relpath": str(rel),
            "dataset_abspath": str(csv_path.resolve()),
            "status": "ok",
            "return_code": 0,
            "duration_sec": round(inference_time, 6),
            "result_json": str(result_json),
            "run_log": str(log_path),
            "command": "",
            "error_message": "",
        }
        row.update(parse_infer_json(payload))

    except Exception as exc:
        duration_sec = time.perf_counter() - start
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"Inference (in-process) on {csv_path}\n")
            f.write(f"Duration: {duration_sec:.6f}s\n")
            f.write(f"Status: failed\n")
            f.write(f"Error: {exc}\n")
        row = {
            "dataset_relpath": str(rel),
            "dataset_abspath": str(csv_path.resolve()),
            "status": "failed",
            "return_code": 1,
            "duration_sec": round(duration_sec, 6),
            "result_json": str(result_json) if result_json.exists() else "",
            "run_log": str(log_path),
            "command": "",
            "error_message": str(exc),
        }

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)

    wb = Workbook()
    ws = wb.active

    if not rows:
        wb.save(path)
        return

    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    ws.append(all_keys)

    for row in rows:
        ws.append([row.get(k, None) for k in all_keys])

    wb.save(path)


def main() -> int:
    args = parse_args()

    datasets_root = args.datasets_root.resolve()
    config_file = args.config_file.resolve()
    model_weight = args.model_weight.resolve()

    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")
    if not config_file.exists():
        raise FileNotFoundError(f"config file not found: {config_file}")
    if not model_weight.exists():
        raise FileNotFoundError(f"model weight not found: {model_weight}")

    infer_argv = [
        "--config-file", str(config_file),
        "--model-weight", str(model_weight),
        "--problem-type", args.problem_type,
        "--vehicles-count", str(args.vehicles_count),
        "--veh-capa", str(args.veh_capa),
        "--veh-speed", _format_cli_number(args.veh_speed),
        "--max-print-instances", str(args.max_print_instances),
        "--verify-rollouts", str(args.verify_rollouts),
    ]
    if args.sample:
        infer_argv.append("--sample")
    else:
        infer_argv.append("--greedy")
    if args.no_verify_routes:
        infer_argv.append("--no-verify-routes")
    for extra in args.extra_arg:
        infer_argv.append(extra)

    infer_args = parse_infer_args(infer_argv)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not infer_args.no_cuda else "cpu"
    )
    set_random_seed(infer_args.rng_seed, deterministic=True)

    dataset_cls = _dataset_cls(infer_args.problem_type)
    env_cls = _environment_cls(infer_args.problem_type)
    if dataset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{infer_args.problem_type}'")

    learner = _init_model(infer_args, dataset_cls, env_cls, device)
    learner.eval()
    _load_model_weights_or_raise(infer_args.model_weight, learner)
    learner.eval()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("output") / f"batch_infer_{now_tag()}"
    output_dir = output_dir.resolve()

    per_case_root = output_dir / "per_case_json"
    logs_root = output_dir / "logs"
    ensure_dir(per_case_root)
    ensure_dir(logs_root)

    csv_files = find_csv_files(datasets_root, args.file_glob)
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[:args.max_files]
    if not csv_files:
        print(f"No CSV files found in {datasets_root} using pattern {args.file_glob}")
        return 1

    print(f"Found {len(csv_files)} CSV files")
    print(f"Loaded model once; processing {len(csv_files)} CSVs in-process")
    rows: list[dict[str, Any]] = []
    failed = 0

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] infer: {csv_path}")
        row = run_one_inproc(
            infer_args, learner, dataset_cls, env_cls, device,
            csv_path, datasets_root, per_case_root, logs_root,
        )
        rows.append(row)
        if row.get("status") != "ok":
            failed += 1
            print(f"  -> FAILED: {row.get('error_message', '')}")
            if args.fail_fast:
                break

    summary_csv = output_dir / "summary.csv"
    summary_excel = output_dir / "summary.xlsx"
    summary_json = output_dir / "summary.json"
    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_root": str(datasets_root),
        "infer_script": str(args.infer_script),
        "config_file": str(config_file),
        "model_weight": str(model_weight),
        "problem_type": args.problem_type,
        "vehicles_count": args.vehicles_count,
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "max_print_instances": args.max_print_instances,
        "verify_rollouts": args.verify_rollouts,
        "decode_mode": "sample" if args.sample else "greedy",
        "no_verify_routes": bool(args.no_verify_routes),
        "step_diagnostics": "--save-step-diagnostics" in args.extra_arg,
        "file_glob": args.file_glob,
        "max_files": args.max_files,
        "total_files": len(rows),
        "ok_files": sum(1 for r in rows if r.get("status") == "ok"),
        "failed_files": sum(1 for r in rows if r.get("status") != "ok"),
    }

    write_csv(summary_csv, rows)
    write_excel(summary_excel, rows)
    summary_json.write_text(
        json.dumps({"meta": run_meta, "results": rows}, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"Output dir      : {output_dir}")
    print(f"Per-case JSON   : {per_case_root}")
    print(f"Per-case logs   : {logs_root}")
    print(f"Summary CSV     : {summary_csv}")
    print(f"Summary JSON    : {summary_json}")
    print(f"Succeeded/Total : {run_meta['ok_files']}/{run_meta['total_files']}")
    print(f"Failed          : {run_meta['failed_files']}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
