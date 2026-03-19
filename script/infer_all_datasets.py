#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from openpyxl import Workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MODEL/infer.py over every CSV under a datasets directory and "
            "save per-case outputs plus aggregate summaries."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
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
    }


def run_one(
    args: argparse.Namespace,
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

    cmd = [
        args.python_executable,
        str(args.infer_script),
        "--problem-type",
        args.problem_type,
        "--config-file",
        str(args.config_file),
        "--model-weight",
        str(args.model_weight),
        "--vehicles-count",
        str(args.vehicles_count),
        "--veh-capa",
        str(args.veh_capa),
        "--veh-speed",
        _format_cli_number(args.veh_speed),
        "--max-print-instances",
        str(args.max_print_instances),
        "--verify-rollouts",
        str(args.verify_rollouts),
        "--data-csv",
        str(csv_path),
        "--save-json",
        str(result_json),
    ]
    if args.no_verify_routes:
        cmd.append("--no-verify-routes")
    if args.sample:
        cmd.append("--sample")
    elif args.greedy:
        cmd.append("--greedy")
    for extra in args.extra_arg:
        cmd.append(extra)

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_sec = time.perf_counter() - start

    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.write("RETURN_CODE:\n")
        f.write(str(proc.returncode) + "\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr)

    row: dict[str, Any] = {
        "dataset_relpath": str(rel),
        "dataset_abspath": str(csv_path.resolve()),
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "duration_sec": round(duration_sec, 6),
        "result_json": str(result_json),
        "run_log": str(log_path),
        "command": " ".join(shlex.quote(x) for x in cmd),
        "error_message": "",
    }

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error_message"] = err[-1] if err else "inference failed"
        return row

    if not result_json.exists():
        row["status"] = "failed"
        row["error_message"] = "infer finished without writing save-json output"
        return row

    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        row.update(parse_infer_json(payload))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error_message"] = f"cannot parse infer json: {exc}"
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

    # Collect all keys (preserve order)
    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    # Write header
    ws.append(all_keys)

    # Write rows
    for row in rows:
        ws.append([row.get(k, None) for k in all_keys])

    wb.save(path)


def main() -> int:
    args = parse_args()

    datasets_root = args.datasets_root.resolve()
    infer_script = args.infer_script.resolve()
    config_file = args.config_file.resolve()
    model_weight = args.model_weight.resolve()

    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")
    if not infer_script.exists():
        raise FileNotFoundError(f"infer script not found: {infer_script}")
    if not config_file.exists():
        raise FileNotFoundError(f"config file not found: {config_file}")
    if not model_weight.exists():
        raise FileNotFoundError(f"model weight not found: {model_weight}")

    args.infer_script = infer_script
    args.config_file = config_file
    args.model_weight = model_weight

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("output") / f"batch_infer_{now_tag()}"
    output_dir = output_dir.resolve()

    per_case_root = output_dir / "per_case_json"
    logs_root = output_dir / "logs"
    ensure_dir(per_case_root)
    ensure_dir(logs_root)

    csv_files = find_csv_files(datasets_root, args.file_glob)
    if not csv_files:
        print(f"No CSV files found in {datasets_root} using pattern {args.file_glob}")
        return 1

    print(f"Found {len(csv_files)} CSV files")
    rows: list[dict[str, Any]] = []
    failed = 0

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] infer: {csv_path}")
        row = run_one(args, csv_path, datasets_root, per_case_root, logs_root)
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
        "infer_script": str(infer_script),
        "python_executable": args.python_executable,
        "problem_type": args.problem_type,
        "config_file": str(config_file),
        "model_weight": str(model_weight),
        "vehicles_count": args.vehicles_count,
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "max_print_instances": args.max_print_instances,
        "verify_rollouts": args.verify_rollouts,
        "decode_mode": "sample" if args.sample else "greedy",
        "no_verify_routes": bool(args.no_verify_routes),
        "total_files": len(rows),
        "ok_files": sum(1 for r in rows if r.get("status") == "ok"),
        "failed_files": sum(1 for r in rows if r.get("status") != "ok"),
    }

    write_csv(summary_csv, rows)
    write_excel(summary_excel , rows)
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
