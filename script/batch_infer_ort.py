#!/usr/bin/env python3
"""
Batch OR-Tools inference over all CSV files in a datasets directory tree.

Usage
-----
    # Run on all datasets under data/datasets/ with 5 vehicles
    python script/batch_infer_ort.py \\
        --datasets-root data/datasets \\
        --vehicles-count 5 --veh-capa 1300 --veh-speed 1 \\
        --late-cost 1 --time-limit-ms 10000 \\
        --output-dir output/batch_ort_100

    # Run on a single scale (e.g. 100 customers)
    python script/batch_infer_ort.py \\
        --datasets-root data/datasets/100 \\
        --vehicles-count 5 --veh-capa 1300 --veh-speed 1 \\
        --file-glob "h100*.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch OR-Tools inference over CSV datasets."
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--infer-script",
        type=Path,
        default=Path("script/infer_ort.py"),
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch infer_ort.py",
    )
    parser.add_argument("--problem-type", type=str, default="dvrptw")
    parser.add_argument("--vehicles-count", type=int, required=True)
    parser.add_argument("--veh-capa", type=int, required=True)
    parser.add_argument("--veh-speed", type=float, required=True)
    parser.add_argument("--late-cost", type=float, default=1.0)
    parser.add_argument("--pending-cost", type=float, default=2.0)
    parser.add_argument("--time-limit-ms", type=int, default=10_000)
    parser.add_argument("--max-print-instances", type=int, default=1)
    parser.add_argument("--verify-rollouts", type=int, default=1)
    parser.add_argument(
        "--no-verify-routes",
        action="store_true",
        help="Disable route replay verification in infer_ort.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: output/batch_ort_<timestamp>",
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
        help="Stop at first failure",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel files to evaluate",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to infer_ort.py (can be repeated)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

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
    """Extract the most important metrics from a per-case infer JSON."""
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


# ---------------------------------------------------------------------------
#  Run one instance
# ---------------------------------------------------------------------------

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
        "--problem-type", args.problem_type,
        "--vehicles-count", str(args.vehicles_count),
        "--veh-capa", str(args.veh_capa),
        "--veh-speed", _format_cli_number(args.veh_speed),
        "--late-cost", _format_cli_number(args.late_cost),
        "--pending-cost", _format_cli_number(args.pending_cost),
        "--time-limit-ms", str(args.time_limit_ms),
        "--max-print-instances", str(args.max_print_instances),
        "--verify-rollouts", str(args.verify_rollouts),
        "--data-csv", str(csv_path),
        "--save-json", str(result_json),
    ]
    if args.no_verify_routes:
        cmd.append("--no-verify-routes")
    for extra in args.extra_arg:
        cmd.append(extra)

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_sec = time.perf_counter() - start

    # Write run log
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
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = f"cannot parse infer json: {exc}"

    return row


# ---------------------------------------------------------------------------
#  Output writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, meta: dict[str, Any],
                        rows: list[dict[str, Any]]) -> None:
    """Write a summary JSON in the same format as batch_infer_*/summary.json."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    datasets_root = args.datasets_root.resolve()
    infer_script = args.infer_script.resolve()

    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")
    if not infer_script.exists():
        raise FileNotFoundError(f"infer script not found: {infer_script}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("output") / f"batch_ort_{now_tag()}"
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
    print(f"Infer script: {infer_script}")
    print(f"Output dir:  {output_dir}")
    print(f"Parameters:  {args.vehicles_count} vehicles, "
          f"capa={args.veh_capa}, speed={args.veh_speed}, "
          f"late_cost={args.late_cost}, time_limit={args.time_limit_ms}ms")
    print()

    rows: list[dict[str, Any]] = []
    failed = 0

    if args.parallel > 1:
        print(f"Running parallel evaluation with {args.parallel} processes")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    run_one,
                    args,
                    csv_path,
                    datasets_root,
                    per_case_root,
                    logs_root,
                ): (idx, csv_path)
                for idx, csv_path in enumerate(csv_files, start=1)
            }
            for future in as_completed(futures):
                idx, csv_path = futures[future]
                try:
                    row = future.result()
                    rows.append(row)
                    status = row.get("status", "?")
                    cost = row.get("normalized_cost", "?")
                    viol = row.get("total_tw_violations", "?")
                    dur = float(row.get("duration_sec", 0.0))
                    print(
                        f"[{idx:>4}/{len(csv_files)}] {csv_path.name}  "
                        f"status={status}  cost={cost}  tw_viol={viol}  dur={dur:.1f}s"
                    )
                    if status != "ok":
                        failed += 1
                        print(f"  -> FAILED: {row.get('error_message', '')}")
                        if args.fail_fast:
                            break
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(
                        f"[{idx:>4}/{len(csv_files)}] {csv_path.name}  "
                        f"status=failed  error={exc}"
                    )
                    if args.fail_fast:
                        break
    else:
        for idx, csv_path in enumerate(csv_files, start=1):
            print(f"[{idx:>4}/{len(csv_files)}] {csv_path.name}  ", end="", flush=True)
            row = run_one(args, csv_path, datasets_root, per_case_root, logs_root)
            rows.append(row)

            status = row.get("status", "?")
            cost = row.get("normalized_cost", "?")
            viol = row.get("total_tw_violations", "?")
            dur = float(row.get("duration_sec", 0.0))
            print(f"status={status}  cost={cost}  tw_viol={viol}  dur={dur:.1f}s")

            if status != "ok":
                failed += 1
                print(f"  -> FAILED: {row.get('error_message', '')}")
                if args.fail_fast:
                    break

    # Write outputs
    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_root": str(datasets_root),
        "infer_script": str(infer_script),
        "python_executable": args.python_executable,
        "problem_type": args.problem_type,
        "vehicles_count": args.vehicles_count,
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "late_cost": args.late_cost,
        "pending_cost": args.pending_cost,
        "time_limit_ms": args.time_limit_ms,
        "max_print_instances": args.max_print_instances,
        "verify_rollouts": args.verify_rollouts,
        "no_verify_routes": bool(args.no_verify_routes),
        "parallel": args.parallel,
        "total_files": len(rows),
        "ok_files": sum(1 for r in rows if r.get("status") == "ok"),
        "failed_files": sum(1 for r in rows if r.get("status") != "ok"),
    }

    write_csv(output_dir / "summary.csv", rows)
    write_summary_json(output_dir / "summary.json", run_meta, rows)
    print(f"\nWrote summary to {output_dir / 'summary.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Done: {run_meta['ok_files']} ok, {run_meta['failed_files']} failed "
          f"out of {run_meta['total_files']} total")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
