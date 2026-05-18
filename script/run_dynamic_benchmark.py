#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from problems import DVRPTW_Dataset


CSV_FIELDS = ["x", "y", "demand", "open", "close", "servicetime", "time"]
DEFAULT_DOD_VALUES = (0.1, 0.25, 0.5, 0.75)
DEFAULT_TW_RATIOS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (42,)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    batch_script: Path
    config_file: Path
    model_weight: Path


def parse_float_list(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    try:
        return [float(item) for item in values]
    except ValueError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_int_list(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of ints")
    try:
        return [int(item) for item in values]
    except ValueError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_int_pair(text: str) -> tuple[int, int]:
    values = parse_int_list(text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected exactly 2 comma-separated ints")
    return values[0], values[1]


def format_token(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_instance_csv(path: Path, nodes: torch.Tensor) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in nodes.detach().cpu().tolist():
            writer.writerow({field: f"{float(value):.6f}" for field, value in zip(CSV_FIELDS, row)})


def count_instance_stats(nodes: torch.Tensor) -> dict[str, Any]:
    customer_nodes = nodes[1:]
    dynamic_customers = int((customer_nodes[:, 6] > 0).sum().item())
    tw_customers = int((customer_nodes[:, 3] > 0).sum().item())
    return {
        "node_count": int(nodes.size(0)),
        "customer_count": int(nodes.size(0) - 1),
        "dynamic_customers": dynamic_customers,
        "dynamic_ratio": round(dynamic_customers / max(1, nodes.size(0) - 1), 6),
        "tw_customers": tw_customers,
        "tw_ratio_realized": round(tw_customers / max(1, nodes.size(0) - 1), 6),
    }


def build_dataset_for_cell(
    *,
    seed: int,
    instances_per_cell: int,
    customers_count: int,
    vehicles_count: int,
    veh_capa: int,
    veh_speed: int,
    horizon: int,
    loc_range: tuple[int, int],
    dem_range: tuple[int, int],
    dur_range: tuple[int, int],
    tw_range: tuple[int, int],
    dod: float,
    tw_ratio: float,
    appear_early_ratio: float,
    min_cust_count: int | None,
) -> DVRPTW_Dataset:
    torch.manual_seed(seed)
    return DVRPTW_Dataset.generate(
        batch_size=instances_per_cell,
        cust_count=customers_count,
        veh_count=vehicles_count,
        veh_capa=veh_capa,
        veh_speed=veh_speed,
        min_cust_count=min_cust_count,
        cust_loc_range=loc_range,
        cust_dem_range=dem_range,
        horizon=horizon,
        cust_dur_range=dur_range,
        tw_ratio=tw_ratio,
        cust_tw_range=tw_range,
        dod=dod,
        d_early_ratio=appear_early_ratio,
    )


def generate_benchmark_data(args: argparse.Namespace) -> dict[str, Any]:
    datasets_root = args.datasets_root.resolve()
    if args.clean and datasets_root.exists():
        shutil.rmtree(datasets_root)
    ensure_dir(datasets_root)

    dod_values = args.dod_values
    tw_values = args.tw_values
    seeds = args.seeds

    cells: list[dict[str, Any]] = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    cell_limit = args.max_cells if args.max_cells is not None else None

    for cell_index, (dod, tw_ratio, seed) in enumerate(
        (dod, tw_ratio, seed)
        for seed in seeds
        for dod in dod_values
        for tw_ratio in tw_values
    ):
        if cell_limit is not None and cell_index >= cell_limit:
            break

        cell_id = f"dod_{format_token(dod)}__tw_{format_token(tw_ratio)}__seed_{seed}"
        cell_dir = datasets_root / f"dod_{format_token(dod)}" / f"tw_{format_token(tw_ratio)}" / f"seed_{seed}"
        ensure_dir(cell_dir)

        dataset = build_dataset_for_cell(
            seed=seed,
            instances_per_cell=args.instances_per_cell,
            customers_count=args.customers_count,
            vehicles_count=args.vehicles_count,
            veh_capa=args.veh_capa,
            veh_speed=args.veh_speed,
            horizon=args.horizon,
            loc_range=args.loc_range,
            dem_range=args.dem_range,
            dur_range=args.dur_range,
            tw_range=args.tw_range,
            dod=dod,
            tw_ratio=tw_ratio,
            appear_early_ratio=args.appear_early_ratio,
            min_cust_count=args.min_cust_count,
        )

        instance_rows: list[dict[str, Any]] = []
        for instance_index, nodes in enumerate(dataset.nodes):
            csv_name = f"instance_{instance_index:05d}.csv"
            csv_path = cell_dir / csv_name
            write_instance_csv(csv_path, nodes)

            stats = count_instance_stats(nodes)
            instance_rows.append(
                {
                    "instance_index": instance_index,
                    "csv_relpath": str(csv_path.relative_to(datasets_root)),
                    "csv_abspath": str(csv_path.resolve()),
                    **stats,
                }
            )

        cell_manifest = {
            "cell_id": cell_id,
            "cell_relpath": str(cell_dir.relative_to(datasets_root)),
            "seed": seed,
            "dod": dod,
            "tw_ratio": tw_ratio,
            "appear_early_ratio": args.appear_early_ratio,
            "instances_per_cell": args.instances_per_cell,
            "customers_count": args.customers_count,
            "vehicles_count": args.vehicles_count,
            "veh_capa": args.veh_capa,
            "veh_speed": args.veh_speed,
            "horizon": args.horizon,
            "loc_range": list(args.loc_range),
            "dem_range": list(args.dem_range),
            "dur_range": list(args.dur_range),
            "tw_range": list(args.tw_range),
            "min_cust_count": args.min_cust_count,
            "instances": instance_rows,
        }
        write_json(cell_dir / "cell_manifest.json", cell_manifest)

        cells.append(
            {
                "cell_id": cell_id,
                "cell_relpath": str(cell_dir.relative_to(datasets_root)),
                "seed": seed,
                "dod": dod,
                "tw_ratio": tw_ratio,
                "appear_early_ratio": args.appear_early_ratio,
                "instance_count": args.instances_per_cell,
                "csv_count": len(instance_rows),
                "cell_manifest": str((cell_dir / "cell_manifest.json").relative_to(datasets_root)),
            }
        )

    manifest = {
        "generated_at": generated_at,
        "datasets_root": str(datasets_root),
        "problem_type": "dvrptw",
        "customers_count": args.customers_count,
        "vehicles_count": args.vehicles_count,
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "horizon": args.horizon,
        "loc_range": list(args.loc_range),
        "dem_range": list(args.dem_range),
        "dur_range": list(args.dur_range),
        "tw_range": list(args.tw_range),
        "appear_early_ratio": args.appear_early_ratio,
        "instances_per_cell": args.instances_per_cell,
        "seeds": list(seeds),
        "dod_values": list(dod_values),
        "tw_values": list(tw_values),
        "min_cust_count": args.min_cust_count,
        "cells": cells,
    }
    write_json(datasets_root / "manifest.json", manifest)
    return manifest


def load_manifest(datasets_root: Path) -> dict[str, Any]:
    manifest_path = datasets_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_algorithm_specs(args: argparse.Namespace) -> list[AlgorithmSpec]:
    specs: list[AlgorithmSpec] = []
    if args.vectra_config_file is not None and args.vectra_model_weight is not None:
        specs.append(
            AlgorithmSpec(
                name="vectra",
                batch_script=args.vectra_batch_script.resolve(),
                config_file=args.vectra_config_file.resolve(),
                model_weight=args.vectra_model_weight.resolve(),
            )
        )
    if args.mardam_config_file is not None and args.mardam_model_weight is not None:
        specs.append(
            AlgorithmSpec(
                name="mardam",
                batch_script=args.mardam_batch_script.resolve(),
                config_file=args.mardam_config_file.resolve(),
                model_weight=args.mardam_model_weight.resolve(),
            )
        )
    return specs


def write_run_log(path: Path, *, command: list[str], return_code: int, stdout: str, stderr: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(shlex.quote(part) for part in command) + "\n\n")
        f.write("RETURN_CODE:\n")
        f.write(str(return_code) + "\n\n")
        f.write("STDOUT:\n")
        f.write(stdout)
        f.write("\n\nSTDERR:\n")
        f.write(stderr)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not summary_csv.exists():
        return rows

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"return_code"}:
                    cleaned[key] = _safe_int(value)
                elif key in {
                    "duration_sec",
                    "normalized_cost",
                    "raw_replay_cost",
                    "raw_total_cost",
                    "raw_distance",
                    "raw_late_time",
                    "raw_late_penalty",
                    "raw_skipped_penalty",
                    "normalized_total_cost",
                    "normalized_distance",
                    "normalized_late_time",
                    "normalized_late_penalty",
                    "normalized_skipped_penalty",
                }:
                    cleaned[key] = _safe_float(value)
                elif key in {
                    "total_skipped_customers",
                    "total_tw_violations",
                    "total_appearance_violations",
                    "active_customers",
                    "visited_customers",
                    "visit_steps",
                    "missing_count",
                    "duplicate_count",
                    "extra_count",
                }:
                    cleaned[key] = _safe_int(value)
                else:
                    cleaned[key] = value
            rows.append(cleaned)
    return rows


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return fmean(values), pstdev(values)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "duration_sec",
        "normalized_cost",
        "raw_replay_cost",
        "total_skipped_customers",
        "total_tw_violations",
        "total_appearance_violations",
        "active_customers",
        "visited_customers",
        "visit_steps",
        "missing_count",
        "duplicate_count",
        "extra_count",
        "raw_total_cost",
        "raw_distance",
        "raw_late_time",
        "raw_late_penalty",
        "raw_skipped_penalty",
        "normalized_total_cost",
        "normalized_distance",
        "normalized_late_time",
        "normalized_late_penalty",
        "normalized_skipped_penalty",
    ]

    summary: dict[str, Any] = {"row_count": len(rows)}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        field_mean, field_std = mean_std(values)
        summary[f"{field}_mean"] = field_mean
        summary[f"{field}_std"] = field_std
    return summary


def augment_rows(
    rows: list[dict[str, Any]],
    *,
    algorithm: AlgorithmSpec,
    cell_meta: dict[str, Any],
    batch_output_dir: Path,
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    cell_relpath = Path(str(cell_meta["cell_relpath"]))
    augmented: list[dict[str, Any]] = []
    for row in rows:
        dataset_relpath = Path(str(row.get("dataset_relpath", "")))
        merged = dict(row)
        merged.update(
            {
                "algorithm": algorithm.name,
                "cell_id": cell_meta["cell_id"],
                "cell_relpath": str(cell_relpath),
                "benchmark_relpath": str(cell_relpath / dataset_relpath) if str(dataset_relpath) else str(cell_relpath),
                "dod": cell_meta["dod"],
                "tw_ratio": cell_meta["tw_ratio"],
                "appear_early_ratio": cell_meta.get("appear_early_ratio"),
                "seed": cell_meta["seed"],
                "batch_output_dir": str(batch_output_dir),
                "benchmark_root": str(benchmark_root),
            }
        )
        augmented.append(merged)
    return augmented


def run_algorithm_on_cell(
    *,
    algorithm: AlgorithmSpec,
    cell_meta: dict[str, Any],
    cell_dir: Path,
    run_dir: Path,
    benchmark_root: Path,
    python_executable: str,
    vehicles_count: int,
    veh_capa: int,
    veh_speed: float,
    greedy: bool,
    sample: bool,
    verify_rollouts: int,
    no_verify_routes: bool,
    max_print_instances: int,
    fail_fast: bool,
) -> dict[str, Any]:
    ensure_dir(run_dir)
    cmd = [
        python_executable,
        str(algorithm.batch_script),
        "--datasets-root",
        str(cell_dir),
        "--problem-type",
        "dvrptw",
        "--config-file",
        str(algorithm.config_file),
        "--model-weight",
        str(algorithm.model_weight),
        "--vehicles-count",
        str(vehicles_count),
        "--veh-capa",
        str(veh_capa),
        "--veh-speed",
        str(veh_speed),
        "--max-print-instances",
        str(max_print_instances),
        "--verify-rollouts",
        str(verify_rollouts),
        "--file-glob",
        "*.csv",
        "--output-dir",
        str(run_dir),
    ]
    if no_verify_routes:
        cmd.append("--no-verify-routes")
    if sample:
        cmd.append("--sample")
    elif greedy:
        cmd.append("--greedy")

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_sec = time.perf_counter() - start

    write_run_log(
        run_dir / "run.log",
        command=cmd,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    row: dict[str, Any] = {
        "algorithm": algorithm.name,
        "cell_id": cell_meta["cell_id"],
        "cell_relpath": cell_meta["cell_relpath"],
        "dod": cell_meta["dod"],
        "tw_ratio": cell_meta["tw_ratio"],
        "appear_early_ratio": cell_meta.get("appear_early_ratio"),
        "seed": cell_meta["seed"],
        "cell_dir": str(cell_dir),
        "run_dir": str(run_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "duration_sec": round(duration_sec, 6),
        "command": " ".join(shlex.quote(part) for part in cmd),
    }

    if proc.returncode != 0:
        row["error_message"] = (proc.stderr or proc.stdout or "").strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "inference failed"
        return row

    rows = read_summary_rows(summary_csv)
    if not rows:
        row["status"] = "failed"
        row["error_message"] = "batch inference finished without summary rows"
        return row

    rows = augment_rows(
        rows,
        algorithm=algorithm,
        cell_meta=cell_meta,
        batch_output_dir=run_dir,
        benchmark_root=benchmark_root,
    )
    row.update(summarize_rows(rows))
    row["instance_rows"] = len(rows)
    row["error_message"] = ""
    row["rows"] = rows
    return row


def build_master_tables(run_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    master_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for record in run_records:
        if record.get("status") != "ok":
            continue

        rows = record.get("rows", [])
        for row in rows:
            master_rows.append(dict(row))

        cell_rows.append(
            {
                key: record.get(key)
                for key in [
                    "algorithm",
                    "cell_id",
                    "cell_relpath",
                    "dod",
                    "tw_ratio",
                    "appear_early_ratio",
                    "seed",
                    "cell_dir",
                    "run_dir",
                    "summary_csv",
                    "summary_json",
                    "status",
                    "return_code",
                    "duration_sec",
                    "instance_rows",
                    "row_count",
                    "normalized_cost_mean",
                    "normalized_cost_std",
                    "raw_replay_cost_mean",
                    "raw_replay_cost_std",
                    "total_skipped_customers_mean",
                    "total_skipped_customers_std",
                    "total_tw_violations_mean",
                    "total_tw_violations_std",
                    "total_appearance_violations_mean",
                    "total_appearance_violations_std",
                    "active_customers_mean",
                    "active_customers_std",
                    "visited_customers_mean",
                    "visited_customers_std",
                    "missing_count_mean",
                    "missing_count_std",
                    "duplicate_count_mean",
                    "duplicate_count_std",
                    "extra_count_mean",
                    "extra_count_std",
                    "raw_total_cost_mean",
                    "raw_total_cost_std",
                    "normalized_total_cost_mean",
                    "normalized_total_cost_std",
                    "normalized_distance_mean",
                    "normalized_distance_std",
                    "normalized_late_penalty_mean",
                    "normalized_late_penalty_std",
                    "normalized_skipped_penalty_mean",
                    "normalized_skipped_penalty_std",
                ]
            }
        )

    return master_rows, cell_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate a dynamic DVRPTW benchmark grid.")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets/dvrptw_dynamic_grid"))
    parser.add_argument("--run-root", type=Path, default=Path("output/dynamic_benchmark"))
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--customers-count", type=int, default=50)
    parser.add_argument("--vehicles-count", type=int, default=3)
    parser.add_argument("--veh-capa", type=int, default=200)
    parser.add_argument("--veh-speed", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--loc-range", type=parse_int_pair, default=(0, 101))
    parser.add_argument("--dem-range", type=parse_int_pair, default=(5, 41))
    parser.add_argument("--dur-range", type=parse_int_pair, default=(10, 31))
    parser.add_argument("--tw-range", type=parse_int_pair, default=(30, 91))
    parser.add_argument("--appear-early-ratio", type=float, default=0.5)
    parser.add_argument("--min-cust-count", type=int, default=None)
    parser.add_argument("--instances-per-cell", type=int, default=32)
    parser.add_argument("--dod-values", type=parse_float_list, default=list(DEFAULT_DOD_VALUES))
    parser.add_argument("--tw-values", "--tw-ratios", dest="tw_values", type=parse_float_list, default=list(DEFAULT_TW_RATIOS))
    parser.add_argument("--seeds", type=parse_int_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--verify-rollouts", type=int, default=1)
    parser.add_argument("--no-verify-routes", action="store_true")
    parser.add_argument("--max-print-instances", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--vectra-batch-script", type=Path, default=Path("script/infer_all_datasets.py"))
    parser.add_argument("--mardam-batch-script", type=Path, default=Path("script/infer_all_datasets_mardam.py"))
    parser.add_argument("--vectra-config-file", type=Path, default=None)
    parser.add_argument("--vectra-model-weight", type=Path, default=None)
    parser.add_argument("--mardam-config-file", type=Path, default=None)
    parser.add_argument("--mardam-model-weight", type=Path, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.generate_only and args.run_only:
        raise ValueError("--generate-only and --run-only are mutually exclusive")

    benchmark_manifest = load_manifest(args.datasets_root) if args.run_only else generate_benchmark_data(args)

    if args.generate_only:
        print(f"Generated dataset grid at {args.datasets_root.resolve()}")
        return 0

    algorithm_specs = collect_algorithm_specs(args)
    if not algorithm_specs:
        print("No algorithm checkpoints provided; data generation completed only.")
        return 0

    run_root = args.run_root.resolve()
    ensure_dir(run_root)

    run_records: list[dict[str, Any]] = []
    cells = benchmark_manifest.get("cells", [])
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    for algorithm in algorithm_specs:
        if not algorithm.batch_script.exists():
            raise FileNotFoundError(f"batch script not found: {algorithm.batch_script}")
        if not algorithm.config_file.exists():
            raise FileNotFoundError(f"config file not found: {algorithm.config_file}")
        if not algorithm.model_weight.exists():
            raise FileNotFoundError(f"model weight not found: {algorithm.model_weight}")

        for cell_meta in cells:
            cell_dir = args.datasets_root.resolve() / str(cell_meta["cell_relpath"])
            run_dir = run_root / algorithm.name / str(cell_meta["cell_relpath"])
            print(f"[{algorithm.name}] {cell_meta['cell_id']} -> {run_dir}")
            record = run_algorithm_on_cell(
                algorithm=algorithm,
                cell_meta=cell_meta,
                cell_dir=cell_dir,
                run_dir=run_dir,
                benchmark_root=args.datasets_root.resolve(),
                python_executable=args.python_executable,
                vehicles_count=args.vehicles_count,
                veh_capa=args.veh_capa,
                veh_speed=args.veh_speed,
                greedy=args.greedy,
                sample=args.sample,
                verify_rollouts=args.verify_rollouts,
                no_verify_routes=args.no_verify_routes,
                max_print_instances=args.max_print_instances,
                fail_fast=args.fail_fast,
            )
            run_records.append(record)
            if record.get("status") != "ok" and args.fail_fast:
                break

    master_rows, cell_rows = build_master_tables(run_records)
    master_csv = run_root / "master_summary.csv"
    cell_csv = run_root / "cell_summary.csv"
    master_json = run_root / "master_summary.json"
    overview_json = run_root / "overview.json"

    write_csv(master_csv, master_rows)
    write_csv(cell_csv, cell_rows)
    write_json(master_json, {"manifest": benchmark_manifest, "rows": master_rows})

    overview: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_root": str(args.datasets_root.resolve()),
        "run_root": str(run_root),
        "run_records": run_records,
        "master_row_count": len(master_rows),
        "cell_row_count": len(cell_rows),
    }
    write_json(overview_json, overview)

    print(f"Benchmark root : {args.datasets_root.resolve()}")
    print(f"Run root       : {run_root}")
    print(f"Master CSV     : {master_csv}")
    print(f"Cell CSV       : {cell_csv}")
    print(f"Master rows    : {len(master_rows)}")
    print(f"Cell rows      : {len(cell_rows)}")
    return 0 if all(record.get("status") == "ok" for record in run_records if record.get("status") is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())