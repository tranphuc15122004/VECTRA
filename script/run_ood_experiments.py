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
from statistics import mean, stdev
from typing import Any


VECTRA_MODELS = {
    "vectra": ("MODEL/infer.py", "data/vectra/args.json", "data/vectra/chkpt_best.pyth"),
    "b0": ("MODEL/infer.py", "data/_Ablation/B0/args.json", "data/_Ablation/B0/chkpt_best.pyth"),
    "b1": ("MODEL/infer.py", "data/_Ablation/B1/args.json", "data/_Ablation/B1/chkpt_best.pyth"),
    "b3": ("MODEL/infer.py", "data/_Ablation/B3/args.json", "data/_Ablation/B3/chkpt_best.pyth"),
    "b5": ("MODEL/infer.py", "data/_Ablation/B5/args.json", "data/_Ablation/B5/chkpt_best.pyth"),
    "edgeoff": ("MODEL/infer.py", "data/_Ablation/Edgeoff/args.json", "data/_Ablation/Edgeoff/chkpt_best.pyth"),
    "no_ownership": ("MODEL/infer.py", "output/ablation/no_ownership/seed42/args.json", "output/ablation/no_ownership/seed42/chkpt_best.pyth"),
    "no_lookahead": ("MODEL/infer.py", "output/ablation/no_lookahead/seed42/args.json", "output/ablation/no_lookahead/seed42/chkpt_best.pyth"),
}
MARDAM_MODEL = ("MODEL/infer_mardam.py", "data/mardam/args.json", "data/mardam/chkpt_best.pyth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local checkpoints on ID/OOD .pyth datasets.")
    parser.add_argument("--datasets-dir", type=Path, default=Path("data/test_sets"))
    parser.add_argument("--file-glob", type=str, default="test_dvrptw_*_*.pyth")
    parser.add_argument("--output-dir", type=Path, default=Path("output/ood_eval"))
    parser.add_argument("--models", type=str, default="vectra,mardam,b0,b1,b3,b5,edgeoff")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--no-verify-routes", action="store_true")
    parser.add_argument("--no-normalize", action="store_true", help="Use if generated datasets were already normalized.")
    parser.add_argument("--max-print-instances", type=int, default=0)
    parser.add_argument("--verify-rollouts", type=int, default=1)
    parser.add_argument("--device-cpu", action="store_true", help="Forward --no-cuda for VECTRA/MARDAM inference.")
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def dataset_name(path: Path) -> str:
    name = path.stem
    if name.startswith("test_dvrptw_"):
        name = name[len("test_dvrptw_"):]
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        name = "_".join(parts[:-1])
    return name


def resolve_model(name: str) -> tuple[str, str, str] | None:
    key = name.strip().lower()
    if key == "mardam":
        return MARDAM_MODEL
    return VECTRA_MODELS.get(key)


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    costs = [float(v) for v in payload.get("normalized_costs", [])]
    raw_costs = [float(v) for v in payload.get("raw_replay_costs", [])]
    raw_components = payload.get("raw_cost_components", [])
    norm_components = payload.get("normalized_cost_components", [])

    def component_values(key: str, source: list[dict[str, Any]]) -> list[float]:
        values = []
        for item in source:
            value = maybe_float(item.get(key))
            if value is not None:
                values.append(value)
        return values

    cost_mean, cost_std = mean_std(costs)
    raw_mean, raw_std = mean_std(raw_costs)
    return {
        "normalized_cost_mean": cost_mean,
        "normalized_cost_std": cost_std,
        "raw_replay_cost_mean": raw_mean,
        "raw_replay_cost_std": raw_std,
        "instances": len(costs),
        "total_skipped_customers": payload.get("total_skipped_customers"),
        "total_tw_violations": payload.get("total_tw_violations"),
        "total_appearance_violations": payload.get("total_appearance_violations"),
        "normalized_distance_mean": mean_std(component_values("distance", norm_components))[0],
        "normalized_late_time_mean": mean_std(component_values("late_time", norm_components))[0],
        "normalized_late_penalty_mean": mean_std(component_values("late_penalty", norm_components))[0],
        "normalized_skipped_penalty_mean": mean_std(component_values("skipped_penalty", norm_components))[0],
        "raw_distance_mean": mean_std(component_values("distance", raw_components))[0],
        "raw_late_time_mean": mean_std(component_values("late_time", raw_components))[0],
        "raw_late_penalty_mean": mean_std(component_values("late_penalty", raw_components))[0],
        "raw_skipped_penalty_mean": mean_std(component_values("skipped_penalty", raw_components))[0],
    }


def run_one(args: argparse.Namespace, model_name: str, dataset_path: Path) -> dict[str, Any]:
    model = resolve_model(model_name)
    if model is None:
        return {"model": model_name, "dataset": dataset_name(dataset_path), "status": "skipped", "error_message": "unknown model"}

    infer_script, config_file, model_weight = model
    for path in (infer_script, config_file, model_weight):
        if not Path(path).exists():
            return {"model": model_name, "dataset": dataset_name(dataset_path), "status": "skipped", "error_message": f"missing {path}"}

    result_json = args.output_dir / model_name / f"{dataset_path.stem}.infer.json"
    log_path = args.output_dir / model_name / f"{dataset_path.stem}.log"
    result_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_executable,
        infer_script,
        "--problem-type",
        "dvrptw",
        "--config-file",
        config_file,
        "--model-weight",
        model_weight,
        "--data-file",
        str(dataset_path),
        "--greedy",
        "--max-print-instances",
        str(args.max_print_instances),
        "--verify-rollouts",
        str(args.verify_rollouts),
        "--save-json",
        str(result_json),
    ]
    if args.no_verify_routes:
        cmd.append("--no-verify-routes")
    if args.no_normalize:
        cmd.append("--no-normalize")
    if args.device_cpu:
        cmd.append("--no-cuda")

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start
    log_path.write_text(
        "COMMAND:\n{}\n\nRETURN_CODE:\n{}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}".format(
            " ".join(shlex.quote(part) for part in cmd),
            proc.returncode,
            proc.stdout,
            proc.stderr,
        ),
        encoding="utf-8",
    )

    row: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name(dataset_path),
        "dataset_path": str(dataset_path),
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "duration_sec": round(duration, 6),
        "result_json": str(result_json),
        "run_log": str(log_path),
        "command": " ".join(shlex.quote(part) for part in cmd),
        "error_message": "",
    }
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error_message"] = lines[-1] if lines else "inference failed"
        return row
    if not result_json.exists():
        row["status"] = "failed"
        row["error_message"] = "missing save-json output"
        return row

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    row.update(parse_payload(payload))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    id_rows = [r for r in rows if r.get("dataset") == "id_n50m3" and r.get("status") == "ok"]
    id_cost = {r["model"]: maybe_float(r.get("normalized_cost_mean")) for r in id_rows}

    lines = ["# OOD Evaluation Report", "", f"Generated: `{datetime.now().isoformat(timespec='seconds')}`", ""]
    lines.append("| Model | Dataset | Cost mean | Cost std | Degradation vs ID | TW viol | Skipped | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda r: (str(r.get("model")), str(r.get("dataset")))):
        cost = maybe_float(row.get("normalized_cost_mean"))
        base = id_cost.get(row.get("model"))
        degr = "" if cost is None or base in (None, 0) else f"{(cost - base) / abs(base) * 100.0:.2f}%"
        lines.append(
            "| {model} | {dataset} | {cost} | {std} | {degr} | {tw} | {skip} | {status} |".format(
                model=row.get("model", ""),
                dataset=row.get("dataset", ""),
                cost="" if cost is None else f"{cost:.4f}",
                std="" if maybe_float(row.get("normalized_cost_std")) is None else f"{maybe_float(row.get('normalized_cost_std')):.4f}",
                degr=degr,
                tw=row.get("total_tw_violations", ""),
                skip=row.get("total_skipped_customers", ""),
                status=row.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(args.datasets_dir.glob(args.file_glob))
    if not datasets:
        raise FileNotFoundError(f"No datasets matched {args.datasets_dir / args.file_glob}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for dataset_path in datasets:
        for model in models:
            print(f"[OOD] model={model} dataset={dataset_path}")
            rows.append(run_one(args, model, dataset_path))

    summary_csv = args.output_dir / "ood_summary.csv"
    report_md = args.output_dir / "ood_report.md"
    manifest_json = args.output_dir / "ood_manifest.json"
    write_csv(summary_csv, rows)
    write_report(report_md, rows)
    manifest_json.write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "datasets": [str(p) for p in datasets],
            "models": models,
            "summary_csv": str(summary_csv),
            "report_md": str(report_md),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
