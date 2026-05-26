#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_RESULTS_ROOT = Path("Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark")
DEFAULT_TABLE_ROOT = Path("experimental_results_csv")

METRICS = [
    "duration_sec",
    "normalized_cost",
    "normalized_total_cost",
    "normalized_distance",
    "normalized_late_time",
    "normalized_late_penalty",
    "normalized_skipped_penalty",
    "raw_replay_cost",
    "raw_total_cost",
    "raw_distance",
    "raw_late_time",
    "raw_late_penalty",
    "raw_skipped_penalty",
    "total_skipped_customers",
    "total_tw_violations",
    "total_appearance_violations",
    "active_customers",
    "visited_customers",
    "visit_steps",
    "missing_count",
    "duplicate_count",
    "extra_count",
    "step_diagnostics_count",
]

DISPLAY_NAMES = {
    "vectra": "COAST/VECTRA",
    "mardam": "MARDAM",
    "am": "AM",
    "polynet": "PolyNet",
    "b0": "B0",
    "b1": "B1",
    "b3": "B3",
    "b5": "B5",
    "edgeoff": "EdgeOff",
    "no_ownership": "NoOwnership",
    "noownership": "NoOwnership",
    "no_lookahead": "NoLookahead",
    "nolookahead": "NoLookahead",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-ready experimental artifacts from raw per-instance "
            "inference summaries and audit the current result tables."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--summary-csv",
        action="append",
        default=[],
        help=(
            "Raw summary CSV to include. Use either PATH or algorithm=PATH. "
            "Can be repeated. Defaults to --results-root/master_summary.csv."
        ),
    )
    parser.add_argument(
        "--discover-nested",
        action="store_true",
        help=(
            "Include nested summaries under "
            "{results_root}/{algorithm}/dod_*/tw_*/seed_*/summary.csv."
        ),
    )
    parser.add_argument(
        "--include-summary-glob",
        action="append",
        default=[],
        help="Additional glob relative to --results-root for summary CSVs. Can be repeated.",
    )
    parser.add_argument(
        "--external-summary",
        action="append",
        default=[],
        help=(
            "External summary in TYPE=PATH form, e.g. am=output/am/aggregated.csv. "
            "Used for AM/PolyNet infer_batch outputs and other compatible CSVs."
        ),
    )
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_TABLE_ROOT / "Benchmark.csv")
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_TABLE_ROOT / "chi_tiết_benchmark.csv")
    parser.add_argument("--pyth-csv", type=Path, default=DEFAULT_TABLE_ROOT / "pyth.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ood-summary", type=Path, default=None, help="Optional ood_summary.csv to embed in paper_statistics.md.")
    parser.add_argument("--behavior-summary", type=Path, default=None, help="Optional behavior_summary.csv to embed in paper_statistics.md.")
    parser.add_argument("--base-algorithm", type=str, default="vectra")
    parser.add_argument("--metric", type=str, default="normalized_cost")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260526)
    return parser.parse_args()


def normalize_algorithm(name: Any) -> str:
    raw = str(name or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"", "coast", "coast_vectra", "vectra_full"}:
        return "vectra"
    if raw in {"edge_off", "edge-off"}:
        return "edgeoff"
    if raw in {"poly_net", "poly-net"}:
        return "polynet"
    return raw


def display_algorithm(name: str) -> str:
    return DISPLAY_NAMES.get(normalize_algorithm(name), str(name))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def fmt_num(value: Any, digits: int = 4) -> str:
    num = maybe_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def parse_summary_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        alg, path = spec.split("=", 1)
        return normalize_algorithm(alg), Path(path)
    return None, Path(spec)


def infer_algorithm_from_summary_path(path: Path, results_root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(results_root.resolve())
        parts = rel.parts
    except Exception:
        parts = path.parts
    if len(parts) >= 5 and parts[-1] == "summary.csv":
        return normalize_algorithm(parts[0])
    parent = normalize_algorithm(path.parent.name)
    if parent not in {"seed_42", "seed42", "paper_ready", "dynamic_benchmark"}:
        return parent
    return None


def discover_nested_summary_specs(results_root: Path) -> list[tuple[str | None, Path]]:
    specs = []
    pattern = "* /dod_* /tw_* /seed_* /summary.csv".replace(" ", "")
    for path in sorted(results_root.glob(pattern)):
        algorithm = infer_algorithm_from_summary_path(path, results_root)
        if algorithm:
            specs.append((algorithm, path))
    return specs


def glob_summary_specs(results_root: Path, patterns: list[str]) -> list[tuple[str | None, Path]]:
    specs = []
    for pattern in patterns:
        for path in sorted(results_root.glob(pattern)):
            if path.is_file():
                specs.append((infer_algorithm_from_summary_path(path, results_root), path))
    return specs


def default_summary_specs(results_root: Path, discover_nested: bool = False) -> list[tuple[str | None, Path]]:
    if discover_nested:
        nested = discover_nested_summary_specs(results_root)
        if nested:
            return nested
    path = results_root / "master_summary.csv"
    return [(None, path)] if path.exists() else []


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extract_cli_value(command: str, flag: str) -> str | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for idx, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[idx + 1]
    return None


def parse_cell_from_path(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    dod = re.search(r"dod_([0-9]+)p([0-9]+)", text)
    tw = re.search(r"tw_([0-9]+)p([0-9]+)", text)
    seed = re.search(r"seed[_-]?([0-9]+)", text)
    if dod:
        out["dod"] = f"{int(dod.group(1))}.{dod.group(2)}"
    if tw:
        out["tw_ratio"] = f"{int(tw.group(1))}.{tw.group(2)}"
    if seed:
        out["seed"] = seed.group(1)
    return out


def _instance_name(row: dict[str, Any]) -> str:
    for key in ("benchmark_relpath", "dataset_relpath", "file", "file_path", "dataset_abspath"):
        value = str(row.get(key, "")).strip()
        if value:
            return str(Path(value).name)
    return str(row.get("result_json", "")).strip()


def pair_key(row: dict[str, Any]) -> str:
    dod = str(row.get("dod", "")).strip()
    tw = str(row.get("tw_ratio", "")).strip()
    seed = str(row.get("seed", "")).strip()
    instance = _instance_name(row)
    if dod or tw or seed:
        return "|".join([dod, tw, seed, instance])
    for key in ("benchmark_relpath", "dataset_relpath", "dataset_abspath"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return str(row.get("result_json", "")).strip()


def load_summary(path: Path, forced_algorithm: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    status_counts = Counter()
    commands: list[str] = []
    config_files = Counter()
    model_weights = Counter()
    summary_meta = parse_cell_from_path(str(path))

    for row in rows:
        clean = dict(row)
        algorithm = forced_algorithm or clean.get("algorithm") or path.parent.name
        clean["algorithm"] = normalize_algorithm(algorithm)
        clean["algorithm_display"] = display_algorithm(clean["algorithm"])
        clean["source_summary_csv"] = str(path)

        meta = {}
        for source in (
            clean.get("benchmark_relpath"),
            clean.get("dataset_relpath"),
            clean.get("dataset_abspath"),
            clean.get("result_json"),
            clean.get("run_log"),
            clean.get("command"),
            str(path),
        ):
            meta.update(parse_cell_from_path(str(source or "")))
        meta = {**summary_meta, **meta}
        for key, value in meta.items():
            clean.setdefault(key, value)
            if not str(clean.get(key, "")).strip():
                clean[key] = value
        clean["instance_name"] = _instance_name(clean)
        clean["row_key"] = "|".join([clean["algorithm"], pair_key(clean)])
        clean["pair_key"] = pair_key(clean)

        command = str(clean.get("command", "")).strip()
        if command:
            commands.append(command)
            config = extract_cli_value(command, "--config-file")
            weight = extract_cli_value(command, "--model-weight")
            if config:
                config_files[config] += 1
            if weight:
                model_weights[weight] += 1

        status_counts[str(clean.get("status", "")).strip() or "unknown"] += 1
        out.append(clean)

    meta = {
        "path": str(path),
        "forced_algorithm": forced_algorithm,
        "row_count": len(rows),
        "algorithms": sorted({r["algorithm"] for r in out}),
        "status_counts": dict(status_counts),
        "config_files": dict(config_files),
        "model_weights": dict(model_weights),
        "sample_command": commands[0] if commands else None,
    }
    return out, meta


def load_external_summary(path: Path, forced_algorithm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    status_counts = Counter()

    for row in rows:
        clean = dict(row)
        clean["algorithm"] = normalize_algorithm(row.get("model") or forced_algorithm)
        clean["algorithm_display"] = display_algorithm(clean["algorithm"])
        clean["source_summary_csv"] = str(path)
        clean["status"] = clean.get("status") or "ok"
        clean["dataset_relpath"] = clean.get("dataset_relpath") or clean.get("file") or clean.get("file_path") or ""
        clean["dataset_abspath"] = clean.get("dataset_abspath") or clean.get("file_path") or ""
        clean["normalized_cost"] = clean.get("normalized_cost") or clean.get("mean_cost")
        clean["raw_replay_cost"] = clean.get("raw_replay_cost") or clean.get("mean_raw_replay_cost")
        clean["duration_sec"] = clean.get("duration_sec") or clean.get("elapsed_seconds")
        clean["total_skipped_customers"] = clean.get("total_skipped_customers") or clean.get("total_skipped")
        clean["normalized_distance"] = clean.get("normalized_distance") or clean.get("mean_norm_distance")
        clean["normalized_late_time"] = clean.get("normalized_late_time") or clean.get("mean_norm_late_time")
        clean["normalized_late_penalty"] = clean.get("normalized_late_penalty") or clean.get("mean_norm_late_penalty")
        clean["raw_distance"] = clean.get("raw_distance") or clean.get("mean_distance")
        clean["raw_late_time"] = clean.get("raw_late_time") or clean.get("mean_late_time")
        clean["raw_late_penalty"] = clean.get("raw_late_penalty") or clean.get("mean_late_penalty")
        clean["raw_skipped_orders"] = clean.get("raw_skipped_orders") or clean.get("mean_skipped_orders")
        clean["raw_skipped_penalty"] = clean.get("raw_skipped_penalty") or clean.get("mean_skipped_penalty")

        meta = {}
        for source in (
            clean.get("dataset_relpath"),
            clean.get("dataset_abspath"),
            clean.get("file_path"),
            clean.get("file"),
            str(path),
        ):
            meta.update(parse_cell_from_path(str(source or "")))
        for key, value in meta.items():
            if not str(clean.get(key, "")).strip():
                clean[key] = value

        clean["instance_name"] = _instance_name(clean)
        clean["row_key"] = "|".join([clean["algorithm"], pair_key(clean)])
        clean["pair_key"] = pair_key(clean)
        status_counts[str(clean.get("status", "")).strip() or "unknown"] += 1
        out.append(clean)

    meta = {
        "path": str(path),
        "forced_algorithm": forced_algorithm,
        "external_summary": True,
        "row_count": len(rows),
        "algorithms": sorted({r["algorithm"] for r in out}),
        "status_counts": dict(status_counts),
        "config_files": {},
        "model_weights": {},
        "sample_command": None,
    }
    return out, meta


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_cell_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("algorithm", "")),
            str(row.get("dod", "")),
            str(row.get("tw_ratio", "")),
            str(row.get("appear_early_ratio", "")),
            str(row.get("seed", "")),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for (algorithm, dod, tw_ratio, appear_early_ratio, seed), items in sorted(groups.items()):
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "algorithm_display": display_algorithm(algorithm),
            "dod": dod,
            "tw_ratio": tw_ratio,
            "appear_early_ratio": appear_early_ratio,
            "seed": seed,
            "row_count": len(items),
        }
        for metric in METRICS:
            values = [v for v in (maybe_float(item.get(metric)) for item in items) if v is not None]
            avg, sd = mean_std(values)
            record[f"{metric}_mean"] = "" if avg is None else avg
            record[f"{metric}_std"] = "" if sd is None else sd
        out.append(record)
    return out


def build_detail_metrics(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in cell_rows:
        out.append({
            "algorithm": row.get("algorithm"),
            "algorithm_display": row.get("algorithm_display"),
            "dod": row.get("dod"),
            "tw_ratio": row.get("tw_ratio"),
            "instances": row.get("row_count"),
            "travel_mean": row.get("normalized_distance_mean"),
            "travel_std": row.get("normalized_distance_std"),
            "skipped_mean": row.get("total_skipped_customers_mean"),
            "skipped_std": row.get("total_skipped_customers_std"),
            "late_time_mean": row.get("normalized_late_time_mean"),
            "late_time_std": row.get("normalized_late_time_std"),
            "late_penalty_mean": row.get("normalized_late_penalty_mean"),
            "late_penalty_std": row.get("normalized_late_penalty_std"),
            "tw_violations_mean": row.get("total_tw_violations_mean"),
            "tw_violations_std": row.get("total_tw_violations_std"),
            "appearance_violations_mean": row.get("total_appearance_violations_mean"),
            "appearance_violations_std": row.get("total_appearance_violations_std"),
        })
    return out


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    low_idx = int(0.025 * (len(means) - 1))
    high_idx = int(0.975 * (len(means) - 1))
    return means[low_idx], means[high_idx]


def build_significance(
    rows: list[dict[str, Any]],
    base_algorithm: str,
    metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    base_algorithm = normalize_algorithm(base_algorithm)
    by_alg: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = str(row.get("pair_key", ""))
        value = maybe_float(row.get(metric))
        if key and value is not None:
            by_alg[str(row.get("algorithm"))][key] = row

    base = by_alg.get(base_algorithm, {})
    out = []
    for algorithm in sorted(by_alg):
        if algorithm == base_algorithm:
            continue
        keys = sorted(set(base) & set(by_alg[algorithm]))
        base_values = []
        other_values = []
        diffs = []
        pct_diffs = []
        for key in keys:
            base_value = maybe_float(base[key].get(metric))
            other_value = maybe_float(by_alg[algorithm][key].get(metric))
            if base_value is None or other_value is None:
                continue
            base_values.append(base_value)
            other_values.append(other_value)
            diff = other_value - base_value
            diffs.append(diff)
            if abs(base_value) > 1e-12:
                pct_diffs.append(diff / abs(base_value) * 100.0)

        ci_low, ci_high = bootstrap_ci(diffs, bootstrap_samples, bootstrap_seed)
        pct_low, pct_high = bootstrap_ci(pct_diffs, bootstrap_samples, bootstrap_seed + 17)
        out.append({
            "base_algorithm": base_algorithm,
            "base_display": display_algorithm(base_algorithm),
            "other_algorithm": algorithm,
            "other_display": display_algorithm(algorithm),
            "metric": metric,
            "n_pairs": len(diffs),
            "base_mean": "" if not base_values else mean(base_values),
            "other_mean": "" if not other_values else mean(other_values),
            "mean_gap_other_minus_base": "" if not diffs else mean(diffs),
            "mean_gap_pct": "" if not pct_diffs else mean(pct_diffs),
            "ci95_low": "" if ci_low is None else ci_low,
            "ci95_high": "" if ci_high is None else ci_high,
            "ci95_pct_low": "" if pct_low is None else pct_low,
            "ci95_pct_high": "" if pct_high is None else pct_high,
            "other_worse_count": sum(1 for d in diffs if d > 0),
            "other_better_count": sum(1 for d in diffs if d < 0),
            "tie_count": sum(1 for d in diffs if d == 0),
            "status": "ok" if diffs else "missing_raw_pairs",
        })
    return out


def audit_detail_csv(path: Path) -> list[str]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"Detail metrics CSV not found: {path}")
        return warnings
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    metric_cells = 0
    for row in rows[2:]:
        metric_cells += sum(1 for cell in row[1:] if str(cell).strip())
    if metric_cells == 0:
        warnings.append(
            f"{path} has scenario names but no travel/skipped/late metric values; "
            "do not use it as a paper table until regenerated."
        )
    return warnings


def audit_benchmark_csv(path: Path) -> list[str]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"Benchmark CSV not found: {path}")
        return warnings
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        warnings.append(f"Benchmark CSV is empty: {path}")
        return warnings
    header = rows[0]
    for alg in ("LKH3", "Ortools"):
        if alg in header:
            idx = header.index(alg)
            count = sum(1 for row in rows[1:] if idx < len(row) and str(row[idx]).strip())
            if count == 0:
                warnings.append(f"{path} has an empty {alg} column; avoid OR/LKH claims until rerun.")
    return warnings


def audit_pyth_csv(path: Path) -> list[str]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"pyth CSV not found: {path}")
        return warnings
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    dataset_rows = {row[0]: row for row in rows if row and str(row[0]).startswith("dvrptw_")}
    n20 = dataset_rows.get("dvrptw_n20m1_10240")
    n50 = dataset_rows.get("dvrptw_n50m3_10240")
    if n20 and n50 and len(n20) > 1 and len(n50) > 1 and n20[1] == n50[1]:
        warnings.append(
            f"{path} has identical VECTRA(s) values for n20m1 and n50m3; "
            "verify this before citing the sampling table."
        )
    return warnings


def build_manifest(
    source_meta: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    audit_warnings: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    algorithms = sorted({str(row.get("algorithm")) for row in rows})
    cells = sorted({
        (str(row.get("algorithm")), str(row.get("dod")), str(row.get("tw_ratio")), str(row.get("seed")))
        for row in rows
    })
    verification_disabled = sum(
        1 for row in rows
        if "--no-verify-routes" in str(row.get("command", ""))
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(args.results_root),
        "output_dir": str(args.output_dir),
        "metric": args.metric,
        "base_algorithm": normalize_algorithm(args.base_algorithm),
        "source_summaries": source_meta,
        "row_count": len(rows),
        "algorithms": algorithms,
        "algorithm_count": len(algorithms),
        "cell_count": len(cells),
        "verification_disabled_rows": verification_disabled,
        "audit_warnings": audit_warnings,
    }


def aggregate_by_algorithm(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("algorithm"))].append(row)
    out = []
    for algorithm, items in sorted(groups.items()):
        values = [v for v in (maybe_float(item.get(metric)) for item in items) if v is not None]
        avg, sd = mean_std(values)
        out.append({
            "algorithm": algorithm,
            "algorithm_display": display_algorithm(algorithm),
            "rows": len(items),
            "metric": metric,
            "mean": avg,
            "std": sd,
            "ok_rows": sum(1 for item in items if str(item.get("status", "")).lower() == "ok"),
        })
    return out


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("row_key") or "|".join([str(row.get("algorithm", "")), str(row.get("pair_key", ""))]))
        deduped[key] = row
    return list(deduped.values())


def _has_algorithm(rows: list[dict[str, Any]], algorithm: str) -> bool:
    target = normalize_algorithm(algorithm)
    return any(normalize_algorithm(row.get("algorithm")) == target for row in rows)


def write_paper_statistics(
    path: Path,
    manifest: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]] | None = None,
    behavior_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# COAST/VECTRA Experimental Report")
    lines.append("")
    lines.append(f"Generated: `{manifest['created_at']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Raw per-instance rows: **{manifest['row_count']}**")
    lines.append(f"- Algorithms with raw rows: **{', '.join(display_algorithm(a) for a in manifest['algorithms']) or 'none'}**")
    lines.append(f"- Algorithm-cell groups: **{manifest['cell_count']}**")
    lines.append(f"- Rows run with `--no-verify-routes`: **{manifest['verification_disabled_rows']}**")
    lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Algorithm | Rows | Mean | Std | OK rows |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            "| {alg} | {rows} | {mean} | {std} | {ok} |".format(
                alg=row["algorithm_display"],
                rows=row["rows"],
                mean=fmt_num(row["mean"]),
                std=fmt_num(row["std"]),
                ok=row["ok_rows"],
            )
        )
    lines.append("")

    lines.append("## Paired Significance")
    lines.append("")
    lines.append("| Base | Other | Pairs | Mean gap | Gap % | 95% CI | Status |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    if significance_rows:
        for row in significance_rows:
            ci = "[{}, {}]".format(fmt_num(row.get("ci95_low")), fmt_num(row.get("ci95_high")))
            lines.append(
                "| {base} | {other} | {pairs} | {gap} | {pct} | {ci} | {status} |".format(
                    base=row["base_display"],
                    other=row["other_display"],
                    pairs=row["n_pairs"],
                    gap=fmt_num(row.get("mean_gap_other_minus_base")),
                    pct=fmt_num(row.get("mean_gap_pct"), 2),
                    ci=ci,
                    status=row["status"],
                )
            )
    else:
        lines.append("| | | 0 | | | | No comparable raw baseline rows found |")
    lines.append("")

    lines.append("## Cell Summary")
    lines.append("")
    lines.append("| Algorithm | DoD | TW | Rows | Cost mean | Skipped mean | TW viol mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in cell_rows[:64]:
        lines.append(
            "| {alg} | {dod} | {tw} | {rows} | {cost} | {skip} | {twv} |".format(
                alg=row.get("algorithm_display", ""),
                dod=row.get("dod", ""),
                tw=row.get("tw_ratio", ""),
                rows=row.get("row_count", ""),
                cost=fmt_num(row.get("normalized_cost_mean")),
                skip=fmt_num(row.get("total_skipped_customers_mean"), 2),
                twv=fmt_num(row.get("total_tw_violations_mean"), 2),
            )
        )
    if len(cell_rows) > 64:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... |")
    lines.append("")

    if ood_rows is not None:
        lines.append("## OOD Generalization")
        lines.append("")
        lines.append("| Model | Dataset | Cost mean | Cost std | TW viol | Skipped | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in ood_rows[:80]:
            lines.append(
                "| {model} | {dataset} | {cost} | {std} | {tw} | {skip} | {status} |".format(
                    model=row.get("model", ""),
                    dataset=row.get("dataset", ""),
                    cost=fmt_num(row.get("normalized_cost_mean")),
                    std=fmt_num(row.get("normalized_cost_std")),
                    tw=row.get("total_tw_violations", ""),
                    skip=row.get("total_skipped_customers", ""),
                    status=row.get("status", ""),
                )
            )
        lines.append("")

    if behavior_rows is not None:
        lines.append("## Behavioral Diagnostics")
        lines.append("")
        lines.append("| JSON | Steps | Override rate | Final rank | Valid candidates | Vehicle load std |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in behavior_rows[:80]:
            if int(float(row.get("step_count") or 0)) <= 0:
                continue
            lines.append(
                "| {json} | {steps} | {override} | {rank} | {valid} | {load} |".format(
                    json=Path(row.get("json_path", "")).name,
                    steps=row.get("step_count", ""),
                    override=fmt_num(row.get("attention_override_rate")),
                    rank=fmt_num(row.get("final_rank_mean")),
                    valid=fmt_num(row.get("valid_count_mean")),
                    load=fmt_num(row.get("vehicle_load_std_mean")),
                )
            )
        lines.append("")

    lines.append("## Audit Warnings")
    lines.append("")
    if manifest["audit_warnings"]:
        for warning in manifest["audit_warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No blocking table-coverage warnings detected.")
    lines.append("")

    lines.append("## Next Runs")
    lines.append("")
    lines.append("- Rerun missing baselines with raw per-instance output on the same dynamic grid.")
    lines.append("- Rerun paper-facing dynamic benchmark with route verification enabled.")
    lines.append("- Train/evaluate `no_ownership` and `no_lookahead` for cleaner H1/H2 ablations.")
    lines.append("- Add OOD test sets for n100m5, n50m6, tight TW, burst arrivals, and sparse locations.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.results_root / "paper_ready"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_summary_spec(spec) for spec in args.summary_csv]
    specs.extend(glob_summary_specs(args.results_root, args.include_summary_glob))
    if args.discover_nested:
        specs.extend(discover_nested_summary_specs(args.results_root))
    if not specs:
        specs = default_summary_specs(args.results_root, discover_nested=args.discover_nested)
    if not specs:
        raise FileNotFoundError("No summary CSVs supplied and no default master_summary.csv found")

    all_rows: list[dict[str, Any]] = []
    source_meta: list[dict[str, Any]] = []
    seen_specs = set()
    for forced_algorithm, path in specs:
        spec_key = (forced_algorithm, str(path.resolve()))
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        if not path.exists():
            raise FileNotFoundError(f"summary CSV not found: {path}")
        rows, meta = load_summary(path, forced_algorithm)
        all_rows.extend(rows)
        source_meta.append(meta)
    for spec in args.external_summary:
        forced_algorithm, path = parse_summary_spec(spec)
        if forced_algorithm is None:
            raise ValueError("--external-summary must use TYPE=PATH form")
        if not path.exists():
            raise FileNotFoundError(f"external summary CSV not found: {path}")
        rows, meta = load_external_summary(path, forced_algorithm)
        all_rows.extend(rows)
        source_meta.append(meta)

    all_rows = dedupe_rows(all_rows)

    audit_warnings = []
    audit_warnings.extend(audit_detail_csv(args.detail_csv))
    audit_warnings.extend(audit_benchmark_csv(args.benchmark_csv))
    audit_warnings.extend(audit_pyth_csv(args.pyth_csv))

    algorithms = {str(row.get("algorithm")) for row in all_rows}
    if len(algorithms) <= 1:
        audit_warnings.append(
            "Only one algorithm has raw per-instance rows in the selected summaries; "
            "paired significance against baselines cannot be computed yet."
        )
    for algorithm in ("mardam", "am", "polynet", "b0", "b1", "b3", "b5", "edgeoff"):
        if not _has_algorithm(all_rows, algorithm):
            audit_warnings.append(f"Missing raw per-instance rows for {display_algorithm(algorithm)}.")
    if args.ood_summary is None:
        audit_warnings.append("OOD summary not provided; run script/generate_ood_sets.py and script/run_ood_experiments.py.")
    if args.behavior_summary is None:
        audit_warnings.append("Behavior summary not provided; run inference with --save-step-diagnostics and analyze_behavior_diagnostics.py.")

    if any("--no-verify-routes" in str(row.get("command", "")) for row in all_rows):
        audit_warnings.append(
            "At least one raw row was produced with --no-verify-routes; rerun final paper tables with verification enabled."
        )

    master_path = args.output_dir / "master_summary.csv"
    cell_path = args.output_dir / "cell_summary.csv"
    detail_path = args.output_dir / "detail_metrics.csv"
    sig_path = args.output_dir / "significance.csv"
    manifest_path = args.output_dir / "manifest.json"
    paper_path = args.output_dir / "paper_statistics.md"

    cell_rows = build_cell_summary(all_rows)
    detail_rows = build_detail_metrics(cell_rows)
    significance_rows = build_significance(
        all_rows,
        args.base_algorithm,
        args.metric,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    aggregate_rows = aggregate_by_algorithm(all_rows, args.metric)
    manifest = build_manifest(source_meta, all_rows, audit_warnings, args)
    ood_rows = read_csv_rows(args.ood_summary) if args.ood_summary is not None and args.ood_summary.exists() else None
    behavior_rows = read_csv_rows(args.behavior_summary) if args.behavior_summary is not None and args.behavior_summary.exists() else None

    write_csv(master_path, all_rows)
    write_csv(cell_path, cell_rows)
    write_csv(detail_path, detail_rows)
    write_csv(sig_path, significance_rows, fieldnames=[
        "base_algorithm",
        "base_display",
        "other_algorithm",
        "other_display",
        "metric",
        "n_pairs",
        "base_mean",
        "other_mean",
        "mean_gap_other_minus_base",
        "mean_gap_pct",
        "ci95_low",
        "ci95_high",
        "ci95_pct_low",
        "ci95_pct_high",
        "other_worse_count",
        "other_better_count",
        "tie_count",
        "status",
    ])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_paper_statistics(paper_path, manifest, aggregate_rows, cell_rows, significance_rows, ood_rows, behavior_rows)

    print(f"Wrote master summary : {master_path}")
    print(f"Wrote cell summary   : {cell_path}")
    print(f"Wrote detail metrics : {detail_path}")
    print(f"Wrote significance   : {sig_path}")
    print(f"Wrote manifest       : {manifest_path}")
    print(f"Wrote paper stats    : {paper_path}")
    if audit_warnings:
        print("Audit warnings:")
        for warning in audit_warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
