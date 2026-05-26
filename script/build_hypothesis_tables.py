#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable


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

LOWER_IS_BETTER = {
    "normalized_cost",
    "normalized_cost_mean",
    "normalized_total_cost",
    "normalized_distance",
    "normalized_late_time",
    "normalized_late_time_mean",
    "normalized_late_penalty",
    "normalized_late_penalty_mean",
    "total_skipped_customers",
    "total_tw_violations",
    "duration_sec",
    "vehicle_load_std_mean",
    "duplicate_count_mean",
    "service_region_overlap_proxy",
    "route_customer_overlap_proxy",
    "degradation_pct",
}

HIGHER_IS_BETTER = {
    "visited_customers",
    "visited_steps_mean",
    "lookahead_support_rate",
    "ownership_support_rate",
    "route_centroid_distance_mean",
}

DESCRIPTIVE_METRICS = {
    "attention_override_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H1-H4 hypothesis tables from paper-ready summaries.")
    parser.add_argument("--master-summary", type=Path, required=True)
    parser.add_argument("--ood-summary", type=Path, default=None)
    parser.add_argument("--behavior-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/hypothesis_tables"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260527)
    return parser.parse_args()


def normalize_algorithm(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"coast", "coast_vectra", "vectra_full"}:
        return "vectra"
    if raw in {"edge_off", "edge-off"}:
        return "edgeoff"
    if raw in {"poly_net", "poly-net"}:
        return "polynet"
    if raw == "noownership":
        return "no_ownership"
    if raw == "nolookahead":
        return "no_lookahead"
    return raw


def display_algorithm(value: Any) -> str:
    return DISPLAY_NAMES.get(normalize_algorithm(value), str(value))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def fmt(value: Any, digits: int = 4) -> str:
    num = maybe_float(value)
    return "" if num is None else f"{num:.{digits}f}"


def read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def instance_name(row: dict[str, Any]) -> str:
    for key in ("instance_name", "benchmark_relpath", "dataset_relpath", "dataset_abspath", "json_path", "result_json"):
        value = str(row.get(key, "")).strip()
        if value:
            name = Path(value).name
            if name.endswith(".infer.json"):
                return name[:-len(".infer.json")] + ".csv"
            return name
    return ""


def pair_key(row: dict[str, Any]) -> str:
    key = str(row.get("pair_key", "")).strip()
    if key:
        return key
    return "|".join([
        str(row.get("dod", "")).strip(),
        str(row.get("tw_ratio", "")).strip(),
        str(row.get("seed", "")).strip(),
        instance_name(row),
    ])


def algorithm_of(row: dict[str, Any]) -> str:
    return normalize_algorithm(row.get("algorithm") or row.get("model"))


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(samples):
        draws.append(mean(values[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def expected_direction(metric: str) -> str:
    if metric in DESCRIPTIVE_METRICS:
        return "descriptive"
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    return "lower_is_better"


def claim_status(metric: str, mean_gap: float | None, ci_low: float | None, ci_high: float | None, n: int) -> str:
    if n == 0 or mean_gap is None:
        return "missing_raw_pairs"
    direction = expected_direction(metric)
    if direction == "descriptive":
        return "descriptive"
    if direction == "lower_is_better":
        if ci_low is not None and ci_low > 0:
            return "supports"
        if mean_gap > 0:
            return "mixed_ci_crosses_zero"
        return "not_supported"
    if ci_high is not None and ci_high < 0:
        return "supports"
    if mean_gap < 0:
        return "mixed_ci_crosses_zero"
    return "not_supported"


def paired_rows(
    rows: list[dict[str, Any]],
    base: str,
    other: str,
    metric: str,
    hypothesis: str,
    table: str,
    comparison: str,
    subset: str,
    filter_fn: Callable[[dict[str, Any]], bool],
    samples: int,
    seed: int,
    note: str = "",
) -> dict[str, Any]:
    base = normalize_algorithm(base)
    other = normalize_algorithm(other)
    by_alg: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        alg = algorithm_of(row)
        if alg in {base, other} and filter_fn(row):
            key = pair_key(row)
            value = maybe_float(row.get(metric))
            if key and value is not None:
                by_alg[alg][key] = row

    keys = sorted(set(by_alg.get(base, {})) & set(by_alg.get(other, {})))
    base_values = []
    other_values = []
    gaps = []
    base_wins = 0
    other_wins = 0
    ties = 0
    for key in keys:
        base_value = maybe_float(by_alg[base][key].get(metric))
        other_value = maybe_float(by_alg[other][key].get(metric))
        if base_value is None or other_value is None:
            continue
        base_values.append(base_value)
        other_values.append(other_value)
        gap = other_value - base_value
        gaps.append(gap)
        direction = expected_direction(metric)
        if direction == "descriptive":
            if base_value == other_value:
                ties += 1
        elif direction == "lower_is_better":
            if base_value < other_value:
                base_wins += 1
            elif base_value > other_value:
                other_wins += 1
            else:
                ties += 1
        else:
            if base_value > other_value:
                base_wins += 1
            elif base_value < other_value:
                other_wins += 1
            else:
                ties += 1

    ci_low, ci_high = bootstrap_ci(gaps, samples, seed)
    gap_mean = None if not gaps else mean(gaps)
    return {
        "hypothesis": hypothesis,
        "table": table,
        "comparison": comparison,
        "evidence_type": "paired_instance",
        "subset": subset,
        "base_algorithm": base,
        "base_display": display_algorithm(base),
        "other_algorithm": other,
        "other_display": display_algorithm(other),
        "metric": metric,
        "expected_direction": expected_direction(metric),
        "n_pairs": len(gaps),
        "base_mean": "" if not base_values else mean(base_values),
        "other_mean": "" if not other_values else mean(other_values),
        "mean_gap_other_minus_base": "" if gap_mean is None else gap_mean,
        "ci95_low": "" if ci_low is None else ci_low,
        "ci95_high": "" if ci_high is None else ci_high,
        "base_wins": base_wins,
        "other_wins": other_wins,
        "ties": ties,
        "claim_status": claim_status(metric, gap_mean, ci_low, ci_high, len(gaps)),
        "note": note,
    }


def all_rows(_: dict[str, Any]) -> bool:
    return True


def high_tw(row: dict[str, Any]) -> bool:
    value = maybe_float(row.get("tw_ratio"))
    return value is not None and value >= 0.75


def aggregate_ood(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_model_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        model = normalize_algorithm(row.get("model") or row.get("algorithm"))
        dataset = str(row.get("dataset", "")).strip()
        if model and dataset and row.get("status", "ok") == "ok":
            by_model_dataset[(model, dataset)] = row

    id_cost = {
        model: maybe_float(row.get("normalized_cost_mean"))
        for (model, dataset), row in by_model_dataset.items()
        if dataset == "id_n50m3"
    }
    degradation_rows = []
    for (model, dataset), row in by_model_dataset.items():
        cost = maybe_float(row.get("normalized_cost_mean"))
        base = id_cost.get(model)
        if dataset == "id_n50m3" or cost is None or base in (None, 0):
            continue
        enriched = dict(row)
        enriched["algorithm"] = model
        enriched["pair_key"] = dataset
        enriched["degradation_pct"] = (cost - base) / abs(base) * 100.0
        degradation_rows.append(enriched)

    out = []
    comparisons = [("vectra", "b5"), ("vectra", "mardam"), ("vectra", "b0"), ("vectra", "edgeoff")]
    for base, other in comparisons:
        out.append(paired_rows(
            degradation_rows,
            base,
            other,
            "degradation_pct",
            "H4",
            "OOD Generalization and Fusion",
            f"{display_algorithm(base)} vs {display_algorithm(other)} OOD degradation",
            "ood_non_id",
            all_rows,
            samples,
            seed,
            "Dataset-level OOD aggregates, not per-instance paired rows.",
        ))

    tight_rows = []
    for (model, dataset), row in by_model_dataset.items():
        if dataset == "ood_tight_tw":
            enriched = dict(row)
            enriched["algorithm"] = model
            enriched["pair_key"] = dataset
            tight_rows.append(enriched)
    for metric in ("total_tw_violations", "normalized_late_time_mean", "normalized_late_penalty_mean", "normalized_cost_mean"):
        out.append(paired_rows(
            tight_rows,
            "vectra",
            "edgeoff",
            metric,
            "H3",
            "Edge Awareness Under Tight Constraints",
            "COAST/VECTRA vs EdgeOff on OOD tight TW",
            "ood_tight_tw",
            all_rows,
            samples,
            seed,
            "Single dataset-level aggregate; use as supporting evidence.",
        ))
    return out


def build_hypothesis_rows(master_rows: list[dict[str, Any]], behavior_rows: list[dict[str, Any]], ood_rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dynamic_specs = [
        ("H1", "Coordination Ablation", "COAST/VECTRA vs clean NoOwnership", "vectra", "no_ownership", ["normalized_cost", "total_skipped_customers", "visited_customers"]),
        ("H1", "Coordination Ablation", "B1 memory-only vs B0 no-memory", "b1", "b0", ["normalized_cost", "total_skipped_customers", "visited_customers"]),
        ("H1", "Coordination Ablation", "COAST/VECTRA vs B1 memory-only", "vectra", "b1", ["normalized_cost", "total_skipped_customers", "visited_customers"]),
        ("H2", "Lookahead Intervention Analysis", "COAST/VECTRA vs clean NoLookahead", "vectra", "no_lookahead", ["normalized_cost", "normalized_late_time", "normalized_late_penalty", "total_skipped_customers"]),
        ("H2", "Lookahead Intervention Analysis", "B3 lookahead-only vs B0 no-memory", "b3", "b0", ["normalized_cost", "normalized_late_time", "normalized_late_penalty", "total_skipped_customers"]),
        ("H2", "Lookahead Intervention Analysis", "COAST/VECTRA vs B1 combined ownership/lookahead", "vectra", "b1", ["normalized_cost", "normalized_late_time", "normalized_late_penalty", "total_skipped_customers"]),
        ("H3", "Edge Awareness Under Tight Constraints", "COAST/VECTRA vs EdgeOff on high TW cells", "vectra", "edgeoff", ["total_tw_violations", "normalized_late_time", "normalized_late_penalty", "total_skipped_customers", "normalized_cost"]),
        ("H4", "OOD Generalization and Fusion", "COAST/VECTRA vs B5 linear fusion on ID grid", "vectra", "b5", ["normalized_cost", "total_skipped_customers", "total_tw_violations"]),
        ("H4", "OOD Generalization and Fusion", "COAST/VECTRA vs MARDAM on ID grid", "vectra", "mardam", ["normalized_cost", "duration_sec"]),
        ("H4", "OOD Generalization and Fusion", "COAST/VECTRA vs AM on ID grid", "vectra", "am", ["normalized_cost", "duration_sec"]),
        ("H4", "OOD Generalization and Fusion", "COAST/VECTRA vs PolyNet on ID grid", "vectra", "polynet", ["normalized_cost", "duration_sec"]),
    ]
    for hyp, table, comparison, base, other, metrics in dynamic_specs:
        filt = high_tw if hyp == "H3" else all_rows
        subset = "tw>=0.75" if hyp == "H3" else "all_dynamic_grid"
        note = ""
        if hyp in {"H1", "H2"} and base == "vectra" and other == "b1":
            note = "Single-checkpoint evidence; no clean no_ownership/no_lookahead checkpoint, so interpret as combined mechanism effect."
        if other in {"no_ownership", "no_lookahead"}:
            note = "Clean ablation row; supports the main mechanism claim only after the corresponding checkpoint exists."
        for metric in metrics:
            out.append(paired_rows(master_rows, base, other, metric, hyp, table, comparison, subset, filt, samples, seed, note))

    behavior_specs = [
        ("H1", "Coordination Ablation", "COAST/VECTRA vs clean NoOwnership route behavior", "vectra", "no_ownership", ["vehicle_load_std_mean", "duplicate_count_mean", "service_region_overlap_proxy", "route_customer_overlap_proxy"]),
        ("H1", "Coordination Ablation", "COAST/VECTRA vs B1 route behavior", "vectra", "b1", ["vehicle_load_std_mean", "duplicate_count_mean", "service_region_overlap_proxy", "route_customer_overlap_proxy"]),
        ("H1", "Coordination Ablation", "B1 memory-only vs B0 route behavior", "b1", "b0", ["vehicle_load_std_mean", "duplicate_count_mean", "service_region_overlap_proxy", "route_customer_overlap_proxy"]),
        ("H2", "Lookahead Intervention Analysis", "COAST/VECTRA vs clean NoLookahead decision behavior", "vectra", "no_lookahead", ["attention_override_rate", "lookahead_support_rate", "final_rank_mean"]),
        ("H2", "Lookahead Intervention Analysis", "COAST/VECTRA vs B3 decision behavior", "vectra", "b3", ["attention_override_rate", "lookahead_support_rate", "final_rank_mean"]),
    ]
    for hyp, table, comparison, base, other, metrics in behavior_specs:
        for metric in metrics:
            out.append(paired_rows(behavior_rows, base, other, metric, hyp, table, comparison, "diagnostics_subset", all_rows, samples, seed, "Requires diagnostics JSON for both compared models."))

    out.extend(aggregate_ood(ood_rows, samples, seed))
    return out


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


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# COAST/VECTRA Hypothesis Summary", "", f"Generated: `{datetime.now().isoformat(timespec='seconds')}`", ""]
    lines.append("Claim status uses paired bootstrap CI when per-instance rows are available. OOD rows are dataset-level aggregates.")
    lines.append("")
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_table[str(row.get("table", ""))].append(row)
    for table, table_rows in by_table.items():
        lines.append(f"## {table}")
        lines.append("")
        lines.append("| Hyp | Comparison | Metric | Subset | Pairs | Base | Other | Gap other-base | 95% CI | Status |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")
        for row in table_rows:
            lines.append(
                "| {hyp} | {comp} | {metric} | {subset} | {pairs} | {base} | {other} | {gap} | [{low}, {high}] | {status} |".format(
                    hyp=row.get("hypothesis", ""),
                    comp=row.get("comparison", ""),
                    metric=row.get("metric", ""),
                    subset=row.get("subset", ""),
                    pairs=row.get("n_pairs", ""),
                    base=fmt(row.get("base_mean")),
                    other=fmt(row.get("other_mean")),
                    gap=fmt(row.get("mean_gap_other_minus_base")),
                    low=fmt(row.get("ci95_low")),
                    high=fmt(row.get("ci95_high")),
                    status=row.get("claim_status", ""),
                )
            )
        lines.append("")
    limitations = sorted({str(row.get("note", "")).strip() for row in rows if str(row.get("note", "")).strip()})
    if limitations:
        lines.append("## Notes")
        lines.append("")
        for note in limitations:
            lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master_rows = read_csv(args.master_summary)
    behavior_rows = read_csv(args.behavior_summary)
    ood_rows = read_csv(args.ood_summary)
    rows = build_hypothesis_rows(master_rows, behavior_rows, ood_rows, args.bootstrap_samples, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "hypothesis_summary.csv"
    summary_md = args.output_dir / "hypothesis_summary.md"
    write_csv(summary_csv, rows)
    write_markdown(summary_md, rows)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
