#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

KNOWN_ALGORITHMS = {
    "vectra",
    "b0",
    "b1",
    "b3",
    "b5",
    "edgeoff",
    "mardam",
    "am",
    "polynet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze VECTRA step diagnostics and route behavior.")
    parser.add_argument("--input-root", type=Path, default=Path("output"))
    parser.add_argument("--input-glob", type=str, default="**/*.json")
    parser.add_argument("--output-dir", type=Path, default=Path("output/behavior_analysis"))
    parser.add_argument("--master-summary", type=Path, default=None, help="Optional master_summary.csv used to map JSON files to source CSV datasets.")
    parser.add_argument("--datasets-root", type=Path, default=None, help="Optional dynamic-grid CSV root used to recover route coordinates.")
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return f"{values[0]:.6f}", "0.000000"
    return f"{mean(values):.6f}", f"{stdev(values):.6f}"


def normalize_algorithm(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"coast", "coast_vectra", "vectra_full"}:
        return "vectra"
    if raw in {"edge_off", "edge-off"}:
        return "edgeoff"
    if raw in {"poly_net", "poly-net"}:
        return "polynet"
    return raw


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


def infer_algorithm_from_path(path: Path) -> str:
    parts = [normalize_algorithm(part) for part in path.parts]
    for part in parts:
        if part in KNOWN_ALGORITHMS:
            return part
    joined = "_".join(parts)
    for algorithm in KNOWN_ALGORITHMS:
        if algorithm in joined:
            return algorithm
    return ""


def instance_name_from_json(path: Path) -> str:
    name = path.name
    if name.endswith(".infer.json"):
        return name[:-len(".infer.json")] + ".csv"
    if name.endswith(".json"):
        return name[:-len(".json")] + ".csv"
    return name


def make_pair_key(row: dict[str, Any]) -> str:
    return "|".join([
        str(row.get("dod", "")).strip(),
        str(row.get("tw_ratio", "")).strip(),
        str(row.get("seed", "")).strip(),
        str(row.get("instance_name", "")).strip(),
    ])


def load_master_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            clean = dict(row)
            clean["algorithm"] = normalize_algorithm(clean.get("algorithm"))
            if not str(clean.get("instance_name", "")).strip():
                clean["instance_name"] = instance_name_from_json(Path(str(clean.get("result_json") or clean.get("dataset_relpath") or "")))
            if not str(clean.get("pair_key", "")).strip():
                clean["pair_key"] = make_pair_key(clean)
            result_json = str(clean.get("result_json", "")).strip()
            if result_json:
                index[str(Path(result_json))] = clean
            row_key = "|".join([str(clean.get("algorithm", "")), str(clean.get("pair_key", ""))])
            if row_key.strip("|"):
                index[row_key] = clean
    return index


def metadata_for_json(path: Path, master_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta["algorithm"] = infer_algorithm_from_path(path)
    for key, value in parse_cell_from_path(str(path)).items():
        meta[key] = value
    meta["instance_name"] = instance_name_from_json(path)
    meta["pair_key"] = make_pair_key(meta)
    source = master_index.get(str(path)) or master_index.get("|".join([str(meta.get("algorithm", "")), str(meta.get("pair_key", ""))])) or {}
    for key in ("algorithm", "dod", "tw_ratio", "seed", "dataset_abspath", "dataset_relpath", "instance_name", "pair_key"):
        if source.get(key) and not str(meta.get(key, "")).strip():
            meta[key] = source.get(key)
    for key in ("dataset_abspath", "dataset_relpath"):
        if source.get(key):
            meta[key] = source.get(key)
    meta["algorithm"] = normalize_algorithm(meta.get("algorithm"))
    meta["pair_key"] = make_pair_key(meta)
    return meta


def candidate_dataset_paths(path: Path, meta: dict[str, Any], datasets_root: Path | None) -> list[Path]:
    candidates = []
    for key in ("dataset_abspath", "dataset_relpath"):
        value = str(meta.get(key, "")).strip()
        if value:
            raw = Path(value)
            candidates.append(raw)
            if datasets_root is not None and not raw.is_absolute():
                candidates.append(datasets_root / raw)
    if datasets_root is not None:
        dod = str(meta.get("dod", "")).replace(".", "p")
        tw = str(meta.get("tw_ratio", "")).replace(".", "p")
        seed = str(meta.get("seed", "")).strip()
        instance = str(meta.get("instance_name", "")).strip()
        if dod and tw and seed and instance:
            candidates.append(datasets_root / f"dod_{dod}" / f"tw_{tw}" / f"seed_{seed}" / instance)
            candidates.append(datasets_root / instance)
    return candidates


def load_points(path: Path) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                points[idx] = (float(row["x"]), float(row["y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return points


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def spatial_route_stats(routes: list[Any], points: dict[int, tuple[float, float]]) -> dict[str, Any]:
    if not routes or not points:
        return {}
    bbox_ious = []
    centroid_distances = []
    jaccards = []
    for inst_routes in routes:
        route_sets = []
        bboxes = []
        centroids = []
        for route in inst_routes:
            nodes = [int(node) for node in route if int(node) != 0 and int(node) in points]
            if not nodes:
                continue
            route_sets.append(set(nodes))
            xs = [points[node][0] for node in nodes]
            ys = [points[node][1] for node in nodes]
            bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            centroids.append((mean(xs), mean(ys)))
        for i in range(len(route_sets)):
            for j in range(i + 1, len(route_sets)):
                union = route_sets[i] | route_sets[j]
                jaccards.append(0.0 if not union else len(route_sets[i] & route_sets[j]) / len(union))
                bbox_ious.append(bbox_iou(bboxes[i], bboxes[j]))
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                centroid_distances.append(math.sqrt(dx * dx + dy * dy))
    return {
        "service_region_overlap_proxy": "" if not bbox_ious else mean(bbox_ious),
        "route_customer_overlap_proxy": "" if not jaccards else mean(jaccards),
        "route_centroid_distance_mean": "" if not centroid_distances else mean(centroid_distances),
    }


def rank_desc(values: list[float], idx: int, mask: list[bool] | None = None) -> int | None:
    if idx < 0 or idx >= len(values):
        return None
    valid = []
    for j, value in enumerate(values):
        if mask is not None and j < len(mask) and mask[j]:
            continue
        valid.append((float(value), j))
    valid.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, j) in enumerate(valid, start=1):
        if j == idx:
            return rank
    return None


def argmax_valid(values: list[float], mask: list[bool] | None = None) -> int | None:
    best_idx = None
    best_value = None
    for j, value in enumerate(values):
        if mask is not None and j < len(mask) and mask[j]:
            continue
        value = float(value)
        if best_value is None or value > best_value:
            best_value = value
            best_idx = j
    return best_idx


def route_stats(payload: dict[str, Any], dataset_path: Path | None = None) -> dict[str, Any]:
    routes = payload.get("routes") or []
    route_diags = payload.get("route_diagnostics") or []
    visited_counts = []
    vehicle_load_stds = []
    duplicate_counts = []

    for inst_idx, inst_routes in enumerate(routes):
        loads = [len([node for node in route if node != 0]) for route in inst_routes]
        if loads:
            visited_counts.append(sum(loads))
            vehicle_load_stds.append(0.0 if len(loads) == 1 else stdev(loads))
        if inst_idx < len(route_diags):
            duplicate_counts.append(float(route_diags[inst_idx].get("duplicate_count", 0) or 0))

    stats = {
        "route_instances": len(routes),
        "visited_steps_mean": mean(visited_counts) if visited_counts else "",
        "vehicle_load_std_mean": mean(vehicle_load_stds) if vehicle_load_stds else "",
        "duplicate_count_mean": mean(duplicate_counts) if duplicate_counts else "",
    }
    if dataset_path is not None and dataset_path.exists():
        stats["dataset_csv"] = str(dataset_path)
        stats.update(spatial_route_stats(routes, load_points(dataset_path)))
    return stats


def analyze_file(path: Path, master_index: dict[str, dict[str, Any]] | None = None, datasets_root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = metadata_for_json(path, master_index or {})
    steps = payload.get("step_diagnostics") or []

    att_ranks = []
    owner_ranks = []
    look_ranks = []
    final_ranks = []
    valid_counts = []
    att_overrides = 0
    lookahead_support = 0
    ownership_support = 0

    for step in steps:
        selected = int(step.get("selected_customer", -1))
        mask = step.get("mask") or None
        att = step.get("att_score") or []
        owner = step.get("owner_bias") or []
        look = step.get("lookahead") or []
        final = step.get("final_score") or []

        for values, ranks in ((att, att_ranks), (owner, owner_ranks), (look, look_ranks), (final, final_ranks)):
            rank = rank_desc(values, selected, mask)
            if rank is not None:
                ranks.append(float(rank))
        if "valid_count" in step:
            valid_counts.append(float(step["valid_count"]))

        att_top = argmax_valid(att, mask)
        look_top = argmax_valid(look, mask)
        owner_top = argmax_valid(owner, mask)
        if att_top is not None and selected != att_top:
            att_overrides += 1
            if look_top == selected:
                lookahead_support += 1
            if owner_top == selected:
                ownership_support += 1

    att_rank_mean, att_rank_std = mean_std(att_ranks)
    owner_rank_mean, owner_rank_std = mean_std(owner_ranks)
    look_rank_mean, look_rank_std = mean_std(look_ranks)
    final_rank_mean, final_rank_std = mean_std(final_ranks)
    valid_mean, valid_std = mean_std(valid_counts)
    dataset_path = None
    for candidate in candidate_dataset_paths(path, meta, datasets_root):
        if candidate.exists():
            dataset_path = candidate
            break
    route = route_stats(payload, dataset_path)

    return {
        "json_path": str(path),
        **meta,
        "step_count": len(steps),
        "att_rank_mean": att_rank_mean,
        "att_rank_std": att_rank_std,
        "owner_rank_mean": owner_rank_mean,
        "owner_rank_std": owner_rank_std,
        "lookahead_rank_mean": look_rank_mean,
        "lookahead_rank_std": look_rank_std,
        "final_rank_mean": final_rank_mean,
        "final_rank_std": final_rank_std,
        "valid_count_mean": valid_mean,
        "valid_count_std": valid_std,
        "attention_override_count": att_overrides,
        "attention_override_rate": "" if not steps else att_overrides / len(steps),
        "lookahead_support_count": lookahead_support,
        "lookahead_support_rate": "" if not att_overrides else lookahead_support / att_overrides,
        "ownership_support_count": ownership_support,
        "ownership_support_rate": "" if not att_overrides else ownership_support / att_overrides,
        **route,
    }


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
    analyzed = [r for r in rows if int(r.get("step_count") or 0) > 0]
    route_analyzed = [r for r in rows if int(r.get("route_instances") or 0) > 0]
    lines = ["# Behavioral Diagnostics Report", "", f"Generated: `{datetime.now().isoformat(timespec='seconds')}`", ""]
    lines.append(f"- JSON files scanned: **{len(rows)}**")
    lines.append(f"- JSON files with step diagnostics: **{len(analyzed)}**")
    lines.append(f"- JSON files with route diagnostics: **{len(route_analyzed)}**")
    if analyzed:
        override_rates = [float(r["attention_override_rate"]) for r in analyzed if r.get("attention_override_rate") != ""]
        valid_counts = [float(r["valid_count_mean"]) for r in analyzed if r.get("valid_count_mean") != ""]
        final_ranks = [float(r["final_rank_mean"]) for r in analyzed if r.get("final_rank_mean") != ""]
        lines.append(f"- Mean attention override rate: **{mean(override_rates):.4f}**" if override_rates else "- Mean attention override rate: n/a")
        lines.append(f"- Mean valid candidates per step: **{mean(valid_counts):.2f}**" if valid_counts else "- Mean valid candidates per step: n/a")
        lines.append(f"- Mean selected final-score rank: **{mean(final_ranks):.2f}**" if final_ranks else "- Mean selected final-score rank: n/a")
    lines.append("")
    lines.append("| JSON | Alg | Steps | Override rate | Final rank | Valid candidates | Load std | Region overlap |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in analyzed[:40]:
        lines.append(
            "| {name} | {alg} | {steps} | {override} | {rank} | {valid} | {load} | {overlap} |".format(
                name=Path(row["json_path"]).name,
                alg=row.get("algorithm", ""),
                steps=row.get("step_count", ""),
                override=row.get("attention_override_rate", ""),
                rank=row.get("final_rank_mean", ""),
                valid=row.get("valid_count_mean", ""),
                load=row.get("vehicle_load_std_mean", ""),
                overlap=row.get("service_region_overlap_proxy", ""),
            )
        )
    if route_analyzed and not analyzed:
        lines.append("")
        lines.append("| JSON | Alg | Load std | Region overlap | Customer overlap |")
        lines.append("|---|---|---:|---:|---:|")
        for row in route_analyzed[:40]:
            lines.append(
                "| {name} | {alg} | {load} | {region} | {customer} |".format(
                    name=Path(row["json_path"]).name,
                    alg=row.get("algorithm", ""),
                    load=row.get("vehicle_load_std_mean", ""),
                    region=row.get("service_region_overlap_proxy", ""),
                    customer=row.get("route_customer_overlap_proxy", ""),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master_index = load_master_index(args.master_summary)
    paths = sorted(args.input_root.glob(args.input_glob))
    rows = []
    for path in paths:
        try:
            rows.append(analyze_file(path, master_index, args.datasets_root))
        except Exception as exc:  # noqa: BLE001
            rows.append({"json_path": str(path), "step_count": 0, "error_message": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "behavior_summary.csv"
    hypothesis_csv = args.output_dir / "hypothesis_behavior_summary.csv"
    report_md = args.output_dir / "behavior_report.md"
    write_csv(summary_csv, rows)
    write_csv(hypothesis_csv, rows)
    write_report(report_md, rows)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {hypothesis_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
