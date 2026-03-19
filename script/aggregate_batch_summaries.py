#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = {
    "duration_sec",
    "normalized_cost",
    "raw_replay_cost",
    "normalized_total_cost",
    "normalized_distance",
    "normalized_late_penalty",
    "normalized_skipped_penalty",
    "raw_total_cost",
    "raw_distance",
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
}

IMPORTANT_COLUMNS = [
    "batch_name",
    "dataset_scale",
    "instance_class",
    "instance_id",
    "instance_name",
    "status",
    "duration_sec",
    "normalized_total_cost",
    "normalized_distance",
    "normalized_late_penalty",
    "normalized_skipped_penalty",
    "total_tw_violations",
    "total_skipped_customers",
    "active_customers",
    "visited_customers",
    "missing_count",
    "duplicate_count",
    "extra_count",
    "dataset_relpath",
    "result_json",
    "run_log",
    "source_summary_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate output/batch_infer_*/summary.csv into one readable CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root folder containing batch_infer_* folders",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="batch_infer_*/summary.csv",
        help="Glob pattern relative to --output-root",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/batch_infer_summary_aggregate.csv"),
        help="Where to write the merged CSV",
    )
    parser.add_argument(
        "--include-all-columns",
        action="store_true",
        help="Include every discovered source column in the output CSV",
    )
    return parser.parse_args()


def parse_instance_meta(dataset_relpath: str) -> dict[str, Any]:
    rel = Path(dataset_relpath)
    dataset_scale = rel.parts[0] if rel.parts else "unknown"
    instance_name = rel.stem

    instance_class = "unknown"
    instance_id: int | None = None
    m = re.search(r"(rc|r|c)(\d+)$", instance_name.lower())
    if m:
        instance_class = m.group(1)
        instance_id = int(m.group(2))

    return {
        "dataset_scale": dataset_scale,
        "instance_name": instance_name,
        "instance_class": instance_class,
        "instance_id": instance_id,
    }


def maybe_number(value: Any, key: str) -> Any:
    if value in (None, ""):
        return value
    if key not in NUMERIC_COLUMNS:
        return value
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value

    if key.endswith("_count") or key in {
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
        return int(num)

    return round(num, 6)


def read_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: maybe_number(v, k) for k, v in row.items()}
            clean_row["batch_name"] = summary_csv.parent.name
            clean_row["source_summary_csv"] = str(summary_csv)
            clean_row.update(parse_instance_meta(str(clean_row.get("dataset_relpath", ""))))
            rows.append(clean_row)
    return rows


def sort_key(row: dict[str, Any]):
    instance_id = row.get("instance_id")
    instance_id_sort = instance_id if isinstance(instance_id, int) else 10**9
    return (
        str(row.get("dataset_scale", "")),
        str(row.get("instance_class", "")),
        instance_id_sort,
        str(row.get("batch_name", "")),
        str(row.get("instance_name", "")),
    )


def ordered_fieldnames(rows: list[dict[str, Any]], include_all_columns: bool) -> list[str]:
    discovered: list[str] = []
    seen = set()

    for key in IMPORTANT_COLUMNS:
        seen.add(key)
        discovered.append(key)

    if not include_all_columns:
        return discovered

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                discovered.append(key)

    return discovered


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    output_root = args.output_root.resolve()
    summary_paths = sorted(p for p in output_root.glob(args.input_glob) if p.is_file())
    if not summary_paths:
        print(f"No files matched: {output_root / args.input_glob}")
        return 1

    all_rows: list[dict[str, Any]] = []
    for path in summary_paths:
        all_rows.extend(read_summary_rows(path))

    all_rows.sort(key=sort_key)
    fieldnames = ordered_fieldnames(all_rows, args.include_all_columns)
    write_csv(args.output_csv.resolve(), all_rows, fieldnames)

    print(f"Found summary files : {len(summary_paths)}")
    print(f"Merged rows         : {len(all_rows)}")
    print(f"Output CSV          : {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
