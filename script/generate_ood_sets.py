#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from problems import DVRPTW_Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DVRPTW ID/OOD test sets for COAST experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_sets"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--smoke-size", type=int, default=None, help="Override batch size for quick smoke tests.")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--normalize", action="store_true", help="Save normalized datasets. Default saves raw data.")
    return parser.parse_args()


def generate_case(batch_size: int, seed: int, params: dict[str, Any]) -> DVRPTW_Dataset:
    torch.manual_seed(seed)
    return DVRPTW_Dataset.generate(
        batch_size=batch_size,
        cust_count=params["cust_count"],
        veh_count=params["veh_count"],
        veh_capa=params.get("veh_capa", 200),
        veh_speed=params.get("veh_speed", 1),
        min_cust_count=params.get("min_cust_count"),
        cust_loc_range=params.get("cust_loc_range", (0, 101)),
        cust_dem_range=params.get("cust_dem_range", (5, 41)),
        horizon=params.get("horizon", 480),
        cust_dur_range=params.get("cust_dur_range", (10, 31)),
        tw_ratio=params.get("tw_ratio", [0.25, 0.5, 0.75, 1.0]),
        cust_tw_range=params.get("cust_tw_range", [30, 91]),
        dod=params.get("dod", [0.1, 0.25, 0.5, 0.75]),
        d_early_ratio=params.get("d_early_ratio", [0.0, 0.5, 0.75, 1.0]),
    )


def case_matrix() -> dict[str, dict[str, Any]]:
    return {
        "id_n50m3": {
            "description": "In-distribution reference: 50 customers, 3 vehicles",
            "cust_count": 50,
            "veh_count": 3,
        },
        "ood_n100m5": {
            "description": "Scale shift: 100 customers, 5 vehicles",
            "cust_count": 100,
            "veh_count": 5,
        },
        "ood_n50m6": {
            "description": "Fleet shift: 50 customers, 6 vehicles",
            "cust_count": 50,
            "veh_count": 6,
        },
        "ood_tight_tw": {
            "description": "Tighter time-window distribution",
            "cust_count": 50,
            "veh_count": 3,
            "tw_ratio": [0.8, 0.9, 1.0],
            "cust_tw_range": [10, 40],
        },
        "ood_burst_dynamic": {
            "description": "High degree of dynamism with later arrivals",
            "cust_count": 50,
            "veh_count": 3,
            "dod": [0.5, 0.75, 1.0],
            "d_early_ratio": [0.0, 0.25, 0.5],
        },
        "ood_sparse_spatial": {
            "description": "Sparse spatial distribution with larger location range",
            "cust_count": 50,
            "veh_count": 3,
            "cust_loc_range": (0, 201),
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.smoke_size if args.smoke_size is not None else args.batch_size

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_size": batch_size,
        "base_seed": args.seed,
        "normalized": bool(args.normalize),
        "datasets": [],
    }

    for offset, (name, params) in enumerate(case_matrix().items()):
        data = generate_case(batch_size, args.seed - offset, params)
        if args.normalize:
            data.normalize()
        out_path = args.output_dir / f"test_dvrptw_{name}_{batch_size}.pyth"
        torch.save(data, out_path)
        manifest["datasets"].append({
            "name": name,
            "path": str(out_path),
            "description": params["description"],
            "seed": args.seed - offset,
            "params": params,
            "nodes_shape": list(data.nodes.size()),
        })
        print(f"Saved {name}: {out_path} shape={tuple(data.nodes.size())}")

    manifest_path = args.output_dir / f"ood_manifest_{batch_size}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
