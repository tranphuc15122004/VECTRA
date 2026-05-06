"""
Single-instance inference with OR-Tools.

Usage
-----
    python script/infer_ort.py --problem-type dvrptw \\
        --data-csv data/datasets/100/h100c101.csv \\
        --vehicles-count 5 --veh-capa 1300 --veh-speed 1 \\
        --save-json output/ort_h100c101_result.json

All common arguments from ``parse_args()`` (see ``utils/_args.py``) are also
accepted and forwarded to the environment (e.g. ``--pending-cost``,
``--late-cost``).
"""

import os
import sys
import time

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Re-use the shared inference helpers from infer_lkh.py
from script.infer_lkh import (
    parse_infer_args,
    _dataset_cls,
    _environment_cls,
    _build_dataset,
    _clone_dataset,
    _build_env_params,
    _print_routes,
    _save_json,
    _route_diag_for_instance,
    _verify_routes_cost,
    _replay_routes_cost,
    _check_route_constraints,
    _compute_cost_components,
)
from externals._ort import ort_solve
from utils import eval_apriori_routes, set_random_seed


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    set_random_seed(args.rng_seed, deterministic=True)

    dataset_cls = _dataset_cls(args.problem_type)
    env_cls = _environment_cls(args.problem_type)
    if dataset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{args.problem_type}'")

    # ---- load & prepare data ---------------------------------------------
    data = _build_dataset(args, dataset_cls)       # raw CSV values
    raw_data = _clone_dataset(data)                # keep raw for the solver
    if not args.no_normalize:
        data.normalize()                           # model env uses normalized

    env_params = _build_env_params(args)
    env = env_cls(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)

    # ---- OR-Tools --------------------------------------------------------
    print("Running OR-Tools ...")
    print(f"Dataset has {raw_data.batch_size} instance(s)")
    print(f"Nodes shape: {raw_data.nodes.shape}")
    print(f"Vehicles: {raw_data.veh_count}, Capacity: {raw_data.veh_capa}, "
          f"Speed: {raw_data.veh_speed}")

    start_t = time.perf_counter()
    routes = ort_solve(
        raw_data,
        late_cost=args.late_cost,
        time_limit_ms=getattr(args, "ort_time_limit_ms", 10_000),
    )
    solve_time = time.perf_counter() - start_t
    print(f"OR-Tools completed in {solve_time:.4f} seconds.")

    # ---- evaluate routes -------------------------------------------------
    costs = _replay_routes_cost(data, env_cls, env_params, routes, rollouts=1)
    raw_replay_costs = _replay_routes_cost(
        raw_data, env_cls, env_params, routes, rollouts=args.verify_rollouts
    )
    route_diagnostics = [
        _route_diag_for_instance(data, routes, idx)
        for idx in range(len(routes))
    ]
    total_skipped = sum(d["missing_count"] for d in route_diagnostics)
    constraint_diagnostics = _check_route_constraints(raw_data, routes)
    raw_cost_components = _compute_cost_components(
        raw_data, routes, args.pending_cost, args.late_cost
    )
    normalized_cost_components = _compute_cost_components(
        data, routes, args.pending_cost, args.late_cost
    )
    total_tw_viol = sum(d["tw_violation_count"]
                        for d in constraint_diagnostics)
    total_appear_viol = sum(d["appearance_violation_count"]
                            for d in constraint_diagnostics)

    mean = costs.mean().item()
    std = costs.std().item() if costs.numel() > 1 else 0.0
    print(f"Inference done on {costs.numel()} instance(s): "
          f"mean={mean:.4f}, std={std:.4f}")

    for idx in range(min(3, costs.numel())):
        print(f"  Instance #{idx} | normalized_cost={costs[idx].item():.4f} "
              f"| raw_replay_cost={raw_replay_costs[idx].item():.4f}")

    print(f"Skipped customers: total={total_skipped}")
    print(f"Constraint violations: total_tw={total_tw_viol} "
          f"| total_appearance={total_appear_viol}")

    _print_routes(routes, costs, args.max_print_instances)

    if args.verify_routes:
        _verify_routes_cost(data, env_cls, env_params, routes, costs,
                            rollouts=args.verify_rollouts)

    if args.save_json is not None:
        _save_json(
            args.save_json,
            routes,
            costs,
            raw_replay_costs,
            route_diagnostics,
            constraint_diagnostics,
            raw_cost_components,
            normalized_cost_components,
        )
        print(f"Saved inference outputs to '{args.save_json}'")


if __name__ == "__main__":
    # Add OR-Tools-specific CLI arguments on top of the common ones.
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--time-limit-ms", type=int, default=10_000,
                         help="OR-Tools solver time limit per instance (ms)")
    _ort_args, _ = _parser.parse_known_args()
    args = parse_infer_args()
    args.ort_time_limit_ms = _ort_args.time_limit_ms
    main(args)
