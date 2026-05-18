"""
Dynamic LKH3 solver for DVRPTW — truly online, seeing only visible customers.

Uses LKH3 as an online rolling-horizon policy:
- At each decision point, LKH3 sees ONLY currently visible + unserved customers
- Multi-vehicle CVRPTW is solved, but only the active vehicle's next action is used
- Re-optimization happens at every step (when a vehicle needs a new action)
- Each vehicle makes exactly ONE trip

This provides a FAIR comparison with learned policies (COAST/VECTRA):
same information (visible customers only), same trip constraint.
"""
from __future__ import annotations

import torch
from dep import LKH_BIN

import subprocess
import tempfile
import os
import sys


def _call_lkh_subset(
    nodes: torch.Tensor,
    veh_count: int,
    veh_capa: float,
    customer_indices: list[int],
    prefix: str = "/tmp/mardan_lkh_subset",
) -> list[list[int]]:
    """
    Run multi-vehicle LKH3 on a SUBSET of customers.

    Only the specified customer_indices are included (plus depot at index 0).
    Returns routes in terms of ORIGINAL customer indices.

    Parameters
    ----------
    nodes : Tensor[V, 7]
        Full node data in original scale.
    veh_count : int
        Number of vehicles.
    veh_capa : float
        Vehicle capacity.
    customer_indices : list[int]
        Which customers (original indices) to include.
    prefix : str
        Temp file prefix.

    Returns
    -------
    list[list[int]]
        One route per vehicle (original customer indices, depot=0 excluded).
    """
    if LKH_BIN is None or not customer_indices:
        return [[] for _ in range(veh_count)]

    # Build mapping: new_index -> original_index
    # Node 0 = depot (always index 0)
    # Node 1..N = customers in customer_indices order
    n_cust = len(customer_indices)

    tsp_path = f"{prefix}.tsp"
    with open(tsp_path, 'w') as f:
        f.write("NAME: dyn_subset\n")
        f.write("TYPE: CVRPTW\n")
        f.write(f"DIMENSION: {n_cust + 1}\n")
        f.write(f"VEHICLES: {min(veh_count, n_cust)}\n")
        f.write(f"CAPACITY: {int(veh_capa)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("EDGE_WEIGHT_FORMAT: FUNCTION\n")
        f.write("NODE_COORD_TYPE: TWOD_COORDS\n")
        f.write("NODE_COORD_SECTION\n")

        # Depot
        f.write(f"1 {float(nodes[0,0]):.0f} {float(nodes[0,1]):.0f}\n")
        # Customers
        for j, orig_idx in enumerate(customer_indices, start=2):
            x = float(nodes[orig_idx, 0])
            y = float(nodes[orig_idx, 1])
            f.write(f"{j} {x:.0f} {y:.0f}\n")

        f.write("DEPOT_SECTION\n1\n-1\n")

        f.write("DEMAND_SECTION\n")
        f.write(f"1 0\n")
        for j, orig_idx in enumerate(customer_indices, start=2):
            f.write(f"{j} {float(nodes[orig_idx, 2]):.0f}\n")

        f.write("SERVICE_TIME_SECTION\n")
        f.write(f"1 0\n")
        for j, orig_idx in enumerate(customer_indices, start=2):
            f.write(f"{j} {float(nodes[orig_idx, 5]):.0f}\n")

        f.write("TIME_WINDOW_SECTION\n")
        depot_due = float(nodes[0, 4])
        f.write(f"1 0 {depot_due:.0f}\n")
        for j, orig_idx in enumerate(customer_indices, start=2):
            # ready = max(open, appear) to respect dynamic constraint
            ready = max(float(nodes[orig_idx, 3]), float(nodes[orig_idx, 6]))
            due = float(nodes[orig_idx, 4])
            f.write(f"{j} {ready:.0f} {due:.0f}\n")

    par_path = f"{prefix}.par"
    tr_path = f"{prefix}.tour"
    with open(par_path, 'w') as f:
        f.write("SPECIAL\n")
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"MTSP_SOLUTION_FILE = {tr_path}\n")
        f.write("MAX_TRIALS = 300\n")
        f.write("RUNS = 1\n")
        f.write("TIME_LIMIT = 10\n")

    try:
        res = subprocess.run(
            [LKH_BIN, par_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, Exception):
        return [[] for _ in range(veh_count)]

    if res.returncode != 0:
        return [[] for _ in range(veh_count)]

    # Parse MTSP_SOLUTION_FILE (same format as original _call_lkh)
    try:
        with open(tr_path, 'r') as f:
            lines = f.readlines()
        # Format: each line after header is a vehicle route
        # Route: "1 5 6 1 (#2)  Cost: 67" or similar
        # Original parser: split by '(', take [0], split, skip first, int-1
        routes = []
        for line in lines[2:]:  # Skip NAME and "The tours..." header
            line = line.strip()
            if not line:
                continue
            before_paren = line.split('(', 1)[0]
            parts = before_paren.split()
            if not parts:
                continue
            # parts[0] is the starting depot (1), parts[1:] are customer nodes
            route = []
            for p in parts[1:]:
                try:
                    node_id = int(p) - 1  # 1-based to 0-based
                except ValueError:
                    continue
                if node_id <= 0:
                    continue  # Skip depot
                orig_idx = customer_indices[node_id - 1]  # node_id 1-based customer
                route.append(orig_idx)
            routes.append(route)
        # Pad to veh_count
        while len(routes) < veh_count:
            routes.append([])
        return routes[:veh_count]
    except (FileNotFoundError, ValueError, IndexError):
        return [[] for _ in range(veh_count)]


def lkh_dynamic_solve_single(
    raw_nodes: torch.Tensor,
    veh_count: int,
    veh_capa: int,
    veh_speed: float,
    pending_cost: float = 2.0,
    late_cost: float = 1.0,
) -> tuple[list[list[int]], float]:
    """
    Solve ONE DVRPTW instance with LKH3 as a rolling-horizon policy.

    At each step, LKH3 re-plans for ALL vehicles using only the currently
    visible + unserved customers. The active vehicle takes its first action
    from the new plan.
    """
    nodes = raw_nodes.clone()
    num_nodes = nodes.size(0)
    depot_pos = (float(nodes[0, 0]), float(nodes[0, 1]))
    depot_due = float(nodes[0, 4])

    veh_pos = [depot_pos for _ in range(veh_count)]
    veh_time = [0.0 for _ in range(veh_count)]
    veh_cap_rem = [float(veh_capa) for _ in range(veh_count)]
    veh_started = [False for _ in range(veh_count)]  # Has vehicle left depot?
    veh_done = [False for _ in range(veh_count)]
    veh_routes: list[list[int]] = [[] for _ in range(veh_count)]

    served = [False] * num_nodes
    served[0] = True

    total_distance = 0.0
    total_late = 0.0

    # Current plan from last LKH3 call
    current_plan: list[list[int]] = [[] for _ in range(veh_count)]

    with tempfile.TemporaryDirectory(prefix="mardan_lkh_online") as tmp_dir:
        call_count = 0

        while True:
            # Find active vehicle with minimum time
            active = [(i, veh_time[i]) for i in range(veh_count) if not veh_done[i]]
            if not active:
                break

            veh_idx, cur_time = min(active, key=lambda x: x[1])

            # Collect visible + unserved customers
            visible_ids = []
            for c in range(1, num_nodes):
                if not served[c] and float(nodes[c, 6]) <= cur_time:
                    visible_ids.append(c)

            # If no visible customers, advance time
            if not visible_ids:
                future = [float(nodes[c, 6]) for c in range(1, num_nodes)
                         if not served[c] and float(nodes[c, 6]) > cur_time]
                if future:
                    next_t = min(future)
                    for i in range(veh_count):
                        if not veh_done[i]:
                            veh_time[i] = max(veh_time[i], next_t)
                    current_plan = [[] for _ in range(veh_count)]  # Invalidate plan
                    continue
                else:
                    break

            # Re-optimize with visible subset
            call_count += 1
            current_plan = _call_lkh_subset(
                nodes,
                veh_count,
                veh_capa,
                visible_ids,
                os.path.join(tmp_dir, f"step{call_count}"),
            )

            # Get action for active vehicle from plan
            plan_for_veh = current_plan[veh_idx] if veh_idx < len(current_plan) else []
            if not plan_for_veh:
                # No planned customers → go to depot
                if veh_started[veh_idx]:
                    dist = ((veh_pos[veh_idx][0] - depot_pos[0])**2 +
                            (veh_pos[veh_idx][1] - depot_pos[1])**2)**0.5
                    total_distance += dist
                    arr = veh_time[veh_idx] + dist / veh_speed
                    total_late += max(0.0, arr - depot_due)
                veh_done[veh_idx] = True
                continue

            cust_id = plan_for_veh[0]
            current_plan[veh_idx] = plan_for_veh[1:]  # Consume action

            # Execute: travel to customer
            cx, cy = float(nodes[cust_id, 0]), float(nodes[cust_id, 1])
            dist = ((veh_pos[veh_idx][0] - cx)**2 + (veh_pos[veh_idx][1] - cy)**2)**0.5
            travel_time = dist / veh_speed
            raw_arrival = veh_time[veh_idx] + travel_time

            ready_t = float(nodes[cust_id, 3])
            due_t = float(nodes[cust_id, 4])
            service_t = float(nodes[cust_id, 5])
            appear_t = float(nodes[cust_id, 6])
            demand = float(nodes[cust_id, 2])

            start_service = max(raw_arrival, ready_t, appear_t)
            late = max(0.0, start_service - due_t)

            if veh_cap_rem[veh_idx] < demand:
                # Can't serve, skip
                continue

            veh_cap_rem[veh_idx] -= demand
            served[cust_id] = True
            veh_routes[veh_idx].append(cust_id)
            veh_started[veh_idx] = True

            total_distance += dist
            total_late += late

            veh_time[veh_idx] = start_service + service_t
            veh_pos[veh_idx] = (cx, cy)

            # Invalidate plan for other vehicles (state changed)
            current_plan = [[] for _ in range(veh_count)]

    # Force remaining vehicles to return to depot
    for i in range(veh_count):
        if not veh_done[i] and veh_started[i]:
            dist = ((veh_pos[i][0] - depot_pos[0])**2 +
                    (veh_pos[i][1] - depot_pos[1])**2)**0.5
            total_distance += dist
            arr = veh_time[i] + dist / veh_speed
            total_late += max(0.0, arr - depot_due)
            veh_done[i] = True

    skipped = sum(1 for c in range(1, num_nodes) if not served[c])
    total_cost = total_distance + late_cost * total_late + pending_cost * skipped

    return veh_routes, total_cost


def lkh_dynamic_solve(
    raw_data,
    pending_cost: float = 2.0,
    late_cost: float = 1.0,
) -> tuple[list[list[list[int]]], torch.Tensor]:
    """Batch dynamic LKH3 solver."""
    from tqdm import tqdm

    all_routes = []
    all_costs = []
    for inst_idx, nodes in enumerate(
        tqdm(raw_data.nodes, desc="LKH3 online dynamic")
    ):
        routes, cost = lkh_dynamic_solve_single(
            nodes, raw_data.veh_count, raw_data.veh_capa,
            raw_data.veh_speed, pending_cost, late_cost,
        )
        all_routes.append(routes)
        all_costs.append(cost)

    costs = torch.tensor(all_costs, dtype=torch.float)
    return all_routes, costs


# ---- Replay functions (used by a-priori mode) ----

def replay_lkh_routes_dynamic(
    raw_nodes: torch.Tensor,
    routes: list[list[int]],
    veh_count: int,
    veh_capa: int,
    veh_speed: float,
    pending_cost: float = 2.0,
    late_cost: float = 1.0,
) -> tuple[float, dict]:
    """Replay pre-computed routes in dynamic simulation."""
    nodes = raw_nodes.clone()
    num_nodes = nodes.size(0)
    total_distance = 0.0
    total_late = 0.0
    visited = set()

    for veh_idx in range(veh_count):
        route = routes[veh_idx] if veh_idx < len(routes) else []
        cur_pos = nodes[0, :2].clone()
        cur_time = 0.0
        cur_cap = float(veh_capa)

        for cust_id in route:
            if cust_id <= 0 or cust_id >= num_nodes:
                continue
            dest = nodes[cust_id]
            dist = float(torch.norm(cur_pos - dest[:2]))
            raw_arrival = cur_time + dist / veh_speed
            start_service = max(raw_arrival, float(dest[3]), float(dest[6]))
            late = max(0.0, start_service - float(dest[4]))
            if cur_cap < float(dest[2]):
                continue
            cur_cap -= float(dest[2])
            total_distance += dist
            total_late += late
            visited.add(cust_id)
            cur_time = start_service + float(dest[5])
            cur_pos = dest[:2].clone()

        if route:
            dist_depot = float(torch.norm(cur_pos - nodes[0, :2]))
            total_distance += dist_depot
            depot_arr = cur_time + dist_depot / veh_speed
            total_late += max(0.0, depot_arr - float(nodes[0, 4]))

    active = set(range(1, num_nodes))
    skipped = len(active - visited)
    total_cost = total_distance + late_cost * total_late + pending_cost * skipped
    return total_cost, {
        "distance": total_distance, "late_time": total_late,
        "late_penalty": late_cost * total_late,
        "skipped_orders": skipped,
        "skipped_penalty": pending_cost * skipped,
        "total_cost": total_cost, "visited_count": len(visited),
    }


def replay_lkh_routes_dynamic_batch(raw_data, routes, pending_cost=2.0, late_cost=1.0):
    """Batch dynamic replay."""
    from tqdm import tqdm
    all_costs = []
    for inst_idx, nodes in enumerate(tqdm(raw_data.nodes, desc="Dynamic replay")):
        inst_routes = routes[inst_idx] if inst_idx < len(routes) else []
        cost, _ = replay_lkh_routes_dynamic(
            nodes, inst_routes, raw_data.veh_count, raw_data.veh_capa,
            raw_data.veh_speed, pending_cost, late_cost,
        )
        all_costs.append(cost)
    return torch.tensor(all_costs, dtype=torch.float)


