"""
OR-Tools solver for VRP / VRPTW / DVRPTW.

Wraps Google OR-Tools' routing solver.  Works with **original-scale
(unnormalized)** data – coordinates, demands, time windows, service times,
vehicle capacity and speed should all be in the original problem units.

OR-Tools uses integer arithmetic, so all raw values are rounded to the
nearest integer before being passed to the solver.  This preserves enough
precision for the homberger / generated benchmarks in this project.

Usage
-----
    from externals._ort import ort_solve

    routes = ort_solve(raw_dataset, late_cost=1.0, time_limit_ms=30_000)
"""

import torch
from tqdm import tqdm
from multiprocessing import Pool

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ---------------------------------------------------------------------------
#  Single-instance solver  (runs inside each worker process)
# ---------------------------------------------------------------------------

def _solve_one(nodes: torch.Tensor, veh_count: int, veh_capa: int,
               veh_speed: float, late_cost: float,
               time_limit_ms: int = 30_000) -> list[list[int]]:
    """
    Build and solve one OR-Tools routing model.

    Parameters
    ----------
    nodes : Tensor[V, F]
        Node features (unnormalized): [x, y, demand, ready, due, service, *].
    veh_count : int
        Number of vehicles.
    veh_capa : int
        Vehicle capacity (same units as demand).
    veh_speed : float
        Speed  (distance unit / time unit).
    late_cost : float
        Penalty per unit of time-window violation (soft upper bound).
    time_limit_ms : int
        Solver time limit in milliseconds.

    Returns
    -------
    list[list[int]]
        One route per vehicle.  Each route is a list of 0-based customer
        indices (depot = 0 is not included).  Empty list means the vehicle
        was not used.
    """
    num_nodes = nodes.size(0)
    manager = pywrapcp.RoutingIndexManager(num_nodes, veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # ---- helpers ---------------------------------------------------------
    def _r(x: float) -> int:
        return int(round(x))

    def _euclid(i: int, j: int) -> float:
        dx = float(nodes[i, 0]) - float(nodes[j, 0])
        dy = float(nodes[i, 1]) - float(nodes[j, 1])
        return (dx * dx + dy * dy) ** 0.5

    # ---- distance callback (arc cost = Euclidean distance) ---------------
    def dist_cb(from_idx, to_idx):
        return _r(_euclid(manager.IndexToNode(from_idx),
                          manager.IndexToNode(to_idx)))

    dist_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    # ---- demand callback & capacity dimension ----------------------------
    def dem_cb(idx):
        return _r(float(nodes[manager.IndexToNode(idx), 2]))

    dem_cb_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_cb_idx, 0, [veh_capa] * veh_count, True, "Capacity"
    )

    # ---- disjunction: allow skipping customers with penalty -------------
    # Without this, ALL customers must be visited → likely infeasible
    # for large problems with tight time windows.
    pending_penalty = _r(float(veh_capa))  # high penalty to encourage serving
    for j in range(1, num_nodes):
        routing.AddDisjunction([manager.NodeToIndex(j)], pending_penalty)

    # ---- time dimension (only for problems with time windows) ------------
    has_tw = nodes.size(1) > 3

    if has_tw:
        horizon = _r(float(nodes[0, 4]))  # depot close time

        def time_cb(from_idx, to_idx):
            src = manager.IndexToNode(from_idx)
            dst = manager.IndexToNode(to_idx)
            # service time of the *source* node (standard OR-Tools pattern).
            # The depot (node 0) has no service time at departure.
            svc = float(nodes[src, 5]) if src != 0 else 0.0
            travel = _euclid(src, dst) / veh_speed
            return _r(svc + travel)

        time_cb_idx = routing.RegisterTransitCallback(time_cb)
        # slack_max = horizon  →  vehicles can wait for ready times
        # capacity  = max(2 * horizon, num_nodes * horizon)
        #   → enough headroom for any feasible route, even with 400+ nodes
        max_route_time = max(2 * horizon, num_nodes * horizon)
        routing.AddDimension(time_cb_idx, horizon,
                             max_route_time, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Soft time-window constraints  (penalty = late_cost per unit late)
        for j in range(1, num_nodes):
            ready = _r(float(nodes[j, 3]))
            due   = _r(float(nodes[j, 4]))
            idx = manager.NodeToIndex(j)
            time_dim.CumulVar(idx).SetMin(ready)
            time_dim.SetCumulVarSoftUpperBound(idx, due, _r(late_cost))

        # Penalty for returning to depot after the horizon
        for v in range(veh_count):
            time_dim.SetCumulVarSoftUpperBound(
                routing.End(v), horizon, _r(late_cost)
            )

    # Allow some vehicles to stay unused (penalty-free)
    for v in range(veh_count):
        routing.AddDisjunction([routing.Start(v)], 0)

    # ---- search parameters -----------------------------------------------
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_ms // 1000
    params.time_limit.nanos = (time_limit_ms % 1000) * 1_000_000

    assignment = routing.SolveWithParameters(params)
    if assignment is None:
        return [[] for _ in range(veh_count)]

    # ---- extract routes --------------------------------------------------
    routes: list[list[int]] = []
    for v in range(veh_count):
        route: list[int] = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            idx = assignment.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        # Remove trailing depot (it is implicit)
        if route and route[-1] == 0:
            route.pop()
        routes.append(route)

    return routes


# ---------------------------------------------------------------------------
#  Batch solver  (spawns a worker pool)
# ---------------------------------------------------------------------------

def ort_solve(data, late_cost: float = 1.0,
              time_limit_ms: int = 30_000) -> list[list[list[int]]]:
    """
    Solve all instances in *data* with OR-Tools in parallel.

    Parameters
    ----------
    data : Dataset
        Must have attributes: .batch_size, .veh_count, .veh_capa,
        .veh_speed, .nodes_gen().
    late_cost : float
        Penalty per unit of time-window violation.
    time_limit_ms : int
        Per-instance solver time limit in milliseconds.

    Returns
    -------
    list[list[list[int]]]
        ``result[i][v]`` = route of vehicle *v* for instance *i*.
    """
    if data.batch_size == 0:
        return []

    with Pool() as pool:
        results = []
        with tqdm(desc="OR-Tools", total=data.batch_size) as pbar:
            for nodes in data.nodes_gen():
                results.append(pool.apply_async(
                    _solve_one,
                    (nodes, data.veh_count, data.veh_capa,
                     data.veh_speed, late_cost, time_limit_ms),
                    callback=lambda _: pbar.update(),
                ))
            routes = [res.get() for res in results]

    return routes
