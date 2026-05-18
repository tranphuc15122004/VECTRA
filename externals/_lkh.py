from dep import LKH_BIN , LKH_ENABLED
from tqdm import tqdm

from multiprocessing import Pool
import subprocess
import tempfile
import os
import os.path
import sys

def _call_lkh(nodes, veh_count, veh_capa, prefix = "/tmp/mardan_lkh0"):
    if LKH_BIN is None:
        raise RuntimeError("LKH binary not found. Please install LKH-3 solver.")
    tsp_path = "{}.tsp".format(prefix)
    with open(tsp_path, 'w') as tsp_f:
        tsp_f.write("NAME: temp\n")
        tsp_f.write("TYPE: {}\n".format("CVRPTW" if nodes.size(1) > 3 else "CVRP"))
        tsp_f.write("DIMENSION: {}\n".format(nodes.size(0)))
        tsp_f.write("VEHICLES: {}\n".format(veh_count))
        tsp_f.write("CAPACITY: {}\n".format(veh_capa))
        tsp_f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        tsp_f.write("EDGE_WEIGHT_FORMAT: FUNCTION\n")
        tsp_f.write("NODE_COORD_TYPE: TWOD_COORDS\n")
        tsp_f.write("NODE_COORD_SECTION\n")
        for j, (x, y) in enumerate(nodes[:, :2], start=1):
            tsp_f.write("{} {:.0f} {:.0f}\n".format(j, x, y))
        tsp_f.write("DEPOT_SECTION\n1\n-1\n")
        tsp_f.write("DEMAND_SECTION\n")
        for j, q in enumerate(nodes[:, 2], start=1):
            tsp_f.write("{} {:.0f}\n".format(j, q))
        if nodes.size(1) > 3:
            tsp_f.write("SERVICE_TIME_SECTION\n")
            for j, d in enumerate(nodes[:, 5], start=1):
                tsp_f.write("{} {:.0f}\n".format(j, d))
            tsp_f.write("TIME_WINDOW_SECTION\n")
            # Use max(ready_time, appearance_time) so LKH3 respects
            # dynamic appearance constraints when solving the static snapshot
            for j, (e, l, a) in enumerate(zip(nodes[:, 3], nodes[:, 4], nodes[:, 6]), start=1):
                tsp_f.write("{} {:.0f} {:.0f}\n".format(j, max(float(e), float(a)), l))

    par_path = "{}.par".format(prefix)
    tr_path = "{}.tour".format(prefix)
    with open(par_path, "w") as par_f:
        par_f.write("SPECIAL\n")
        par_f.write("PROBLEM_FILE = {}\n".format(tsp_path))
        par_f.write("MTSP_SOLUTION_FILE = {}\n".format(tr_path))
        par_f.write("MAX_TRIALS = 4000\n")
        par_f.write("RUNS = 2\n")
        par_f.write("TIME_LIMIT = 300\n")

    try:
        res = subprocess.run([LKH_BIN, par_path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=360)
    except subprocess.TimeoutExpired:
        print(f'LKH timeout for {prefix}', file=sys.stderr)
        return [[] for _ in range(veh_count)]
    except Exception as e:
        print(f'LKH execution error for {prefix}: {e}', file=sys.stderr)
        return [[] for _ in range(veh_count)]
    
    if res.returncode != 0:
        print(f'LKH Error for {prefix}: {res.returncode}', file=sys.stderr)

    try:
        with open(tr_path, 'r') as tr_f:
            lines = tr_f.readlines()
        return [[int(j)-1 for j in l.split('(',1)[0].split()[1:]] for l in lines[2:]]
    except FileNotFoundError:
        # LKH failed to find a feasible solution or write the tour
        return [[] for _ in range(veh_count)]


def _get_lkh_worker_count():
    value = os.environ.get("MARDAM_LKH_WORKERS", "")
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 0
    return max(1, workers)


def lkh_solve(data):
    if LKH_BIN is None:
        print("WARNING: LKH binary not found. Returning empty routes.", file=sys.stderr)
        return [[] for _ in range(data.batch_size)]

    with Pool(processes=_get_lkh_worker_count()) as p:
        with tqdm(desc="Calling LKH3", total=data.batch_size) as pbar:
            with tempfile.TemporaryDirectory(prefix="mardan_lkh") as tmp_dir:
                results = [p.apply_async(_call_lkh, (nodes, data.veh_count, data.veh_capa,
                    os.path.join(tmp_dir, str(b))), callback=lambda _: pbar.update())
                    for b, nodes in enumerate(data.nodes_gen())]
                routes = []
                for i, res in enumerate(results):
                    try:
                        route = res.get(timeout=600)
                        routes.append(route)
                    except Exception as e:
                        print(f"Error getting result for instance {i}: {e}", file=sys.stderr)
                        routes.append([])
    return routes
