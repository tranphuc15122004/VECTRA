import torch
import math
import os.path
from itertools import repeat, zip_longest

def actions_to_routes(actions, batch_size, veh_count):
    routes = [[[] for i in range(veh_count)] for b in range(batch_size)]
    for veh_idx, cust_idx in actions:
        for b, (i,j) in enumerate(zip(veh_idx, cust_idx)):
            routes[b][i.item()].append(j.item())
    return routes


def routes_to_string(routes):
    return '\n'.join(
        '0 -> ' + ' -> '.join(str(j) for j in route)
        for route in routes
        )


""" def export_train_test_stats(args, start_ep, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
            ))
        for ep, (tr,te) in enumerate( zip_longest(train_stats, test_stats, fillvalue=float('nan')), start = start_ep):
            f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(8)) + '\n').format(
                ep, *tr, *te)) """


def export_train_test_stats(args, start_ep, train_stats, test_stats):
    os.makedirs(args.output_dir, exist_ok=True)
    fpath = os.path.join(args.output_dir, "loss_gap.csv")

    header = [
        "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM",
        "#TEST_MU", "#TEST_STD", "#TEST_GAP"
    ]

    write_header = not os.path.exists(fpath)

    def safe_vals(vals, n):
        """Pad hoặc cắt để đủ n giá trị"""
        if vals is None or (isinstance(vals, float) and math.isnan(vals)):
            return [float("nan")] * n
        vals = list(vals)
        return (vals + [float("nan")] * n)[:n]

    with open(fpath, "a") as f:
        if write_header:
            f.write(" ".join(f"{h:>16}" for h in header) + "\n")

        for ep, (tr, te) in enumerate(
            zip_longest(train_stats, test_stats, fillvalue=None),
            start=start_ep
        ):
            tr_vals = safe_vals(tr, 5)
            te_vals = safe_vals(te, 3)

            f.write(
                f"{ep:>16d}" +
                "".join(f"{v:>16.3g}" for v in (*tr_vals, *te_vals)) +
                "\n"
            )
    
def update_train_test_stats(args, ep, train_stats, val_stats):
    os.makedirs(args.output_dir, exist_ok=True)
    fpath = os.path.join(args.output_dir, "train_statistics.csv")

    header = [
        "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM",
        "#VAL_MU", "#VAL_STD"
    ]

    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().item()
        try:
            return float(x)
        except Exception:
            return float("nan")

    def get_latest(vals):
        """Lấy phần tử mới nhất nếu là list"""
        if vals is None:
            return None
        if isinstance(vals, (list, tuple)):
            return vals[-1]
        return vals

    def safe_vals(vals, n):
        vals = get_latest(vals)
        if vals is None:
            return [float("nan")] * n

        vals = list(vals)
        vals = [to_float(v) for v in vals]

        return (vals + [float("nan")] * n)[:n]

    write_header = not os.path.exists(fpath)

    # ---- đúng với dữ liệu của bạn ----
    tr_vals = safe_vals(train_stats, 5)
    va_vals = safe_vals(val_stats, 2)

    with open(fpath, "a") as f:
        if write_header:
            f.write(" ".join(f"{h:>16}" for h in header) + "\n")

        f.write(
            f"{ep:>16d}" +
            "".join(f"{v:>16.3g}" for v in (*tr_vals, *va_vals)) +
            "\n"
        )

def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)


def eval_apriori_routes(dyna, routes, rollout_count):
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch_size)
    for c in range(rollout_count):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in routes]
        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
                for n,i in enumerate(dyna.cur_veh_idx)], dtype = torch.int64)
            rewards.append( dyna.step(cust_idx) )
        mean_cost += -torch.stack(rewards).sum(dim = 0).squeeze(-1)
    return mean_cost / rollout_count


def load_old_weights(learner, state_dict):
    learner.load_state_dict(state_dict)
    for layer in learner.cust_encoder.children():
        layer.mha._inv_sqrt_d = layer.mha.key_size_per_head**0.5
    learner.fleet_attention._inv_sqrt_d = learner.fleet_attention.key_size_per_head**0.5
    learner.veh_attention._inv_sqrt_d = learner.veh_attention.key_size_per_head**0.5

