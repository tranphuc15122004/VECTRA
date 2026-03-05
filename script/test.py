from _learner import AttentionLearner
from MODEL.model import *
from problems import *
from baselines import *
from externals import *
from dep import *
from utils import *
from layers import reinforce_loss
from  baselines._base import Baseline
import torch
from torch.optim import Adam 
from torch.optim.lr_scheduler import LambdaLR
from problems import *
import time
import os
from torch.nn.parameter import UninitializedParameter


def _count_params(model):
    total = 0
    for p in model.parameters():
        if not p.requires_grad or isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total

def test_epoch(args, test_env, learner, ref_costs):
    learner.eval()
    with torch.no_grad():   # 🔥 BẮT BUỘC
        
        if args.problem_type[0] == "s":
            costs = test_env.nodes.new_zeros(test_env.minibatch_size)
            for _ in range(100):
                _, _, rewards = learner(test_env)
                costs -= torch.stack(rewards).sum(0).squeeze(-1)
            costs = costs / 100
        else:
            _, _, rs = learner(test_env)
            costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
        mean = costs.mean()
        std = costs.std()
        gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()

def val_epoch(args, test_env, learner):
    learner.eval()
    with torch.no_grad():   # 🔥 BẮT BUỘC
        if args.problem_type[0] == "s":
            costs = test_env.nodes.new_zeros(test_env.minibatch_size)
            for _ in range(100):
                _, _, rewards = learner(test_env)
                costs -= torch.stack(rewards).sum(0).squeeze(-1)
            costs = costs / 100
        else:
            _, _, rs = learner(test_env)
            costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
        mean = costs.mean()
        std = costs.std()

    print("Cost on test dataset: {:5.2f} +- {:5.2f}".format(mean, std))
    return mean.item(), std.item()


def print_forward_profiling(learner):
    summary, total = learner.get_forward_profiling_summary()
    if not summary:
        return
    print(f"Forward profiling (total {total * 1000:.2f} ms)")
    for name, duration, pct in summary:
        print(f"  {name:<24}: {duration * 1000:.2f} ms ({pct:4.1f}%)")

def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_random_seed(args.rng_seed, deterministic = True)

    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs): pass

    # PROBLEM
    Dataset = {
            "vrp": VRP_Dataset,
            "vrptw": VRPTW_Dataset,
            "svrptw": VRPTW_Dataset,
            "sdvrptw": SDVRPTW_Dataset,
            "dvrptw" : DVRPTW_Dataset
            }.get(args.problem_type)
    gen_params = [
            args.customers_count,
            args.vehicles_count,
            args.veh_capa,
            args.veh_speed,
            args.min_cust_count,
            args.loc_range,
            args.dem_range
            ]
    if args.problem_type != "vrp":
        gen_params.extend( [args.horizon, args.dur_range, args.tw_ratio, args.tw_range] )
    if args.problem_type == "sdvrptw" or  args.problem_type == "dvrptw":
        gen_params.extend( [args.deg_of_dyna, args.appear_early_ratio] )

    # TRAIN DATA
    verbose_print("Generating {} {} samples of training data...".format(
        args.iter_count * args.batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    train_data = Dataset.generate(
            args.iter_count * args.batch_size,
            *gen_params
            )
    train_data.normalize()
    verbose_print("Done.")

    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    test_data = Dataset.generate(
            args.test_batch_size,
            *gen_params
            )
    verbose_print("Done.")

    if ORTOOLS_ENABLED:
        ref_routes = ort_solve(test_data)
    elif LKH_ENABLED:
        ref_routes = lkh_solve(test_data)
    else:
        ref_routes = None
        print("Warning! No external solver found to compute gaps for test.")
    test_data.normalize()

    # ENVIRONMENT
    Environment = {
            "vrp": VRP_Environment,
            "vrptw": VRPTW_Environment,
            "svrptw": SVRPTW_Environment,
            "sdvrptw": SDVRPTW_Environment,
            "dvrptw": DVRPTW_Environment
            }.get(args.problem_type)
    env_params = [args.pending_cost]
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type != "vrptw" and args.problem_type != "dvrptw":
            env_params.extend( [args.speed_var, args.late_prob, args.slow_down, args.late_var] )
    print(env_params)
    test_env = Environment(test_data, None, None, *env_params)

    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == 's' else 1)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    test_env.nodes = test_env.nodes.to(dev)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    # MODEL
    verbose_print("Initializing attention model...",
        end = " ", flush = True)
    learner : torch.Module = VECTRA(
            Dataset.CUST_FEAT_SIZE,
            Environment.VEH_STATE_SIZE,
            model_size = args.model_size,
            layer_count = args.layer_count,
            head_count = args.head_count,
            ff_size = args.ff_size,
            tanh_xplor = args.tanh_xplor,
            greedy = False,
            edge_feat_size = args.edge_feat_size,
            cust_k = args.cust_k,
            memory_size = args.memory_size,
            lookahead_hidden = args.lookahead_hidden,
            dropout = args.dropout,
            adaptive_depth = args.adaptive_depth,
            adaptive_min_layers = args.adaptive_min_layers,
            adaptive_easy_ratio = args.adaptive_easy_ratio,
            latent_bottleneck = args.latent_bottleneck,
            latent_tokens = args.latent_tokens,
            latent_min_nodes = args.latent_min_nodes,
            )


    """ learner : torch.Module = AttentionLearner(
		Dataset.CUST_FEAT_SIZE,
        Environment.VEH_STATE_SIZE,
		128,
		3,
		8,
		512,
		args.tanh_xplor,
	) """
    learner.to(dev)
    verbose_print("Done.")

    # Initialize lazy parameters before reporting parameter count.
    with torch.no_grad():
        dyna_warmup = Environment(train_data, train_data.nodes[:1].to(dev), None, *env_params)
        _ = learner(dyna_warmup)
    
    print("VECTRA params:", _count_params(learner))

    # BASELINE
    verbose_print("Initializing '{}' baseline...".format(
        args.baseline_type),
        end = " ", flush = True)
    baseline : Baseline = None
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
    baseline.to(dev)
    verbose_print("Done.")

    # CHECKPOINTING
    verbose_print("Creating output dir...",
        end = " ", flush = True)
    args.output_dir = "./output/{}n{}m{}_{}".format(
            args.problem_type.upper(),
            args.customers_count,
            args.vehicles_count,
            time.strftime("%y%m%d-%H%M")
            ) if args.output_dir is None else args.output_dir
    os.makedirs(args.output_dir, exist_ok = True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))
        
    
    load_model_weights(args , learner )
    

    verbose_print("Running...")

    if hasattr(learner, "reset_forward_profiling"):
        learner.reset_forward_profiling()
    val_epoch(args , test_env , learner)
    if hasattr(learner, "get_forward_profiling_summary"):
        print_forward_profiling(learner)


if __name__ == "__main__":
    main(parse_args())
