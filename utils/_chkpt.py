import torch
import os.path

def save_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_ep{}.pyth".format(ep+1)))

KEEP_LAST = 5

def save_checkpoint_in_train(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    out_name = "chkpt_ep{}.pyth".format(ep+1)
    out_path = os.path.join(args.output_dir, out_name)
    torch.save(checkpoint, out_path)

    # keep only the latest KEEP_LAST checkpoints (or args.keep_checkpoints if provided)
    try:
        keep = getattr(args, 'keep_checkpoints', KEEP_LAST)
        if keep is None:
            keep = KEEP_LAST
        files = [f for f in os.listdir(args.output_dir) if f.startswith("chkpt_ep") and f.endswith('.pyth')]

        def epoch_from_name(name):
            try:
                s = name[len("chkpt_ep"):-len(".pyth")]
                return int(s)
            except Exception:
                return -1

        files_sorted = sorted(files, key=epoch_from_name, reverse=True)
        for old in files_sorted[int(keep):]:
            p = os.path.join(args.output_dir, old)
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        # do not fail saving checkpoint because of cleanup errors
        pass

def load_checkpoint(args, learner, optim, baseline = None, lr_sched = None):
    checkpoint = torch.load(args.resume_state)
    learner.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic":
        baseline.load_state_dict(checkpoint["critic"])
    return checkpoint["ep"]

def load_model_weights(args, learner, device="cpu", strict=True):
    """
    Chỉ load trọng số model (policy network), dùng cho:
    - inference
    - fine-tune
    - eval
    - warm-start training (optimizer reset)

    Args:
        args.resume_state: đường dẫn checkpoint
        learner: nn.Module
        device: "cpu" | "cuda"
        strict: load strict hay không
    """
    checkpoint = torch.load(args.model_weight, map_location=device)

    # Trường hợp checkpoint lưu full dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Trường hợp checkpoint chỉ là state_dict
        state_dict = checkpoint

    missing, unexpected = learner.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(f"[load_model_weights] missing keys: {len(missing)}")
        print(f"[load_model_weights] unexpected keys: {len(unexpected)}")

    return learner