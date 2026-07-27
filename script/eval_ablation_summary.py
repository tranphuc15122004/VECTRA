#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path

def get_node_count(path: Path) -> int:
    name = path.name
    try:
        part = name.split("_n")[1]
        return int(part.split("m")[0])
    except Exception:
        return 9999

def main():
    import argparse
    parser = argparse.ArgumentParser("Chạy ablation và thu thập báo cáo")
    parser.add_argument("--datasets-root", type=Path, default="datasets")
    parser.add_argument("--models-dir", type=Path, default="output/ablation")
    parser.add_argument("--output-dir", type=Path, default="infer_ablations")
    parser.add_argument("--ablations", nargs="+", default=["vectra", "b0", "b1", "b3", "b5", "edgeoff"])
    parser.add_argument("--skip_infer", action="store_true", help="Chỉ hiển thị kết quả đã có, không chạy infer")
    args = parser.parse_args()

    # Find pyth datasets
    datasets = sorted([p for p in args.datasets_root.glob("*.pyth")], key=get_node_count)
    if not datasets:
        print("Không tìm thấy file .pyth trong", args.datasets_root)
        return

    # Map ablations to models
    ablation_dirs = {}
    for abl in args.ablations:
        # User defined structure might vary. Try to discover args.json and chkpt_best.pyth
        # Specifically handle "vectra" if it's placed differently
        abl_path = args.models_dir / abl / "seed42"
        if not abl_path.exists():
            abl_path = args.models_dir / abl  # fallback

        args_json = abl_path / "args.json"
        chkpt = abl_path / "chkpt_best.pyth"
        if args_json.exists() and chkpt.exists():
            ablation_dirs[abl] = (args_json, chkpt)
        else:
            print(f"CẢNH BÁO: Không tìm thấy model cho {abl} tại {abl_path}", file=sys.stderr)

    results = {}
    inference_script = Path("script/infer_all_pyth.py")

    for abl, (cfg_path, model_path) in ablation_dirs.items():
        abl_out_dir = args.output_dir / abl
        results[abl] = {}
        
        if not args.skip_infer:
            print(f"[{abl}] Bắt đầu infer trên {len(datasets)} bộ...", file=sys.stderr)
            cmd = [
                sys.executable, str(inference_script),
                "--datasets-root", str(args.datasets_root),
                "--config-file", str(cfg_path),
                "--model-weight", str(model_path),
                "--output-dir", str(abl_out_dir),
                "--greedy"
            ]
            subprocess.run(cmd)

        for ds in datasets:
            json_file = abl_out_dir / f"{ds.stem}.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                    stats = data.get("summary_stats", {})
                    mean = stats.get("mean", 0.0)
                    std = stats.get("std", 0.0)
                    time = data.get("inference_time", 0.0)
                    results[abl][ds.name] = {"mean": mean, "std": std, "time": time}
            else:
                results[abl][ds.name] = None

    # In báo cáo theo định dạng yêu cầu
    print("\n" + "="*40)
    print("THỐNG KÊ CHI PHÍ (mean ± std)")
    print("="*40)
    for abl in ablation_dirs.keys():
        print(abl)
        for ds in datasets:
            res = results.get(abl, {}).get(ds.name)
            if res is not None:
                print(f"{res['mean']:.4f} \u00b1 {res['std']:.4f}")
            else:
                print("N/A")

    print("\n" + "="*40)
    print("THỜI GIAN (giây)")
    print("="*40)
    for abl in ablation_dirs.keys():
        print(abl)
        for ds in datasets:
            res = results.get(abl, {}).get(ds.name)
            if res is not None:
                # Format exactly as requested (using comma decimal optionally, but we use . or standard format)
                # Dùng str thay thế . bằng , 
                time_str = f"{res['time']:.1f}".replace(".", ",")
                print(time_str)
            else:
                print("N/A")

if __name__ == "__main__":
    main()
