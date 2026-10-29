"""
Merge the metrics.json files from several finetune_clip_fusion.py runs into one table.

Usage:
    python compare_runs.py                                   # reads runs/*/metrics.json
    python compare_runs.py --runs "runs/v*/metrics.json" --out comparison.csv
"""
import argparse
import glob
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/head_model_runs/*/metrics.json", help="glob for metrics.json files")
ap.add_argument("--out", default="runs/comparison_tables/run_head_comparison.csv")
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(args.runs)):
    m = json.load(open(path))
    best = m["history"][m["best_epoch"] - 1]          # the epoch that was kept
    rows.append({
        "run": path.replace("\\", "/").split("/")[-2],
        "vision_layers": m["n_vision_layers"],
        "text_layers": m["n_text_layers"],
        "epochs": m["args"]["epochs"],
        "best_epoch": m["best_epoch"],
        "dev_f1": m["best_dev"],
        "dev_loss": best["dev_loss"],
        "dev_acc": best["dev_acc"],
        "test_acc": m["test"]["acc"],
        "test_f1": m["test"]["f1_weighted"],
        "test_macro_f1": m["test"]["f1_macro"],
        "params_M": m["n_params"] / 1e6,
        "minutes": m["minutes"],
    })

if not rows:
    raise SystemExit(f"no files matched {args.runs}")

df = pd.DataFrame(rows).sort_values("dev_f1", ascending=False).reset_index(drop=True)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv(args.out, index=False)
print(f"\nSaved to {args.out}\n")
print("What to look at:")
print("  - Rank by dev_f1 (higher is better); differences under ~0.01 are probably noise, so prefer the smaller model.")
print("  - Quote test_acc / test_f1 only for the run you chose on dev. Always-guess-informative scores ~0.715 test accuracy.")