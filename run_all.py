"""End-to-end: dataset -> baselines/models -> experiments A/B/C -> figures -> results/summary.md

    python run_all.py                      # simulated labels
    python run_all.py --data data/measured_a100.csv   # real labels from measure_gpu.py
"""
import argparse
import os
import sys
import pandas as pd

from rooflinelab.simulate import build_sim_dataset
from rooflinelab.experiments import run_all, summarize, MODEL_NAMES
from rooflinelab import plots


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sim")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-launch-dominated", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figdir = os.path.join(a.out, "figs")
    os.makedirs(figdir, exist_ok=True)

    if a.data == "sim":
        df = build_sim_dataset(a.seed)
        source = "SIMULATED labels (see rooflinelab/simulate.py) - not hardware ground truth"
    else:
        df = pd.read_csv(a.data)
        source = f"measured labels from {a.data}"
    n0 = len(df)
    if not a.keep_launch_dominated and "launch_frac" in df and df["launch_frac"].notna().any():
        df = df[df["launch_frac"] <= 0.5].reset_index(drop=True)
    df.to_csv(os.path.join(a.out, "dataset_used.csv"), index=False)
    print(f"{source}\nrows: {n0} -> {len(df)} after dropping launch-dominated kernels")
    print("label balance (binary):", df.y_bin.value_counts(normalize=True).round(3).to_dict())

    res = pd.concat([run_all(df, 2), run_all(df, 3)], ignore_index=True)
    res.to_csv(os.path.join(a.out, "all_folds.csv"), index=False)
    summ = summarize(res)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    for c in (2, 3):
        for f in ("full", "raw"):
            plots.fig_accuracy(summ, figdir, c, f, "accuracy")
    plots.fig_accuracy(summ, figdir, 2, "full", "near_ridge_acc", "near_ridge_2class_full.png")
    plots.fig_accuracy(summ, figdir, 2, "raw", "near_ridge_acc", "near_ridge_2class_raw.png")
    plots.fig_per_group(res, figdir, "A_unseen_workload", 2, "raw")
    plots.fig_per_group(res, figdir, "B_unseen_hardware", 2, "raw")
    if "r_ratio" in df:
        plots.fig_ratio_scatter(df, figdir)
    if "ach_gflops" in df:
        plots.fig_roofline(df, figdir, "A100-40", "fp16")
    imp = plots.fig_importance(df, figdir)

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(f"# Results\n\nData: {source}\n\nRows used: {len(df)} (of {n0}).\n\n")
        for c in (2, 3):
            for feats in ("full", "raw"):
                f.write(f"## {c}-class, features = {feats}\n\n")
                t = summ[(summ.classes == c) & (summ.features == feats)].drop(columns=["classes", "features"])
                f.write(md_table(t) + "\n\n")
        f.write("## Permutation importance (GradBoost, raw features, held-out pairs)\n\n")
        for k, v in list(imp.items())[:10]:
            f.write(f"- {k}: {v:.4f}\n")
    print(open(os.path.join(a.out, "summary.md")).read())


if __name__ == "__main__":
    sys.exit(main())
