"""Figures for the real-data study (run after train_real.py):  python make_real_figs.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rooflinelab.realdata import OPS, load_real

COL = {"gemm": "#0072B2", "gemm_fp8": "#56B4E9", "attn": "#D55E00", "rmsnorm": "#009E73",
       "silu_mul": "#E69F00", "moe_w13": "#CC79A7"}
MCOL = {"Roofline": "#000000", "ConstEff/op": "#999999", "PowerLaw/op": "#E69F00", "GradBoost": "#0072B2"}
OUT = "results/real/figs"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_eff_vs_ratio(df):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(-4, 4, 25)
    for op in OPS:
        d = df[df.op == op]
        if d.empty:
            continue
        c = pd.cut(d.log_ratio, bins)
        g = d.groupby(c, observed=True).eff.agg(["median", "size"])
        g = g[g["size"] >= 50]
        x = [iv.mid for iv in g.index]
        ax.plot(x, g["median"], color=COL[op], lw=2, label=op)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("log10(arithmetic intensity / ridge point)   [<0 memory side | >0 compute side]")
    ax.set_ylabel("median efficiency = Roofline ideal time / measured time")
    ax.set_title("How far real kernels fall below Roofline (measured, 11 GPUs)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficiency_vs_ratio.png"), dpi=140)
    plt.close(fig)


def fig_errors(summ, split):
    s = summ[(summ.split == split) & (summ.op != "ALL")]
    ops = [o for o in OPS if o in set(s.op)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for i, m in enumerate(["Roofline", "ConstEff/op", "PowerLaw/op", "GradBoost"]):
        v = [s[(s.op == o) & (s.method == m)].mape_median.mean() * 100 for o in ops]
        ax.bar(np.arange(len(ops)) + (i - 1.5) * w, v, w * 0.9, color=MCOL[m], label=m)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops)
    ax.set_ylabel("median relative time error (%)")
    ax.set_title(f"{split}: error of predicted kernel time (lower is better)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"error_{split}.png"), dpi=140)
    plt.close(fig)


def fig_pred_vs_actual(preds, split):
    p = preds[preds.split == split]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hb = ax.hexbin(p.y, p.yhat, gridsize=70, bins="log", cmap="Blues", mincnt=1, extent=(-0.5, 3, -0.5, 3))
    ax.plot([-0.5, 3], [-0.5, 3], color="black", lw=1)
    ax.set_xlabel("measured  log10(time / Roofline ideal)")
    ax.set_ylabel("predicted  log10(time / Roofline ideal)")
    ax.set_title(f"{split}: predicted vs measured (0 = on the roofline)")
    fig.colorbar(hb, ax=ax, label="kernels (log)")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pred_vs_actual_{split}.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_real("data/real", log=lambda *_: None)
    summ = pd.read_csv("results/real/summary.csv")
    preds = pd.read_csv("results/real/cv_predictions.csv.gz")
    fig_eff_vs_ratio(df)
    for sp in ("unseen_arch", "unseen_shape"):
        fig_errors(summ, sp)
        fig_pred_vs_actual(preds, sp)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
