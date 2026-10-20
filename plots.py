"""Figures. Colour-blind-safe (Okabe-Ito) palette, fixed model order, one y-axis per chart."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from .experiments import MODEL_NAMES, make_model, heldout_pairs
from .features import FEATURES_FULL, FEATURES_RAW
from .hw_specs import HARDWARE

COL = {"Roofline": "#000000", "LogReg": "#E69F00", "RandomForest": "#009E73",
       "GradBoost": "#0072B2", "MLP": "#CC79A7"}
LAB = {0: "memory-bound", 1: "compute-bound"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_accuracy(summary, out, classes=2, features="full", metric="accuracy", fname=None):
    s = summary[(summary.classes == classes) & (summary.features == features)]
    exps = list(dict.fromkeys(s.experiment))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.16
    for i, m in enumerate(MODEL_NAMES):
        vals = [s[(s.experiment == e) & (s.model == m)][metric].mean() for e in exps]
        ax.bar(np.arange(len(exps)) + (i - 2) * w, vals, w * 0.9, color=COL[m], label=m)
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels(exps, fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    lo = np.nanmin(s[metric]) if len(s) else 0
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    ax.set_title(f"{classes}-class, features={features}")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname or f"acc_{classes}class_{features}_{metric}.png"), dpi=140)
    plt.close(fig)


def fig_per_group(res, out, experiment, classes=2, features="full"):
    r = res[(res.classes == classes) & (res.features == features) & (res.experiment == experiment)]
    folds = sorted(r.fold.unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    models = ["Roofline", "LogReg", "GradBoost"]
    w = 0.26
    for i, m in enumerate(models):
        vals = [r[(r.fold == f) & (r.model == m)].accuracy.mean() for f in folds]
        ax.bar(np.arange(len(folds)) + (i - 1) * w, vals, w * 0.9, color=COL[m], label=m)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(folds, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"{experiment}: accuracy on each held-out group")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"per_group_{experiment}.png"), dpi=140)
    plt.close(fig)


def fig_ratio_scatter(df, out):
    """Predicted (analytic AI/ridge) vs measured ratio r = t_c/t_m. Off-diagonal = Roofline error."""
    d = df.sample(min(len(df), 6000), random_state=0)
    x = d["log_ai_over_ridge"]
    y = np.log10(d["r_ratio"])
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(x, y, s=4, alpha=0.35, color="#0072B2", linewidths=0)
    lim = [-4, 4]
    ax.plot(lim, lim, color="black", lw=1.2, label="Roofline: measured = AI/ridge")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("log10(AI / ridge point)   [Roofline prediction]")
    ax.set_ylabel("log10(compute time / memory time)   [measured]")
    ax.text(-3.9, 3.4, "Roofline says memory,\nmeasured compute", fontsize=8, color="#444")
    ax.text(1.2, -3.8, "Roofline says compute,\nmeasured memory", fontsize=8, color="#444")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.88))
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roofline_vs_measured.png"), dpi=140)
    plt.close(fig)


def fig_roofline(df, out, hw_name="A100-40", precision="fp16"):
    hw = HARDWARE[hw_name]
    d = df[(df.hw == hw_name) & (df.precision == precision)]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ai = np.logspace(-2, 4, 200)
    for peak, lab, ls in ((hw.fp16_tensor_tflops, "tensor peak", "-"), (hw.fp16_vec_tflops, "vector peak", "--")):
        if peak > 0:
            ax.plot(ai, np.minimum(peak * 1e3, ai * hw.bw_gbs), color="black", lw=1.4, ls=ls, label=lab)
    for y, c, lab in ((0, "#D55E00", "measured memory-bound"), (1, "#0072B2", "measured compute-bound")):
        s = d[d.y_bin == y]
        ax.scatter(10 ** s["log_ai"], s["ach_gflops"], s=9, alpha=0.6, color=c, linewidths=0, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte, compulsory traffic)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Roofline plot, {hw_name} {precision} (simulated data)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"roofline_{hw_name}.png"), dpi=140)
    plt.close(fig)


def fig_importance(df, out, target="y_bin"):
    """Permutation importance of a GradBoost trained WITHOUT the roofline-derived features."""
    _, tr_idx, te_idx = heldout_pairs(df, n_repeats=1, seed=1)[0]
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    m = make_model("GradBoost").fit(tr[FEATURES_RAW].values, tr[target].values)
    pi = permutation_importance(m, te[FEATURES_RAW].values, te[target].values, n_repeats=5,
                                scoring="balanced_accuracy", random_state=0)
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(np.array(FEATURES_RAW)[order], pi.importances_mean[order], xerr=pi.importances_std[order],
            color="#0072B2", height=0.6)
    ax.set_xlabel("drop in balanced accuracy when permuted")
    ax.set_title("GradBoost feature importance (held-out pairs, no roofline features)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_importance.png"), dpi=140)
    plt.close(fig)
    return dict(zip(np.array(FEATURES_RAW)[order][::-1], pi.importances_mean[order][::-1]))
