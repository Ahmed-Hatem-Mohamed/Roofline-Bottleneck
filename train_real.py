"""Train + evaluate the REAL-data efficiency model (how far below Roofline does a kernel run?).

    python fetch_real_data.py        # once
    python train_real.py             # ~10-20 min on a laptop; --fast for a quick sanity run

Target: y = log10(measured_time / roofline_ideal_time).  Predicted time = ideal * 10**y_hat.
Quantile models (10% / 50% / 90%) give an 80% prediction interval.

Evaluation (every split trains on training rows only):
  random   : rows split 80/20  (LEAKY: same shape on other GPUs sits in training) - for contrast only
  shape    : GroupKFold(5) over problem shapes - test shapes never seen on ANY GPU
  gpu      : leave-one-GPU-out (11 folds) - test GPU never seen: the hardware-generalisation test
Baselines: Roofline (efficiency = 1), per-op constant efficiency, per-op linear fit t = a*ideal + b.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, train_test_split

from rooflinelab.realdata import FEATURES, OP_NOTES, OPS, PEAK_FIX, load_real

QS = (0.1, 0.5, 0.9)
# unseen_gpu leaves out one GPU but its same-die siblings (H100/H200/H800, L20/L40/RTX6000Ada, A40/A6000)
# stay in training, which flatters hardware generalisation. unseen_arch removes the whole family.
ARCH = {"H100": "Hopper", "H200": "Hopper", "H800": "Hopper", "H20": "Hopper",
        "L20": "Ada", "L40": "Ada", "RTX6000Ada": "Ada",
        "A100-80": "Ampere", "A40": "Ampere", "RTXA6000": "Ampere", "RTX-PRO-6000": "Blackwell"}


def make(q, fast):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=120 if fast else 300,
                                         learning_rate=0.1 if fast else 0.08, max_leaf_nodes=63,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                                         random_state=0)


def fit_models(tr, fast, qs=QS, cap=None, rng=None):
    if cap and len(tr) > cap:
        tr = tr.iloc[rng.choice(len(tr), cap, replace=False)]
    X, y = tr[FEATURES].values, tr["y"].values
    return {q: make(q, fast).fit(X, y) for q in qs}


def baselines(tr, te):
    out = {"Roofline": np.zeros(len(te))}
    med, pw = np.zeros(len(te)), np.zeros(len(te))
    for op in te.op.unique():
        t, m = tr[tr.op == op], (te.op == op).values
        if len(t) == 0:                       # op never seen in this training fold
            med[m] = pw[m] = tr.y.median()
            continue
        med[m] = t.y.median()
        # power law per op: log10(t/ideal) = c0 + c1*log10(ideal)  (captures the latency floor on small kernels)
        c1, c0 = np.polyfit(t.log_t_ideal.values, t.y.values, 1)
        pw[m] = c0 + c1 * te.log_t_ideal.values[m]
    out["ConstEff/op"], out["PowerLaw/op"] = med, pw
    return out


def metrics(te, yhat, lo=None, hi=None):
    err = np.abs(10 ** (yhat - te.y.values) - 1)          # |t_hat - t| / t
    r = dict(n=len(te), mape_median=float(np.median(err)), within25=float(np.mean(err <= 0.25)),
             log10_mae=float(np.mean(np.abs(yhat - te.y.values))))
    big = (te.t_s.values >= 20e-6)
    r["mape_median_ge20us"] = float(np.median(err[big])) if big.any() else np.nan
    if lo is not None:
        r["cover80"] = float(np.mean((te.y.values >= lo) & (te.y.values <= hi)))
        r["interval_width_x"] = float(np.median(10 ** (hi - lo)))
    return r


def run_split(df, name, splits, fast, rng, cap, log):
    rows, preds = [], []
    for fold, tr_i, te_i in splits:
        t0 = time.time()
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        ms = fit_models(tr, fast, cap=cap, rng=rng)
        X = te[FEATURES].values
        p = {q: ms[q].predict(X) for q in QS}
        # enforce monotone quantiles
        lo, mid, hi = np.minimum(p[0.1], p[0.5]), p[0.5], np.maximum(p[0.9], p[0.5])
        methods = baselines(tr, te)
        methods["GradBoost"] = mid
        for mname, yh in methods.items():
            m = metrics(te, yh, lo, hi) if mname == "GradBoost" else metrics(te, yh)
            rows.append(dict(split=name, fold=fold, method=mname, op="ALL", **m))
            for op in te.op.unique():
                mk = (te.op == op).values
                sub = te[mk]
                mm = metrics(sub, yh[mk], lo[mk], hi[mk]) if mname == "GradBoost" else metrics(sub, yh[mk])
                rows.append(dict(split=name, fold=fold, method=mname, op=op, **mm))
        preds.append(pd.DataFrame(dict(split=name, fold=fold, op=te.op.values, gpu=te.gpu.values,
                                       y=te.y.values, yhat=mid, lo=lo, hi=hi, ratio=te.ratio.values,
                                       t_s=te.t_s.values)))
        log(f"  {name}/{fold}: {len(te)} test rows, {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def make_meta(df, summ):
    """Everything predict_real.py needs to judge applicability and quote validation numbers."""
    ranges = {op: {c: [float(g[c].min()), float(g[c].max())] for c in ("log_peak", "log_bw", "log_sms")}
              for op, g in df.groupby("op")}
    keep = summ[summ.method.isin(["GradBoost", "Roofline", "PowerLaw/op"])]
    cv = keep[["split", "op", "method", "mape_median", "within25", "cover80", "n"]].round(3)
    return dict(n_rows=len(df), gpus=sorted(df.gpu.unique()), ops=sorted(df.op.unique()), op_notes=OP_NOTES,
                peak_fix=PEAK_FIX, hw_range_per_op=ranges, arch=ARCH,
                gpus_per_op=df.groupby("op").gpu.unique().map(sorted).to_dict(), files=sorted(df.file.unique()),
                source="PipeWeave dataset (github.com/zksainx/pipeweave, arXiv 2601.14910), measured latencies",
                cv=cv.where(cv.notna(), None).to_dict("records"))


def refresh_meta(a):
    df = load_real(a.data)
    summ = pd.read_csv(os.path.join(a.results, "summary.csv"))
    b = joblib.load(a.out)
    b["meta"] = make_meta(df, summ)
    joblib.dump(b, a.out)
    print("meta refreshed in", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-meta", action="store_true", help="only rebuild metadata from saved results")
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--out", default="models/real_efficiency.joblib")
    ap.add_argument("--results", default="results/real")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cv-cap", type=int, default=150000, help="max training rows per CV fold (speed)")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    if a.refresh_meta:
        return refresh_meta(a)
    rng = np.random.default_rng(0)
    df = load_real(a.data)
    print(f"rows {len(df)}; ops {df.groupby('op').size().to_dict()}; GPUs {sorted(df.gpu.unique())}")

    idx = np.arange(len(df))
    splits_random = [("80/20",) + tuple(train_test_split(idx, test_size=0.2, random_state=0))]
    gkf = GroupKFold(n_splits=5)
    splits_shape = [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(idx, groups=df.shape_key.values))]
    splits_gpu = [(g, np.where(df.gpu != g)[0], np.where(df.gpu == g)[0]) for g in sorted(df.gpu.unique())]
    df["arch"] = df.gpu.map(ARCH)
    splits_arch = [(ar, np.where(df.arch != ar)[0], np.where(df.arch == ar)[0]) for ar in sorted(df.arch.unique())]

    res, preds = [], []
    for name, sp in (("random(leaky)", splits_random), ("unseen_shape", splits_shape),
                     ("unseen_gpu", splits_gpu), ("unseen_arch", splits_arch)):
        r, p = run_split(df, name, sp, a.fast, rng, a.cv_cap, print)
        res.append(r)
        preds.append(p)
    res, preds = pd.concat(res, ignore_index=True), pd.concat(preds, ignore_index=True)
    res.to_csv(os.path.join(a.results, "metrics_by_fold.csv"), index=False)
    preds.to_csv(os.path.join(a.results, "cv_predictions.csv.gz"), index=False)

    # summaries: row-weighted over folds
    cols = ("mape_median", "within25", "log10_mae", "mape_median_ge20us", "cover80", "interval_width_x")

    def agg(g):
        w = g.n.values.astype(float)
        out = {}
        for c in cols:
            v = g[c].values.astype(float) if c in g else np.full(len(g), np.nan)
            ok = ~np.isnan(v)
            out[c] = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
        out["n"] = int(w.sum())
        return pd.Series(out)
    summ = res.groupby(["split", "op", "method"]).apply(agg).reset_index()
    summ.to_csv(os.path.join(a.results, "summary.csv"), index=False)

    # final models on all rows
    print("fitting final models on all rows...")
    final = fit_models(df, a.fast)
    meta = make_meta(df, summ)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    joblib.dump(dict(models=final, features=FEATURES, meta=meta), a.out)
    json.dump(meta, open(os.path.join(a.results, "meta.json"), "w"), indent=2, default=str)
    print(f"saved {a.out}")

    # markdown report
    f = open(os.path.join(a.results, "summary.md"), "w")
    f.write("# Real-data efficiency model\n\nTarget: measured time / Roofline ideal time. Metric: median |t_hat - t| / t "
            "(lower is better) and share of kernels predicted within 25%.\n\n")
    for sp in ("unseen_arch", "unseen_gpu", "unseen_shape", "random(leaky)"):
        for op in ["ALL"] + OPS:
            s = summ[(summ.split == sp) & (summ.op == op)]
            if s.empty:
                continue
            f.write(f"### {sp} / {op}\n\n| method | median err | within 25% | log10 MAE | median err (>=20us) |"
                    + (" 80% interval coverage |" if "cover80" in s and s.cover80.notna().any() else "")
                    + "\n|---|---|---|---|---|" + ("---|" if "cover80" in s and s.cover80.notna().any() else "") + "\n")
            for _, r in s.iterrows():
                cov = f" {r.cover80:.2f} |" if "cover80" in s and pd.notna(r.get("cover80")) else (
                    " - |" if "cover80" in s and s.cover80.notna().any() else "")
                f.write(f"| {r.method} | {r.mape_median:.2f} | {r.within25:.2f} | {r.log10_mae:.2f} | "
                        f"{r.mape_median_ge20us:.2f} |{cov}\n")
            f.write("\n")
    f.close()
    print(open(os.path.join(a.results, "summary.md")).read()[:6000])


if __name__ == "__main__":
    main()
