"""Model ladder + the generalisation experiments (random-split contrast, A, B, C)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import roofline_predict
from .features import FEATURES_FULL, FEATURES_RAW, NEAR_RIDGE_LOG10

MODEL_NAMES = ["Roofline", "LogReg", "RandomForest", "GradBoost", "MLP"]


def make_model(name):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)
    if name == "GradBoost":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=500, early_stopping=True,
                                                             random_state=0))
    raise ValueError(name)


def _metrics(y, p, near):
    out = dict(accuracy=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p),
               macro_f1=f1_score(y, p, average="macro"), n_test=len(y))
    out["near_ridge_acc"] = accuracy_score(y[near], p[near]) if near.sum() >= 10 else np.nan
    out["n_near_ridge"] = int(near.sum())
    return out


def _fit_predict(name, tr, te, feats, target, classes):
    if name == "Roofline":
        return roofline_predict(te, classes)
    m = make_model(name)
    m.fit(tr[feats].values, tr[target].values)
    return m.predict(te[feats].values)


def run_splits(df, splits, feats, target, classes, experiment, models=MODEL_NAMES):
    rows = []
    for fold_name, tr_idx, te_idx in splits:
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        y = te[target].values
        near = (te["log_ai_over_ridge"].abs() < NEAR_RIDGE_LOG10).values
        for name in models:
            if name == "Roofline" and feats is FEATURES_RAW:
                continue  # Roofline has no features; reported once under the full set
            p = _fit_predict(name, tr, te, feats, target, classes)
            rows.append(dict(experiment=experiment, fold=fold_name, model=name, **_metrics(y, p, near)))
    return pd.DataFrame(rows)


def random_split(df, seed=0):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=df["y_bin"])
    return [("random80/20", tr, te)]


def leave_one_family_out(df):
    return [(f, np.where(df["family"] != f)[0], np.where(df["family"] == f)[0])
            for f in sorted(df["family"].unique())]


def leave_one_hw_out(df):
    return [(h, np.where(df["hw"] != h)[0], np.where(df["hw"] == h)[0])
            for h in sorted(df["hw"].unique())]


def heldout_pairs(df, n_repeats=5, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pair = (df["family"] + "|" + df["hw"]).values
    pairs = np.unique(pair)
    splits = []
    for r in range(n_repeats):
        test_pairs = set(rng.choice(pairs, size=int(frac * len(pairs)), replace=False))
        mask = np.array([p in test_pairs for p in pair])
        splits.append((f"rep{r}", np.where(~mask)[0], np.where(mask)[0]))
    return splits


def run_all(df, classes=2, models=MODEL_NAMES, log=print):
    target = "y_bin" if classes == 2 else "y_3"
    results = []
    for fs_name, feats in (("full", FEATURES_FULL), ("raw", FEATURES_RAW)):
        for exp, splits in (("random_split(leaky)", random_split(df)),
                            ("A_unseen_workload", leave_one_family_out(df)),
                            ("B_unseen_hardware", leave_one_hw_out(df)),
                            ("C_unseen_pair", heldout_pairs(df))):
            log(f"  [{classes}-class] {exp} / features={fs_name}")
            r = run_splits(df, splits, feats, target, classes, exp, models)
            r["features"] = fs_name
            results.append(r)
    res = pd.concat(results, ignore_index=True)
    # Roofline uses no features: copy its rows to the 'raw' view so tables line up.
    rl = res[(res.model == "Roofline") & (res.features == "full")].copy()
    rl["features"] = "raw"
    res = pd.concat([res, rl], ignore_index=True)
    res["classes"] = classes
    return res


def summarize(res):
    g = res.groupby(["classes", "features", "experiment", "model"], sort=False)
    s = g[["accuracy", "bal_acc", "macro_f1", "near_ridge_acc"]].mean().reset_index()
    s["model"] = pd.Categorical(s["model"], MODEL_NAMES, ordered=True)
    return s.sort_values(["classes", "features", "experiment", "model"]).reset_index(drop=True)
