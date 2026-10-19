"""Train and SAVE the bottleneck classifier used by predict.py.

    python train.py                                            # simulated labels only
    python train.py --real data/measured_1650ti.csv            # simulated + your real measurements
    python train.py --real data/a.csv data/b.csv --real-weight 8

Model: gradient boosting, 3 classes (0 = memory-bound, 1 = mixed, 2 = compute-bound),
trained on FEATURES_FULL. Probabilities come from predict_proba.

When real data is supplied, two honest checks are printed BEFORE the final blended fit:
  1. How well do Roofline and the simulation-trained model do on your real rows? This is the
     true test of whether the simulator resembles your hardware.
  2. Leave-one-family-out on the real rows with the blended training set: how well does the
     blend do on a kernel family it has never measured?
The final model is then fit on simulated + real rows, with real rows weighted up.
"""
import argparse
import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from rooflinelab.baseline import roofline_predict
from rooflinelab.features import FEATURES_FULL
from rooflinelab.simulate import build_sim_dataset


def new_model():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)


def fit(df, w=None):
    m = new_model()
    m.fit(df[FEATURES_FULL].values, df["y_3"].values, sample_weight=w)
    return m


def report(tag, y, p):
    print(f"  {tag:<42} acc={accuracy_score(y, p):.3f}  bal_acc={balanced_accuracy_score(y, p):.3f}  n={len(y)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[], help="CSV(s) written by measure_gpu.py")
    ap.add_argument("--real-weight", type=float, default=5.0)
    ap.add_argument("--out", default="models/model.joblib")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sim = build_sim_dataset(a.seed)
    sim = sim[sim["launch_frac"] <= 0.5].reset_index(drop=True)
    sim["src"] = "sim"
    print(f"simulated rows: {len(sim)}")
    meta = dict(trained=datetime.datetime.now().isoformat(timespec="seconds"), n_sim=len(sim),
                n_real=0, real_gpus=[], real_weight=None, eval={})

    if not a.real:
        model = fit(sim)
        meta["label_source"] = "SIMULATED only (rooflinelab/simulate.py) - not measured hardware behaviour"
    else:
        real = pd.concat([pd.read_csv(p) for p in a.real], ignore_index=True)
        real = real[real["launch_frac"] <= 0.5].reset_index(drop=True)
        real["src"] = "real"
        print(f"real rows (after dropping launch-dominated): {len(real)} from GPUs {sorted(real.hw.unique())}")
        meta.update(n_real=len(real), real_gpus=sorted(real.hw.unique()), real_weight=a.real_weight)

        print("\n1) Simulation-trained model vs Roofline on REAL rows:")
        sim_model = fit(sim)
        y = real["y_3"].values
        r_rl = roofline_predict(real, 3)
        r_ml = sim_model.predict(real[FEATURES_FULL].values)
        report("Roofline", y, r_rl)
        report("ML trained on simulation only", y, r_ml)
        meta["eval"]["real_roofline_acc"] = float(accuracy_score(y, r_rl))
        meta["eval"]["real_simonly_acc"] = float(accuracy_score(y, r_ml))

        print("\n2) Blended model (sim + real of OTHER families), leave-one-family-out on real rows:")
        preds = np.empty(len(real), dtype=int)
        for fam in sorted(real.family.unique()):
            te = real.family == fam
            tr = pd.concat([sim, real[~te]], ignore_index=True)
            w = np.where(tr["src"] == "real", a.real_weight, 1.0)
            preds[te.values] = fit(tr, w).predict(real.loc[te, FEATURES_FULL].values)
        report("Blended, leave-one-family-out", y, preds)
        meta["eval"]["real_blend_lofo_acc"] = float(accuracy_score(y, preds))

        allr = pd.concat([sim, real], ignore_index=True)
        w = np.where(allr["src"] == "real", a.real_weight, 1.0)
        model = fit(allr, w)
        meta["label_source"] = (f"simulated + REAL measurements ({len(real)} rows on {meta['real_gpus']}, "
                                f"weight x{a.real_weight})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    joblib.dump(dict(model=model, features=FEATURES_FULL, meta=meta), a.out)
    print(f"\nsaved {a.out}\nlabel source: {meta['label_source']}")


if __name__ == "__main__":
    main()
