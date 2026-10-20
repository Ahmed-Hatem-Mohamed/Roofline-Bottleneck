"""Analytical Roofline baseline: classify purely from AI vs ridge point."""
import numpy as np

BAND = np.log10(1.5)


def roofline_predict(df, classes=2):
    x = df["log_ai_over_ridge"].values
    if classes == 2:
        return (x > 0).astype(int)
    return np.where(x > BAND, 2, np.where(x < -BAND, 0, 1))
