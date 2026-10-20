"""Feature construction shared by the simulator and the real-GPU harness."""
import math
import numpy as np
from .hw_specs import effective_peak

ROOFLINE_FEATS = ["log_ai", "log_ridge", "log_ai_over_ridge"]

BASE_FEATS = [
    "log_flops", "log_bytes", "log_ws", "irregular", "log_row_bytes", "log_tiles",
    "log_min_dim", "tensor_eligible", "is_fp16",
    "log_peak", "log_bw", "log_l2", "log_sms", "log_mem", "has_tensor", "used_tensor",
    "log_ws_over_l2", "log_tiles_per_sm",
]
FEATURES_FULL = BASE_FEATS + ROOFLINE_FEATS
FEATURES_RAW = BASE_FEATS  # roofline-derived columns removed (ablation)

NEAR_RIDGE_LOG10 = math.log10(3.0)  # "near the ridge" = within a factor of 3


def featurize(w, hw):
    P, used_tc = effective_peak(hw, w["precision"], bool(w["tensor_eligible"]))
    B = hw.bw_gbs * 1e9
    L2 = hw.l2_mb * 2**20
    ai = w["flops"] / w["bytes"]
    ridge = P / B
    l10 = math.log10
    return dict(
        log_flops=l10(max(w["flops"], 1.0)), log_bytes=l10(w["bytes"]), log_ws=l10(w["ws"]),
        irregular=w["irregular"], log_row_bytes=l10(max(w["row_bytes"], 1)),
        log_tiles=l10(w["tiles"]), log_min_dim=l10(max(w["min_dim"], 1)),
        tensor_eligible=w["tensor_eligible"], is_fp16=int(w["precision"] == "fp16"),
        log_peak=l10(P), log_bw=l10(B), log_l2=l10(L2), log_sms=l10(hw.sms), log_mem=l10(hw.mem_gb),
        has_tensor=int(hw.fp16_tensor_tflops > 0), used_tensor=int(used_tc),
        log_ws_over_l2=l10(w["ws"] / L2), log_tiles_per_sm=l10(w["tiles"] / hw.sms),
        log_ai=l10(max(ai, 1e-9)), log_ridge=l10(ridge),
        log_ai_over_ridge=l10(max(ai, 1e-9) / ridge),
    )


def labels_from_ratio(r):
    """r = (compute-side time or utilisation) / (memory-side time or utilisation).

    Binary: compute-bound iff r > 1.
    Three-class: 'mixed' iff both sides are within 1.5x of each other (0=mem, 1=mixed, 2=compute).
    """
    r = np.asarray(r, dtype=float)
    y_bin = (r > 1).astype(int)
    y_3 = np.where(r > 1.5, 2, np.where(r < 1 / 1.5, 0, 1))
    return y_bin, y_3
