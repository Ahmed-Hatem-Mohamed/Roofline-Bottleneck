"""Smoke tests for the no-GPU demo's core analysis path."""
from rooflinelab.advisor import analyze, roofline
from rooflinelab.features import FEATURES_FULL
from rooflinelab.hw_specs import HARDWARE
from rooflinelab.simulate import build_sim_dataset
from rooflinelab.workloads import make_workload
from train import fit


def demo_bundle():
    df = build_sim_dataset(seed=0)
    df = df[df["launch_frac"] <= 0.5].reset_index(drop=True)
    return {"model": fit(df), "features": FEATURES_FULL,
            "meta": {"label_source": "SIMULATED only"}}


def test_roofline_compute_and_memory_times_are_consistent():
    hw = HARDWARE["A100-40"]
    w = make_workload("gemm", "fp32", M=512, N=512, K=512)
    r = roofline(w, hw)
    assert r["ai"] > 0
    assert r["ridge"] > 0
    assert r["t_ideal"] == max(r["t_compute"], r["t_memory"])
    assert r["attainable_flops"] <= r["peak_flops"]


def test_end_to_end_analysis_includes_ml_whatifs_and_advice():
    bundle = demo_bundle()
    hw = HARDWARE["A100-40"]
    w = make_workload("gemm", "fp16", M=1, N=8192, K=8192)
    result = analyze(bundle, w, hw)
    assert result["roofline"]["bound3"] in {"memory-bound", "mixed", "compute-bound"}
    assert result["ml"]["label"] in {"memory-bound", "mixed", "compute-bound"}
    assert "2x memory bandwidth" in result["what_if"]
    assert "2x compute peak" in result["what_if"]
    assert isinstance(result["advice"], list)
