"""Interactive Roofline + simulation-trained ML bottleneck demo.

Run with: streamlit run app.py
No GPU or downloaded dataset is required. The ML classifier is trained in memory
from the project's simulator; it is not a model validated on measured hardware labels.
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rooflinelab.advisor import analyze
from rooflinelab.hw_specs import HARDWARE, custom_hw
from rooflinelab.simulate import build_sim_dataset
from rooflinelab.workloads import make_workload
from train import fit


st.set_page_config(page_title="Roofline Bottleneck Explorer", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Preparing the CPU-only demo model…")
def load_demo_model() -> dict[str, Any]:
    """Train a deterministic model locally so no model artifact or GPU is needed."""
    df = build_sim_dataset(seed=0)
    df = df[df["launch_frac"] <= 0.5].reset_index(drop=True)
    model = fit(df)
    from rooflinelab.features import FEATURES_FULL

    return {
        "model": model,
        "features": FEATURES_FULL,
        "meta": {
            "label_source": "SIMULATED only (rooflinelab/simulate.py) — not measured hardware behaviour",
            "n_sim": len(df),
            "trained": "on first app launch, deterministic seed 0",
        },
    }


def positive_int(label: str, value: int, key: str, minimum: int = 1, maximum: int = 1_000_000_000) -> int:
    return int(st.number_input(label, min_value=minimum, max_value=maximum, value=value, step=1, key=key))


def build_kernel() -> dict[str, Any]:
    st.subheader("2. Define the kernel")
    op_labels = {
        "gemm": "GEMM / linear layer",
        "conv": "2D convolution",
        "depthwise": "Depthwise convolution",
        "attn_scores": "Attention score matrix (Q × Kᵀ)",
        "softmax": "Softmax",
        "layernorm": "LayerNorm",
        "add": "Elementwise add",
        "gelu": "GELU activation",
        "silu": "SiLU activation",
        "sum": "Reduction / sum",
        "gather": "Embedding-style gather",
        "custom": "Custom kernel (FLOPs + bytes supplied by you)",
    }
    op = st.selectbox("Kernel / operator", list(op_labels), format_func=lambda x: op_labels[x], key="op")
    precision = st.selectbox("Data precision", ["fp32", "fp16"], format_func=lambda x: x.upper(), key="precision")

    params: dict[str, int] = {}
    if op == "gemm":
        c = st.columns(4)
        for col, name, default in zip(c, ["M", "N", "K", "Batch"], [4096, 4096, 4096, 1]):
            with col:
                params[name if name != "Batch" else "batch"] = positive_int(
                    name, default, f"gemm_{name}"
                )
    elif op == "conv":
        c = st.columns(3)
        for col, name, default in zip(c, ["N (batch)", "C (input channels)", "H (input height/width)"], [32, 256, 14]):
            with col:
                params[{"N (batch)": "N", "C (input channels)": "C", "H (input height/width)": "H"}[name]] = positive_int(name, default, f"conv_{name}")
        c = st.columns(3)
        for col, name, default in zip(c, ["K (output channels)", "R (kernel width)", "Stride"], [256, 3, 1]):
            with col:
                params[{"K (output channels)": "K", "R (kernel width)": "R", "Stride": "stride"}[name]] = positive_int(name, default, f"conv_{name}")
    elif op == "depthwise":
        c = st.columns(4)
        for col, name, default in zip(c, ["N", "C", "H", "R"], [32, 128, 56, 3]):
            with col:
                params[name] = positive_int(name, default, f"depthwise_{name}")
    elif op == "attn_scores":
        c = st.columns(4)
        for col, name, default in zip(c, ["B", "Heads", "Sequence length L", "Head dimension d"], [1, 32, 2048, 128]):
            with col:
                key = {"B": "B", "Heads": "heads", "Sequence length L": "L", "Head dimension d": "d"}[name]
                params[key] = positive_int(name, default, f"attn_{key}")
    elif op == "softmax":
        c = st.columns(2)
        for col, name, default, key in zip(c, ["Rows", "Elements per row"], [98304, 2048], ["rows", "length"]):
            with col:
                params[key] = positive_int(name, default, f"softmax_{key}")
    elif op == "layernorm":
        c = st.columns(2)
        for col, name, default, key in zip(c, ["Tokens", "Hidden dimension H"], [4096, 4096], ["tokens", "H"]):
            with col:
                params[key] = positive_int(name, default, f"layernorm_{key}")
    elif op in {"add", "gelu", "silu", "sum"}:
        params["n"] = positive_int("Number of elements", 16_777_216, f"{op}_n")
    elif op == "gather":
        c = st.columns(2)
        with c[0]:
            params["rows"] = positive_int("Rows gathered", 65_536, "gather_rows")
        with c[1]:
            params["D"] = positive_int("Elements per row D", 128, "gather_D")
    else:
        c = st.columns(3)
        with c[0]:
            flops = float(st.number_input("Total FLOPs", min_value=1.0, value=2_000_000_000.0, format="%.0f", key="custom_flops"))
        with c[1]:
            byte_count = float(st.number_input("Bytes transferred (estimated)", min_value=1.0, value=400_000_000.0, format="%.0f", key="custom_bytes"))
        with c[2]:
            ws = float(st.number_input("Working set bytes", min_value=1.0, value=400_000_000.0, format="%.0f", key="custom_ws"))
        c = st.columns(2)
        with c[0]:
            irregular = st.checkbox("Irregular / random memory access", value=False)
        with c[1]:
            tensor_eligible = st.checkbox("Eligible for tensor-core acceleration", value=False)
        return dict(family="custom", desc=f"custom kernel ({flops:.3g} FLOP, {byte_count:.3g} B)",
                    precision=precision, flops=flops, bytes=byte_count, ws=ws, reload=0.0,
                    irregular=int(irregular), row_bytes=128, tiles=max(1, int(ws // 16384)),
                    min_dim=1024, tensor_eligible=int(tensor_eligible), base_eff=0.5, params=None)

    return make_workload(op, precision, **params)


def build_hardware():
    st.subheader("1. Describe the architecture")
    mode = st.radio("Hardware input", ["Use a preset architecture", "Enter custom architecture details"], horizontal=True)
    if mode == "Use a preset architecture":
        names = list(HARDWARE)
        selected = st.selectbox("Architecture preset", names, index=names.index("A100-40"))
        hw = HARDWARE[selected]
        st.caption("Preset figures are approximate theoretical specifications. Select custom input to enter values from a datasheet or measured profile.")
        return hw

    st.caption("Enter sustained/credible ceilings for the precision you want to model. The result is only as good as these inputs.")
    c = st.columns(3)
    with c[0]:
        name = st.text_input("Architecture name", "My accelerator")
        fp32 = float(st.number_input("FP32 peak (TFLOP/s)", min_value=0.001, value=30.0, step=1.0))
        fp16_vec = float(st.number_input("FP16 vector peak (TFLOP/s)", min_value=0.001, value=60.0, step=1.0))
    with c[1]:
        fp16_tensor = float(st.number_input("FP16 tensor-core peak (TFLOP/s; 0 if none)", min_value=0.0, value=120.0, step=1.0))
        bw = float(st.number_input("Memory bandwidth (GB/s)", min_value=0.001, value=900.0, step=25.0))
    with c[2]:
        l2 = float(st.number_input("L2 cache (MiB)", min_value=0.001, value=40.0, step=1.0))
        sms = int(st.number_input("Compute units / SMs (proxy)", min_value=1, value=80, step=1))
        mem = float(st.number_input("Device memory capacity (GB)", min_value=0.001, value=40.0, step=1.0))
    return custom_hw(fp32, bw, l2, sms, mem, fp16_vec, fp16_tensor, name=name or "custom")


def roofline_plot(result: dict[str, Any]):
    rf = result["roofline"]
    ai = rf["ai"]
    peak = rf["peak_flops"] / 1e12
    bw = rf["bandwidth"] / 1e9
    x_min = max(1e-9, min(1e-2, ai / 10, rf["ridge"] / 10))
    x_max = max(1e5, ai * 10, rf["ridge"] * 10)
    x = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    y = np.minimum(peak, x * bw)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.loglog(x, y, label="Roofline ceiling")
    ax.scatter([ai], [rf["attainable_flops"] / 1e12], s=65, label="Kernel's Roofline point", zorder=3)
    ax.axvline(rf["ridge"], linestyle="--", label=f"Ridge = {rf['ridge']:.3g} FLOP/byte")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable throughput (TFLOP/s)")
    ax.set_title("Analytical Roofline — an upper bound, not measured throughput")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Roofline Bottleneck Explorer")
st.write("Give the tool an architecture and a kernel. Compare the analytical Roofline verdict with an ML verdict, inspect optimization ideas, and explore what-if hardware changes — **no GPU required**.")
st.warning("Research-demo caveat: the bottleneck classifier is trained on this project's simulated labels, not measured hardware bottleneck labels. Its verdict is a hypothesis to compare with Roofline, not proof of real-device behaviour.", icon="⚠️")

with st.sidebar:
    st.header("How to read the results")
    st.markdown("- **Roofline**: analytical ceiling from peak compute, bandwidth, FLOPs and bytes.\n- **ML**: patterns learned from the simulator's labels.\n- **What-if**: recomputed estimates under changed architecture assumptions.\n- **Optimizations**: heuristic suggestions; speed-ups are idealized bounds, not promises.")
    st.markdown("**Arithmetic intensity** = FLOPs / bytes.\n\n**Ridge point** = peak FLOP/s / memory bandwidth.")

left, right = st.columns([1, 1])
with left:
    hw = build_hardware()
with right:
    w = build_kernel()

run = st.button("Analyze kernel", type="primary", use_container_width=True)
if run:
    try:
        bundle = load_demo_model()
        result = analyze(bundle, w, hw)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

    st.divider()
    st.header("3. Verdicts")
    c = st.columns(4)
    c[0].metric("Roofline verdict", result["roofline"]["bound3"].upper())
    c[1].metric("ML verdict", result["ml"]["label"].upper() if result["ml"] else "Unavailable")
    c[2].metric("Arithmetic intensity", f"{result['roofline']['ai']:.3g} FLOP/B")
    c[3].metric("AI / ridge ratio", f"{result['roofline']['ratio']:.3g}×")

    tab_roof, tab_ml, tab_whatif, tab_opt = st.tabs(["Roofline analysis", "ML explanation", "What-if scenarios", "Optimization advisor"])
    with tab_roof:
        c = st.columns(3)
        c[0].metric("Compute-side ideal time", f"{result['roofline']['t_compute'] * 1e6:.3g} µs")
        c[1].metric("Memory-side ideal time", f"{result['roofline']['t_memory'] * 1e6:.3g} µs")
        c[2].metric("Roofline ideal time", f"{result['roofline']['t_ideal'] * 1e6:.3g} µs")
        c = st.columns(2)
        c[0].metric("Compute ceiling", f"{result['roofline']['peak_flops'] / 1e12:.3g} TFLOP/s")
        c[1].metric("Memory bandwidth", f"{hw.bw_gbs:.3g} GB/s")
        st.pyplot(roofline_plot(result), clear_figure=True)
        st.caption("The plotted point is the analytical attainable-performance bound. It is not a measured kernel throughput point.")
        with st.expander("Kernel and architecture inputs"):
            st.json({"hardware": result["hardware"], "workload": result["workload"]})
    with tab_ml:
        ml = result["ml"]
        if ml:
            st.write(f"**ML verdict: {ml['label'].upper()}** — class probability {ml['confidence']:.1%}.")
            probs = pd.DataFrame({"Class": list(ml["probs"].keys()), "Probability": list(ml["probs"].values())})
            st.bar_chart(probs.set_index("Class"))
            st.info("These probabilities describe the classifier's output for a simulator-trained model. They are not calibrated probabilities of the true physical bottleneck.")
        st.write("**Structural diagnostics (heuristic estimates, not measurements)**")
        dg = result["diagnostics"]
        st.dataframe(pd.DataFrame([
            {"Diagnostic": "Working set covered by L2", "Estimate": f"{min(dg['l2_cover'], 1):.1%}"},
            {"Diagnostic": "Estimated DRAM traffic / compulsory bytes", "Estimate": f"{dg['dram_vs_compulsory']:.2f}×"},
            {"Diagnostic": "Wave efficiency", "Estimate": f"{dg['wave_eff']:.1%}"},
            {"Diagnostic": "Dimension penalty factor", "Estimate": f"{dg['dim_pen']:.3f}"},
            {"Diagnostic": "Heuristic runtime", "Estimate": f"{dg['est_time_s'] * 1e6:.3g} µs"},
        ]), hide_index=True, use_container_width=True)
    with tab_whatif:
        rows = []
        for scenario, values in result["what_if"].items():
            rows.append({
                "Scenario": scenario,
                "Roofline verdict": values["roofline"],
                "ML verdict": values["ml"] or "—",
                "Ideal speed-up upper bound": (f"{values['speedup_ub']:.2f}×" if values.get("speedup_ub") is not None else "not comparable"),
                "Per-item upper bound": (f"{values['per_item_ub']:.2f}×" if values.get("per_item_ub") is not None else "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("The what-if engine includes 2× memory bandwidth, 2× compute, 4× L2, and applicable precision/fusion/batch changes. A bound is not a prediction of realized speed-up.")
    with tab_opt:
        if not result["advice"]:
            st.write("No specific optimization was ranked for this configuration. Inspect the Roofline ratio and validate with profiling when available.")
        for i, item in enumerate(result["advice"], 1):
            with st.container(border=True):
                gain = f" · ideal upper bound ≤ {item['speedup_ub']:.2f}×" if item.get("speedup_ub") else ""
                st.markdown(f"**{i}. {item['title']}**{gain}")
                st.write(item["detail"])
    if result["notes"]:
        st.subheader("Caveats for this case")
        for note in result["notes"]:
            st.warning(note)

st.divider()
st.caption("Built around the project's existing rooflinelab advisor. For defensible real-hardware verdicts, the next step is to collect profiler-counter labels on any accessible GPU; training the current demo classifier itself requires only a CPU.")
