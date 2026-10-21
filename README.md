# Roofline Bottleneck Explorer
### AI-Based Hardware Performance Predictor

[![Open Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-0ea5e9?style=for-the-badge)](https://roofline-bottleneck-48zezmtbygadxxxwg4yfzt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

**[Launch the interactive app →](https://roofline-bottleneck-48zezmtbygadxxxwg4yfzt.streamlit.app/)**

Roofline Bottleneck Explorer is an interactive tool for reasoning about AI-kernel performance from the perspective of hardware architecture. It combines analytical Roofline estimates with a machine-learning prediction path, then turns the results into optimization guidance and what-if comparisons.

> **Live demo:** Replace `https://roofline-bottleneck-48zezmtbygadxxxwg4yfzt.streamlit.app/` above with the deployed Streamlit Community Cloud URL. The badge and launch link then become direct entry points from the repository page.

---

## Aim

Make the relationship between AI workloads and hardware limits easier to inspect: **is a kernel limited primarily by compute throughput or by data movement, and what architectural or workload changes could shift that limit?**

## Objectives

- Model representative AI kernels using workload and hardware parameters.
- Estimate attainable performance with the Roofline model.
- Present the ML model's prediction alongside the analytical estimate.
- Identify likely bottlenecks and surface actionable optimization ideas.
- Compare hypothetical changes—such as doubling compute throughput or memory bandwidth—without requiring a local GPU.

## What the application does

### 1. Configure the hardware
Enter the relevant hardware characteristics, including peak compute throughput and memory bandwidth. These values define the hardware ceiling used by the analytical model.

### 2. Define a workload
Choose a kernel family and its dimensions. Supported workload families include:

- GEMM / linear layers
- 2D convolution and depthwise convolution
- Attention score matrices (`Q × Kᵀ`)
- Softmax and LayerNorm
- Elementwise add, GELU and SiLU
- Reduction / sum
- Embedding-style gather
- Custom kernels defined by FLOPs and estimated bytes transferred

### 3. Compare analytical and ML outputs
The interface places the Roofline estimate and the ML prediction side by side so that the user can inspect where their conclusions agree or differ.

### 4. Explore what-if scenarios
Change one assumption at a time—for example, peak compute, memory bandwidth, precision, cache capacity, or workload dimensions—and inspect how the estimated bottleneck and performance ceiling change.

### 5. Review optimization guidance
Use the analysis to consider strategies such as improving data reuse, increasing arithmetic intensity, reducing memory traffic, selecting a suitable precision, or changing the workload configuration.

---

## How the Roofline model works

The Roofline model relates attainable performance to operational intensity and hardware limits:

\[
P_{attainable} \leq \min(P_{peak},\ I \times B)
\]

where:

- \(P_{attainable}\) is the performance ceiling estimated by the model.
- \(P_{peak}\) is the hardware's peak compute throughput.
- \(I\) is operational intensity: operations performed per byte transferred.
- \(B\) is memory bandwidth.

The ridge point is:

\[
I^* = \frac{P_{peak}}{B}
\]

Below the ridge point, the bandwidth ceiling is lower than peak compute throughput, so the workload is classified as memory-bound under the model. Above it, peak compute is the limiting ceiling and the workload is compute-bound. Real kernels can also be limited by latency, occupancy, synchronization, cache behavior, launch overhead, and other effects that a simple Roofline bound does not fully represent.

## System flow

```text
Hardware specification ─┐
                        ├─> Feature / workload construction
Kernel parameters ──────┘              │
                                       ├─> Roofline analysis
                                       ├─> ML prediction
                                       └─> What-if scenarios
                                                │
                                                v
                              Bottleneck comparison + recommendations
                                                │
                                                v
                                    Streamlit interactive interface
```

## Run locally

### Requirements
- Python 3.10 or newer (use a Python version supported by the pinned dependencies)
- Git
- The packages listed in `requirements.txt`

### Setup

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux / WSL
source .venv/bin/activate
```

Install dependencies and launch:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deployment

The application is deployed through Streamlit Community Cloud and connected to this GitHub repository. Commits pushed to the configured deployment branch can trigger a rebuild.

To deploy your own copy:
1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app using the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy and add the resulting public URL to the badge and launch link at the top of this README.

## Repository map

```text
.
├── app.py                 # Streamlit interface
├── rooflinelab/           # Workload definitions and analytical logic
├── requirements.txt       # Python dependencies
├── tests/                 # Automated tests, if present in this version
├── train.py               # Training pipeline, if included
├── train_real.py          # Alternate data-backed training pipeline, if included
├── predict.py             # Prediction entry point, if included
└── predict_real.py        # Data-backed prediction entry point, if included
```

The exact contents depend on the version committed to the repository. Keep this tree aligned with the files actually tracked by Git.

## Current deliverable

The current application provides:
- An interactive browser-based interface.
- Configurable hardware and workload inputs.
- Analytical Roofline results.
- An ML prediction view.
- What-if comparisons for hardware and workload changes.
- Optimization guidance tied to the modeled workload.
- A deployment path that lets users open the interface through a public URL.

## Scope of the results

The Roofline output is an analytical estimate based on the parameters supplied by the user. The ML output is the prediction of the model included with the project. These outputs are useful for comparing assumptions and exploring bottleneck hypotheses; they should not be interpreted as physical measurements unless a result has been validated against measurements from the target hardware.

## Technology

- Python
- Streamlit
- NumPy / pandas / scikit-learn / Matplotlib (as used by the selected project modules)

## References

1. S. Williams, A. Waterman, and D. Patterson, “Roofline: An Insightful Visual Performance Model for Multicore Architectures,” *Communications of the ACM*, 2009.
2. T. Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” *NeurIPS*, 2022.

---

**Project:** AI-Based Hardware Performance Predictor  
**Interface:** Roofline Bottleneck Explorer
