# Roofline Bottleneck Explorer

### AI-Based Hardware Performance Predictor

[**Launch the application →**](https://roofline-bottleneck-48zezmtbygadxxxwg4yfzt.streamlit.app/)

Roofline Bottleneck Explorer is an interactive application for analyzing AI workload performance across different hardware configurations.

The application combines the analytical Roofline model with machine-learning predictions to investigate the relationship between computational workloads, hardware capabilities, and performance bottlenecks.

## Features

**1. Hardware Configuration**

Define hardware characteristics such as peak computational throughput and memory bandwidth to explore how different architectures affect workload performance.

**2. AI Workload Analysis**

Select from a range of AI kernels and configure their parameters, including dimensions, batch sizes, sequence lengths, and numerical precision.

Supported workloads include:

* Matrix multiplication (GEMM)
* Convolution and depthwise convolution
* Attention
* Softmax and LayerNorm
* Elementwise operations (Add, GELU, SiLU)
* Reductions and Gather
* Custom workloads

**3. Roofline Analysis**

Calculate operational intensity, estimate attainable performance, and determine whether a workload is primarily compute-bound or memory-bound.

**4. Machine Learning Predictions**

Use a trained ML model to predict workload performance characteristics and identify potential bottlenecks alongside the analytical Roofline results.

**5. What-if Analysis**

Experiment with changes to hardware and workload parameters, such as increasing compute throughput, memory bandwidth, or changing numerical precision, and compare their effects on predicted performance.

**6. Optimization Suggestions**

Explore potential optimizations based on the identified bottleneck, including improving data reuse, reducing memory traffic, and increasing computational efficiency.

## Technology

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

**Roofline Bottleneck Explorer**
AI-Based Hardware Performance Predictor
