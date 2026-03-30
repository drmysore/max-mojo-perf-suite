# MAX + Mojo Performance Engineering Suite

**Author: Dr. Mysore Supreeth**

Inference benchmarking and GPU kernel optimization on NVIDIA and AMD hardware, built on Modular's MAX platform and the Mojo programming language.

---

## Background

Most published LLM inference benchmarks test a single model on a single GPU and do not document their methodology well enough for others to reproduce the results. Cross-vendor comparisons are worse: they frequently conflate software stack differences with hardware differences, making it impossible to attribute performance deltas to the silicon itself.

I wrote this suite to address both problems. It consists of two projects:

1. **MAX-Inference-Bench** -- a configuration-driven pipeline that systematically sweeps models, quantization schemes, and hardware targets through Modular's `max serve` and `max benchmark` tooling, producing structured JSON results that can be diffed across runs.

2. **Mojo-GPU-Kernels** -- a set of GPU kernels written in Mojo (vector addition, tiled matrix multiplication, fused softmax, parallel reduction) that illustrate the optimization patterns underlying fast inference. Writing these by hand clarified for me what the MAX compiler handles automatically and where custom operations remain necessary.

Both projects are designed around reproducibility: pinned container images, recorded driver and runtime versions, single-command execution.

---

## Repository Layout

```
max-mojo-perf-suite/
|-- max-inference-bench/            Project 1: Inference Benchmarking
|   |-- configs/
|   |   |-- models.yaml             Model and quantization matrix
|   |   |-- benchmark_profiles.yaml Test profiles (quick through stress)
|   |   +-- hardware.yaml           NVIDIA / AMD Docker and env config
|   |-- scripts/
|   |   |-- run_benchmark.py        Orchestrator: server lifecycle + max benchmark
|   |   |-- compare_hardware.py     NVIDIA vs AMD comparison tables
|   |   +-- quantization_sweep.py   Quantization Pareto analysis
|   |-- analysis/
|   |   +-- plot_results.py         Matplotlib visualization
|   +-- results/                    JSON output (gitignored)
|
|-- mojo-gpu-kernels/               Project 2: Custom GPU Kernels
|   |-- kernels/
|   |   |-- vector_add.mojo         Grid/block decomposition baseline
|   |   |-- matmul_tiled.mojo       Tiled matmul with shared memory
|   |   |-- softmax.mojo            Fused online softmax (single-pass)
|   |   +-- reduction.mojo          Parallel reduction patterns
|   |-- benchmarks/
|   |   +-- kernel_bench.py         Compile and time all kernels
|   |-- tests/
|   +-- analysis/
|
|-- docker-compose.nvidia.yaml      Single-command NVIDIA stack
|-- docker-compose.amd.yaml         Single-command AMD stack
|-- requirements.txt
+-- docs/
```

---

## Project 1: MAX-Inference-Bench

### Overview

This project measures LLM inference performance across the full quantization spectrum on both NVIDIA and AMD GPUs using Modular's MAX serving infrastructure. The orchestrator manages the complete lifecycle: launching a containerised MAX server, polling for readiness, executing benchmarks at multiple concurrency levels, and collecting structured results.

### Model and Quantization Coverage

| Model | BF16 | Q4_K | Q6_K | GPTQ-INT4 | FP8 |
|-------|:----:|:----:|:----:|:---------:|:---:|
| Llama 3.1 8B Instruct | Y | Y | Y | Y | Y |
| Mistral Nemo 12B | Y | Y | Y | -- | -- |
| Gemma 3 27B | Y | Y | -- | -- | -- |

### Metrics

| Category | Collected |
|----------|-----------|
| Latency | TTFT (P50/P95/P99), TPOT (P50/P95), ITL (P50/P99) |
| Throughput | Tokens/sec, Requests/sec |
| Resources | GPU utilization (mean), GPU memory peak, power draw |

### Usage

```bash
export HF_TOKEN="hf_..."

# Dry run to verify configuration
python max-inference-bench/scripts/run_benchmark.py \
  --model llama3_1_8b \
  --profile quick \
  --hardware nvidia \
  --dry-run

# Full benchmark run
python max-inference-bench/scripts/run_benchmark.py \
  --model llama3_1_8b \
  --profile standard \
  --hardware nvidia amd

# Docker Compose (NVIDIA)
docker compose -f docker-compose.nvidia.yaml up

# Post-run analysis
python max-inference-bench/scripts/compare_hardware.py
python max-inference-bench/scripts/quantization_sweep.py
python max-inference-bench/analysis/plot_results.py
```

### Benchmark Profiles

| Profile | Prompts | Dataset | Concurrency | Purpose |
|---------|---------|---------|-------------|---------|
| quick | 50 | sonnet | 1, 4 | Validation |
| standard | 500 | sonnet | 1--32 | Production baseline |
| stress | 2000 | sharegpt | 16--128 | Peak throughput, tail latency |
| long_context | 200 | arxiv | 1--8 | Long-sequence behaviour |

---

## Project 2: Mojo-GPU-Kernels

### Overview

These are GPU kernel implementations in Mojo that correspond to the optimization patterns used inside MAX's inference pipeline. Each kernel isolates a specific concern, making it straightforward to measure the impact of a single transformation (e.g., tiling, fusion, or address pattern changes) in isolation.

### Kernel Inventory

| Kernel | Optimisation Pattern | Relevance |
|--------|---------------------|-----------|
| vector_add.mojo | Grid/block decomposition, device memory management | Validates the programming model and memory transfer correctness |
| matmul_tiled.mojo | Shared memory tiling, data reuse | Core operation in attention and feed-forward layers |
| softmax.mojo | Online (single-pass) vs naive (three-pass) | Illustrates the kernel fusion opportunity that compilers target |
| reduction.mojo | Tree reduction, sequential addressing | Fundamental primitive for normalisation and loss computation |

### Running

```bash
# Single kernel
mojo run mojo-gpu-kernels/kernels/vector_add.mojo

# Full suite
python mojo-gpu-kernels/benchmarks/kernel_bench.py
```

---

## Hardware Targets

| Vendor | Datacenter | Consumer |
|--------|-----------|----------|
| NVIDIA | H100, H200, B200 | RTX 4090, RTX 3090 |
| AMD | MI300X, MI325X, MI355X | RDNA3/RDNA4 |

Both vendors use the same codebase. MAX and Mojo abstract the vendor-specific runtime (CUDA vs ROCm) behind a single programming model, so benchmark configurations and kernel source transfer across hardware without modification.

---

## Design Rationale

### MAX for benchmarking

- OpenAI-compatible serving API, so results apply directly to production serving analysis.
- Single binary across NVIDIA and AMD -- the same configuration files run on both vendors without changes.
- Native metric collection: `max benchmark` reports TTFT, TPOT, ITL, and GPU resource statistics without external instrumentation.
- Quantisation support: BF16, GGUF variants (Q4_K, Q6_K), GPTQ, and FP8 are all accessible from the same CLI.

### Mojo for kernel work

- Python-family syntax reduces friction compared to writing raw CUDA or HIP.
- Vendor-portable GPU code via compile-time dispatch -- one source file targets both NVIDIA and AMD.
- Direct integration with MAX's custom operation pipeline.
- Full systems-level control when needed: manual memory management, SIMD, tile scheduling.

### Methodology

The benchmarking methodology is documented in detail in `docs/METHODOLOGY.md`. The key principles:

1. **Isolation** -- each run starts a fresh Docker container with no leftover state.
2. **Warmup** -- configurable warmup requests precede every measurement window.
3. **Concurrency sweep** -- multiple concurrency levels per profile, since single-stream numbers are not representative of production behaviour.
4. **Percentile reporting** -- P50, P95, and P99 for all latency metrics. Tail latency matters for SLA compliance; reporting only averages obscures the problems that matter most.
5. **Reproducibility** -- pinned container image tags, recorded driver and runtime versions, one-command replay from any commit.

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| pyyaml | Configuration file parsing |
| matplotlib | Result visualisation |
| numpy | Numerical utilities |
| requests | HTTP health checks |
| openai | OpenAI-compatible client for ad-hoc inference testing |

System requirements: Docker, Modular `max` CLI, Mojo compiler, NVIDIA or AMD GPU with current drivers.

---

## Contact

Dr. Mysore Supreeth -- issues and pull requests are welcome. I am happy to discuss the methodology, walk through specific results, or explore collaboration.

## License

MIT
