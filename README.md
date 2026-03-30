# MAX + Mojo Performance Engineering Suite

**Author: Dr. Mysore Supreeth**

Systematic inference benchmarking and GPU kernel optimization on NVIDIA and AMD hardware using Modular's MAX platform and Mojo.

> *"I've been building benchmarking and optimization projects with MAX and Mojo. **MAX-Inference-Bench** measures inference performance across quantization schemes on NVIDIA and AMD hardware—exactly the kind of systematic performance engineering that drives Modular's compiler roadmap. **Mojo-GPU-Kernels** demonstrates kernel-level optimization patterns (tiled matmul, fused softmax, parallel reduction) that map directly to the custom op pipeline. The code is on GitHub; I'd be happy to walk through it."*
>
> — Dr. Mysore Supreeth

---

## Motivation

I built this suite because I kept running into the same problem: there was no clean, reproducible way to compare how different quantization schemes actually perform across NVIDIA and AMD GPUs under realistic serving conditions. Most benchmarks I found online either tested a single model on a single GPU, or buried the methodology so deeply that results couldn't be replicated.

My goal here is twofold:

1. **MAX-Inference-Bench** — a config-driven pipeline that sweeps models, quantizations, and hardware targets, then produces structured JSON you can actually diff across runs.
2. **Mojo-GPU-Kernels** — hand-written GPU kernels in Mojo that explore the optimization patterns (tiling, fusion, reduction) underpinning fast inference. Writing these helped me understand what the MAX compiler does automatically and where custom ops are still necessary.

Everything is designed to be reproducible: pinned container images, recorded driver versions, one-command execution.

---

## Repository Structure

```
max-mojo-perf-suite/
├── max-inference-bench/          # Project 1: Inference Benchmarking
│   ├── configs/
│   │   ├── models.yaml           # Model + quantization matrix
│   │   ├── benchmark_profiles.yaml   # Test profiles (quick → stress)
│   │   └── hardware.yaml         # NVIDIA / AMD Docker + env config
│   ├── scripts/
│   │   ├── run_benchmark.py      # Orchestrator: server lifecycle + max benchmark
│   │   ├── compare_hardware.py   # NVIDIA vs AMD comparison tables
│   │   └── quantization_sweep.py # Quant scheme Pareto analysis
│   ├── analysis/
│   │   └── plot_results.py       # matplotlib visualization
│   └── results/                  # JSON output (gitignored)
│
├── mojo-gpu-kernels/             # Project 2: Custom GPU Kernels
│   ├── kernels/
│   │   ├── vector_add.mojo       # Baseline: grid/block decomposition
│   │   ├── matmul_tiled.mojo     # Tiled matmul with shared memory
│   │   ├── softmax.mojo          # Fused online softmax (single-pass)
│   │   └── reduction.mojo        # Parallel reduction patterns
│   ├── benchmarks/
│   │   └── kernel_bench.py       # Compile + time all kernels
│   ├── tests/
│   └── analysis/
│
├── docker-compose.nvidia.yaml    # One-command NVIDIA stack
├── docker-compose.amd.yaml       # One-command AMD stack
├── requirements.txt
└── docs/
```

---

## Project 1: MAX-Inference-Bench

### What it does

I use this to measure LLM inference performance across the full quantization spectrum on both NVIDIA and AMD GPUs, using Modular's `max serve` + `max benchmark` pipeline. The orchestrator handles the full lifecycle — spinning up Docker containers, waiting for server health, running benchmarks at multiple concurrency levels, and collecting structured JSON results.

### Models × Quantizations

| Model | BF16 | Q4_K | Q6_K | GPTQ-INT4 | FP8 |
|-------|:----:|:----:|:----:|:---------:|:---:|
| Llama 3.1 8B Instruct | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mistral Nemo 12B | ✓ | ✓ | ✓ | — | — |
| Gemma 3 27B | ✓ | ✓ | — | — | — |

### Metrics Collected

| Category | Metrics |
|----------|---------|
| **Latency** | TTFT (P50/P95/P99), TPOT (P50/P95), ITL (P50/P99) |
| **Throughput** | Tokens/sec, Requests/sec |
| **Resources** | GPU utilization (mean), GPU memory peak, power draw |

### Quick Start

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_..."

# NVIDIA — dry run to verify configuration
python max-inference-bench/scripts/run_benchmark.py \
  --model llama3_1_8b \
  --profile quick \
  --hardware nvidia \
  --dry-run

# Full run (launches Docker, waits for health, benchmarks)
python max-inference-bench/scripts/run_benchmark.py \
  --model llama3_1_8b \
  --profile standard \
  --hardware nvidia amd

# Docker Compose shortcut (NVIDIA)
docker compose -f docker-compose.nvidia.yaml up

# Compare results
python max-inference-bench/scripts/compare_hardware.py
python max-inference-bench/scripts/quantization_sweep.py
python max-inference-bench/analysis/plot_results.py
```

### Benchmark Profiles

| Profile | Prompts | Dataset | Concurrency | Purpose |
|---------|---------|---------|-------------|---------|
| `quick` | 50 | sonnet | 1, 4 | Smoke test |
| `standard` | 500 | sonnet | 1–32 | Production baseline |
| `stress` | 2000 | sharegpt | 16–128 | Peak throughput + tail latency |
| `long_context` | 200 | arxiv | 1–8 | Long-sequence behavior |

---

## Project 2: Mojo-GPU-Kernels

### What I'm exploring

These are custom GPU kernel implementations in Mojo that plug into MAX's custom op pipeline. I wrote each one to understand a specific optimization pattern — from basic grid/block decomposition up through tiled matmul and fused softmax — the same patterns that matter in compiler and runtime engineering.

### Kernels

| Kernel | Optimization Pattern | Why It Matters |
|--------|---------------------|----------------|
| `vector_add.mojo` | Grid/block decomposition, device memory management | Validates GPU programming model correctness |
| `matmul_tiled.mojo` | Shared memory tiling, data reuse | Core of attention and FFN layers; memory hierarchy is everything |
| `softmax.mojo` | Online (single-pass) vs naive (three-pass) | Kernel fusion opportunity the compiler should find; custom ops must do it manually |
| `reduction.mojo` | Tree reduction, sequential addressing | Fundamental primitive for normalization layers and loss computation |

### Running Kernels

```bash
# Individual kernel
mojo run mojo-gpu-kernels/kernels/vector_add.mojo

# Full benchmark suite
python mojo-gpu-kernels/benchmarks/kernel_bench.py
```

---

## Hardware Targets

| Vendor | Datacenter | Consumer |
|--------|-----------|----------|
| **NVIDIA** | H100, H200, B200 | RTX 4090, RTX 3090 |
| **AMD** | MI300X, MI325X, MI355X | RDNA3/RDNA4 |

Both targets use the same codebase — MAX and Mojo abstract the vendor-specific runtime (CUDA vs ROCm) behind a unified programming model, which is one of the things I find most compelling about this stack.

---

## Architecture & Design Decisions

### Why I chose MAX for benchmarking

- **OpenAI-compatible API** — results transfer directly to production serving analysis
- **Single binary, multi-vendor** — same benchmark configs work across NVIDIA and AMD without rewriting anything
- **Built-in metrics** — `max benchmark` collects TTFT, TPOT, ITL, GPU stats natively; I don't have to instrument the server myself
- **Quantization matrix** — BF16, GGUF (Q4_K, Q6_K), GPTQ, FP8 from the same CLI

### Why Mojo for kernels

- **Python-family syntax** — significantly lower friction than raw CUDA/HIP for prototyping
- **Vendor-portable GPU code** — compile-time dispatch for NVIDIA vs AMD; I write one kernel, it runs on both
- **MAX custom op integration** — kernels plug directly into the inference graph
- **Systems-level control** — manual memory management, SIMD, tile scheduling when I need it

### Methodology

I follow a few principles that I've found matter more than any single benchmark number:

1. **Isolation**: Each benchmark run starts a fresh Docker container — no leftover state
2. **Warmup**: Configurable warmup requests before measurement begins
3. **Concurrency sweep**: Multiple concurrency levels per profile, because single-stream numbers lie
4. **Percentile reporting**: P50/P95/P99 — tails matter for SLAs, averages hide problems
5. **Reproducibility**: Pinned container images, recorded driver/runtime versions, one-command replay

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `pyyaml` | Config parsing |
| `matplotlib` | Result visualization |
| `numpy` | Numerical utilities |
| `requests` | HTTP health checks |
| `openai` | OpenAI-compatible client for ad-hoc inference |

**System requirements**: Docker, `max` CLI (from Modular), Mojo compiler, NVIDIA/AMD GPU with drivers.

---

## Contact

**Dr. Mysore Supreeth** — feel free to open an issue or reach out if you'd like to discuss the methodology, results, or potential collaboration.

## License

MIT
