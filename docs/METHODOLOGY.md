# Benchmarking Methodology

**Author: Dr. Mysore Supreeth**

## Metrics Definitions

I align all metrics with the definitions established by NVIDIA GenAI-Perf and Modular's `max benchmark`, so results are directly comparable to published industry benchmarks:

### TTFT — Time to First Token
Wall-clock time from request submission to the first non-empty output token arriving at the client. Dominated by prompt prefill; scales with input length. Measures perceived "startup" latency.

### TPOT — Time Per Output Token
Average time between consecutive output tokens during the decode phase (excludes TTFT). Characterizes decode throughput from the user's perspective.

### ITL — Inter-Token Latency
Time gap between adjacent output tokens, reported as percentiles. Differs from TPOT in that individual ITL values capture jitter (scheduling delays, batching interference), not just the average.

### Throughput (tok/s)
Total output tokens generated across all concurrent requests divided by the measurement window. Window starts at first request submission, ends at last token delivery.

### GPU Metrics
Collected via `--collect-gpu-stats` when client and server share a machine. Includes utilization percentage (mean over measurement window) and peak memory allocation.

## Controlled Variables

Each benchmark run records:
- MAX container image tag and digest
- GPU driver version (NVIDIA: `nvidia-smi`, AMD: `rocm-smi`)
- Model revision (HuggingFace commit hash)
- Quantization encoding
- Benchmark profile (prompt count, dataset, concurrency levels)
- Timestamp

## Comparison Fairness

For NVIDIA vs AMD comparisons, I hold everything constant except the hardware:
- Identical model revisions and quantization encodings
- Same benchmark profile and concurrency sweep
- Same MAX version (pinned container tag)
- Same dataset and prompt distribution

The only variable is the hardware target and its corresponding Docker image. This matters because most published comparisons conflate software stack differences with hardware differences.

## Quantization Tradeoff Analysis

For each model, BF16 serves as the accuracy and memory baseline. Lower-precision schemes are evaluated on:

1. **Throughput delta** — speedup vs BF16 at matched concurrency
2. **Latency delta** — TTFT/ITL change, especially P99 tails
3. **Memory savings** — peak GPU memory reduction
4. **Quality** — perplexity or downstream task accuracy (measured separately via lm-evaluation-harness if needed)

The Pareto frontier identifies quantization schemes that improve throughput and memory without unacceptable latency or quality regression. In my experience, this is where the interesting engineering decisions live — the "right" quantization is always workload-dependent.
