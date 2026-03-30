# Benchmarking Methodology

**Author: Dr. Mysore Supreeth**

## Metric Definitions

All metrics align with the definitions used by NVIDIA GenAI-Perf and Modular's `max benchmark`, so that results produced by this suite are directly comparable to published industry benchmarks.

### TTFT -- Time to First Token
Wall-clock time from request submission to the arrival of the first non-empty output token at the client. Dominated by prompt prefill; scales with input sequence length. This is the metric that determines perceived startup latency in interactive applications.

### TPOT -- Time Per Output Token
Mean time between consecutive output tokens during the decode phase, excluding TTFT. Characterises decode throughput from the end user's perspective.

### ITL -- Inter-Token Latency
Time gap between adjacent output tokens, reported as percentiles. Where TPOT captures the average decode rate, ITL percentiles (particularly P95 and P99) expose jitter caused by scheduling delays, batching interference, or garbage collection pauses.

### Throughput (tok/s)
Total output tokens generated across all concurrent requests, divided by the measurement window. The window begins at first request submission and ends at last token delivery.

### GPU Metrics
Collected via `--collect-gpu-stats` when the benchmark client and the MAX server share a machine. Includes mean GPU utilisation over the measurement window and peak memory allocation.

## Controlled Variables

Each benchmark run records the following for reproducibility:
- MAX container image tag and digest
- GPU driver version (NVIDIA: `nvidia-smi`, AMD: `rocm-smi`)
- Model revision (HuggingFace commit hash)
- Quantisation encoding
- Benchmark profile (prompt count, dataset, concurrency levels)
- Timestamp

## Cross-Vendor Comparison Fairness

For NVIDIA vs AMD comparisons, every variable is held constant except the hardware target:
- Identical model revisions and quantisation encodings
- Same benchmark profile and concurrency sweep
- Same MAX version (pinned container tag)
- Same dataset and prompt distribution

The only difference is the hardware and its corresponding Docker image. This isolation is deliberate: most published cross-vendor comparisons conflate software stack maturity with silicon capability, which makes it impossible to attribute performance differences to the hardware itself.

## Quantisation Tradeoff Analysis

For each model, the BF16 result serves as the accuracy and memory baseline. Lower-precision schemes are evaluated along four axes:

1. **Throughput delta** -- speedup relative to BF16 at matched concurrency.
2. **Latency delta** -- change in TTFT and ITL, with particular attention to P99 tails.
3. **Memory savings** -- reduction in peak GPU memory allocation.
4. **Quality** -- perplexity or downstream task accuracy, measured separately via lm-evaluation-harness when needed.

The Pareto frontier identifies quantisation schemes that improve throughput and reduce memory without introducing unacceptable latency regression or quality degradation. In practice, the right choice is always workload-dependent, which is precisely why having a repeatable, structured sweep matters.
