#!/usr/bin/env python3
"""Cross-hardware comparison report generator.

Author: Dr. Mysore Supreeth

Reads benchmark result JSONs and produces comparative tables
showing NVIDIA vs AMD performance across quantization schemes.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "results"

METRIC_LABELS = {
    "ttft_p50_ms": "TTFT P50 (ms)",
    "ttft_p95_ms": "TTFT P95 (ms)",
    "ttft_p99_ms": "TTFT P99 (ms)",
    "tpot_p50_ms": "TPOT P50 (ms)",
    "tpot_p95_ms": "TPOT P95 (ms)",
    "itl_p50_ms":  "ITL P50 (ms)",
    "itl_p99_ms":  "ITL P99 (ms)",
    "throughput_tok_s": "Throughput (tok/s)",
    "throughput_req_s": "Throughput (req/s)",
    "gpu_util_mean": "GPU Util %",
    "gpu_mem_peak_gb": "GPU Mem Peak (GB)",
}


def load_results(pattern: str = "summary_*.json") -> list[dict]:
    entries = []
    for p in sorted(RESULTS.glob(pattern)):
        with open(p) as f:
            data = json.load(f)
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)
    return entries


def group_by(entries: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        k = tuple(e.get(key, "?") for key in keys)
        grouped[k].append(e)
    return grouped


def fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def print_comparison_table(entries: list[dict]):
    by_model_quant = group_by(entries, "model", "quantization")

    for (model, quant), runs in sorted(by_model_quant.items()):
        nvidia_runs = [r for r in runs if r["hardware"] == "nvidia"]
        amd_runs = [r for r in runs if r["hardware"] == "amd"]

        if not nvidia_runs and not amd_runs:
            continue

        print(f"\n{'='*72}")
        print(f"  {model}  |  Quantization: {quant}")
        print(f"{'='*72}")
        header = f"{'Metric':<25} {'NVIDIA':>12} {'AMD':>12} {'Delta':>12}"
        print(header)
        print("-" * len(header))

        nv = nvidia_runs[0] if nvidia_runs else {}
        am = amd_runs[0] if amd_runs else {}

        for key, label in METRIC_LABELS.items():
            nv_val = nv.get(key)
            am_val = am.get(key)
            delta = ""
            if isinstance(nv_val, (int, float)) and isinstance(am_val, (int, float)):
                diff = am_val - nv_val
                pct = (diff / nv_val * 100) if nv_val != 0 else 0
                sign = "+" if diff >= 0 else ""
                delta = f"{sign}{pct:.1f}%"
            print(f"{label:<25} {fmt(nv_val):>12} {fmt(am_val):>12} {delta:>12}")


def main():
    entries = load_results()
    if not entries:
        print("No results found. Run benchmarks first.")
        sys.exit(1)
    print_comparison_table(entries)


if __name__ == "__main__":
    main()
