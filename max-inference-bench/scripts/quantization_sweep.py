#!/usr/bin/env python3
"""Quantization sweep analysis.

Author: Dr. Mysore Supreeth

Generates a report comparing inference quality and performance
across all quantization schemes for a given model, highlighting
the accuracy-performance Pareto frontier.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_all_results() -> list[dict]:
    entries = []
    for p in sorted(RESULTS.glob("summary_*.json")):
        with open(p) as f:
            data = json.load(f)
            entries.extend(data if isinstance(data, list) else [data])
    return entries


def compute_speedup(baseline_tok_s: float, current_tok_s: float) -> str:
    if not baseline_tok_s or not current_tok_s:
        return "—"
    ratio = current_tok_s / baseline_tok_s
    return f"{ratio:.2f}x"


def compute_memory_savings(baseline_gb: float, current_gb: float) -> str:
    if not baseline_gb or not current_gb:
        return "—"
    saved = (1 - current_gb / baseline_gb) * 100
    return f"{saved:.1f}%"


def print_sweep(entries: list[dict]):
    by_model_hw = defaultdict(list)
    for e in entries:
        key = (e["model"], e["hardware"])
        by_model_hw[key].append(e)

    for (model, hw), runs in sorted(by_model_hw.items()):
        print(f"\n{'='*80}")
        print(f"  {model} on {hw.upper()}")
        print(f"{'='*80}")

        baseline = next(
            (r for r in runs if r["quantization"] == "bfloat16"), None
        )
        base_tok = baseline["throughput_tok_s"] if baseline else None
        base_mem = baseline["gpu_mem_peak_gb"] if baseline else None

        header = (
            f"{'Quantization':<18} {'Tok/s':>10} {'Speedup':>10} "
            f"{'TTFT P50':>10} {'TTFT P99':>10} {'Mem GB':>10} {'Mem Saved':>10}"
        )
        print(header)
        print("-" * len(header))

        for r in sorted(runs, key=lambda x: x.get("throughput_tok_s") or 0, reverse=True):
            tok_s = r.get("throughput_tok_s")
            speedup = compute_speedup(base_tok, tok_s) if base_tok else "—"
            ttft50 = f"{r['ttft_p50_ms']:.1f}" if r.get("ttft_p50_ms") else "—"
            ttft99 = f"{r['ttft_p99_ms']:.1f}" if r.get("ttft_p99_ms") else "—"
            mem = f"{r['gpu_mem_peak_gb']:.1f}" if r.get("gpu_mem_peak_gb") else "—"
            mem_saved = compute_memory_savings(base_mem, r.get("gpu_mem_peak_gb")) if base_mem else "—"
            tok_str = f"{tok_s:.1f}" if tok_s else "—"

            print(
                f"{r['quantization']:<18} {tok_str:>10} {speedup:>10} "
                f"{ttft50:>10} {ttft99:>10} {mem:>10} {mem_saved:>10}"
            )


def main():
    entries = load_all_results()
    if not entries:
        print("No results found. Run benchmarks first.")
        sys.exit(1)
    print_sweep(entries)


if __name__ == "__main__":
    main()
