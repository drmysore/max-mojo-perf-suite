#!/usr/bin/env python3
"""Benchmark result visualization.

Author: Dr. Mysore Supreeth

Generates publication-quality charts comparing inference performance
across quantization schemes and hardware targets.
Outputs PNG files suitable for README embedding or reports.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("Install dependencies: pip install matplotlib numpy")
    sys.exit(1)

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUTPUT = Path(__file__).resolve().parent / "figures"


def load_results() -> list[dict]:
    entries = []
    for p in sorted(RESULTS.glob("summary_*.json")):
        with open(p) as f:
            data = json.load(f)
            entries.extend(data if isinstance(data, list) else [data])
    return entries


def plot_throughput_by_quant(entries: list[dict]):
    """Bar chart: throughput (tok/s) per quantization, grouped by hardware."""
    by_model = defaultdict(list)
    for e in entries:
        by_model[e["model"]].append(e)

    for model, runs in by_model.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        quants = sorted(set(r["quantization"] for r in runs))
        hardware = sorted(set(r["hardware"] for r in runs))
        x = np.arange(len(quants))
        width = 0.35

        for i, hw in enumerate(hardware):
            vals = []
            for q in quants:
                match = [r for r in runs if r["quantization"] == q and r["hardware"] == hw]
                vals.append(match[0].get("throughput_tok_s", 0) if match else 0)
            offset = (i - len(hardware) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=hw.upper(), alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0f}", ha="center", va="bottom", fontsize=8)

        short_name = model.split("/")[-1][:30]
        ax.set_title(f"Inference Throughput — {short_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Tokens / second")
        ax.set_xticks(x)
        ax.set_xticklabels(quants, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        OUTPUT.mkdir(parents=True, exist_ok=True)
        safe_name = short_name.replace("/", "_").replace(" ", "_")
        fig.savefig(OUTPUT / f"throughput_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved throughput chart for {short_name}")


def plot_latency_profile(entries: list[dict]):
    """Grouped bar chart: TTFT P50/P95/P99 across quantizations."""
    by_model = defaultdict(list)
    for e in entries:
        by_model[e["model"]].append(e)

    for model, runs in by_model.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        quants = sorted(set(r["quantization"] for r in runs))
        x = np.arange(len(quants))
        width = 0.25

        for i, (key, label) in enumerate([
            ("ttft_p50_ms", "TTFT P50"),
            ("ttft_p95_ms", "TTFT P95"),
            ("ttft_p99_ms", "TTFT P99"),
        ]):
            vals = []
            for q in quants:
                match = [r for r in runs if r["quantization"] == q]
                vals.append(match[0].get(key, 0) if match else 0)
            ax.bar(x + (i - 1) * width, vals, width, label=label, alpha=0.85)

        short_name = model.split("/")[-1][:30]
        ax.set_title(f"TTFT Latency Distribution — {short_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Milliseconds")
        ax.set_xticks(x)
        ax.set_xticklabels(quants, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        OUTPUT.mkdir(parents=True, exist_ok=True)
        safe_name = short_name.replace("/", "_").replace(" ", "_")
        fig.savefig(OUTPUT / f"latency_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved latency chart for {short_name}")


def plot_memory_efficiency(entries: list[dict]):
    """Scatter: throughput vs memory, colored by quantization."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.Set2
    quants = sorted(set(e["quantization"] for e in entries))
    colors = {q: cmap(i / max(len(quants), 1)) for i, q in enumerate(quants)}

    for e in entries:
        tok_s = e.get("throughput_tok_s")
        mem = e.get("gpu_mem_peak_gb")
        if tok_s and mem:
            ax.scatter(mem, tok_s, c=[colors[e["quantization"]]],
                       s=100, alpha=0.8, edgecolors="black", linewidth=0.5)
            ax.annotate(
                f"{e['quantization']}\n{e['hardware']}",
                (mem, tok_s), fontsize=7, ha="center", va="bottom"
            )

    ax.set_xlabel("Peak GPU Memory (GB)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Memory-Performance Efficiency Frontier", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "memory_efficiency.png", dpi=150)
    plt.close(fig)
    print("Saved memory efficiency chart")


def main():
    entries = load_results()
    if not entries:
        print("No results found. Run benchmarks first, or place sample results in results/.")
        sys.exit(1)

    plot_throughput_by_quant(entries)
    plot_latency_profile(entries)
    plot_memory_efficiency(entries)
    print("\nAll charts saved to", OUTPUT)


if __name__ == "__main__":
    main()
