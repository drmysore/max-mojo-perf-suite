#!/usr/bin/env python3
"""Kernel benchmark harness.

Author: Dr. Mysore Supreeth

Compiles Mojo GPU kernels and collects timing data across
problem sizes and data types, producing structured JSON output
suitable for regression tracking.
"""

import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional

KERNELS = Path(__file__).resolve().parent.parent / "kernels"
RESULTS = Path(__file__).resolve().parent.parent / "analysis"


@dataclass
class KernelRun:
    kernel: str
    problem_size: str
    dtype: str
    elapsed_ms: float
    gflops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    timestamp: str = ""


@dataclass
class BenchmarkReport:
    gpu_name: str
    driver_version: str
    mojo_version: str
    timestamp: str
    runs: list[dict] = field(default_factory=list)


def get_gpu_info() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return parts[0], parts[1] if len(parts) > 1 else "unknown"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip(), "ROCm"
    except Exception:
        pass

    return "unknown", "unknown"


def get_mojo_version() -> str:
    try:
        result = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compile_and_run(kernel_path: Path, timeout: int = 300) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(
        ["mojo", "run", str(kernel_path)],
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = (time.perf_counter() - start) * 1000

    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"Kernel failed: {output}")

    return elapsed, output


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    gpu_name, driver = get_gpu_info()
    mojo_ver = get_mojo_version()
    ts = datetime.now().isoformat()

    report = BenchmarkReport(
        gpu_name=gpu_name,
        driver_version=driver,
        mojo_version=mojo_ver,
        timestamp=ts,
    )

    kernels = sorted(KERNELS.glob("*.mojo"))
    if not kernels:
        print("No kernels found in", KERNELS)
        sys.exit(1)

    print(f"GPU: {gpu_name} | Driver: {driver} | Mojo: {mojo_ver}")
    print(f"Found {len(kernels)} kernel(s)\n")

    for kpath in kernels:
        print(f"Running {kpath.name}...")
        try:
            elapsed, output = compile_and_run(kpath)
            run = KernelRun(
                kernel=kpath.stem,
                problem_size="default",
                dtype="float32",
                elapsed_ms=elapsed,
                timestamp=ts,
            )
            report.runs.append(asdict(run))
            print(f"  Completed in {elapsed:.1f}ms")
            for line in output.strip().split("\n")[-5:]:
                print(f"    {line}")
        except Exception as e:
            print(f"  FAILED: {e}")
            report.runs.append({
                "kernel": kpath.stem,
                "error": str(e),
                "timestamp": ts,
            })

    out_path = RESULTS / f"kernel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
