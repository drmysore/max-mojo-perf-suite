#!/usr/bin/env python3
"""MAX Inference Benchmark Orchestrator

Author: Dr. Mysore Supreeth

Systematically measures LLM inference performance across quantization
schemes on NVIDIA and AMD hardware using Modular's MAX platform.
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
import os
import signal
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"


@dataclass
class BenchmarkResult:
    model: str
    quantization: str
    hardware: str
    profile: str
    timestamp: str
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    itl_p50_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    throughput_tok_s: Optional[float] = None
    throughput_req_s: Optional[float] = None
    gpu_util_mean: Optional[float] = None
    gpu_mem_peak_gb: Optional[float] = None
    error: Optional[str] = None


def load_config(name: str) -> dict:
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_docker_cmd(hw_cfg: dict, model_name: str, quant: dict, vendor: str) -> list[str]:
    img = hw_cfg[vendor]["docker_image"]
    flags = hw_cfg[vendor]["docker_flags"]
    common = hw_cfg["common"]

    cmd = ["docker", "run", "-d", "--name", "max-bench-server"]
    cmd.extend(flags.split())

    for vol in common["volumes"]:
        expanded = os.path.expandvars(vol)
        cmd.extend(["-v", expanded])

    for port in common["ports"]:
        cmd.extend(["-p", port])

    all_env = {**common.get("env", {}), **hw_cfg[vendor].get("env", {})}
    for k, v in all_env.items():
        cmd.extend(["-e", f"{k}={os.path.expandvars(v)}"])

    cmd.append(img)
    serve_args = ["--model", quant.get("model_override", model_name)]

    if quant.get("weight_path"):
        serve_args.extend(["--weight-path", quant["weight_path"]])
    if quant["encoding"] != "bfloat16":
        serve_args.extend(["--quantization-encoding", quant["encoding"]])

    cmd.extend(serve_args)
    return cmd


def wait_for_server(timeout: int = 300) -> bool:
    """Poll the MAX server health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["curl", "-sf", "http://localhost:8000/health"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(5)
    return False


def build_benchmark_cmd(
    model_name: str, quant: dict, profile: dict, collect_gpu: bool
) -> list[str]:
    cmd = [
        "max", "benchmark",
        "--model", quant.get("model_override", model_name),
        "--backend", "modular",
        "--endpoint", "/v1/chat/completions",
        "--num-prompts", str(profile["num_prompts"]),
    ]

    if profile.get("dataset"):
        cmd.extend(["--dataset-name", profile["dataset"]])
    if profile.get("sonnet_input_len"):
        cmd.extend(["--sonnet-input-len", str(profile["sonnet_input_len"])])
    if profile.get("output_lengths"):
        cmd.extend(["--output-lengths", str(profile["output_lengths"])])
    if profile.get("sonnet_prefix_len"):
        cmd.extend(["--sonnet-prefix-len", str(profile["sonnet_prefix_len"])])
    if collect_gpu:
        cmd.append("--collect-gpu-stats")

    return cmd


def run_single_benchmark(
    model_key: str,
    model_cfg: dict,
    quant: dict,
    profile_name: str,
    profile: dict,
    vendor: str,
    hw_cfg: dict,
    dry_run: bool = False,
) -> BenchmarkResult:
    ts = datetime.now().isoformat()
    label = f"{model_key}_{quant['encoding']}_{vendor}_{profile_name}"
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"  Model: {model_cfg['name']} | Quant: {quant['label']}")
    print(f"  Hardware: {vendor.upper()} | Profile: {profile_name}")
    print(f"{'='*60}")

    result = BenchmarkResult(
        model=model_cfg["name"],
        quantization=quant["encoding"],
        hardware=vendor,
        profile=profile_name,
        timestamp=ts,
    )

    docker_cmd = build_docker_cmd(hw_cfg, model_cfg["name"], quant, vendor)
    bench_cmds = []
    for conc in profile.get("concurrency_levels", [1]):
        cmd = build_benchmark_cmd(
            model_cfg["name"], quant, profile, hw_cfg[vendor].get("gpu_stats", False)
        )
        cmd.extend(["--concurrency", str(conc)])
        result_file = RESULTS / f"{label}_c{conc}_{ts.replace(':', '-')}.json"
        cmd.extend(["--save-result", "--result-filename", str(result_file)])
        bench_cmds.append((conc, cmd))

    if dry_run:
        print(f"  [DRY RUN] Docker: {' '.join(docker_cmd)}")
        for conc, cmd in bench_cmds:
            print(f"  [DRY RUN] Bench (c={conc}): {' '.join(cmd)}")
        return result

    try:
        subprocess.run(
            ["docker", "rm", "-f", "max-bench-server"],
            capture_output=True
        )
        print("  Starting MAX server...")
        subprocess.run(docker_cmd, check=True)

        if not wait_for_server():
            result.error = "Server failed to start within timeout"
            return result

        for conc, cmd in bench_cmds:
            print(f"  Running benchmark (concurrency={conc})...")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                print(f"  WARN: Benchmark exited {proc.returncode}")
                result.error = proc.stderr[:500]
    except Exception as e:
        result.error = str(e)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", "max-bench-server"],
            capture_output=True
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MAX Inference Benchmark Orchestrator"
    )
    parser.add_argument(
        "--model", nargs="*", default=None,
        help="Model keys to benchmark (default: all)"
    )
    parser.add_argument(
        "--profile", default="quick",
        choices=["quick", "standard", "stress", "long_context"],
    )
    parser.add_argument(
        "--hardware", nargs="*", default=["nvidia"],
        choices=["nvidia", "amd"],
    )
    parser.add_argument(
        "--quantization", nargs="*", default=None,
        help="Filter quantization encodings (e.g. bfloat16 q4_k gptq)"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_cfg = load_config("models")
    profiles_cfg = load_config("benchmark_profiles")
    hw_cfg = load_config("hardware")

    profile = profiles_cfg["profiles"][args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_keys = args.model or list(models_cfg["models"].keys())
    all_results = []

    for mkey in model_keys:
        mcfg = models_cfg["models"][mkey]
        for quant in mcfg["quantizations"]:
            if args.quantization and quant["encoding"] not in args.quantization:
                continue
            for vendor in args.hardware:
                result = run_single_benchmark(
                    mkey, mcfg, quant, args.profile,
                    profile, vendor, hw_cfg["hardware"],
                    dry_run=args.dry_run,
                )
                all_results.append(asdict(result))

    summary_path = RESULTS / f"summary_{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
