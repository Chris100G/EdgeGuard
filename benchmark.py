"""
EdgeGuard - Model Benchmark Utility
Compares INT8 Quantized vs FP32 TFLite/LiteRT Person Detection Models.
Measures real hardware latency, standard deviation, throughput (FPS), and memory footprint.
"""

import sys
import os
import time
from typing import Dict, List, Any
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_REGISTRY
from core.detector import PersonDetector, BACKEND_NAME


def run_benchmark(iterations: int = 40) -> List[Dict[str, Any]]:
    """
    Benchmark all available models in MODEL_REGISTRY on the current system.
    Returns structured benchmark results.
    """
    print("\n" + "=" * 78)
    print(f" EDGEGUARD TINYML BENCHMARK SUITE")
    print(f" Execution Backend: {BACKEND_NAME}")
    print(f" Iterations per model: {iterations} (with 5 warmup cycles)")
    print("=" * 78)

    results = []

    for model_key, meta in MODEL_REGISTRY.items():
        model_path = meta["path"]
        if not model_path.exists():
            print(f"[SKIP] Model file not found: {model_path}")
            continue

        print(f"\nBenchmarking '{meta['name']}'...")
        print(f"  File: {meta['filename']} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")
        print(f"  Quantization: {meta['precision']} (Quantized: {meta['quantized']})")
        print(f"  Input Resolution: {meta['input_size'][0]}x{meta['input_size'][1]}")

        try:
            detector = PersonDetector(model_key=model_key)
            stats = detector.benchmark(iterations=iterations)
            results.append(stats)
            print(f"  -> Mean Latency:  {stats['mean_latency_ms']:.2f} ms")
            print(f"  -> Min / Max:      {stats['min_latency_ms']:.2f} ms / {stats['max_latency_ms']:.2f} ms")
            print(f"  -> Latency Std:    {stats['std_latency_ms']:.2f} ms")
            print(f"  -> Throughput:     {stats['approx_fps']:.1f} FPS")
        except Exception as e:
            print(f"  [ERROR] Failed to benchmark {model_key}: {e}")

    # Print Summary Comparison Table
    print("\n" + "=" * 78)
    print(" BENCHMARK COMPARISON TABLE")
    print("=" * 78)
    headers = ["Model Architecture", "Precision", "Size (MB)", "Mean (ms)", "Min (ms)", "FPS"]
    row_fmt = "{:<32} {:<10} {:<10} {:<11} {:<10} {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 78)

    for r in results:
        print(row_fmt.format(
            r["model_name"][:30],
            r["precision"],
            f"{r['model_size_mb']:.2f}",
            f"{r['mean_latency_ms']:.2f}",
            f"{r['min_latency_ms']:.2f}",
            f"{r['approx_fps']:.1f}",
        ))

    print("=" * 78)
    print("Note: Latency is measured using time.perf_counter on the local host CPU.\n")
    return results


if __name__ == "__main__":
    run_benchmark(iterations=35)

