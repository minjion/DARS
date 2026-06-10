"""Benchmark DARS inference latency, throughput, and memory usage.

Measures the FULL pipeline: feature extraction + model inference + rule scoring
+ SHAP explanation. Includes proper CUDA warmup to avoid cold-start bias.

Simulates concurrent agent loads of 1, 10, 50, and 100 agents.

Run from the repository root:
    python scripts/scalability_benchmark.py
"""

import gc
import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier
from src.scoring.risk import calculate_rule_risk, blend_risk_scores
from src.explainability.shap_analyzer import ExplainableRiskAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models_saved" / "dars_model_real.pt"
TEST_DATA = ROOT / "data" / "test_traces.jsonl"
RESULTS_PATH = ROOT / "scripts" / "scalability_results.txt"
CONCURRENT_LOADS = [1, 10, 50, 100]
WARMUP_ITERS = 10
MEASURE_ITERS = 50


def load_checkpoint(path, device):
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = DARSClassifier(
        input_dim=config.get("input_dim", len(FEATURE_NAMES)),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    extractor = DARSFeatureExtractor.from_state(checkpoint["extractor_state"])
    return model, extractor, checkpoint


# ---------------------------------------------------------------------------
# Full single-trace pipeline (with proper warmup)
# ---------------------------------------------------------------------------

def benchmark_single_trace_pipeline(model, extractor, traces, device, shap_bg=None):
    """Measure each stage of the DARS pipeline with CUDA warmup."""
    analyzer = ExplainableRiskAnalyzer()

    # Pick 5 diverse traces for measurement
    sample_indices = np.linspace(0, len(traces) - 1, 5, dtype=int)
    sample_traces = [traces[i] for i in sample_indices]

    # === WARMUP (critical for CUDA) ===
    print("  Warming up CUDA...")
    for _ in range(WARMUP_ITERS):
        for trace in sample_traces:
            features = extractor.extract_trace_features(trace)
            x = features.unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(x).cpu().item()
            summary = extractor.summarize_trace(trace)
            _ = calculate_rule_risk(trace, summary)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("  Warmup complete.")

    # === MEASURE ===
    times_extract = []
    times_infer = []
    times_rule = []
    times_shap = []
    times_total = []

    for _ in range(MEASURE_ITERS):
        for trace in sample_traces:
            # 1. Feature extraction
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            features = extractor.extract_trace_features(trace)
            summary = extractor.summarize_trace(trace)
            t1 = time.perf_counter()
            times_extract.append((t1 - t0) * 1000)

            # 2. Model inference
            x = features.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                learned_score = model(x).cpu().item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            times_infer.append((t3 - t2) * 1000)

            # 3. Rule scoring + blending
            t4 = time.perf_counter()
            rule_score = calculate_rule_risk(trace, summary)
            final_score = blend_risk_scores(learned_score, rule_score)
            t5 = time.perf_counter()
            times_rule.append((t5 - t4) * 1000)

            # 4. SHAP explanation
            t6 = time.perf_counter()
            explanation = analyzer.permutation_attribution(model, features.to(device))
            t7 = time.perf_counter()
            times_shap.append((t7 - t6) * 1000)

            # Total
            times_total.append((t1 - t0 + t3 - t2 + t5 - t4 + t7 - t6) * 1000)

    return {
        "extract_ms": float(np.median(times_extract)),
        "infer_ms": float(np.median(times_infer)),
        "rule_ms": float(np.median(times_rule)),
        "shap_ms": float(np.median(times_shap)),
        "total_ms": float(np.median(times_total)),
        "total_mean_ms": float(np.mean(times_total)),
        "total_p95_ms": float(np.percentile(times_total, 95)),
    }


# ---------------------------------------------------------------------------
# Batch throughput benchmark (concurrent agents)
# ---------------------------------------------------------------------------

def infer_batch(model, batch_tensor, device):
    with torch.no_grad():
        return model(batch_tensor.to(device)).cpu().numpy().flatten()


def benchmark_concurrent(model, dataset, device, n_agents):
    """Simulate n_agents concurrent inference with proper measurement."""
    data_tensor = torch.stack([dataset[i][0] for i in range(len(dataset))])
    total_traces = len(dataset)

    chunk_size = max(1, total_traces // n_agents)
    chunks = [
        data_tensor[i * chunk_size: min((i + 1) * chunk_size, total_traces)]
        for i in range(n_agents)
    ]
    if total_traces > chunk_size * n_agents:
        remaining = total_traces - chunk_size * n_agents
        chunks[-1] = torch.cat([chunks[-1], data_tensor[-remaining:]], dim=0)

    # Warmup
    for _ in range(WARMUP_ITERS):
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = [pool.submit(infer_batch, model, c, device) for c in chunks]
            for f in as_completed(futures):
                f.result()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    gc.collect()
    tracemalloc.start()
    latencies = []
    for _ in range(10):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = [pool.submit(infer_batch, model, c, device) for c in chunks]
            for f in as_completed(futures):
                f.result()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        latencies.append(elapsed)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_wall = float(np.mean(latencies))
    return {
        "n_agents": n_agents,
        "total_traces": total_traces,
        "avg_latency_ms": (avg_wall / total_traces) * 1000,
        "throughput_traces_sec": total_traces / avg_wall,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("DARS - Full Pipeline Scalability Benchmark")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, extractor, checkpoint = load_checkpoint(MODEL_PATH, device)
    raw_traces = load_traces(str(TEST_DATA))
    dataset = AgentDojoDataset(TEST_DATA, extractor=extractor)
    print(f"Test traces: {len(dataset)}")
    print(f"Warmup iters: {WARMUP_ITERS}, Measure iters: {MEASURE_ITERS}")
    print()

    # --- Single-trace full pipeline ---
    print("=== Single-Trace Full Pipeline (with CUDA warmup) ===")
    single = benchmark_single_trace_pipeline(model, extractor, raw_traces, device)

    single_lines = [
        "=== Single-Trace Full Pipeline Latency (median) ===",
        f"  Feature extraction : {single['extract_ms']:.3f} ms",
        f"  Model inference    : {single['infer_ms']:.3f} ms",
        f"  Rule scoring       : {single['rule_ms']:.3f} ms",
        f"  SHAP explanation   : {single['shap_ms']:.3f} ms",
        f"  ---",
        f"  Total (median)     : {single['total_ms']:.3f} ms",
        f"  Total (mean)       : {single['total_mean_ms']:.3f} ms",
        f"  Total (p95)        : {single['total_p95_ms']:.3f} ms",
    ]
    for line in single_lines:
        print(line)
    print()

    # --- Concurrent benchmark ---
    print("=== Concurrent Inference Benchmark (model only) ===")
    header = f"{'Agents':>8s}  {'Traces':>8s}  {'Latency (ms)':>14s}  {'Throughput (t/s)':>18s}  {'Peak Mem (MB)':>14s}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    results = []
    for n_agents in CONCURRENT_LOADS:
        r = benchmark_concurrent(model, dataset, device, n_agents)
        results.append(r)
        print(
            f"{r['n_agents']:>8d}  {r['total_traces']:>8d}  "
            f"{r['avg_latency_ms']:>14.3f}  "
            f"{r['throughput_traces_sec']:>18.1f}  "
            f"{r['peak_memory_mb']:>14.2f}"
        )

    # --- Save ---
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for line in single_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write("=== Concurrent Inference Benchmark (model only) ===\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in results:
            f.write(
                f"{r['n_agents']:>8d}  {r['total_traces']:>8d}  "
                f"{r['avg_latency_ms']:>14.3f}  "
                f"{r['throughput_traces_sec']:>18.1f}  "
                f"{r['peak_memory_mb']:>14.2f}\n"
            )
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
