"""
Enhanced Scalability Benchmark with CPU/RAM/GPU Metrics.

Benchmarks DARS model inference at various concurrent loads,
measuring latency, throughput, CPU/RAM/GPU usage, and scaling efficiency.
Also benchmarks the full pipeline (feature extraction + inference + rule scoring).
"""

import os
import sys
import time
import gc

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
import torch
import psutil
from torch.utils.data import DataLoader

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.scoring.risk import calculate_rule_risk
from evaluate import load_checkpoint, predict_model, summarize_features


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def warmup_cuda(model, dataset, device, iterations=10):
    """CUDA warmup with dummy inference."""
    if not torch.cuda.is_available():
        return
    print("Running CUDA warmup...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
    torch.cuda.synchronize()
    print(f"CUDA warmup complete ({iterations} iterations)")


def benchmark_batch(model, dataset, device, batch_size, num_runs=5):
    """Run inference with a given batch size multiple times and collect metrics."""
    process = psutil.Process()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_traces = len(dataset)

    latencies = []
    cpu_usages = []
    ram_usages = []
    gpu_memories = []

    model.eval()

    for run in range(num_runs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Measure CPU usage over inference period
        psutil.cpu_percent(interval=None)  # reset
        reset_gpu_memory()

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        gpu_mem = get_gpu_memory_mb()

        latencies.append(elapsed)
        cpu_usages.append(cpu_usage)
        ram_usages.append(ram_usage)
        gpu_memories.append(gpu_mem)

    avg_elapsed = np.mean(latencies)
    avg_latency_ms = (avg_elapsed / max(n_traces, 1)) * 1000.0
    throughput = n_traces / avg_elapsed if avg_elapsed > 0 else 0

    return {
        "batch_size": batch_size,
        "n_traces": n_traces,
        "avg_total_time_s": avg_elapsed,
        "avg_latency_ms": avg_latency_ms,
        "throughput_traces_s": throughput,
        "cpu_percent": np.mean(cpu_usages),
        "ram_mb": np.mean(ram_usages),
        "gpu_memory_mb": np.mean(gpu_memories),
    }


def benchmark_concurrent_loads(model, dataset, device, loads):
    """Simulate concurrent loads by increasing batch size."""
    results = []
    for load in loads:
        print(f"  Benchmarking concurrent load = {load}...")
        result = benchmark_batch(model, dataset, device, batch_size=load, num_runs=5)
        results.append(result)
    return results


def benchmark_full_pipeline(extractor, model, device, trace):
    """Benchmark each stage of the full pipeline on a single trace."""
    process = psutil.Process()

    # Stage 1: Feature extraction
    gc.collect()
    reset_gpu_memory()
    start = time.perf_counter()
    features = extractor.extract_trace_features(trace)
    feat_time = time.perf_counter() - start
    feat_time_ms = feat_time * 1000.0

    # Stage 2: Model inference
    gc.collect()
    reset_gpu_memory()
    model.eval()
    x = features.unsqueeze(0).to(device)

    # Warmup for single trace
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    reset_gpu_memory()
    start = time.perf_counter()
    with torch.no_grad():
        score = model(x).detach().cpu().numpy().flatten()[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - start
    infer_time_ms = infer_time * 1000.0
    gpu_mem_infer = get_gpu_memory_mb()

    # Stage 3: Rule scoring
    gc.collect()
    summary = extractor.summarize_trace(trace)
    start = time.perf_counter()
    rule_score = calculate_rule_risk(trace, summary)
    rule_time = time.perf_counter() - start
    rule_time_ms = rule_time * 1000.0

    total_time_ms = feat_time_ms + infer_time_ms + rule_time_ms
    ram_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "feature_extraction_ms": feat_time_ms,
        "model_inference_ms": infer_time_ms,
        "rule_scoring_ms": rule_time_ms,
        "total_pipeline_ms": total_time_ms,
        "model_score": float(score),
        "rule_score": float(rule_score),
        "gpu_memory_inference_mb": gpu_mem_infer,
        "ram_mb": ram_mb,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model_path = "models_saved/dars_model_real.pt"
    model, extractor, config = load_checkpoint(model_path, device)
    print(f"Model loaded from {model_path}")

    # Load test dataset
    test_path = "data/test_traces.jsonl"
    test_dataset = AgentDojoDataset(test_path, extractor=extractor)
    test_traces = test_dataset.traces_raw
    print(f"Test traces loaded: {len(test_traces)}")

    # CUDA warmup
    warmup_cuda(model, test_dataset, device, iterations=10)

    lines = []
    lines.append("=" * 100)
    lines.append("DARS Enhanced Scalability Benchmark")
    lines.append("=" * 100)
    lines.append(f"Device: {device}")
    lines.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append(f"Model: {model_path}")
    lines.append(f"Test traces: {len(test_traces)}")
    lines.append(f"CPU cores: {psutil.cpu_count(logical=True)}")
    lines.append(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    lines.append("")

    # --- Concurrent load benchmark ---
    concurrent_loads = [1, 10, 50, 100]
    print("\n=== Concurrent Load Benchmark ===")
    lines.append("=== Concurrent Load Benchmark ===")
    lines.append("(Simulated via batch size; 5 runs averaged per load)")
    lines.append("")

    results = benchmark_concurrent_loads(model, test_dataset, device, concurrent_loads)

    header = (
        f"{'Load':>6s} {'Latency(ms)':>12s} {'Throughput':>12s} "
        f"{'CPU%':>7s} {'RAM(MB)':>10s} {'GPU(MB)':>10s} {'Efficiency':>11s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Base throughput for efficiency calculation
    base_throughput = results[0]["throughput_traces_s"] if results else 1.0

    for i, r in enumerate(results):
        load = r["batch_size"]
        efficiency = r["throughput_traces_s"] / (load * base_throughput) if (load * base_throughput) > 0 else 0
        eff_str = f"{efficiency:.3f}"
        line = (
            f"{load:>6d} {r['avg_latency_ms']:>12.3f} {r['throughput_traces_s']:>12.1f} "
            f"{r['cpu_percent']:>7.1f} {r['ram_mb']:>10.1f} {r['gpu_memory_mb']:>10.1f} {eff_str:>11s}"
        )
        lines.append(line)
        print(f"  Load={load}: latency={r['avg_latency_ms']:.3f}ms, "
              f"throughput={r['throughput_traces_s']:.1f} traces/s, "
              f"CPU={r['cpu_percent']:.1f}%, RAM={r['ram_mb']:.1f}MB, "
              f"GPU={r['gpu_memory_mb']:.1f}MB, efficiency={efficiency:.3f}")

    lines.append("")
    lines.append("Scaling Efficiency = Throughput_n / (n × Throughput_1)")
    lines.append(f"Base throughput (load=1): {base_throughput:.1f} traces/s")
    lines.append("")

    # --- Detailed resource usage ---
    lines.append("=== Detailed Resource Usage per Load ===")
    lines.append("")
    for r in results:
        lines.append(f"--- Load = {r['batch_size']} ---")
        lines.append(f"  Avg total time:  {r['avg_total_time_s']*1000:.3f} ms")
        lines.append(f"  Avg latency:     {r['avg_latency_ms']:.3f} ms/trace")
        lines.append(f"  Throughput:      {r['throughput_traces_s']:.1f} traces/s")
        lines.append(f"  CPU usage:       {r['cpu_percent']:.1f}%")
        lines.append(f"  RAM usage:       {r['ram_mb']:.1f} MB")
        lines.append(f"  GPU memory:      {r['gpu_memory_mb']:.1f} MB")
        lines.append("")

    # --- Full pipeline benchmark ---
    print("\n=== Full Pipeline Benchmark ===")
    lines.append("=== Full Pipeline Benchmark (Single Trace) ===")
    lines.append("(With CUDA warmup; measures feature extraction, inference, rule scoring)")
    lines.append("")

    # Pick a representative trace (first one)
    if len(test_traces) > 0:
        sample_trace = test_traces[0]
        pipeline_result = benchmark_full_pipeline(extractor, model, device, sample_trace)

        lines.append(f"Sample trace scenario: {sample_trace.get('scenario', 'unknown')}")
        lines.append(f"Sample trace label: {sample_trace.get('label', 'unknown')}")
        lines.append(f"Tool calls: {len(sample_trace.get('tool_calls', []))}")
        lines.append("")
        lines.append(f"{'Stage':<25s} {'Time (ms)':>12s} {'% of Total':>12s}")
        lines.append("-" * 50)

        total_ms = pipeline_result["total_pipeline_ms"]
        for stage, key in [
            ("Feature Extraction", "feature_extraction_ms"),
            ("Model Inference", "model_inference_ms"),
            ("Rule Scoring", "rule_scoring_ms"),
        ]:
            t = pipeline_result[key]
            pct = (t / total_ms * 100) if total_ms > 0 else 0
            lines.append(f"{stage:<25s} {t:>12.3f} {pct:>11.1f}%")

        lines.append("-" * 50)
        lines.append(f"{'TOTAL':<25s} {total_ms:>12.3f} {'100.0%':>12s}")
        lines.append("")
        lines.append(f"Model score: {pipeline_result['model_score']:.4f}")
        lines.append(f"Rule score:  {pipeline_result['rule_score']:.4f}")
        lines.append(f"GPU memory (inference):  {pipeline_result['gpu_memory_inference_mb']:.1f} MB")
        lines.append(f"RAM usage:              {pipeline_result['ram_mb']:.1f} MB")

        print(f"  Feature extraction: {pipeline_result['feature_extraction_ms']:.3f} ms")
        print(f"  Model inference:    {pipeline_result['model_inference_ms']:.3f} ms")
        print(f"  Rule scoring:       {pipeline_result['rule_scoring_ms']:.3f} ms")
        print(f"  Total pipeline:     {total_ms:.3f} ms")
        print(f"  Model score: {pipeline_result['model_score']:.4f}")
        print(f"  Rule score:  {pipeline_result['rule_score']:.4f}")
    else:
        lines.append("No test traces available for pipeline benchmark.")

    lines.append("")

    # --- Multi-trace pipeline benchmark ---
    print("\n=== Multi-Trace Pipeline Benchmark ===")
    lines.append("=== Multi-Trace Pipeline Benchmark ===")
    lines.append("(Full pipeline over all test traces)")
    lines.append("")

    gc.collect()
    reset_gpu_memory()
    process = psutil.Process()

    # Feature extraction for all
    start = time.perf_counter()
    summaries = summarize_features(extractor, test_traces)
    feat_total = time.perf_counter() - start

    # Model inference for all
    reset_gpu_memory()
    labels, scores, infer_ms = predict_model(model, test_dataset, device)
    infer_total = infer_ms * len(test_traces) / 1000.0  # convert avg ms back to total seconds

    # Rule scoring for all
    from src.scoring.risk import calculate_rule_risks
    start = time.perf_counter()
    rule_scores = calculate_rule_risks(test_traces, summaries)
    rule_total = time.perf_counter() - start

    gpu_mem_total = get_gpu_memory_mb()
    ram_total = process.memory_info().rss / (1024 * 1024)

    total_pipeline = feat_total + infer_total + rule_total
    n = len(test_traces)

    lines.append(f"{'Stage':<25s} {'Total (s)':>10s} {'Per Trace (ms)':>15s} {'% of Total':>12s}")
    lines.append("-" * 65)
    for stage, total_s in [
        ("Feature Extraction", feat_total),
        ("Model Inference", infer_total),
        ("Rule Scoring", rule_total),
    ]:
        per_trace = (total_s / n * 1000) if n > 0 else 0
        pct = (total_s / total_pipeline * 100) if total_pipeline > 0 else 0
        lines.append(f"{stage:<25s} {total_s:>10.4f} {per_trace:>15.3f} {pct:>11.1f}%")
    per_trace_total = (total_pipeline / n * 1000) if n > 0 else 0
    lines.append("-" * 65)
    lines.append(f"{'TOTAL':<25s} {total_pipeline:>10.4f} {per_trace_total:>15.3f} {'100.0%':>12s}")
    lines.append("")
    lines.append(f"Throughput (full pipeline): {n / total_pipeline:.1f} traces/s")
    lines.append(f"GPU memory: {gpu_mem_total:.1f} MB")
    lines.append(f"RAM usage:  {ram_total:.1f} MB")

    print(f"  Feature extraction: {feat_total:.4f}s ({feat_total/n*1000:.3f} ms/trace)")
    print(f"  Model inference:    {infer_total:.4f}s ({infer_ms:.3f} ms/trace)")
    print(f"  Rule scoring:       {rule_total:.4f}s ({rule_total/n*1000:.3f} ms/trace)")
    print(f"  Total pipeline:     {total_pipeline:.4f}s ({per_trace_total:.3f} ms/trace)")
    print(f"  GPU memory: {gpu_mem_total:.1f} MB, RAM: {ram_total:.1f} MB")

    output = "\n".join(lines)
    print("\n" + output)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "enhanced_scalability_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
