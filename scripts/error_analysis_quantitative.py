"""
Quantitative Error Analysis on DARS Test Set.

Loads the trained DARS model, runs inference on test traces,
classifies each prediction as TP/FP/TN/FN at threshold=0.5,
and produces per-scenario error breakdowns.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
import torch

from src.data.dataset import AgentDojoDataset
from evaluate import load_checkpoint, predict_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and extractor
    model_path = "models_saved/dars_model_real.pt"
    model, extractor, config = load_checkpoint(model_path, device)
    print(f"Model loaded from {model_path}")

    # Load test dataset
    test_path = "data/test_traces.jsonl"
    test_dataset = AgentDojoDataset(test_path, extractor=extractor)
    test_traces = test_dataset.traces_raw
    print(f"Test traces loaded: {len(test_traces)}")

    # Run inference
    labels, scores, avg_ms = predict_model(model, test_dataset, device)
    print(f"Inference complete. Avg latency: {avg_ms:.3f} ms/trace")

    # Threshold
    threshold = 0.5
    preds = (scores >= threshold).astype(int)

    # Classify each trace
    tp_mask = (preds == 1) & (labels == 1)
    fp_mask = (preds == 1) & (labels == 0)
    tn_mask = (preds == 0) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    # Overall counts
    total_tp = int(tp_mask.sum())
    total_fp = int(fp_mask.sum())
    total_tn = int(tn_mask.sum())
    total_fn = int(fn_mask.sum())
    total = len(labels)

    # Group by scenario
    scenario_data = defaultdict(lambda: {"total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for i, trace in enumerate(test_traces):
        scenario = trace.get("scenario", "unknown")
        scenario_data[scenario]["total"] += 1
        if tp_mask[i]:
            scenario_data[scenario]["tp"] += 1
        elif fp_mask[i]:
            scenario_data[scenario]["fp"] += 1
        elif tn_mask[i]:
            scenario_data[scenario]["tn"] += 1
        elif fn_mask[i]:
            scenario_data[scenario]["fn"] += 1

    # Build output
    lines = []
    lines.append("=" * 100)
    lines.append("DARS Quantitative Error Analysis — Test Set")
    lines.append("=" * 100)
    lines.append(f"Model: {model_path}")
    lines.append(f"Test data: {test_path}")
    lines.append(f"Total traces: {total}")
    lines.append(f"Threshold: {threshold}")
    lines.append(f"Device: {device}")
    lines.append("")

    # Per-scenario table
    header = f"{'Scenario':<35s} {'Total':>6s} {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s} {'FP%':>7s} {'FN%':>7s}"
    sep = "-" * len(header)
    lines.append("=== Per-Scenario Error Breakdown ===")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    # Sort by scenario name
    for scenario in sorted(scenario_data.keys()):
        d = scenario_data[scenario]
        t = d["total"]
        tp_c = d["tp"]
        fp_c = d["fp"]
        tn_c = d["tn"]
        fn_c = d["fn"]
        fp_pct = (fp_c / t * 100) if t > 0 else 0.0
        fn_pct = (fn_c / t * 100) if t > 0 else 0.0
        lines.append(f"{scenario:<35s} {t:>6d} {tp_c:>5d} {fp_c:>5d} {tn_c:>5d} {fn_c:>5d} {fp_pct:>6.1f}% {fn_pct:>6.1f}%")

    lines.append(sep)
    # Overall row
    fp_pct_overall = (total_fp / total * 100) if total > 0 else 0.0
    fn_pct_overall = (total_fn / total * 100) if total > 0 else 0.0
    lines.append(f"{'OVERALL':<35s} {total:>6d} {total_tp:>5d} {total_fp:>5d} {total_tn:>5d} {total_fn:>5d} {fp_pct_overall:>6.1f}% {fn_pct_overall:>6.1f}%")
    lines.append("")

    # Summary statistics
    lines.append("=== Overall Summary ===")
    lines.append(f"Total TP: {total_tp}")
    lines.append(f"Total FP: {total_fp} ({fp_pct_overall:.1f}%)")
    lines.append(f"Total TN: {total_tn}")
    lines.append(f"Total FN: {total_fn} ({fn_pct_overall:.1f}%)")
    lines.append(f"Total Errors (FP+FN): {total_fp + total_fn}")
    lines.append(f"Error Rate: {(total_fp + total_fn) / total * 100:.1f}%")
    accuracy = (total_tp + total_tn) / total if total > 0 else 0
    lines.append(f"Accuracy: {accuracy:.4f}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall: {recall:.4f}")
    lines.append(f"F1: {f1:.4f}")
    lines.append("")

    # Most errors by scenario
    lines.append("=== Scenarios with Most Errors (FP+FN, descending) ===")
    error_by_scenario = []
    for scenario, d in scenario_data.items():
        total_errors = d["fp"] + d["fn"]
        error_by_scenario.append((scenario, total_errors, d["fp"], d["fn"], d["total"]))
    error_by_scenario.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"{'Scenario':<35s} {'Errors':>7s} {'FP':>5s} {'FN':>5s} {'Total':>6s}")
    lines.append("-" * 60)
    for scenario, err, fp_c, fn_c, t in error_by_scenario:
        lines.append(f"{scenario:<35s} {err:>7d} {fp_c:>5d} {fn_c:>5d} {t:>6d}")
    lines.append("")

    # Scenarios with highest FP rate
    lines.append("=== Scenarios with Highest FP Rate ===")
    fp_rate_list = []
    for scenario, d in scenario_data.items():
        negatives = d["tn"] + d["fp"]
        if negatives > 0:
            fp_rate = d["fp"] / negatives
            fp_rate_list.append((scenario, fp_rate, d["fp"], negatives))
    fp_rate_list.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"{'Scenario':<35s} {'FP Rate':>8s} {'FP':>5s} {'Neg':>5s}")
    lines.append("-" * 55)
    for scenario, rate, fp_c, neg in fp_rate_list:
        lines.append(f"{scenario:<35s} {rate:>7.1%} {fp_c:>5d} {neg:>5d}")
    lines.append("")

    # Scenarios with highest FN rate
    lines.append("=== Scenarios with Highest FN Rate ===")
    fn_rate_list = []
    for scenario, d in scenario_data.items():
        positives = d["tp"] + d["fn"]
        if positives > 0:
            fn_rate = d["fn"] / positives
            fn_rate_list.append((scenario, fn_rate, d["fn"], positives))
    fn_rate_list.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"{'Scenario':<35s} {'FN Rate':>8s} {'FN':>5s} {'Pos':>5s}")
    lines.append("-" * 55)
    for scenario, rate, fn_c, pos in fn_rate_list:
        lines.append(f"{scenario:<35s} {rate:>7.1%} {fn_c:>5d} {pos:>5d}")
    lines.append("")

    # Score statistics for error types
    lines.append("=== Score Distribution by Classification ===")
    for name, mask in [("TP", tp_mask), ("FP", fp_mask), ("TN", tn_mask), ("FN", fn_mask)]:
        s = scores[mask]
        if len(s) > 0:
            lines.append(f"{name}: count={len(s)}, mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")
        else:
            lines.append(f"{name}: count=0")
    lines.append("")

    output = "\n".join(lines)
    print(output)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "error_analysis_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
