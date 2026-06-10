"""Analyse statistical similarity between seed traces and synthetic traces.

For each of the 6 DARS features, computes:
- Mean and standard deviation for seed vs synthetic distributions
- Two-sample Kolmogorov–Smirnov test statistic and p-value

Results are printed and saved to ``scripts/distribution_results.txt``.

Run from the repository root:
    python scripts/distribution_analysis.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED_PATHS = [
    ROOT / "data" / "real_seed_traces.jsonl",
    ROOT / "data" / "seed_traces.jsonl",
]
SYNTHETIC_PATH = ROOT / "data" / "train_traces.jsonl"
MODEL_PATH = ROOT / "models_saved" / "dars_model_real.pt"
RESULTS_PATH = ROOT / "scripts" / "distribution_results.txt"

PRETTY_NAMES = {
    "bdi_deviation": "BDI Deviation",
    "privilege_level": "Privilege Level",
    "privilege_escalation": "Privilege Escalation",
    "transition_anomaly": "Transition Anomaly",
    "sequence_anomaly": "Sequence Anomaly",
    "token_burst": "Token Burst",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_extractor_from_checkpoint(path: Path):
    """Load only the fitted extractor from a saved checkpoint."""
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if "extractor_state" in checkpoint:
        return DARSFeatureExtractor.from_state(checkpoint["extractor_state"])
    return None


def extract_feature_matrix(
    extractor: DARSFeatureExtractor, traces: list[dict]
) -> np.ndarray:
    """Return (N, 6) matrix of summarised per-trace feature values."""
    rows = []
    for trace in traces:
        summary = extractor.summarize_trace(trace)
        rows.append([summary[name] for name in FEATURE_NAMES])
    return np.array(rows, dtype=np.float64)


def find_seed_file() -> Path:
    """Return the first existing seed file from the candidate list."""
    for path in SEED_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No seed file found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in SEED_PATHS)
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse_distributions(
    seed_features: np.ndarray,
    synth_features: np.ndarray,
) -> list[dict]:
    """Per-feature comparison with KS test."""
    results = []
    for idx, name in enumerate(FEATURE_NAMES):
        seed_col = seed_features[:, idx]
        synth_col = synth_features[:, idx]
        ks_stat, ks_p = ks_2samp(seed_col, synth_col)
        results.append({
            "feature": name,
            "seed_mean": float(np.mean(seed_col)),
            "seed_std": float(np.std(seed_col)),
            "synth_mean": float(np.mean(synth_col)),
            "synth_std": float(np.std(synth_col)),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        })
    return results


def overall_similarity(per_feature: list[dict]) -> dict:
    """Aggregate distributional similarity across all features."""
    ks_stats = [r["ks_stat"] for r in per_feature]
    ks_ps = [r["ks_p"] for r in per_feature]
    n_pass = sum(1 for p in ks_ps if p > 0.05)
    return {
        "mean_ks_stat": float(np.mean(ks_stats)),
        "mean_ks_p": float(np.mean(ks_ps)),
        "median_ks_p": float(np.median(ks_ps)),
        "features_passing_05": n_pass,
        "total_features": len(per_feature),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

COL_W = {
    "feature": 22,
    "seed_mean": 11,
    "seed_std": 11,
    "synth_mean": 11,
    "synth_std": 11,
    "ks_stat": 10,
    "ks_p": 12,
    "sig": 5,
}

HEADER = (
    f"{'Feature':<{COL_W['feature']}s}"
    f"{'Seed Mean':>{COL_W['seed_mean']}s}"
    f"{'Seed Std':>{COL_W['seed_std']}s}"
    f"{'Synth Mean':>{COL_W['synth_mean']}s}"
    f"{'Synth Std':>{COL_W['synth_std']}s}"
    f"{'KS Stat':>{COL_W['ks_stat']}s}"
    f"{'KS p-value':>{COL_W['ks_p']}s}"
    f"{'Sig?':>{COL_W['sig']}s}"
)
SEP = "-" * len(HEADER)


def format_row(r: dict) -> str:
    pretty = PRETTY_NAMES.get(r["feature"], r["feature"])
    sig = "***" if r["ks_p"] < 0.001 else ("**" if r["ks_p"] < 0.01 else ("*" if r["ks_p"] < 0.05 else ""))
    return (
        f"{pretty:<{COL_W['feature']}s}"
        f"{r['seed_mean']:>{COL_W['seed_mean']}.4f}"
        f"{r['seed_std']:>{COL_W['seed_std']}.4f}"
        f"{r['synth_mean']:>{COL_W['synth_mean']}.4f}"
        f"{r['synth_std']:>{COL_W['synth_std']}.4f}"
        f"{r['ks_stat']:>{COL_W['ks_stat']}.4f}"
        f"{r['ks_p']:>{COL_W['ks_p']}.6f}"
        f"{sig:>{COL_W['sig']}s}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("========================================")
    print("DARS - Seed vs Synthetic Distribution Analysis")
    print("========================================")

    # Load extractor (prefer checkpoint state; fallback to fitting on synthetic)
    extractor = None
    if MODEL_PATH.exists():
        extractor = load_extractor_from_checkpoint(MODEL_PATH)
        if extractor is not None:
            print(f"Loaded fitted extractor from {MODEL_PATH.name}")
    if extractor is None:
        print("No checkpoint extractor; fitting on synthetic training data.")
        synth_traces = load_traces(str(SYNTHETIC_PATH))
        extractor = DARSFeatureExtractor()
        extractor.fit(synth_traces)

    # Load seed traces
    seed_path = find_seed_file()
    seed_traces = load_traces(str(seed_path))
    print(f"Seed traces : {len(seed_traces)} from {seed_path.name}")

    # Load synthetic traces
    synth_traces = load_traces(str(SYNTHETIC_PATH))
    print(f"Synth traces: {len(synth_traces)} from {SYNTHETIC_PATH.name}")

    # Extract features
    print("\nExtracting features ...")
    seed_features = extract_feature_matrix(extractor, seed_traces)
    synth_features = extract_feature_matrix(extractor, synth_traces)

    # Analyse
    per_feature = analyse_distributions(seed_features, synth_features)
    overall = overall_similarity(per_feature)

    # Print results
    print(f"\n{'=== Per-Feature Distribution Comparison ==='}")
    print(HEADER)
    print(SEP)
    for r in per_feature:
        print(format_row(r))
    print(SEP)

    print("\n=== Overall Similarity ===")
    summary_lines = [
        f"  Mean KS statistic         : {overall['mean_ks_stat']:.4f}",
        f"  Mean KS p-value           : {overall['mean_ks_p']:.6f}",
        f"  Median KS p-value         : {overall['median_ks_p']:.6f}",
        f"  Features passing a=0.05   : {overall['features_passing_05']}/{overall['total_features']}",
    ]
    for line in summary_lines:
        print(line)

    interpretation = (
        "The synthetic data closely mirrors the seed distribution."
        if overall["features_passing_05"] >= overall["total_features"] - 1
        else "Some features diverge significantly; review data generation parameters."
    )
    print(f"\n  Interpretation: {interpretation}")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("DARS - Seed vs Synthetic Distribution Analysis\n")
        f.write(f"Seed file  : {seed_path.name} ({len(seed_traces)} traces)\n")
        f.write(f"Synth file : {SYNTHETIC_PATH.name} ({len(synth_traces)} traces)\n\n")
        f.write("=== Per-Feature Distribution Comparison ===\n")
        f.write(HEADER + "\n")
        f.write(SEP + "\n")
        for r in per_feature:
            f.write(format_row(r) + "\n")
        f.write(SEP + "\n\n")
        f.write("=== Overall Similarity ===\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nInterpretation: {interpretation}\n")
        f.write("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001\n")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
