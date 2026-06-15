"""
Dataset Validation: Synthetic Data vs Seed Distribution
Validates that synthetic data preserves the seed distribution.

Computes:
  - Jensen-Shannon Divergence per feature
  - Wasserstein Distance per feature
  - t-SNE visualization (seed vs synthetic)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE

from src.data.dataset import load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES


SEED_PATH  = "data/real_seed_traces.jsonl"
TRAIN_PATH = "data/train_traces.jsonl"
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def compute_histogram(values, bins=50, range_min=0.0, range_max=1.0):
    """Compute a normalized histogram (probability distribution)."""
    counts, _ = np.histogram(values, bins=bins, range=(range_min, range_max))
    total = counts.sum()
    if total == 0:
        return np.ones(bins) / bins
    return counts / total


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("DATASET VALIDATION: Seed vs Synthetic Distribution")
    print("=" * 60)

    # 1. Load traces
    print("\nLoading seed traces...")
    seed_traces = load_traces(SEED_PATH)
    print(f"  Seed traces: {len(seed_traces)}")

    print("Loading training traces...")
    train_traces = load_traces(TRAIN_PATH)
    print(f"  Training traces: {len(train_traces)}")

    # 2. Fit extractor on training benign traces
    print("\nFitting feature extractor on training benign traces...")
    extractor = DARSFeatureExtractor(max_seq_len=20)
    extractor.fit(train_traces)

    # 3. Summarize traces
    print("Extracting per-trace feature summaries...")
    seed_summaries = [extractor.summarize_trace(t) for t in seed_traces]
    train_summaries = [extractor.summarize_trace(t) for t in train_traces]

    # Build feature matrices
    seed_matrix = np.array(
        [[s[name] for name in FEATURE_NAMES] for s in seed_summaries], dtype=float
    )
    train_matrix = np.array(
        [[s[name] for name in FEATURE_NAMES] for s in train_summaries], dtype=float
    )

    print(f"  Seed feature matrix shape:  {seed_matrix.shape}")
    print(f"  Train feature matrix shape: {train_matrix.shape}")

    # 4. Compute per-feature divergence metrics
    print("\n" + "-" * 60)
    print("PER-FEATURE DISTRIBUTION COMPARISON")
    print("-" * 60)

    n_bins = 50
    header = f"{'Feature':<25s} {'JSD':>10s} {'Wasserstein':>12s} {'Seed Mean':>10s} {'Train Mean':>10s} {'Seed Std':>10s} {'Train Std':>10s}"
    print(header)
    print("-" * len(header))

    feature_results = []
    for i, name in enumerate(FEATURE_NAMES):
        seed_vals = seed_matrix[:, i]
        train_vals = train_matrix[:, i]

        # Determine range for histograms
        all_vals = np.concatenate([seed_vals, train_vals])
        range_min = float(all_vals.min())
        range_max = float(all_vals.max()) + 1e-9

        # Histograms for JSD
        p = compute_histogram(seed_vals, bins=n_bins, range_min=range_min, range_max=range_max)
        q = compute_histogram(train_vals, bins=n_bins, range_min=range_min, range_max=range_max)

        jsd = float(jensenshannon(p, q) ** 2)  # JSD (squared Jensen-Shannon distance = divergence)
        wd = float(wasserstein_distance(seed_vals, train_vals))

        seed_mean = float(np.mean(seed_vals))
        train_mean = float(np.mean(train_vals))
        seed_std = float(np.std(seed_vals))
        train_std = float(np.std(train_vals))

        print(f"{name:<25s} {jsd:>10.6f} {wd:>12.6f} {seed_mean:>10.4f} {train_mean:>10.4f} {seed_std:>10.4f} {train_std:>10.4f}")

        feature_results.append({
            "feature": name,
            "jsd": jsd,
            "wasserstein": wd,
            "seed_mean": seed_mean,
            "train_mean": train_mean,
            "seed_std": seed_std,
            "train_std": train_std,
        })

    # Summary statistics
    jsds = [r["jsd"] for r in feature_results]
    wds = [r["wasserstein"] for r in feature_results]
    print(f"\n{'Average JSD:':<25s} {np.mean(jsds):.6f}")
    print(f"{'Max JSD:':<25s} {np.max(jsds):.6f} ({feature_results[np.argmax(jsds)]['feature']})")
    print(f"{'Average Wasserstein:':<25s} {np.mean(wds):.6f}")
    print(f"{'Max Wasserstein:':<25s} {np.max(wds):.6f} ({feature_results[np.argmax(wds)]['feature']})")

    # 5. t-SNE visualization
    print("\nComputing t-SNE embedding...")
    combined_matrix = np.vstack([seed_matrix, train_matrix])
    labels = np.array(["Seed"] * len(seed_matrix) + ["Synthetic"] * len(train_matrix))

    # Handle potential issues: add small noise if features are constant
    combined_matrix = combined_matrix + np.random.normal(0, 1e-8, combined_matrix.shape)

    perplexity = min(30, len(combined_matrix) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embedding = tsne.fit_transform(combined_matrix)

    seed_mask = labels == "Seed"
    synth_mask = labels == "Synthetic"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(
        embedding[synth_mask, 0], embedding[synth_mask, 1],
        c="steelblue", alpha=0.4, s=20, label=f"Synthetic (n={synth_mask.sum()})", zorder=1,
    )
    ax.scatter(
        embedding[seed_mask, 0], embedding[seed_mask, 1],
        c="crimson", alpha=0.8, s=50, label=f"Seed (n={seed_mask.sum()})", edgecolors="black",
        linewidths=0.5, zorder=2,
    )
    ax.set_title("t-SNE: Seed vs Synthetic Traces (Feature Space)", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "tsne_seed_vs_synthetic.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  t-SNE figure saved to {fig_path}")

    # 6. Save results
    output_path = os.path.join(os.path.dirname(__file__), "dataset_validation_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("DATASET VALIDATION RESULTS: Seed vs Synthetic Distribution\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Seed traces:     {len(seed_traces)}\n")
        f.write(f"Training traces: {len(train_traces)}\n\n")

        f.write("PER-FEATURE DISTRIBUTION COMPARISON\n")
        f.write("-" * 90 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in feature_results:
            f.write(f"{r['feature']:<25s} {r['jsd']:>10.6f} {r['wasserstein']:>12.6f} "
                    f"{r['seed_mean']:>10.4f} {r['train_mean']:>10.4f} "
                    f"{r['seed_std']:>10.4f} {r['train_std']:>10.4f}\n")
        f.write("-" * len(header) + "\n\n")

        f.write(f"Average JSD:          {np.mean(jsds):.6f}\n")
        f.write(f"Max JSD:              {np.max(jsds):.6f} ({feature_results[np.argmax(jsds)]['feature']})\n")
        f.write(f"Average Wasserstein:  {np.mean(wds):.6f}\n")
        f.write(f"Max Wasserstein:      {np.max(wds):.6f} ({feature_results[np.argmax(wds)]['feature']})\n\n")

        f.write("INTERPRETATION GUIDE\n")
        f.write("-" * 50 + "\n")
        f.write("JSD range: [0, ln(2)] ≈ [0, 0.693]. Lower = more similar.\n")
        f.write("  JSD < 0.05:  Excellent preservation\n")
        f.write("  JSD < 0.10:  Good preservation\n")
        f.write("  JSD > 0.20:  Significant divergence\n\n")
        f.write("Wasserstein distance: Lower = more similar distributions.\n")
        f.write("  Values depend on feature scale (all features are [0, 1]).\n\n")

        f.write(f"t-SNE figure saved to: {fig_path}\n")

    print(f"\nResults saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
