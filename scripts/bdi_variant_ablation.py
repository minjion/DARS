"""
BDI Variant Ablation Study
Tests whether the full BDI (Entropy + SVI + ADS) is better than its individual components.

Variants:
  V1: Entropy only       (alpha=1.0, beta=0.0, gamma=0.0)
  V2: Entropy + SVI      (alpha=0.5, beta=0.5, gamma=0.0)
  V3: Entropy + ADS      (alpha=0.5, beta=0.0, gamma=0.5)
  V4: Full BDI (default) (alpha=0.4, beta=0.4, gamma=0.2)
"""
import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.bdi_calculator import BDICalculator
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier
from train import train_classifier, set_seed
from evaluate import classification_metrics, expected_calibration_error, spearman_rank_correlation, severity_values


# ── Configuration ──────────────────────────────────────────────────────────
VARIANTS = {
    "V1: Entropy only":  {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "V2: Entropy+SVI":   {"alpha": 0.5, "beta": 0.5, "gamma": 0.0},
    "V3: Entropy+ADS":   {"alpha": 0.5, "beta": 0.0, "gamma": 0.5},
    "V4: Full BDI":      {"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
}

TRAIN_PATH = "data/train_traces.jsonl"
VAL_PATH   = "data/val_traces.jsonl"
TEST_PATH  = "data/test_traces.jsonl"

LR        = 1e-3
BATCH_SIZE = 16
EPOCHS     = 50
PATIENCE   = 8
NU         = 0.5
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS  = 4
INPUT_DIM  = 6  # len(FEATURE_NAMES)
SEED       = 42


def predict_all(model, dataset, device):
    """Return (labels, scores) arrays."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            scores = model(batch_x).detach().cpu().numpy().flatten()
            labels = batch_y.numpy().flatten()
            all_scores.extend(scores)
            all_labels.extend(labels)
    return np.array(all_labels, dtype=int), np.array(all_scores, dtype=float)


def run_variant(name, weights, device):
    """Train & evaluate one BDI variant. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {name}  (alpha={weights['alpha']}, beta={weights['beta']}, gamma={weights['gamma']})")
    print(f"{'='*60}")

    set_seed(SEED)

    # 1. Create modified BDI calculator
    bdi_calc = BDICalculator(alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"])

    # 2. Create extractor and monkey-patch its bdi_calc
    extractor = DARSFeatureExtractor(max_seq_len=20)
    extractor.bdi_calc = bdi_calc

    # 3. Fit on training data
    train_traces = load_traces(TRAIN_PATH)
    extractor.fit(train_traces)

    # 4. Create datasets
    train_dataset = AgentDojoDataset(TRAIN_PATH, extractor=extractor, max_seq_len=20)
    val_dataset   = AgentDojoDataset(VAL_PATH,   extractor=extractor, max_seq_len=20)
    test_dataset  = AgentDojoDataset(TEST_PATH,  extractor=extractor, max_seq_len=20)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # 5. Train fresh model
    model = DARSClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    ).to(device)

    model, best_val_loss = train_classifier(
        name, model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR, brier_weight=NU, patience=PATIENCE,
    )
    print(f"  Best val loss: {best_val_loss:.4f}")

    # 6. Evaluate
    labels, scores = predict_all(model, test_dataset, device)
    test_traces = test_dataset.traces_raw

    cls_metrics = classification_metrics(labels, scores, threshold=0.5)
    ece = expected_calibration_error(labels, scores)
    severities = severity_values(test_traces, labels)
    spearman = spearman_rank_correlation(severities, scores)

    return {
        "F1":       cls_metrics["f1"],
        "AUC":      cls_metrics["roc_auc"],
        "ECE":      ece,
        "Spearman": spearman,
        "Val Loss": best_val_loss,
        "Acc":      cls_metrics["accuracy"],
        "Prec":     cls_metrics["precision"],
        "Rec":      cls_metrics["recall"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input dim: {INPUT_DIM}, Hidden: {HIDDEN_DIM}, Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}")
    print(f"LR: {LR}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, Patience: {PATIENCE}, Nu: {NU}")

    results = {}
    for name, weights in VARIANTS.items():
        results[name] = run_variant(name, weights, device)

    # ── Print formatted table ──────────────────────────────────────────
    header_metrics = ["F1", "AUC", "ECE", "Spearman", "Acc", "Prec", "Rec", "Val Loss"]
    col_w = 10

    print("\n" + "=" * 100)
    print("BDI VARIANT ABLATION RESULTS")
    print("=" * 100)

    header = f"{'Variant':<22s}" + "".join(f"{m:>{col_w}s}" for m in header_metrics)
    print(header)
    print("-" * len(header))

    for name, metrics in results.items():
        row = f"{name:<22s}"
        for m in header_metrics:
            row += f"{metrics[m]:>{col_w}.4f}"
        print(row)

    print("-" * len(header))

    # Identify best variant for each metric (higher is better, except ECE and Val Loss)
    higher_better = {"F1", "AUC", "Spearman", "Acc", "Prec", "Rec"}
    lower_better  = {"ECE", "Val Loss"}
    print("\nBest per metric:")
    for m in header_metrics:
        vals = {name: r[m] for name, r in results.items()}
        if m in higher_better:
            best = max(vals, key=vals.get)
        else:
            best = min(vals, key=vals.get)
        print(f"  {m:>10s}: {best} ({vals[best]:.4f})")

    # ── Save results to file ───────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "bdi_variant_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BDI VARIANT ABLATION RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Config: input_dim={INPUT_DIM}, hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}, "
                f"num_heads={NUM_HEADS}, lr={LR}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
                f"patience={PATIENCE}, nu={NU}, seed={SEED}\n\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name, metrics in results.items():
            row = f"{name:<22s}"
            for m in header_metrics:
                row += f"{metrics[m]:>{col_w}.4f}"
            f.write(row + "\n")
        f.write("-" * len(header) + "\n")

        f.write("\nBest per metric:\n")
        for m in header_metrics:
            vals = {name: r[m] for name, r in results.items()}
            if m in higher_better:
                best = max(vals, key=vals.get)
            else:
                best = min(vals, key=vals.get)
            f.write(f"  {m:>10s}: {best} ({vals[best]:.4f})\n")

        f.write("\nVariant Details:\n")
        for name, weights in VARIANTS.items():
            f.write(f"  {name}: alpha={weights['alpha']}, beta={weights['beta']}, gamma={weights['gamma']}\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
