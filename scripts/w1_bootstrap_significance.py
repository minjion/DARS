"""W1: Bootstrap significance test for Spearman (Full DARS vs w/o BDI)"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "d:/DARS-Github-Repo-V2")

from src.data.dataset import AgentDojoDataset
from src.feature_extraction.extractor import FEATURE_NAMES
from evaluate import (
    load_checkpoint, load_ablation_models, predict_model,
    severity_values, spearman_rank_correlation, classification_metrics,
    load_traces
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and data
model, extractor, config = load_checkpoint("d:/DARS-Github-Repo-V2/models_saved/dars_model_real.pt", device)
test_ds = AgentDojoDataset("d:/DARS-Github-Repo-V2/data/test_traces.jsonl", extractor, max_seq_len=20)
test_traces = test_ds.traces_raw

# Full DARS scores
labels_full, scores_full, _ = predict_model(model, test_ds, device)
sevs = severity_values(test_traces, labels_full)

# w/o BDI scores
ablation_models = load_ablation_models("d:/DARS-Github-Repo-V2/models_saved/dars_model_real.pt", device)
if "bdi_deviation" in ablation_models:
    abl_model, abl_mask = ablation_models["bdi_deviation"]
    labels_abl, scores_abl, _ = predict_model(abl_model, test_ds, device, abl_mask)
else:
    print("ERROR: bdi_deviation ablation model not found")
    sys.exit(1)

# Bootstrap significance test
N_BOOTSTRAP = 1000
np.random.seed(42)
n = len(sevs)

spearman_full = spearman_rank_correlation(sevs, scores_full)
spearman_abl = spearman_rank_correlation(sevs, scores_abl)
observed_diff = spearman_full - spearman_abl

boot_diffs = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n, size=n, replace=True)
    s_full = spearman_rank_correlation(sevs[idx], scores_full[idx])
    s_abl = spearman_rank_correlation(sevs[idx], scores_abl[idx])
    boot_diffs.append(s_full - s_abl)

boot_diffs = np.array(boot_diffs)
ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)
p_value = np.mean(boot_diffs <= 0)

print("=" * 60)
print("W1: Bootstrap Significance Test (N=1000)")
print("=" * 60)
print(f"Full DARS Spearman:   {spearman_full:.4f}")
print(f"w/o BDI Spearman:     {spearman_abl:.4f}")
print(f"Observed Delta:       {observed_diff:.4f}")
print(f"95% CI for Delta:     [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"p-value (one-sided):  {p_value:.4f}")
print(f"Significant (p<0.05)? {'YES' if p_value < 0.05 else 'NO'}")

m_full = classification_metrics(labels_full, scores_full)
m_abl = classification_metrics(labels_abl, scores_abl)
print(f"\nFull DARS: F1={m_full['f1']:.3f}, AUC={m_full['roc_auc']:.3f}, ECE={m_full['ece']:.3f}, Spearman={m_full['spearman']:.3f}")
print(f"w/o BDI:  F1={m_abl['f1']:.3f}, AUC={m_abl['roc_auc']:.3f}, ECE={m_abl['ece']:.3f}, Spearman={m_abl['spearman']:.3f}")

with open("d:/DARS-Github-Repo-V2/scripts/w1_bootstrap_results.txt", "w") as f:
    f.write(f"Full DARS Spearman: {spearman_full:.4f}\n")
    f.write(f"w/o BDI Spearman: {spearman_abl:.4f}\n")
    f.write(f"Delta: {observed_diff:.4f}\n")
    f.write(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    f.write(f"p-value: {p_value:.4f}\n")
print("\nSaved to scripts/w1_bootstrap_results.txt")
