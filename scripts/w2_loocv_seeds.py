"""W2: Leave-One-Out Cross-Validation on 48 seed traces"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "d:/DARS-Github-Repo-V2")

from src.data.dataset import load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier
from evaluate import load_checkpoint
from train import combined_bce_brier_loss, set_seed
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load extractor from trained checkpoint (fitted on training benign data)
_, extractor, _ = load_checkpoint("d:/DARS-Github-Repo-V2/models_saved/dars_model_real.pt", device)

# Load seed traces
seeds = load_traces("d:/DARS-Github-Repo-V2/data/real_seed_traces.jsonl")
print(f"Loaded {len(seeds)} seed traces")

# Extract features for all seeds using the pre-fitted extractor
all_features = []
all_labels_raw = []
for t in seeds:
    feat = extractor.extract_trace_features(t)  # returns tensor (seq_len, 6)
    all_features.append(feat)
    all_labels_raw.append(float(t.get("label", 0)))

all_preds = []
all_labels = []
all_scores = []

for i in range(len(seeds)):
    set_seed(42)
    
    # Split
    train_feats = all_features[:i] + all_features[i+1:]
    train_labels = all_labels_raw[:i] + all_labels_raw[i+1:]
    test_feat = all_features[i]
    test_label = all_labels_raw[i]
    
    # Create tensors
    X_train = torch.stack(train_feats)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    
    train_ds = TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    # Train a small model
    model = DARSClassifier(
        input_dim=len(FEATURE_NAMES), hidden_dim=64, num_layers=1, num_heads=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = combined_bce_brier_loss(pred, by, 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        tx = test_feat.unsqueeze(0).to(device)
        score = model(tx).item()
    
    label = int(test_label)
    pred_label = 1 if score >= 0.5 else 0
    all_scores.append(score)
    all_labels.append(label)
    all_preds.append(pred_label)
    
    status = "OK" if pred_label == label else "MISS"
    scenario = seeds[i].get('scenario', '?')
    print(f"  [{i+1:2d}/48] {status:4s} label={label} score={score:.3f} pred={pred_label} scenario={scenario}")

all_labels = np.array(all_labels)
all_scores = np.array(all_scores)
all_preds = np.array(all_preds)

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, zero_division=0)
try:
    auc = roc_auc_score(all_labels, all_scores)
except:
    auc = 0.0

print("\n" + "=" * 60)
print("W2: LOOCV on 48 Seed Traces")
print("=" * 60)
print(f"Accuracy:  {acc:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"AUC:       {auc:.3f}")
print(f"Correct:   {int(np.sum(all_preds == all_labels))}/48")
print(f"Errors:    {int(np.sum(all_preds != all_labels))}/48")

errors = []
for i in range(len(seeds)):
    if all_preds[i] != all_labels[i]:
        errors.append(f"  Trace {i}: label={all_labels[i]}, pred={all_preds[i]}, score={all_scores[i]:.3f}, scenario={seeds[i].get('scenario','?')}")

if errors:
    print("\nMisclassified traces:")
    for e in errors:
        print(e)

with open("d:/DARS-Github-Repo-V2/scripts/w2_loocv_results.txt", "w") as f:
    f.write(f"LOOCV on 48 Seed Traces\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"F1: {f1:.3f}\n")
    f.write(f"AUC: {auc:.3f}\n")
    f.write(f"Correct: {int(np.sum(all_preds == all_labels))}/48\n")
    if errors:
        f.write("\nMisclassified:\n")
        for e in errors:
            f.write(e + "\n")
print("\nSaved to scripts/w2_loocv_results.txt")
