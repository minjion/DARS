"""W3: XGBoost baseline on the same 6 DARS features"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, "d:/DARS-Github-Repo-V2")

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from evaluate import (
    load_checkpoint, classification_metrics, severity_values,
    spearman_rank_correlation, expected_calibration_error
)

# Try import xgboost
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Installing xgboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBClassifier

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load extractor from checkpoint
model, extractor, config = load_checkpoint("d:/DARS-Github-Repo-V2/models_saved/dars_model_real.pt", device)

# Load data
train_traces = load_traces("d:/DARS-Github-Repo-V2/data/train_traces.jsonl")
val_traces = load_traces("d:/DARS-Github-Repo-V2/data/val_traces.jsonl")
test_traces = load_traces("d:/DARS-Github-Repo-V2/data/test_traces.jsonl")

def extract_features(traces, ext):
    """Extract per-trace summary features (6-dim vector)"""
    X = []
    y = []
    for t in traces:
        summary = ext.summarize_trace(t)
        features = [summary[name] for name in FEATURE_NAMES]
        X.append(features)
        y.append(int(t.get("label", 0)))
    return np.array(X), np.array(y)

print("Extracting features...")
X_train, y_train = extract_features(train_traces, extractor)
X_val, y_val = extract_features(val_traces, extractor)
X_test, y_test = extract_features(test_traces, extractor)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Combine train+val for XGBoost (it has its own regularization)
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

# Train XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_trainval, y_trainval)

# Predict
start_time = time.perf_counter()
scores_xgb = xgb.predict_proba(X_test)[:, 1]
elapsed = time.perf_counter() - start_time
avg_ms = (elapsed / len(X_test)) * 1000.0

preds_xgb = (scores_xgb >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, preds_xgb)
prec = precision_score(y_test, preds_xgb, zero_division=0)
rec = recall_score(y_test, preds_xgb, zero_division=0)
f1 = f1_score(y_test, preds_xgb, zero_division=0)
auc = roc_auc_score(y_test, scores_xgb)
tp = int(((preds_xgb == 1) & (y_test == 1)).sum())
fp = int(((preds_xgb == 1) & (y_test == 0)).sum())
fdr = fp / max(tp + fp, 1)
ece = expected_calibration_error(y_test, scores_xgb)

sevs = severity_values(test_traces, y_test)
spearman = spearman_rank_correlation(sevs, scores_xgb)

print("\n" + "=" * 60)
print("W3: XGBoost Baseline Results")
print("=" * 60)
print(f"Accuracy:   {acc:.3f}")
print(f"Precision:  {prec:.3f}")
print(f"Recall:     {rec:.3f}")
print(f"F1:         {f1:.3f}")
print(f"AUC:        {auc:.3f}")
print(f"FDR:        {fdr:.3f}")
print(f"ECE:        {ece:.3f}")
print(f"Spearman:   {spearman:.3f}")
print(f"Latency:    {avg_ms:.2f} ms/trace")

# Feature importance
print("\nFeature Importance:")
for name, imp in sorted(zip(FEATURE_NAMES, xgb.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:25s}: {imp:.4f}")

# Save
with open("d:/DARS-Github-Repo-V2/scripts/w3_xgboost_results.txt", "w") as f:
    f.write("XGBoost Baseline on DARS 6 Features\n")
    f.write(f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}\n")
    f.write(f"AUC={auc:.3f} FDR={fdr:.3f} ECE={ece:.3f} Spearman={spearman:.3f}\n")
    f.write(f"Latency={avg_ms:.2f} ms/trace\n")
    f.write(f"\nFor Table 4 row:\n")
    f.write(f"XGBoost | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {auc:.3f} | {fdr:.3f} | {avg_ms:.2f}\n")
print("\nSaved to scripts/w3_xgboost_results.txt")
