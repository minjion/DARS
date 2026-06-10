import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, ndcg_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier, StandardLSTMClassifier
from src.scoring.risk import calculate_rule_risks


DISALLOWED_SOURCES = {"agentdojo_seed_synthetic", "template", "synthetic_template"}


def validate_paper_dataset(traces: list[dict], path: str) -> None:
    bad_sources = sorted(
        {
            str(trace.get("source", "")).lower()
            for trace in traces
            if str(trace.get("source", "")).lower() in DISALLOWED_SOURCES
        }
    )
    if bad_sources:
        raise ValueError(
            f"{path} contains non-paper seed sources {bad_sources}. "
            "Regenerate data from data/real_seed_traces.jsonl collected via AgentDojo/Groq."
        )


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= start) & (scores < end)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += (np.sum(mask) / len(labels)) * abs(accuracy - confidence)
    return ece


def spearman_rank_correlation(labels: np.ndarray, scores: np.ndarray) -> float:
    label_rank = np.argsort(np.argsort(labels))
    score_rank = np.argsort(np.argsort(scores))
    if np.std(label_rank) == 0 or np.std(score_rank) == 0:
        return 0.0
    return float(np.corrcoef(label_rank, score_rank)[0, 1])


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "fdr": fp / max(tp + fp, 1),
        "ece": expected_calibration_error(labels, scores),
        "spearman": spearman_rank_correlation(labels, scores),
        "ndcg@10": ndcg_score([labels], [scores], k=10),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, scores)
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def severity_values(traces: list[dict], labels: np.ndarray) -> np.ndarray:
    values = []
    defaults = {
        "benign": 0.0,
        "benign_suspicious": 0.25,
        "S1_prompt_injection": 0.70,
        "S2_privilege_escalation": 0.85,
        "S3_malicious_api_usage": 0.75,
        "S4_data_exfiltration": 0.90,
        "S5_denial_of_wallet": 0.65,
        "S6_stealth_mimicry": 0.80,
    }
    for trace, label in zip(traces, labels):
        if "severity" in trace:
            values.append(float(trace["severity"]))
        else:
            values.append(defaults.get(trace.get("scenario", "benign"), float(label)))
    return np.array(values, dtype=float)


def ranking_metrics(labels: np.ndarray, severities: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "ece": expected_calibration_error(labels, scores),
        "spearman": spearman_rank_correlation(severities, scores),
        "ndcg@10": ndcg_score([severities], [scores], k=10),
    }


def load_checkpoint(path: str, device: torch.device) -> tuple[DARSClassifier, DARSFeatureExtractor, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint or "extractor_state" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain extractor_state. Regenerate data and run train.py "
            "so evaluation can use the same benign baselines."
        )
    config = checkpoint.get("config", {})
    model = DARSClassifier(
        input_dim=config.get("input_dim", len(FEATURE_NAMES)),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    extractor = DARSFeatureExtractor.from_state(checkpoint["extractor_state"])
    return model, extractor, config


def load_standard_lstm(checkpoint_path: str, device: torch.device) -> tuple[StandardLSTMClassifier | None, torch.Tensor | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "standard_lstm_state" not in checkpoint:
        return None, None
    config = checkpoint.get("config", {})
    model = StandardLSTMClassifier(
        input_dim=config.get("input_dim", len(FEATURE_NAMES)),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
    )
    model.load_state_dict(checkpoint["standard_lstm_state"])
    model.to(device)
    mask_values = config.get("standard_lstm_feature_mask")
    mask = torch.tensor(mask_values, dtype=torch.float32) if mask_values else None
    return model, mask


def load_ablation_models(checkpoint_path: str, device: torch.device) -> dict[str, tuple[DARSClassifier, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    states = checkpoint.get("ablation_states", {})
    models = {}
    for feature_name, payload in states.items():
        model = DARSClassifier(
            input_dim=config.get("input_dim", len(FEATURE_NAMES)),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 4),
        )
        model.load_state_dict(payload["model_state"])
        model.to(device)
        mask = torch.tensor(payload["feature_mask"], dtype=torch.float32)
        models[feature_name] = (model, mask)
    return models


def predict_model(
    model: torch.nn.Module,
    dataset: AgentDojoDataset,
    device: torch.device,
    feature_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_scores = []
    all_labels = []
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            if feature_mask is not None:
                batch_x = batch_x * feature_mask.to(device).view(1, 1, -1)
            scores = model(batch_x).detach().cpu().numpy().flatten()
            labels = batch_y.numpy().flatten()
            all_scores.extend(scores)
            all_labels.extend(labels)
    elapsed = time.perf_counter() - start_time
    avg_ms = (elapsed / max(len(all_scores), 1)) * 1000.0
    return np.array(all_labels, dtype=int), np.array(all_scores, dtype=float), avg_ms


def summarize_features(extractor: DARSFeatureExtractor, traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        rows.append(extractor.summarize_trace(trace))
    return rows


def feature_matrix(feature_summaries: list[dict]) -> np.ndarray:
    return np.array([[summary[name] for name in FEATURE_NAMES] for summary in feature_summaries], dtype=float)


def log_parser_scores(traces: list[dict]) -> np.ndarray:
    risky_terms = {"exploit", "payload", "root_shell", "transfer_funds", "delete_user", "change_permissions"}
    scores = []
    for trace in traces:
        blob = str(trace).lower()
        score = 0.85 if any(term in blob for term in risky_terms) else 0.15
        scores.append(score)
    return np.array(scores, dtype=float)


def sandbox_monitor_scores(traces: list[dict]) -> np.ndarray:
    policy_violations = {"root_shell", "delete_user", "transfer_funds", "change_permissions"}
    scores = []
    for trace in traces:
        tools = {call.get("tool") for call in trace.get("tool_calls", [])}
        scores.append(0.95 if tools & policy_violations else 0.05)
    return np.array(scores, dtype=float)


# ---------------------------------------------------------------------------
# DeepLog baseline: LSTM that predicts the next tool in a sequence.
# Anomaly score = average cross-entropy loss per trace.
# ---------------------------------------------------------------------------


class DeepLogModel(nn.Module):
    """Simple LSTM language model over tool-call sequences (DeepLog-style)."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) of token ids -> (batch, seq_len, vocab_size) logits."""
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)


def build_tool_vocab(traces: list[dict]) -> dict[str, int]:
    """Build tool -> index mapping. Index 0 is reserved for padding."""
    tools: set[str] = set()
    for trace in traces:
        for call in trace.get("tool_calls", []):
            tools.add(call.get("tool", "unknown"))
    vocab = {"<PAD>": 0}
    for idx, tool in enumerate(sorted(tools), start=1):
        vocab[tool] = idx
    return vocab


def traces_to_sequences(traces: list[dict], vocab: dict[str, int], max_len: int = 20) -> list[list[int]]:
    """Convert traces into integer-encoded tool sequences (capped at max_len)."""
    unk_id = max(vocab.values()) + 1  # out-of-vocab fallback
    seqs: list[list[int]] = []
    for trace in traces:
        seq = [vocab.get(call.get("tool", "unknown"), unk_id) for call in trace.get("tool_calls", [])]
        seqs.append(seq[:max_len])
    return seqs


def train_deeplog(
    benign_traces: list[dict],
    device: torch.device,
    max_len: int = 20,
    epochs: int = 12,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 4,
) -> tuple[DeepLogModel, dict[str, int]]:
    """Train a DeepLog model on benign traces and return (model, vocab)."""
    vocab = build_tool_vocab(benign_traces)
    vocab_size = len(vocab) + 1  # +1 for possible unk
    seqs = traces_to_sequences(benign_traces, vocab, max_len)

    # Build input/target pairs: input = seq[:-1], target = seq[1:]
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for seq in seqs:
        if len(seq) < 2:
            continue
        inputs.append(torch.tensor(seq[:-1], dtype=torch.long))
        targets.append(torch.tensor(seq[1:], dtype=torch.long))

    if not inputs:
        # Not enough data – return untrained model
        model = DeepLogModel(vocab_size).to(device)
        return model, vocab

    # Pad to equal length within this batch
    max_seq = max(t.size(0) for t in inputs)
    padded_in = torch.zeros(len(inputs), max_seq, dtype=torch.long)
    padded_tgt = torch.zeros(len(targets), max_seq, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_in[i, : inp.size(0)] = inp
        padded_tgt[i, : tgt.size(0)] = tgt

    dataset = TensorDataset(padded_in, padded_tgt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepLogModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_in, batch_tgt in loader:
            batch_in, batch_tgt = batch_in.to(device), batch_tgt.to(device)
            logits = model(batch_in)  # (B, T, V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, vocab


def deeplog_anomaly_scores(
    model: DeepLogModel,
    traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    max_len: int = 20,
) -> np.ndarray:
    """Score each trace by average next-tool prediction cross-entropy."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    seqs = traces_to_sequences(traces, vocab, max_len)
    raw_scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for seq in seqs:
            if len(seq) < 2:
                raw_scores.append(0.0)
                continue
            inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([seq[1:]], dtype=torch.long, device=device)
            logits = model(inp)  # (1, T, V)
            per_step = loss_fn(logits.squeeze(0), tgt.squeeze(0))  # (T,)
            # Average over non-padding positions
            mask = tgt.squeeze(0) != 0
            if mask.any():
                raw_scores.append(float(per_step[mask].mean().item()))
            else:
                raw_scores.append(0.0)

    raw = np.array(raw_scores, dtype=float)
    # Normalize to [0, 1]
    mn, mx = float(raw.min()), float(raw.max())
    if mx - mn < 1e-9:
        return np.full_like(raw, 0.5)
    return (raw - mn) / (mx - mn)


def isolation_forest_scores(train_features: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    detector = IsolationForest(contamination=0.15, random_state=42)
    detector.fit(train_features)
    raw = -detector.decision_function(test_features)
    min_raw, max_raw = float(raw.min()), float(raw.max())
    if max_raw - min_raw < 1e-9:
        return np.zeros_like(raw)
    return (raw - min_raw) / (max_raw - min_raw)


def timed_scores(function, *args) -> tuple[np.ndarray, float]:
    start_time = time.perf_counter()
    scores = function(*args)
    elapsed = time.perf_counter() - start_time
    return scores, (elapsed / max(len(scores), 1)) * 1000.0


def print_metric_row(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    avg_ms: float | None = None,
    threshold: float = 0.5,
) -> None:
    metrics = classification_metrics(labels, scores, threshold=threshold)
    time_suffix = f" Time={avg_ms:.3f}ms" if avg_ms is not None else ""
    print(
        f"{name:18s} "
        f"Acc={metrics['accuracy']:.3f} "
        f"Prec={metrics['precision']:.3f} "
        f"Rec={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} "
        f"AUC={metrics['roc_auc']:.3f} "
        f"FDR={metrics['fdr']:.3f}"
        f"{time_suffix}"
    )


def print_scenario_f1(traces: list[dict], labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> None:
    grouped = defaultdict(list)
    preds = (scores >= threshold).astype(int)
    benign_indices = [idx for idx, label in enumerate(labels) if int(label) == 0]
    for idx, trace in enumerate(traces):
        scenario = trace.get("scenario", "unknown")
        if int(labels[idx]) == 0:
            continue
        grouped[scenario].append(idx)
    print("\n=== F1 by Scenario ===")
    for scenario in sorted(grouped):
        indices = benign_indices + grouped[scenario]
        scenario_f1 = f1_score(labels[indices], preds[indices], zero_division=0)
        print(
            f"{scenario:28s} F1={scenario_f1:.3f} "
            f"n_pos={len(grouped[scenario])} n_benign={len(benign_indices)}"
        )


def tune_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(labels, (scores >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def tune_blend_and_threshold(
    labels: np.ndarray,
    learned_scores: np.ndarray,
    rule_scores: np.ndarray,
) -> tuple[float, float, float]:
    best_alpha = 1.0
    best_threshold = 0.5
    best_f1 = -1.0
    for alpha in np.linspace(0.0, 1.0, 41):
        scores = alpha * learned_scores + (1.0 - alpha) * rule_scores
        threshold, score = tune_threshold(labels, scores)
        if score > best_f1:
            best_alpha = float(alpha)
            best_threshold = threshold
            best_f1 = score
    return best_alpha, best_threshold, float(best_f1)


def calibrate_scores(val_labels: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_scores, val_labels)
    return np.asarray(calibrator.transform(test_scores), dtype=float)


def _compute_s6_f1(
    traces: list[dict], labels: np.ndarray, scores: np.ndarray, threshold: float,
) -> float:
    """Return F1 evaluated on S6 (stealth mimicry) positives + all benign negatives."""
    preds = (scores >= threshold).astype(int)
    benign_idx = [i for i, lbl in enumerate(labels) if int(lbl) == 0]
    s6_idx = [
        i for i, trace in enumerate(traces)
        if trace.get("scenario", "").startswith("S6") and int(labels[i]) == 1
    ]
    if not s6_idx:
        return float("nan")
    subset = benign_idx + s6_idx
    return float(f1_score(labels[subset], preds[subset], zero_division=0))


def run_ablation(
    ablation_models: dict[str, tuple[DARSClassifier, torch.Tensor]],
    dataset: AgentDojoDataset,
    labels: np.ndarray,
    device: torch.device,
    traces: list[dict] | None = None,
) -> None:
    print("\n=== Feature Ablation ===")
    if not ablation_models:
        print("Ablation models unavailable; rerun train.py without --skip_ablations.")
        return
    for feature_name in FEATURE_NAMES:
        if feature_name not in ablation_models:
            print(f"w/o {feature_name:20s} unavailable")
            continue
        model, mask = ablation_models[feature_name]
        _, scores, _ = predict_model(model, dataset, device, mask)
        threshold, _ = tune_threshold(labels, scores)
        metrics = classification_metrics(labels, scores, threshold=threshold)
        # Extended ablation metrics
        ece = expected_calibration_error(labels, scores)
        spearman = spearman_rank_correlation(labels, scores)
        s6_f1_str = "N/A"
        if traces is not None:
            s6_val = _compute_s6_f1(traces, labels, scores, threshold)
            s6_f1_str = f"{s6_val:.3f}" if not np.isnan(s6_val) else "N/A"
        print(
            f"w/o {feature_name:20s} "
            f"F1={metrics['f1']:.3f} AUC={metrics['roc_auc']:.3f} "
            f"S6_F1={s6_f1_str} ECE={ece:.3f} Spearman={spearman:.3f} "
            f"threshold={threshold:.3f}"
        )
    print("w/o shap                 F1 unchanged; SHAP affects explanation, not classifier output.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DARS and paper-aligned baselines.")
    parser.add_argument("--input", type=str, default="data/test_traces.jsonl")
    parser.add_argument("--train", type=str, default="data/train_traces.jsonl")
    parser.add_argument("--val", type=str, default="data/val_traces.jsonl")
    parser.add_argument("--model", type=str, default="models_saved/dars_model_real.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("========================================")
    print("DARS Framework - Evaluation")
    print("========================================")
    print(f"Device: {device}")

    model, extractor, _ = load_checkpoint(args.model, device)
    test_dataset = AgentDojoDataset(args.input, extractor=extractor)
    val_dataset = AgentDojoDataset(args.val, extractor=extractor)
    validate_paper_dataset(test_dataset.traces_raw, args.input)
    validate_paper_dataset(val_dataset.traces_raw, args.val)
    labels, learned_scores, learned_ms = predict_model(model, test_dataset, device)
    val_labels, val_learned_scores, _ = predict_model(model, val_dataset, device)
    test_traces = test_dataset.traces_raw

    train_traces = load_traces(args.train)
    validate_paper_dataset(train_traces, args.train)
    train_summaries = summarize_features(extractor, train_traces)
    train_features = feature_matrix(train_summaries)
    benign_train_features = train_features[
        np.array([int(trace.get("label", 0)) for trace in train_traces]) == 0
    ]
    test_summaries = summarize_features(extractor, test_traces)
    test_features = feature_matrix(test_summaries)
    val_traces = val_dataset.traces_raw
    val_summaries = summarize_features(extractor, val_traces)

    rule_scores, rule_ms = timed_scores(calculate_rule_risks, test_traces, test_summaries)
    val_rule_scores, _ = timed_scores(calculate_rule_risks, val_traces, val_summaries)
    if_scores, if_ms = timed_scores(isolation_forest_scores, benign_train_features, test_features)
    logp_scores, logp_ms = timed_scores(log_parser_scores, test_traces)
    sandbox_scores, sandbox_ms = timed_scores(sandbox_monitor_scores, test_traces)

    # --- DeepLog baseline ---
    benign_train_traces = [t for t in train_traces if int(t.get("label", 0)) == 0]
    print("Training DeepLog baseline on benign tool sequences...")
    deeplog_model, deeplog_vocab = train_deeplog(benign_train_traces, device)
    deeplog_scores, deeplog_ms = timed_scores(
        deeplog_anomaly_scores, deeplog_model, test_traces, deeplog_vocab, device,
    )
    blend_alpha, dars_threshold, val_blend_f1 = tune_blend_and_threshold(
        val_labels,
        val_learned_scores,
        val_rule_scores,
    )
    raw_dars_scores = blend_alpha * learned_scores + (1.0 - blend_alpha) * rule_scores
    val_raw_dars_scores = blend_alpha * val_learned_scores + (1.0 - blend_alpha) * val_rule_scores
    dars_scores = calibrate_scores(val_labels, val_raw_dars_scores, raw_dars_scores)
    dars_threshold, _ = tune_threshold(val_labels, calibrate_scores(val_labels, val_raw_dars_scores, val_raw_dars_scores))
    dars_ms = learned_ms + rule_ms

    standard_lstm, standard_mask = load_standard_lstm(args.model, device)
    standard_scores = None
    standard_ms = None
    if standard_lstm is not None:
        _, standard_scores, standard_ms = predict_model(standard_lstm, test_dataset, device, standard_mask)
    ablation_models = load_ablation_models(args.model, device)

    print("\n=== Overall Classification Performance ===")
    print_metric_row("Rule-based", labels, rule_scores, rule_ms)
    print_metric_row("Isolation Forest", labels, if_scores, if_ms)
    print_metric_row("Log Parser", labels, logp_scores, logp_ms)
    print_metric_row("Sandbox Monitor", labels, sandbox_scores, sandbox_ms)
    print_metric_row("DeepLog", labels, deeplog_scores, deeplog_ms)
    if standard_scores is not None:
        print_metric_row("Standard LSTM", labels, standard_scores, standard_ms)
    else:
        print("Standard LSTM     unavailable: checkpoint has no standard_lstm_state; rerun train.py.")
    print_metric_row("DARS learned", labels, learned_scores, learned_ms)
    print_metric_row("DARS", labels, dars_scores, dars_ms, threshold=dars_threshold)
    print(
        f"DARS validation tuning: learned_weight={blend_alpha:.2f} "
        f"rule_weight={1.0 - blend_alpha:.2f} threshold={dars_threshold:.3f} "
        f"val_f1={val_blend_f1:.3f}"
    )

    severities = severity_values(test_traces, labels)
    dars_metrics = ranking_metrics(labels, severities, dars_scores)
    print("\n=== Risk Scoring Quality ===")
    print(f"ECE={dars_metrics['ece']:.3f}")
    print(f"Spearman={dars_metrics['spearman']:.3f}")
    print(f"NDCG@10={dars_metrics['ndcg@10']:.3f}")

    print_scenario_f1(test_traces, labels, dars_scores, threshold=dars_threshold)
    run_ablation(ablation_models, test_dataset, labels, device, traces=test_traces)


if __name__ == "__main__":
    main()
