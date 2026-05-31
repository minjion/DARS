import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier, StandardLSTMClassifier


DISALLOWED_SOURCES = {"agentdojo_seed_synthetic", "template", "synthetic_template"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def combined_bce_brier_loss(predictions: torch.Tensor, labels: torch.Tensor, nu: float) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(predictions, labels)
    brier = torch.mean((predictions - labels) ** 2)
    return bce + nu * brier


def apply_feature_mask(batch_x: torch.Tensor, feature_mask: torch.Tensor | None) -> torch.Tensor:
    if feature_mask is None:
        return batch_x
    return batch_x * feature_mask.to(batch_x.device).view(1, 1, -1)


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    nu: float,
    feature_mask: torch.Tensor | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = apply_feature_mask(batch_x.to(device), feature_mask)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            total_loss += combined_bce_brier_loss(predictions, batch_y, nu).item()
    return total_loss / max(len(loader), 1)


def train_classifier(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    brier_weight: float,
    patience: int,
    feature_mask: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0

    print(f"Starting {name} training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = apply_feature_mask(batch_x.to(device), feature_mask)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            loss = combined_bce_brier_loss(predictions, batch_y, brier_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = evaluate_loss(model, val_loader, device, brier_weight, feature_mask)
        print(f"{name} epoch [{epoch}/{epochs}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"{name} early stopping after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def build_shap_background(dataset: AgentDojoDataset, max_samples: int = 100) -> torch.Tensor:
    benign_features = [
        features
        for features, label in zip(dataset.traces, dataset.labels)
        if int(label) == 0
    ]
    source = benign_features or dataset.traces
    if not source:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    return torch.stack(source[:max_samples]).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DARS model.")
    parser.add_argument("--train", type=str, default="data/train_traces.jsonl")
    parser.add_argument("--val", type=str, default="data/val_traces.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--brier_weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models_saved/dars_model_real.pt")
    parser.add_argument(
        "--skip_ablations",
        action="store_true",
        help="Skip retraining feature-ablation models. Use only for quick smoke tests.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========================================")
    print("DARS Framework - Model Training")
    print("========================================")
    print(f"Device: {device}")
    print("Fitting feature baselines on benign train traces only...")

    train_traces = load_traces(args.train)
    validate_paper_dataset(train_traces, args.train)
    validate_paper_dataset(load_traces(args.val), args.val)
    extractor = DARSFeatureExtractor(max_seq_len=args.max_seq_len)
    extractor.fit(train_traces)

    train_dataset = AgentDojoDataset(args.train, extractor=extractor, max_seq_len=args.max_seq_len)
    val_dataset = AgentDojoDataset(args.val, extractor=extractor, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DARSClassifier(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    model, best_val_loss = train_classifier(
        "DARS",
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.brier_weight,
        args.patience,
    )

    no_bdi_mask = torch.ones(len(FEATURE_NAMES), dtype=torch.float32)
    no_bdi_mask[FEATURE_NAMES.index("bdi_deviation")] = 0.0
    standard_lstm = StandardLSTMClassifier(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    standard_lstm, lstm_val_loss = train_classifier(
        "Standard LSTM (w/o BDI and attention)",
        standard_lstm,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.brier_weight,
        args.patience,
        no_bdi_mask,
    )

    ablation_states = {}
    ablation_val_losses = {}
    if not args.skip_ablations:
        for feature_name in FEATURE_NAMES:
            feature_mask = torch.ones(len(FEATURE_NAMES), dtype=torch.float32)
            feature_mask[FEATURE_NAMES.index(feature_name)] = 0.0
            ablation_model = DARSClassifier(
                input_dim=len(FEATURE_NAMES),
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
            ).to(device)
            ablation_model, ablation_loss = train_classifier(
                f"DARS ablation w/o {feature_name}",
                ablation_model,
                train_loader,
                val_loader,
                device,
                args.epochs,
                args.lr,
                args.brier_weight,
                args.patience,
                feature_mask,
            )
            ablation_states[feature_name] = {
                "model_state": ablation_model.state_dict(),
                "feature_mask": feature_mask.tolist(),
            }
            ablation_val_losses[feature_name] = ablation_loss

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "standard_lstm_state": standard_lstm.state_dict(),
        "ablation_states": ablation_states,
        "extractor_state": extractor.to_state(),
        "shap_background": build_shap_background(train_dataset, max_samples=100),
        "config": {
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "max_seq_len": args.max_seq_len,
            "feature_names": FEATURE_NAMES,
            "standard_lstm_feature_mask": no_bdi_mask.tolist(),
            "dars_rule_weight": 0.30,
            "dars_learned_weight": 0.70,
            "ablation_val_losses": ablation_val_losses,
        },
    }
    torch.save(checkpoint, args.output)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Standard LSTM baseline validation loss: {lstm_val_loss:.4f}")
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
