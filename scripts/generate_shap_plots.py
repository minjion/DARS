"""Generate SHAP visualisation figures from the trained DARS model.

Produces three figures in ``scripts/figures/``:
1. **shap_summary_plot.png** - beeswarm plot showing per-feature impact
2. **shap_feature_importance.png** - mean |SHAP| bar chart
3. **shap_waterfall_plot.png** - waterfall for a specific S4 data-exfiltration trace

Run from the repository root:
    python scripts/generate_shap_plots.py
"""

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import AgentDojoDataset, load_traces
from src.feature_extraction.extractor import FEATURE_NAMES
from src.models.dars_model import DARSClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models_saved" / "dars_model_real.pt"
TEST_DATA = ROOT / "data" / "test_traces.jsonl"
FIGURES_DIR = ROOT / "scripts" / "figures"

PRETTY_NAMES = {
    "bdi_deviation": "BDI Deviation",
    "privilege_level": "Privilege Level",
    "privilege_escalation": "Privilege Escalation",
    "transition_anomaly": "Transition Anomaly",
    "sequence_anomaly": "Sequence Anomaly",
    "token_burst": "Token Burst",
}


# ---------------------------------------------------------------------------
# Load helpers (mirrors evaluate.py patterns)
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | Path, device: torch.device):
    """Return (model, extractor, checkpoint) from a saved DARS checkpoint."""
    from src.feature_extraction.extractor import DARSFeatureExtractor

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    if "model_state" not in checkpoint or "extractor_state" not in checkpoint:
        raise ValueError("Checkpoint missing model_state or extractor_state.")

    config = checkpoint.get("config", {})
    model = DARSClassifier(
        input_dim=config.get("input_dim", len(FEATURE_NAMES)),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    extractor = DARSFeatureExtractor.from_state(checkpoint["extractor_state"])
    return model, extractor, checkpoint


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: torch.nn.Module,
    data_tensor: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
) -> np.ndarray | None:
    """Attempt DeepExplainer; return SHAP values (N, seq_len, features) or None."""
    try:
        import shap

        model.eval()
        explainer = shap.DeepExplainer(model, background.to(device))
        shap_vals = explainer.shap_values(data_tensor.to(device))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return np.asarray(shap_vals, dtype=np.float64)
    except Exception as exc:
        print(f"[WARN] DeepExplainer failed ({exc}); falling back to permutation attribution.")
        return None


def permutation_attribution(
    model: torch.nn.Module,
    data_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Deterministic occlusion-based attribution - shape (N, seq_len, features)."""
    model.eval()
    data = data_tensor.to(device)
    n_samples, seq_len, n_features = data.shape
    attributions = np.zeros((n_samples, seq_len, n_features), dtype=np.float64)

    with torch.no_grad():
        baseline_scores = model(data).cpu().numpy().flatten()
        for feat_idx in range(n_features):
            masked = data.clone()
            masked[:, :, feat_idx] = 0.0
            masked_scores = model(masked).cpu().numpy().flatten()
            # Attribute the drop uniformly across timesteps
            diff = baseline_scores - masked_scores
            attributions[:, :, feat_idx] = diff[:, np.newaxis] / seq_len

    return attributions


def aggregate_per_feature(shap_values: np.ndarray) -> np.ndarray:
    """Collapse (N, seq_len, features) -> (N, features) via mean over timesteps."""
    return shap_values.mean(axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Use a clean, publication-friendly style
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
})


def plot_summary_beeswarm(
    shap_per_feature: np.ndarray,
    feature_values: np.ndarray,
    save_path: Path,
) -> None:
    """Create a beeswarm-style summary plot."""
    n_features = shap_per_feature.shape[1]
    pretty = [PRETTY_NAMES.get(f, f) for f in FEATURE_NAMES]

    # Sort by mean |SHAP|
    mean_abs = np.abs(shap_per_feature).mean(axis=0)
    order = np.argsort(mean_abs)  # ascending; plot bottom-to-top

    fig, ax = plt.subplots(figsize=(9, 5))
    for plot_idx, feat_idx in enumerate(order):
        vals = shap_per_feature[:, feat_idx]
        colours = feature_values[:, feat_idx]
        jitter = np.random.default_rng(42).normal(0, 0.12, size=len(vals))
        sc = ax.scatter(
            vals,
            np.full_like(vals, plot_idx) + jitter,
            c=colours,
            cmap="coolwarm",
            alpha=0.55,
            s=10,
            edgecolors="none",
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([pretty[i] for i in order])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Summary - Feature Impact on Risk Score")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Feature value")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {save_path.name}")


def plot_feature_importance(
    shap_per_feature: np.ndarray,
    save_path: Path,
) -> None:
    """Bar chart of mean |SHAP| per feature."""
    mean_abs = np.abs(shap_per_feature).mean(axis=0)
    pretty = [PRETTY_NAMES.get(f, f) for f in FEATURE_NAMES]
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = plt.cm.viridis(np.linspace(0.3, 0.85, len(order)))
    bars = ax.barh(
        range(len(order)),
        mean_abs[order],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")

    for bar, idx in zip(bars, order):
        ax.text(
            bar.get_width() + mean_abs.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_abs[idx]:.4f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {save_path.name}")


def plot_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    save_path: Path,
    title: str = "SHAP Waterfall",
    trace_info: str = "",
) -> None:
    """Waterfall chart for a single sample showing cumulative SHAP contribution."""
    pretty = [PRETTY_NAMES.get(f, f) for f in FEATURE_NAMES]
    order = np.argsort(np.abs(shap_values))[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    cumulative = base_value
    y_positions = []
    widths = []
    lefts = []
    colours_list = []

    for rank, feat_idx in enumerate(order):
        val = shap_values[feat_idx]
        y_positions.append(rank)
        widths.append(abs(val))
        lefts.append(min(cumulative, cumulative + val))
        colours_list.append("#e74c3c" if val > 0 else "#3498db")
        cumulative += val

    ax.barh(y_positions, widths, left=lefts, color=colours_list, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{pretty[i]}  ({shap_values[i]:+.4f})" for i in order])
    ax.axvline(base_value, color="grey", linewidth=0.8, linestyle="--", label=f"Base: {base_value:.3f}")
    ax.axvline(cumulative, color="black", linewidth=1.2, linestyle="-", label=f"Output: {cumulative:.3f}")
    ax.set_xlabel("Model output (risk score)")
    full_title = title
    if trace_info:
        full_title += f"\n{trace_info}"
    ax.set_title(full_title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {save_path.name}")


def _find_trace_index(traces: list[dict], scenario: str, label: int) -> int | None:
    """Find the first trace index matching scenario and label."""
    for i, trace in enumerate(traces):
        if trace.get("scenario", "") == scenario and int(trace.get("label", 0)) == label:
            return i
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("========================================")
    print("DARS - SHAP Visualisation")
    print("========================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and data
    model, extractor, checkpoint = load_checkpoint(MODEL_PATH, device)
    raw_traces = load_traces(str(TEST_DATA))
    dataset = AgentDojoDataset(TEST_DATA, extractor=extractor)
    print(f"Loaded {len(dataset)} test traces from {TEST_DATA.name}")

    # Build data tensor (N, seq_len, features)
    data_tensor = torch.stack([dataset[i][0] for i in range(len(dataset))])

    # Feature values for coloring the beeswarm plot
    feature_values_flat = aggregate_per_feature(data_tensor.numpy())

    # Background for DeepExplainer
    background = checkpoint.get("shap_background", None)
    if background is not None:
        if isinstance(background, np.ndarray):
            background = torch.tensor(background, dtype=torch.float32)
        background = background.to(device)
        print(f"SHAP background shape: {tuple(background.shape)}")
    else:
        print("[INFO] No shap_background in checkpoint; using first 50 test samples.")
        background = data_tensor[:50].to(device)

    # Compute SHAP values
    print("Computing SHAP values ...")
    shap_values = compute_shap_values(model, data_tensor, background, device)
    if shap_values is None:
        print("Using permutation attribution as fallback.")
        shap_values = permutation_attribution(model, data_tensor, device)

    # Aggregate over timesteps -> (N, features)
    shap_per_feature = aggregate_per_feature(shap_values)

    # Base value (mean model output across all test traces)
    with torch.no_grad():
        all_scores = model(data_tensor.to(device)).cpu().numpy().flatten()
    base_value = float(all_scores.mean())

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to {FIGURES_DIR}/")

    # 1) Summary beeswarm plot
    plot_summary_beeswarm(
        shap_per_feature,
        feature_values_flat,
        FIGURES_DIR / "shap_summary_plot.png",
    )

    # 2) Feature importance bar chart
    plot_feature_importance(
        shap_per_feature,
        FIGURES_DIR / "shap_feature_importance.png",
    )

    # 3) Waterfall plots for multiple scenarios
    waterfall_configs = [
        {
            "scenario": "S2_privilege_escalation",
            "label": 1,
            "title": "SHAP Waterfall - S2 Privilege Escalation",
            "filename": "shap_waterfall_s2.png",
        },
        {
            "scenario": "S4_data_exfiltration",
            "label": 1,
            "title": "SHAP Waterfall - S4 Data Exfiltration",
            "filename": "shap_waterfall_s4.png",
        },
        {
            "scenario": "S6_stealth_mimicry",
            "label": 1,
            "title": "SHAP Waterfall - S6 Stealth Mimicry",
            "filename": "shap_waterfall_s6.png",
        },
    ]

    for cfg in waterfall_configs:
        idx = _find_trace_index(raw_traces, cfg["scenario"], cfg["label"])
        if idx is not None:
            agent_id = raw_traces[idx].get("agent_id", "?")
            score = float(all_scores[idx])
            trace_info = f"agent_id={agent_id}  |  risk_score={score:.3f}"
            plot_waterfall(
                shap_per_feature[idx],
                base_value,
                FIGURES_DIR / cfg["filename"],
                title=cfg["title"],
                trace_info=trace_info,
            )
        else:
            print(f"  [SKIP] No {cfg['scenario']} trace found")

    # 4) Waterfall for a benign trace (for comparison)
    benign_idx = _find_trace_index(raw_traces, "benign", 0)
    if benign_idx is None:
        # fallback: first trace with label=0
        for i, t in enumerate(raw_traces):
            if int(t.get("label", 0)) == 0:
                benign_idx = i
                break
    if benign_idx is not None:
        agent_id = raw_traces[benign_idx].get("agent_id", "?")
        score = float(all_scores[benign_idx])
        trace_info = f"agent_id={agent_id}  |  risk_score={score:.3f}"
        plot_waterfall(
            shap_per_feature[benign_idx],
            base_value,
            FIGURES_DIR / "shap_waterfall_benign.png",
            title="SHAP Waterfall - Benign Trace",
            trace_info=trace_info,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

