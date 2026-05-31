from typing import Dict, Iterable

import torch

from src.feature_extraction.extractor import FEATURE_NAMES


class ExplainableRiskAnalyzer:
    def __init__(self, feature_names: Iterable[str] | None = None):
        self.feature_names = list(feature_names or FEATURE_NAMES)

    def permutation_attribution(self, model: torch.nn.Module, features: torch.Tensor) -> Dict[str, float]:
        """Model-based feature attribution used as a deterministic SHAP fallback."""
        model.eval()
        with torch.no_grad():
            baseline_score = float(model(features.unsqueeze(0)).item())
            values = {}
            for feature_index, feature_name in enumerate(self.feature_names):
                masked = features.clone()
                masked[:, feature_index] = 0.0
                masked_score = float(model(masked.unsqueeze(0)).item())
                values[feature_name] = baseline_score - masked_score
        return values

    def deep_shap_attribution(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        background: torch.Tensor,
    ) -> Dict[str, float]:
        try:
            import shap

            model.eval()
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(features.unsqueeze(0))
            values = shap_values[0] if isinstance(shap_values, list) else shap_values
            tensor_values = torch.as_tensor(values, dtype=torch.float32)
            if tensor_values.ndim == 3:
                tensor_values = tensor_values.squeeze(0)
            feature_scores = tensor_values.abs().mean(dim=0)
            return {
                name: float(feature_scores[index].item())
                for index, name in enumerate(self.feature_names)
            }
        except Exception:
            return self.permutation_attribution(model, features)

    def explain(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        background: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if background is not None and background.numel() > 0:
            return self.deep_shap_attribution(model, features, background)
        return self.permutation_attribution(model, features)

    def generate_narrative(self, risk_score: float, attributions: Dict[str, float]) -> str:
        if risk_score < 0.4:
            return "Session appears benign with stable diversity, privilege, transition, and token signals."

        positive = [(name, value) for name, value in attributions.items() if value > 0]
        if not positive:
            positive = list(attributions.items())
        ranked = sorted(positive, key=lambda item: item[1], reverse=True)
        top = ranked[:2]
        reason = ", combined with ".join(
            f"{name.replace('_', ' ')} (phi={value:+.2f})" for name, value in top
        )
        return f"Risk is elevated due to {reason}."
