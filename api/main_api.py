import os
import sys
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability.shap_analyzer import ExplainableRiskAnalyzer
from src.feature_extraction.extractor import DARSFeatureExtractor, FEATURE_NAMES
from src.models.dars_model import DARSClassifier
from src.scoring.risk import (
    blend_risk_scores,
    calculate_rule_risk,
    get_severity_tier,
    temporal_smooth,
    triggered_static_rules,
)


MODEL_PATH = os.getenv("DARS_MODEL_PATH", "models_saved/dars_model_real.pt")

app = FastAPI(
    title="DARS Framework API",
    description="Diversity-Aware Risk Scoring for AI Agents",
    version="1.1.0",
)

DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DARS Security Dashboard</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #172033; }
    header { background: #172033; color: white; padding: 16px 24px; }
    main { display: grid; grid-template-columns: minmax(320px, 440px) 1fr; gap: 16px; padding: 16px; }
    section { background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; }
    textarea { width: 100%; min-height: 360px; font-family: Consolas, monospace; font-size: 13px; }
    button { background: #1b66c9; color: white; border: 0; border-radius: 6px; padding: 10px 14px; cursor: pointer; }
    pre { white-space: pre-wrap; word-break: break-word; background: #eef2f8; padding: 12px; border-radius: 6px; }
    .metric { display: inline-block; min-width: 130px; margin: 8px 8px 8px 0; padding: 10px; background: #eef2f8; border-radius: 6px; }
    .critical { color: #b00020; font-weight: 700; }
    @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>DARS Security Dashboard</h1></header>
  <main>
    <section>
      <h2>Session Input</h2>
      <textarea id="payload">{
  "session_id": "demo-session",
  "domain": "workspace",
  "tool_calls": [
    {"tool": "read_database", "tokens": 180, "success": true},
    {"tool": "send_email", "tokens": 220, "success": true}
  ]
}</textarea>
      <p><button onclick="score()">Score Session</button></p>
    </section>
    <section>
      <h2>Risk Result</h2>
      <div id="metrics"></div>
      <h3>Triggered Rules</h3>
      <pre id="rules">No result yet.</pre>
      <h3>Attributions</h3>
      <pre id="attrs">No result yet.</pre>
      <h3>Narrative</h3>
      <pre id="narrative">No result yet.</pre>
    </section>
  </main>
  <script>
    async function score() {
      const response = await fetch('/api/v1/score_session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: document.getElementById('payload').value
      });
      const data = await response.json();
      if (!response.ok) {
        document.getElementById('metrics').innerHTML = '<span class="critical">' + JSON.stringify(data) + '</span>';
        return;
      }
      document.getElementById('metrics').innerHTML =
        '<span class="metric">Risk: ' + data.risk_score.toFixed(3) + '</span>' +
        '<span class="metric">Learned: ' + data.learned_score.toFixed(3) + '</span>' +
        '<span class="metric">Rule: ' + data.rule_score.toFixed(3) + '</span>' +
        '<span class="metric">Tier: ' + data.severity_tier + '</span>';
      document.getElementById('rules').textContent = JSON.stringify(data.triggered_rules, null, 2);
      document.getElementById('attrs').textContent = JSON.stringify(data.attributions, null, 2);
      document.getElementById('narrative').textContent = data.narrative;
    }
  </script>
</body>
</html>
"""


class ToolCall(BaseModel):
    tool: str
    privilege: int | None = None
    tokens: int = 0
    timestamp: int | None = None
    success: bool = True
    input: Dict | None = None


class SessionRequest(BaseModel):
    session_id: str
    domain: str = "unknown"
    tool_calls: List[ToolCall]
    previous_risk_score: float | None = None


class ScoringResponse(BaseModel):
    session_id: str
    risk_score: float
    learned_score: float
    rule_score: float
    severity_tier: str
    attributions: Dict[str, float]
    triggered_rules: List[str]
    narrative: str


def load_runtime() -> tuple[DARSClassifier, DARSFeatureExtractor, torch.Tensor | None]:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "model_state" not in checkpoint or "extractor_state" not in checkpoint:
        raise RuntimeError("Model checkpoint is missing extractor_state; rerun train.py.")
    config = checkpoint.get("config", {})
    model = DARSClassifier(
        input_dim=config.get("input_dim", len(FEATURE_NAMES)),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    extractor = DARSFeatureExtractor.from_state(checkpoint["extractor_state"])
    background = checkpoint.get("shap_background")
    if isinstance(background, torch.Tensor) and background.numel() > 0:
        background = background.to("cpu")
    else:
        background = None
    return model, extractor, background


MODEL = None
EXTRACTOR = None
SHAP_BACKGROUND = None
ANALYZER = ExplainableRiskAnalyzer()
SESSION_RISK_STATE: Dict[str, float] = {}


@app.on_event("startup")
def startup() -> None:
    global MODEL, EXTRACTOR, SHAP_BACKGROUND
    if os.path.exists(MODEL_PATH):
        MODEL, EXTRACTOR, SHAP_BACKGROUND = load_runtime()


@app.get("/health")
async def health() -> Dict[str, bool | str]:
    return {
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "shap_background_loaded": SHAP_BACKGROUND is not None,
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML


@app.post("/api/v1/score_session", response_model=ScoringResponse)
async def score_session(request: SessionRequest):
    if MODEL is None or EXTRACTOR is None:
        raise HTTPException(status_code=503, detail="DARS model is not loaded. Run train.py first.")
    if not request.tool_calls:
        raise HTTPException(status_code=400, detail="Tool calls sequence cannot be empty.")

    trace = {
        "session_id": request.session_id,
        "domain": request.domain,
        "tool_calls": [call.model_dump() for call in request.tool_calls],
        "label": 0,
    }
    features = EXTRACTOR.extract_trace_features(trace)
    feature_summary = EXTRACTOR.summarize_trace(trace)
    with torch.no_grad():
        learned_score = float(MODEL(features.unsqueeze(0)).item())
    rule_score = calculate_rule_risk(trace, feature_summary)
    blended_score = blend_risk_scores(learned_score, rule_score)
    previous_score = request.previous_risk_score
    if previous_score is None:
        previous_score = SESSION_RISK_STATE.get(request.session_id)
    risk_score = temporal_smooth(previous_score, blended_score)
    SESSION_RISK_STATE[request.session_id] = risk_score

    attributions = ANALYZER.explain(MODEL, features, SHAP_BACKGROUND)
    triggered_rules = [rule["name"] for rule in triggered_static_rules(trace, feature_summary)]
    narrative = ANALYZER.generate_narrative(risk_score, attributions)

    return ScoringResponse(
        session_id=request.session_id,
        risk_score=risk_score,
        learned_score=learned_score,
        rule_score=rule_score,
        severity_tier=get_severity_tier(risk_score),
        attributions=attributions,
        triggered_rules=triggered_rules,
        narrative=narrative,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
