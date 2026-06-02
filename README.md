# DARS: Diversity-Aware Risk Scoring Framework for Detecting Malicious Tool Usage in AI Agents

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)]()
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-green)]()
[![AgentDojo](https://img.shields.io/badge/AgentDojo-Groq-purple)]()

**Author:** Nguyen Hoang Minh — SE0001 K49  
**Supervisor:** Dr. Nguyen Quoc Hung  
**Institution:** University of Economics Ho Chi Minh City (UEH)  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results](#2-key-results)
3. [System Requirements](#3-system-requirements)
4. [Installation](#4-installation)
5. [Project Structure](#5-project-structure)
6. [Dataset and Seed Trace Setup](#6-dataset-and-seed-trace-setup)
7. [Model Training](#7-model-training)
8. [Evaluation](#8-evaluation)
9. [Running the API](#9-running-the-api)
10. [AgentDojo/Groq Data Collection](#10-agentdojogroq-data-collection)
11. [Artifact Validation](#11-artifact-validation)
12. [Troubleshooting](#12-troubleshooting)
13. [Dependencies](#13-dependencies)

---

## 1. Project Overview

DARS is a risk-scoring framework for AI agents that use external tools. It monitors tool-call traces and produces a continuous risk score in `[0, 1]`, together with severity tiers and explanation signals.

The implementation follows the paper design:

- Behavioral Diversity Index (BDI) deviation from benign tool behavior
- Privilege-level and dynamic privilege-escalation features
- Markov transition anomaly with Laplace smoothing
- N-gram sequence-context anomaly for multi-step tool-use behavior
- Token burst and cumulative token-consumption anomaly
- BiLSTM plus 4-head Transformer attention classifier
- Rule-based risk fusion with 42 static security rules
- SHAP/permutation-style attributions for risk explanations
- FastAPI scoring endpoint for online session analysis

The repository supports two workflows:

- **Paper reproduction:** use the included 48 balanced AgentDojo/Groq seed traces to synthesize the benchmark, train, evaluate, and validate artifacts.
- **Live data collection:** collect more AgentDojo traces through a Groq-compatible API and append/deduplicate them incrementally.

---

## 2. Key Results

Current evaluation on the generated paper benchmark:

| Model | Accuracy | Precision | Recall | F1 | AUC | FDR |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based | 0.340 | 0.340 | 1.000 | 0.507 | 0.847 | 0.660 |
| Isolation Forest | 0.842 | 0.720 | 0.874 | 0.789 | 0.919 | 0.280 |
| Log Parser | 0.717 | 1.000 | 0.167 | 0.286 | 0.583 | 0.000 |
| Sandbox Monitor | 0.717 | 1.000 | 0.167 | 0.286 | 0.583 | 0.000 |
| Standard LSTM | 0.946 | 0.960 | 0.878 | 0.917 | 0.988 | 0.040 |
| **DARS** | **0.952** | **0.946** | **0.911** | **0.928** | **0.985** | **0.054** |

Risk scoring quality:

| Metric | Value |
|---|---:|
| Expected Calibration Error (ECE) | 0.015 |
| Spearman ranking correlation | 0.691 |
| NDCG@10 | 0.851 |

Scenario-level F1:

| Scenario | F1 |
|---|---:|
| S1 prompt injection | 0.832 |
| S2 privilege escalation | 0.865 |
| S3 malicious API usage | 0.865 |
| S4 data exfiltration | 0.843 |
| S5 denial of wallet | 0.865 |
| S6 stealth mimicry | 0.612 |

Robustness checks are available through `scripts/robustness_experiment.py`:

| Condition | F1 | AUC | FDR |
|---|---:|---:|---:|
| Baseline | 0.928 | 0.985 | 0.054 |
| 20% token noise | 0.934 | 0.986 | 0.046 |
| 10% telemetry dropout | 0.857 | 0.950 | 0.141 |
| 10% benign insertions | 0.894 | 0.966 | 0.116 |
| Adaptive S6 mimicry | 0.671 | 0.871 | 0.048 |

These numbers are produced by `evaluate.py` after training `models_saved/dars_model_real.pt`.

---

## 3. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk Space | 2 GB | 5 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| GPU | Not required | NVIDIA GPU with CUDA for faster training |

The paper environment can run on CPU, but the current local setup has been verified with CUDA on `venv311`.

---

## 4. Installation

### Step 1 - Clone or Extract the Project

If using a Git checkout or ZIP submission, open the repository root:

```bash
cd DARS-Github-Repo-V2
```

### Step 2 - Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\activate
```

macOS / Linux:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

### Step 3 - Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the wheel matching your CUDA environment from the official PyTorch index before running training.

### Step 4 - Verify Installation

```bash
python -c "import torch, sklearn, fastapi, shap; print('Dependencies OK'); print('CUDA:', torch.cuda.is_available())"
```

Expected output includes:

```text
Dependencies OK
CUDA: True
```

`CUDA: False` is acceptable for CPU-only runs.

---

## 5. Project Structure

```text
DARS-Github-Repo-V2/
|
|-- api/
|   `-- main_api.py                    # FastAPI inference service and dashboard
|
|-- data/
|   |-- collect_real_traces.py          # AgentDojo/Groq collection wrapper
|   |-- generate_data.py                # Paper dataset generator from 48 seeds
|   |-- real_traces.jsonl               # Parsed accumulated real traces
|   |-- real_seed_traces.jsonl          # 48 balanced paper seed traces
|   |-- train_traces.jsonl              # Generated train split
|   |-- val_traces.jsonl                # Generated validation split
|   `-- test_traces.jsonl               # Generated test split
|
|-- models_saved/
|   `-- dars_model_real.pt              # Trained DARS checkpoint
|
|-- scripts/
|   |-- check_collection_coverage.py    # Incremental real-trace coverage report
|   |-- check_paper_artifacts.py        # Structural artifact validation
|   |-- parse_agentdojo_logs.py         # AgentDojo log parser
|   |-- robustness_experiment.py        # Token-noise, dropout, insertion, and S6 stress tests
|   |-- run_groq_benchmark.ps1          # Single AgentDojo/Groq benchmark launcher
|   `-- run_groq_stepwise.ps1           # Stepwise collection helper
|
|-- src/
|   |-- data/                           # JSONL loading and dataset wrappers
|   |-- explainability/                 # Explanation helper
|   |-- feature_extraction/             # BDI, transition, privilege, token, sequence features
|   |-- models/                         # DARS and baseline neural models
|   `-- scoring/                        # Rule scoring, fusion, severity, smoothing
|
|-- train.py                            # Training entry point
|-- evaluate.py                         # Evaluation, baselines, ablations
|-- requirements.txt                    # Python dependencies
|-- README-template.md                  # Original README template
`-- README.md                           # This file
```

---

## 6. Dataset and Seed Trace Setup

The generated benchmark is derived from 48 balanced genuine AgentDojo/Groq seed traces:

| Requirement | Value |
|---|---:|
| Total seed traces | 48 |
| Benign seed traces | 24 |
| Malicious seed traces | 24 |
| Benign traces per domain | 6 |
| Malicious traces per scenario | 4 |
| Domains | workspace, slack, data_manipulation, external_api |
| Scenarios | S1, S2, S3, S4, S5, S6 |

The malicious scenario taxonomy is:

- `S1_prompt_injection`
- `S2_privilege_escalation`
- `S3_malicious_api_usage`
- `S4_data_exfiltration`
- `S5_denial_of_wallet`
- `S6_stealth_mimicry`

Generate the paper dataset:

```bash
python data/generate_data.py --seed-file data/real_seed_traces.jsonl --seed 42
```

Expected split sizes:

| Split | Benign | Malicious | Total |
|---|---:|---:|---:|
| Train | 2450 | 1260 | 3710 |
| Validation | 525 | 270 | 795 |
| Test | 525 | 270 | 795 |

The generator refuses incomplete or structurally invalid seed coverage unless `--allow-synthetic-seeds` is explicitly passed. Do not use that escape hatch for paper results.

---

## 7. Model Training

Train the full DARS model, Standard LSTM baseline, and feature-ablation models:

```bash
python train.py
```

Windows with the local CUDA environment:

```powershell
.\venv311\Scripts\python.exe train.py
```

Default training configuration:

| Parameter | Default |
|---|---:|
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Hidden dimension | 128 |
| LSTM layers | 2 |
| Attention heads | 4 |
| Max sequence length | 20 |
| Brier loss weight | 0.5 |
| Early-stopping patience | 8 |

Output:

```text
models_saved/dars_model_real.pt
```

For a faster smoke test:

```bash
python train.py --epochs 2 --patience 2 --skip_ablations
```

---

## 8. Evaluation

Run the full paper-aligned evaluation:

```bash
python evaluate.py
```

Windows:

```powershell
.\venv311\Scripts\python.exe evaluate.py
```

The evaluation reports:

- Overall classification performance for DARS and baselines
- Risk scoring quality: ECE, Spearman, NDCG@10
- F1 by malicious scenario S1-S6
- Feature ablation results
- Runtime per trace in milliseconds

Optional arguments:

```bash
python evaluate.py \
  --input data/test_traces.jsonl \
  --train data/train_traces.jsonl \
  --val data/val_traces.jsonl \
  --model models_saved/dars_model_real.pt
```

Run robustness experiments:

```bash
python scripts/robustness_experiment.py
```

Windows:

```powershell
.\venv311\Scripts\python.exe scripts\robustness_experiment.py
```

---

## 9. Running the API

Start the FastAPI service after training:

```bash
uvicorn api.main_api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/dashboard
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Scoring endpoint:

```text
POST /api/v1/score_session
```

Example request:

```json
{
  "session_id": "demo-session",
  "domain": "workspace",
  "tool_calls": [
    {
      "tool": "read_database",
      "tokens": 180,
      "success": true
    },
    {
      "tool": "send_email",
      "tokens": 220,
      "success": true
    }
  ]
}
```

Example response fields:

```json
{
  "session_id": "demo-session",
  "risk_score": 0.42,
  "learned_score": 0.31,
  "rule_score": 0.55,
  "severity_tier": "medium",
  "attributions": {},
  "triggered_rules": [],
  "narrative": "..."
}
```

To load a different checkpoint:

```bash
set DARS_MODEL_PATH=models_saved/dars_model_real.pt
uvicorn api.main_api:app --reload --port 8000
```

On PowerShell:

```powershell
$env:DARS_MODEL_PATH="models_saved/dars_model_real.pt"
uvicorn api.main_api:app --reload --port 8000
```

---

## 10. AgentDojo/Groq Data Collection

Live collection requires a valid Groq API key:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

Collect benign traces for all default suites:

```powershell
.\venv311\Scripts\python.exe data\collect_real_traces.py `
  --output data\real_traces.jsonl `
  --seed-output data\real_seed_traces.jsonl
```

Default suites:

```text
workspace, slack, travel, banking
```

The parser maps these to paper domains:

| AgentDojo suite | Paper domain |
|---|---|
| workspace | workspace |
| slack | slack |
| travel | data_manipulation |
| banking | external_api |

Collect malicious traces by attack:

```powershell
.\venv311\Scripts\python.exe data\collect_real_traces.py `
  --suite workspace `
  --attack direct `
  --output data\real_traces.jsonl `
  --seed-output data\real_seed_traces.jsonl
```

Useful attack names:

```text
direct
tool_knowledge
dos
injecagent
```

Because Groq quota can expire mid-run, collection is append/deduplicated. You can safely run the same suite or attack multiple times. Existing valid traces are kept.

Check incremental coverage:

```powershell
.\venv311\Scripts\python.exe scripts\check_collection_coverage.py
```

Refresh the 48-trace seed file from accumulated parsed traces without launching a new benchmark:

```powershell
.\venv311\Scripts\python.exe data\collect_real_traces.py `
  --skip-benchmark `
  --output data\real_traces.jsonl `
  --seed-output data\real_seed_traces.jsonl `
  --force-seed-refresh
```

When the coverage checker says the seed set is ready, regenerate the dataset and retrain.

---

## 11. Artifact Validation

Before reporting paper results, run:

```bash
python scripts/check_paper_artifacts.py
```

Windows:

```powershell
.\venv311\Scripts\python.exe scripts\check_paper_artifacts.py
```

The checker validates:

- Feature contract includes all six paper features
- Static rule count is 42
- Seed file has 48 balanced traces
- Train/validation/test splits have expected label counts
- Model checkpoint contains the expected state keys and SHAP background

Expected final line:

```text
All paper artifacts are present and structurally valid.
```

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'sklearn'`

Install dependencies into the active environment:

```bash
python -m pip install -r requirements.txt
```

On this repo, prefer `venv311`:

```powershell
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### Training runs on CPU instead of GPU

Check PyTorch CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If it prints `False`, install a CUDA-enabled PyTorch build compatible with your GPU driver and Python version.

### `GROQ_API_KEY` is missing

Set the key before live collection:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

### Groq preflight returns HTTP 403 or code 1010

This usually indicates a network or edge access policy issue. The collector may continue, but benchmark calls can still fail. Try another network, VPN setting, or key/account environment.

### AgentDojo `KeyError: 'tools'`

The installed AgentDojo package may not support a requested suite. Use the supported suites:

```text
workspace, slack, travel, banking
```

Do not pass `tools` as a suite.

### Seed export says S1-S6 coverage is incomplete

Run:

```powershell
.\venv311\Scripts\python.exe scripts\check_collection_coverage.py
```

Then collect the missing suite/scenario combinations. The collector appends and deduplicates, so partial quota-limited runs are safe.

### API returns `503 DARS model is not loaded`

Train first or point `DARS_MODEL_PATH` to an existing checkpoint:

```bash
python train.py
uvicorn api.main_api:app --reload --port 8000
```

---

## 13. Dependencies

Main dependencies from `requirements.txt`:

| Package | Purpose |
|---|---|
| `torch` | DARS neural model and tensor operations |
| `scikit-learn` | Isolation Forest, calibration, metrics |
| `numpy` | Numerical operations |
| `pandas` | Data inspection and tabular utilities |
| `fastapi` | REST API |
| `uvicorn` | ASGI server |
| `pydantic` | API request/response validation |
| `shap` | Explainability support |

Install all dependencies with:

```bash
python -m pip install -r requirements.txt
```

---

## Quick Start Summary

```bash
python -m pip install -r requirements.txt
python data/generate_data.py --seed-file data/real_seed_traces.jsonl --seed 42
python train.py
python evaluate.py
python scripts/check_paper_artifacts.py
uvicorn api.main_api:app --reload --port 8000
```

Windows with `venv311`:

```powershell
.\venv311\Scripts\python.exe data\generate_data.py --seed-file data\real_seed_traces.jsonl --seed 42
.\venv311\Scripts\python.exe train.py
.\venv311\Scripts\python.exe evaluate.py
.\venv311\Scripts\python.exe scripts\check_paper_artifacts.py
uvicorn api.main_api:app --reload --port 8000
```

---

*DARS - Diversity-Aware Risk Scoring Framework for AI-agent tool-use security.*
