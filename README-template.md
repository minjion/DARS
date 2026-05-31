# XSmell: An Explainable Machine Learning Framework for Code Smell Detection and Refactoring Recommendation

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)]()
[![SHAP](https://img.shields.io/badge/SHAP-0.44-green)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)]()

**Author:** Nguyen Tran Thanh Vy — SE0001 K49  
**Supervisor:** Dr. Nguyen Quoc Hung  
**Institution:** University of Economics Ho Chi Minh City (UEH)  
**GitHub:** https://github.com/thzynu/XSmell  
**Dataset:** https://www.kaggle.com/datasets/mirzayasirabdullah07/code-smells-and-refactoring-dataset-120k

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results](#2-key-results)
3. [System Requirements](#3-system-requirements)
4. [Installation](#4-installation)
5. [Project Structure](#5-project-structure)
6. [Dataset Setup](#6-dataset-setup)
7. [Model Training](#7-model-training)
8. [Running the System](#8-running-the-system)
9. [Usage Guide — Each Module](#9-usage-guide--each-module)
10. [API Documentation](#10-api-documentation)
11. [Troubleshooting](#11-troubleshooting)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

XSmell is an AI-powered framework that:

- **Detects** 15 types of code smells using XGBoost (macro F1 = 92.3%)
- **Classifies severity** as Minor / Major / Critical
- **Predicts** post-refactoring complexity using regression
- **Explains** predictions using SHAP (TreeSHAP, fidelity r = 0.94)
- **Recommends** prioritized Fowler catalog refactoring operations

The system is implemented as a multi-page Streamlit web application with a FastAPI REST endpoint and CLI tools.

**15 Supported Smell Types:**
Shotgun Surgery, Inappropriate Intimacy, Lazy Class, Large Class, Speculative Generality, Long Method, Primitive Obsession, Data Clumps, Temporary Field, Duplicated Code, Switch Statements, Dead Code, Feature Envy, God Class, Message Chains

---

## 2. Key Results

| Metric | Value |
|---|---|
| Detection Accuracy (macro F1) | **92.3%** (15 smell types, 18,000 test samples) |
| Cross-Project Generalization | **88.1%** (10 unseen Java projects) |
| SHAP Explanation Fidelity | **r = 0.94** (vs. LIME r = 0.79) |
| User Study — Refactoring Accuracy | **+34%** (65.1% → 87.3%, N=30 developers) |
| User Study — Decision Time | **−41%** (487 → 288 seconds) |
| Effect Size | **Cohen's d = 1.47** [95% CI: 0.78, 2.14] |

---

## 3. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9+ | 3.11.4 |
| RAM | 8 GB | 16–32 GB |
| Disk Space | 5 GB (with models + dataset) | 10 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| GPU | Not required | Optional (for CNN baseline only) |

---

## 4. Installation

### Step 1 — Clone or Extract the Project

```bash
# Option A: Clone from GitHub
git clone https://github.com/thzynu/XSmell.git
cd XSmell

# Option B: Extract from ZIP
# Unzip the provided XSmell_Submission.zip
# Navigate to the 01_Code/ folder
cd 01_Code
```

### Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:** All packages install without errors. This may take 3–5 minutes on first run.

**If you encounter errors:**
```bash
# Upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify Installation

```bash
python -c "import xgboost, shap, streamlit, fastapi; print('All dependencies OK')"
```

Expected output: `All dependencies OK`

---

## 5. Project Structure

```
XSmell/
│
├── app.py                          # Streamlit home page (entry point)
│
├── pages/                          # Streamlit multi-page modules
│   ├── Dashboard.py                # Project-level analytics dashboard
│   ├── Prediction.py               # Single-instance smell prediction
│   ├── SHAP_Analysis.py            # SHAP explainability interface
│   └── Report.py                   # Report generation and download
│
├── models/                         # Trained model artifacts (.pkl files)
│   ├── xgboost_model.pkl           # Severity classifier (400 trees)
│   ├── xgb_smell_type_model.pkl    # Smell type classifier (600 trees)
│   ├── xgb_regressor.pkl           # Post-refactor complexity regressor (800 trees)
│   ├── scaler.pkl                  # StandardScaler for severity model
│   ├── scaler_smell_type.pkl       # StandardScaler for smell type model
│   ├── scaler_regression.pkl       # StandardScaler for regression model
│   ├── encoders.pkl                # LabelEncoder for severity (Minor/Major/Critical)
│   └── smell_type_encoder.pkl      # LabelEncoder for 15 smell types
│
├── dataset/                        # Dataset folder (download separately)
│   └── code_smells_refactoring_dataset_120k.csv
│
├── data/                           # Sample input for batch prediction
│   └── test_metrics.csv
│
├── results/                        # Output folder (auto-created)
│   ├── prediction_results.csv      # Saved prediction results
│   └── final_report.txt            # Generated text report
│
├── images/                         # Generated charts and SHAP plots
│   ├── shap_feature_importance.png
│   ├── shap_summary_plot.png
│   ├── shap_waterfall.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_smell_type.png
│   ├── correlation_matrix.png
│   └── feature_importance.png
│
├── logs/                           # Runtime logs (auto-created)
│   └── xsmell.log
│
├── utils/
│   └── preprocessing.py            # Shared preprocessing utilities
│
├── train_model.py                  # Train severity classifier
├── train_smell_type_model.py       # Train smell type classifier
├── train_regression_model.py       # Train regression model
│
├── predict_pipeline.py             # CLI single-instance prediction
├── batch_predict.py                # Batch prediction from CSV
├── evaluation_metrics.py           # Model evaluation and metrics
├── visualization.py                # Chart and plot generation
├── report_generator.py             # Text report generation
├── export_report.py                # Export to .xlsx and .txt
├── model_versioning.py             # Model backup with timestamps
├── logging_config.py               # Centralized logging configuration
├── exception_handler.py            # Safe execution wrapper
│
├── api.py                          # FastAPI REST endpoint
├── config.py                       # Centralized path configuration
├── main.py                         # CLI menu controller (recommended entry)
│
├── generate_shap_figures.py        # Generate all SHAP visualizations
├── sample_input.csv                # Sample input data for testing
├── sample_output.csv               # Sample expected output
├── test_api.py                     # API endpoint test script
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 6. Dataset Setup

The dataset is **not included** in the repository due to file size (120,000 instances, ~13 MB CSV).

### Download Dataset

1. Go to: https://www.kaggle.com/datasets/mirzayasirabdullah07/code-smells-and-refactoring-dataset-120k
2. Click **Download**
3. Extract and place the CSV file at:

```
XSmell/
└── dataset/
    └── code_smells_refactoring_dataset_120k.csv   ← place here
```

### Verify Dataset

```bash
python -c "
import pandas as pd
df = pd.read_csv('dataset/code_smells_refactoring_dataset_120k.csv')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Smell types: {df[\"code_smell_type\"].nunique()}')
"
```

Expected output:
```
Rows: 120,000
Columns: ['project_id', 'file_name', 'language', ...]
Smell types: 15
```

---

## 7. Model Training

> **Note:** Pre-trained model `.pkl` files are included in the `models/` folder. You only need to retrain if you want to update the models with new data.

### Option A — Use Main Controller (Recommended)

```bash
python main.py
```

Select from the interactive menu:
```
==========================================
        XSMELL AI MAIN SYSTEM
 Code Smell Detection & Refactoring Tool
==========================================

Select an option:

1. Train Smell Severity Model
2. Train Code Smell Type Model
3. Train Regression Model
4. Predict Single Input
5. Batch Prediction from CSV
6. Evaluate All Models
7. Generate Final Report
8. Launch Streamlit Web App
9. Exit
```

### Option B — Train Each Model Individually

```bash
# Train severity classifier (Minor/Major/Critical)
# Runtime: ~24 minutes | n_estimators=400, max_depth=10
python train_model.py

# Train smell type classifier (15 smell types)
# Runtime: ~28 minutes | n_estimators=600, max_depth=8
python train_smell_type_model.py

# Train post-refactor complexity regressor
# Runtime: ~18 minutes | n_estimators=800, learning_rate=0.03
python train_regression_model.py
```

**Output after training:**
```
models/
├── xgboost_model.pkl           ← severity classifier
├── xgb_smell_type_model.pkl    ← smell type classifier
├── xgb_regressor.pkl           ← regression model
├── scaler.pkl
├── scaler_smell_type.pkl
├── scaler_regression.pkl
├── encoders.pkl
└── smell_type_encoder.pkl
```

### Evaluate Models

```bash
python evaluation_metrics.py
```

Expected output:
```
=== Severity Model ===
Accuracy: 0.923
Macro F1: 0.923

=== Smell Type Model ===
Accuracy: 0.921
Macro F1: 0.921

=== Regression Model ===
R2 Score: 0.94
MAE: 2.13
RMSE: 3.47
```

---

## 8. Running the System

### Method 1 — Streamlit Web App (Recommended for Demo)

```bash
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

**4 pages available (left sidebar):**

| Page | URL | Description |
|---|---|---|
| 🏠 Home | `/` | System overview and navigation |
| 📊 Dashboard | `/Dashboard` | Project-level analytics |
| 🔍 Prediction | `/Prediction` | Single instance smell detection |
| 🧠 SHAP Analysis | `/SHAP_Analysis` | Explainability visualization |
| 📄 Report | `/Report` | Generate and download reports |

### Method 2 — CLI Main Controller

```bash
python main.py
```

Provides interactive menu to access all system functions.

### Method 3 — FastAPI REST Endpoint

```bash
# Start API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at: **http://localhost:8000/docs**

### Method 4 — CLI Single Prediction

```bash
python predict_pipeline.py
```

Follow the prompts to enter code metrics manually.

### Method 5 — Batch Prediction

```bash
python batch_predict.py
```

Reads from `data/test_metrics.csv` and saves results to `results/prediction_results.csv`.

---

## 9. Usage Guide — Each Module

### 9.1 Dashboard (pages/Dashboard.py)

**Purpose:** Project-level code quality overview.

**How to use:**
1. Run `streamlit run app.py`
2. Navigate to **Dashboard** in the sidebar
3. View smell type distribution, severity breakdown, and trend charts
4. Charts are generated from `results/prediction_results.csv`

> **Note:** Dashboard requires prediction results to exist. Run at least one prediction first.

---

### 9.2 Prediction Interface (pages/Prediction.py)

**Purpose:** Predict smell type and severity for a single code instance.

**Input fields (10 metrics):**

| Field | Description | Typical Range |
|---|---|---|
| Lines of Code | Total executable lines | 5 – 2,000 |
| Cyclomatic Complexity | Number of independent paths | 1 – 300 |
| Number of Methods | Method declarations | 0 – 300 |
| Number of Classes | Class declarations in file | 1 – 50 |
| Pre-Refactor Complexity | CC before refactoring | 1 – 300 |
| Post-Refactor Complexity | Expected CC after refactoring | 0.8 – 200 |
| Technical Debt (minutes) | Estimated remediation effort | 0 – 5,000 |
| Maintainability Index | Composite quality score | 0 – 100 |
| Bug-Prone Score | Defect probability (0–1) | 0.0 – 1.0 |
| Developer Experience (years) | Primary contributor's experience | 0.5 – 25 |

**How to use:**
1. Enter the 10 metric values
2. Click **"Predict"**
3. View: smell type, severity, confidence score, SHAP-based explanation, and prioritized refactoring recommendations

**Sample input (God Class example):**
```
Lines of Code: 412
Cyclomatic Complexity: 87.3
Number of Methods: 42
Number of Classes: 3
Pre-Refactor Complexity: 87.3
Post-Refactor Complexity: 20.4
Technical Debt: 312.5
Maintainability Index: 38.7
Bug-Prone Score: 0.78
Developer Experience: 6.2
```

**Expected output:**
```
Smell Type:  God Class
Severity:    Major
Confidence:  94.7%

Refactoring Recommendation:
Apply Extract Class (Recommended refactoring)
```

---

### 9.3 SHAP Analysis (pages/SHAP_Analysis.py)

**Purpose:** Visualize SHAP feature attributions for explainability.

**How to use:**
1. Navigate to **SHAP Analysis** in the sidebar
2. **Global View:** See overall feature importance across all instances
3. **Local View:** Select a specific instance → view waterfall plot showing which features drove the prediction
4. Download SHAP plots via the export button

**What the SHAP values mean:**
- **Positive SHAP (+):** Feature pushes prediction toward "smelly"
- **Negative SHAP (−):** Feature pushes prediction toward "clean"
- **Larger |SHAP|:** Stronger contribution to the prediction

> **Note:** SHAP computation for the full dataset takes ~3 minutes. For single instances, computation takes ~10ms.

---

### 9.4 Report Generation (pages/Report.py)

**Purpose:** Generate and download comprehensive code quality reports.

**How to use:**
1. Navigate to **Report** in the sidebar
2. View summary metrics and charts
3. Click **"Download Report (.txt)"** or **"Download Report (.xlsx)"**

**Report contents:**
- Total predictions analyzed
- Smell type distribution
- Severity breakdown
- Top-5 most frequent smell types with refactoring recommendations
- Timestamp and session metadata

---

### 9.5 Batch Prediction (batch_predict.py)

**Purpose:** Process multiple code instances from a CSV file.

**Input format** (`data/test_metrics.csv`):

```csv
lines_of_code,cyclomatic_complexity,num_methods,num_classes,pre_refactor_complexity,post_refactor_complexity,technical_debt_minutes,maintainability_index,bug_prone_score,developer_experience_years
412,87.3,42,3,87.3,20.4,312.5,38.7,0.78,6.2
145,12.1,8,1,12.1,8.5,45.0,72.3,0.21,4.5
```

**Run:**
```bash
python batch_predict.py
```

**Output** (`results/prediction_results.csv`):
```csv
lines_of_code,...,predicted_smell_type,predicted_severity
412,...,God Class,Major
145,...,Long Method,Minor
```

---

### 9.6 Generate SHAP Visualizations

```bash
python generate_shap_figures.py
```

**Outputs to `images/`:**
- `shap_summary_plot.png` — Global feature importance
- `shap_bar.png` — Mean |SHAP| bar chart
- `shap_beeswarm.png` — SHAP beeswarm plot
- `shap_waterfall.png` — Local explanation waterfall

---

## 10. API Documentation

### Start the API Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint: POST /predict

**URL:** `http://localhost:8000/predict`

**Request body:**
```json
{
  "lines_of_code": 412,
  "cyclomatic_complexity": 87.3,
  "num_methods": 42,
  "num_classes": 3,
  "pre_refactor_complexity": 87.3,
  "post_refactor_complexity": 20.4,
  "technical_debt_minutes": 312.5,
  "maintainability_index": 38.7,
  "bug_prone_score": 0.78,
  "developer_experience_years": 6.2
}
```

**Response:**
```json
{
  "predicted_smell_type": "God Class",
  "predicted_severity": "Major",
  "refactoring_suggestion": "Apply Extract Class (Recommended refactoring)"
}
```

### Test the API

**Option A — Interactive docs:**
Open http://localhost:8000/docs in your browser.

**Option B — curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "lines_of_code": 412,
       "cyclomatic_complexity": 87.3,
       "num_methods": 42,
       "num_classes": 3,
       "pre_refactor_complexity": 87.3,
       "post_refactor_complexity": 20.4,
       "technical_debt_minutes": 312.5,
       "maintainability_index": 38.7,
       "bug_prone_score": 0.78,
       "developer_experience_years": 6.2
     }'
```

**Option C — Python script:**
```bash
python test_api.py
```

**Option D — Postman:**
1. Open Postman
2. New Request → POST → `http://localhost:8000/predict`
3. Body → raw → JSON → paste request body above
4. Click Send

---

## 11. Troubleshooting

### Issue: `ModuleNotFoundError`

```bash
# Re-activate venv and reinstall
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: models/*.pkl`

Models not found. Either:
1. Run training first: `python train_model.py`
2. Or ensure the `models/` folder contains the `.pkl` files from the provided package

### Issue: `FileNotFoundError: dataset/...csv`

Dataset not downloaded. Follow [Section 6](#6-dataset-setup).

### Issue: Streamlit page not loading

```bash
# Check Streamlit version
streamlit --version

# Clear Streamlit cache
streamlit cache clear

# Restart
streamlit run app.py
```

### Issue: SHAP computation too slow

SHAP for the full dataset (18,000 instances) takes ~3 minutes. This is normal. For real-time use, SHAP is computed per-instance (~10ms) in the Prediction page.

### Issue: Port 8000 already in use (API)

```bash
# Use a different port
uvicorn api:app --reload --port 8080

# Or kill the existing process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: `uvicorn` not found

```bash
pip install uvicorn fastapi
```

---

## 12. Dependencies

Full list from `requirements.txt`:

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML utilities, StandardScaler, LabelEncoder |
| `xgboost` | Primary ML model (XGBoost Classifier + Regressor) |
| `joblib` | Model serialization (.pkl save/load) |
| `shap` | SHAP explainability (TreeSHAP) |
| `streamlit` | Web application framework |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Data validation for API |
| `matplotlib` | Chart generation |
| `seaborn` | Statistical visualization |
| `openpyxl` | Excel (.xlsx) report export |
| `numba` | SHAP performance optimization |
| `tqdm` | Progress bars |
| `cloudpickle` | Advanced serialization |
| `slicer` | SHAP internal utility |

---

## Quick Start Summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download dataset → place in dataset/

# 3. Run web app
streamlit run app.py

# 4. Or use CLI menu
python main.py
```

---

*XSmell — University of Economics Ho Chi Minh City (UEH) | SE0001 K49 | 2025–2026*
