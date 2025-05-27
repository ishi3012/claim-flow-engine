# ClaimFlowEngine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Made with FastAPI](https://img.shields.io/badge/Made%20with-FastAPI-009688.svg)
![CI](https://github.com/ishi3012/claim-flow-engine/actions/workflows/ci.yml/badge.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linted with Ruff](https://img.shields.io/badge/linting-ruff-yellow)
![Docs](https://img.shields.io/badge/docs-available-blue)
![Status](https://img.shields.io/badge/deploy-ready-green)
![Streamlit UI](https://img.shields.io/badge/streamlit-dashboard-red?logo=streamlit)



> **ClaimFlowEngine** is an ML-powered microservice for automating healthcare claim denial prediction, uncovering root causes through NLP-based clustering, and intelligently routing high-priority claims to resolution teams using reinforcement learning-inspired policies.

---

## Project Overview

**ClaimFlowEngine** simulates real-world **Revenue Cycle Management (RCM)** workflows and serves as a full-stack ML engineering showcase. It enables:

- Predicting whether a claim will be denied and why
  - For model details and evaluation, see the 
- Clustering denied claims into root causes using sentence embeddings
  - For model details and evaluation, see the .
- Routing high-complexity claims to the right team using contextual features
  - For model details and evaluation, see the 

The system demonstrates expertise in ML, RL-style logic, MLOps, and domain-driven healthcare automation.
---

📄 Model Cards
- [Denial Prediction Model Card](docs/model_card.md)
  - Detailed breakdown of model inputs, outputs, metrics, and retraining strategy.

- [Root Cause Clustering Model Card](docs/model_card_cluster.md)
  - Explains the Sentence-BERT + UMAP + HDBSCAN clustering pipeline with real output structure.

- [Routing Policy Engine Card](docs/model_card_policy.md)
  - Describes the priority scoring logic and YAML-driven queue assignment engine.

---
## Key Features

- **Real-time Denial Prediction**  
  Uses XGBoost/LightGBM models trained on structured claim + EHR data.

- **Root Cause Clustering**  
  Embeds denial reasons with Sentence-BERT and applies UMAP + HDBSCAN to discover latent denial drivers.

- **Routing Engine**  
  Applies an offline-RL-inspired policy to prioritize A/R workflows based on claim metadata, denial cluster, and team skills.

- **FastAPI Inference API**  
  Modular endpoints for predicting denial, clustering root causes, and suggesting routing actions.

- **CI/CD + MLOps Ready**  
  Integrated testing, type checks, and style checks with `pytest`, `mypy`, `ruff`, `black`, and GitHub Actions from day one.
---

## System Flow

```mermaid
flowchart TD
    A[Claims Dataset] --> B[Feature Engineering]
    B --> C[Denial Prediction (XGBoost)]
    C --> D[Root Cause Clustering (UMAP + HDBSCAN)]
    D --> E[Routing Engine]
    E --> F[FastAPI Endpoint]
```

## API Usage (FastAPI)

### 🔹 POST /api/fullroute
Submit a CSV file of claims to trigger the full triage pipeline:
```bash
curl -X POST http://localhost:8000/api/fullroute \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_claims.csv"
```
- Input: CSV with claims (columns: claim_id, cpt_code, diagnosis_code, payer_id, etc.)
- Output: JSON list with denial probability, top denial reasons, cluster ID, priority score, and recommended queue.

## Sample Inference (via JSON)

```json
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "12345",
    "cpt_code": "99213",
    "diagnosis_code": "E11.9",
    "payer_id": "AETNA",
    "provider_type": "Cardiologist"
  }'

```
  - ### Response
            ```json
            {
              "denial_probability": 0.82,
              "top_denial_reasons": ["Medical Necessity", "Prior Auth Missing"],
              "model_version": "v1.0.0",
              "routing_cluster_id": "auth_required",
              "explainability_scores": null
            }

            ```

## Sample Inference (via FastAPI)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "12345",
    "cpt_code": "99213",
    "diagnosis_code": "E11.9",
    "payer_id": "AETNA",
    "provider_type": "Cardiologist",
    ...
  }'
```

  - ### Response:
            ```json
            {
              "denial_probability": 0.82,
              "likely_denial_reasons": ["Medical Necessity", "Prior Auth Missing"],
              "denial_cluster": "auth-required",
              "routing_team": "High Complexity A/R Team"
            }
            ```
## Streamlit UI

You can launch a local dashboard to triage claims:

```bash
streamlit run streamlit_app.py
```

### Features
- Upload claims and run full prediction → cluster → route pipeline
- Visualize routed claim volume by team (bar chart)
- Explore denial clusters in UMAP 2D space
- Download results as triaged CSV

  - Runs at: `http://localhost:8501`


## Tech Stack
| Layer                  | Tools & Frameworks                                          |
|------------------------|-------------------------------------------------------------|
| ML Models              | XGBoost, LightGBM, Scikit-learn                             |
| Clustering             | Sentence-BERT, UMAP, HDBSCAN                                |
| API & Infra            | FastAPI, Uvicorn, Docker, GitHub Actions                    |
| Testing & CI           | Pytest, Mypy, Ruff, Black, Pre-commit                       |
| MLOps & Orchestration  | Vertex AI Pipelines, Airflow (optional), Streamlit          |
| Data                   | Simulated healthcare claims + EHR metadata                  |

## Project Structure

```bash
ClaimFlowEngine/
├── claimflowengine/
│   ├── agents/             # Claim routing + appeal agents
│   ├── pipelines/          # Model training & scoring
│   ├── preprocessing/      # Feature engineering modules
│   ├── skills/             # Legacy routing logic
│   ├── utils/              # Config loaders, path helpers
│   └── workflows/          # Orchestrated pipelines
├── data/                   # Raw + processed datasets
├── models/                 # Trained models & metrics
├── deployment/
│   ├── pipelines/          # Kubeflow/Vertex pipeline components
│   ├── serving/            # FastAPI app
│   ├── docker/             # Dockerfiles
│   ├── config/             # YAML configs
│   └── gcloud_scripts/     # Vertex CLI automation
├── notebooks/              # EDA + clustering analysis
├── tests/                  # Unit + integration tests
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Getting Started
### 1. Clone the repo
```bash
git clone https://github.com/ishi3012/ClaimFlowEngine.git
cd ClaimFlowEngine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI server
```bash
uvicorn app.main:app --reload
```

### 4. Prepare input data
```bash
# Place your raw claim dataset in:
data/raw_claims.csv
```
## Sample Results

| Metric                | Score  |
|------------------------|--------|
| AUC (Binary Denial)    | 0.540  |
| F1-Score               | 0.567  |
| Recall                 | 0.593  |
| Accuracy               | 0.537  |
| Composite Score        | 0.553  |

> Trained using 5-fold cross-validation on simulated EHR + claims data (CatBoost model).

⚠️ Results based on simulated EHR + claims dataset.

## Roadmap
### Domain Additions
- HITL feedback loop for routing policy refinement

- Real-time EHR integration

- Interactive audit dashboards for claim lifecycle

### Technical Extensions
- LangChain/CrewAI for multi-agent routing

- Vertex AI Workbench migration

- CI-integrated model retraining pipeline

## Author

**Shilpa Musale**  
[LinkedIn](https://www.linkedin.com/in/shilpamusale) • [GitHub](https://github.com/ishi3012) 
<!-- • [Portfolio](https://ishi3012.github.io/ishi-ai/) -->

## License
This project is licensed under the [MIT License](LICENSE).
