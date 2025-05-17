# ClaimFlowEngine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Made with FastAPI](https://img.shields.io/badge/Made%20with-FastAPI-009688.svg)

> A modular microservice for healthcare claim denial prediction, root cause clustering, and intelligent claim routing.

---

## Project Overview

ClaimFlowEngine is a production-grade FastAPI microservice that predicts healthcare claim denials, clusters denied claims into root causes, and routes complex cases to the most appropriate resolution queues.

This project is designed to simulate **real-world Revenue Cycle Management (RCM)** workflows and showcases end-to-end ML engineering, from data ingestion to smart routing logic.

---

## Key Features

- **Real-time Denial Prediction**  
  Uses XGBoost and LightGBM models trained on structured healthcare claims and EHR metadata.

- **Root Cause Clustering**  
  Combines Sentence-BERT embeddings with UMAP + HDBSCAN to group denied claims by latent denial drivers.

- **Routing Engine**  
  Applies a hybrid rule-based and ML-inspired logic to prioritize claims by complexity, denial type, and team expertise.

- **FastAPI Inference API**  
  Provides REST endpoints for predictions, clustering, and routing in real-time workflows.

- **CI/CD Ready**  
  Integrated with GitHub Actions, pytest, mypy, ruff, black, and pre-commit from Day 1.

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

---

## 🛠 Tech Stack

| Layer              | Tools & Frameworks                                                  |
|-------------------|----------------------------------------------------------------------|
| ML Models          | XGBoost, LightGBM, Scikit-learn                                      |
| Clustering         | Sentence-BERT, UMAP, HDBSCAN                                         |
| API & Infra        | FastAPI, Uvicorn, Docker, GitHub Actions                                |
| Testing & CI       | Pytest, Mypy, Ruff, Black, Pre-commit, Codecov (TBD)                       |
| Data               | Simulated claims and EHR metadata


---

## 📁 Directory Structure

```
ClaimFlowEngine/
├── claimflowengine/ # Core logic: agents, pipelines, preprocessing, utils, workflows
│ ├── agents/ # Agent logic for decision routing and appeals
│ ├── pipelines/ # Model training, evaluation, and composite scoring logic
│ ├── preprocessing/ # Feature engineering, imputers, encoders, datetime transforms
│ ├── skills/ # Legacy logic (can migrate to pipelines or agents)
│ ├── utils/ # Shared utilities like config loading and path management
│ └── workflows/ # Agent task chains or pipeline orchestrations
├── data/ # Raw and processed claims dataset
├── models/ # Trained model binaries and benchmark results
├── deployment/
│ ├── pipelines/ # Vertex AI pipeline components & orchestrators
│ ├── serving/ # FastAPI app or custom prediction containers
│ ├── docker/ # Dockerfiles and startup configs
│ ├── config/ # Pipeline & model YAML configs
│ └── gcloud_scripts/ # CLI scripts for pipeline submission, deploy, etc.
├── notebooks/ # EDA, experiments, embedding analysis
├── tests/ # Unit & integration tests
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```


## 🚀 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/<your-username>/ClaimFlowEngine.git
cd ClaimFlowEngine

```

2. **Install dependencies**

```bash
pip install -r requirements.txt

```
3. **Run the API**

```bash
uvicorn app.main:app --reload
```

---

## 🔬 Future Directions

### Domain features:
- Live data integration with EHR APIs for real-time claim processing.
- Human-in-the-loop (HITL) feedback mechanism for continuous policy tuning and improvement.
- Interactive dashboard for enhanced explainability of predictions, cluster insights, and audit trails of routing decisions.
- Agentic AI logic for denial appeals and policy optimization.

### Tech Enhancements:

- LangChain/CrewAI orchestration for multi-agent workflows
- Vertex AI Workbench for low-code experimentation
---

## 👩‍💻 Author

**Shilpa Musale** – [LinkedIn](https://www.linkedin.com/in/shilpamusale) • [GitHub](https://github.com/ishi3012) • [Portfolio](https://ishi3012.github.io/ishi-ai/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
