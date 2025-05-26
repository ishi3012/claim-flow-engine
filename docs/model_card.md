# Model Card: Denial Prediction Model

**Model Name:** `xgb_denial_predictor_v1`  
**Owner:** Shilpa Musale  
**Created:** May 2025  
**Version:** 1.0

---

## Overview

This model predicts whether a healthcare claim is likely to be **denied** and, if so, identifies the top 3 **likely reasons** for denial. It is used in the **ClaimFlowEngine** system to optimize revenue cycle operations, automate claim triaging, and reduce time to resolution.

---

## Intended Use

**Primary Use Case:**
- Predict denial likelihood for incoming claims
- Identify actionable denial reasons before submission
- Enable downstream clustering and routing decisions

**Target Users:**
- Revenue Cycle Management (RCM) teams
- A/R follow-up agents
- Intelligent workflow routing systems

**Operational Context:**
- Healthcare systems
- Clearinghouses
- Backend claim processors

---

## Model Details

| Aspect             | Details                                   |
|--------------------|--------------------------------------------|
| Model Type         | Gradient Boosted Trees (XGBoost)           |
| Frameworks         | scikit-learn, XGBoost                      |
| Task               | Binary classification + Top-K multi-label |
| Inputs             | 32 structured features (claims + EHR)      |
| Outputs            | Denial probability, top-3 denial reasons   |

---

## Input Features (Sample)

| Feature                    | Type       | Description                                  |
|----------------------------|------------|----------------------------------------------|
| `claim_age_days`           | Numeric    | Days between service and submission          |
| `payer_id`                 | Categorical| Unique identifier for payer                  |
| `procedure_code` (CPT)     | Categorical| Medical procedure performed                  |
| `diagnosis_code` (ICD-10)  | Categorical| Diagnosed condition                          |
| `provider_type`            | Categorical| Type of provider (e.g., Cardiologist)        |
| `historical_denial_rate`   | Numeric    | Rolling average of past denials              |

→ Full feature list available in `feature_config.yaml`.

---

## Evaluation Metrics

| Metric                | Score  |
|------------------------|--------|
| AUC (Binary Denial)    | 0.540  |
| F1-Score               | 0.567  |
| Recall                 | 0.593  |
| Accuracy               | 0.537  |
| Composite Score        | 0.553  |

> Model: **CatBoost**  
> Dataset size: 1000 claims, 21 features  
> Evaluation: 5-fold stratified cross-validation
---

## Limitations

- Trained on simulated data — performance may vary on real EHR pipelines
- Does not capture unstructured data unless explicitly embedded
- Payer behavior drift may affect model stability without retraining
- Predictions are probabilistic — HITL validation is advised before appeals

---

## Retraining Strategy

- **Trigger:** Monthly retraining or performance dip >10%
- **Data Source:** Cleaned claim logs with updated denial labels
- **Pipeline:** Automated via Vertex AI Pipelines + model registry tracking
- **Validation:** Must meet baseline AUC and F1 thresholds

---

## Ethical Considerations

- Model bias risk exists due to payer-specific denial patterns
- Interpretability is partially addressed via SHAP value analysis
- No PHI used — only de-identified metadata

---

## Author

**Shilpa Musale**  
[LinkedIn](https://www.linkedin.com/in/shilpamusale) • [GitHub](https://github.com/ishi3012) 
<!-- • [Portfolio](https://ishi3012.github.io/ishi-ai/) -->

---

## Repo Link

[ClaimFlowEngine GitHub Repository](https://github.com/ishi3012/ClaimFlowEngine)
