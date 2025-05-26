# Model Card: Routing Policy Engine

**Component Name:** `policy.py`  
**Owner:** Shilpa Musale  
**Created:** May 2025  
**Version:** 1.0

---

## Overview

The **Routing Policy Engine** is a configurable, rule-driven scoring module that assigns **priority scores** and **recommended queues** to denied healthcare claims. Inspired by reinforcement learning principles (contextual reward scoring), this component is crucial for **A/R team triage**, ensuring complex claims are routed to the most skilled units for resolution.

---

## Intended Use

**Primary Use Case:**
- Assign priority to denied claims based on structured + semantic features
- Route claims to the appropriate follow-up team based on denial cluster
- Feed into FastAPI for real-time triage or used in batch pipelines

**Target Users:**
- Revenue Cycle Management teams
- A/R follow-up agents
- Claim appeal systems

**Integration Points:**
- Consumes predictions from `denial_predictor.py` and clusters from `cluster.py`
- Returns: `priority_score`, `recommended_queue`, and debug trace

---

## Scoring Policy Logic

| Signal                    | Contribution Logic                                           |
|---------------------------|--------------------------------------------------------------|
| Denial Prediction         | Adds fixed penalty (e.g., `+2.0` if `True`)                  |
| Denial Cluster            | Weighted by cluster ID (`auth_required`: `+1.5`, etc.)       |
| Claim Age (in days)       | `claim_age_days * 0.02`                                      |
| Payer Behavior            | Payers with slow payment history get bonus (`P001`: `+1.5`)  |
| Complex CPT Codes         | Adds additional weight (`99285`: `+2.0`, etc.)               |

These weights are defined in [`routing_config.yaml`](routing_config.yaml) and fallback to `MOCK_WEIGHTS` if config is missing.

---

## Queue Assignment Logic

| Denial Cluster    | Assigned Team |
|-------------------|---------------|
| `auth_required`   | `TeamA`       |
| `expired_coverage`| `TeamA`       |
| `coding_error`    | `TeamB`       |
| `duplicate_claim` | `TeamC`       |
| Unmatched clusters| `DefaultQueue`|

Rules are defined in `team_rules:` block of YAML config.

---

## Summary Statistics

- Claims Routed: **23**
- All routed to: **`DefaultQueue`**
- Suggests cluster labels were missing or unmatched in team rules

| Metric           | Value  |
|------------------|--------|
| Min Score        | ~      |
| Max Score        | ~      |
| Mean Score       | ~      |

> *See routed_claims.csv for full breakdown.*

---

## Example

**Input:**
```json
{
  "denial_prediction": 1,
  "denial_cluster_id": "auth_required",
  "claim_submission_date": "2024-12-01",
  "last_followup_date": "2025-01-15",
  "payer_id": "P001",
  "CPT_codes": "99285"
}
```

**Output:**
```json
{
  "priority_score": 7.1,
  "recommended_queue": "TeamA",
  "debug_notes": [
    "Denied: +2.0",
    "Cluster 'auth_required': +1.5",
    "Claim age 45d: +0.90",
    "Payer 'P001': +1.5",
    "CPT 99285: +2.0"
  ]
}
```

---

## Architecture

```mermaid
flowchart TD
    A[Denial Prediction] --> B[Clustering Output]
    B --> C[PolicyEngine (score_claim)]
    C --> D[Assign Queue (TeamA/B/C)]
    D --> E[priority_score + queue]
```

---

## Limitations

- Fallbacks to `DefaultQueue` if cluster ID is missing/unrecognized
- Requires tight sync with `cluster_labels.json`
- Currently rule-based; no RL agent is learning policy weights dynamically

---

## Maintenance Strategy

- Update YAML weights quarterly with business feedback
- Integrate SHAP or interpretable scores for appeal prioritization
- Option to evolve to a contextual bandit policy (e.g., LinUCB)

---

## Author

**Shilpa Musale**  
[LinkedIn](https://www.linkedin.com/in/shilpamusale) • [GitHub](https://github.com/ishi3012) 
<!-- • [Portfolio](https://ishi3012.github.io/ishi-ai/) -->

---

## Repo Link

[ClaimFlowEngine GitHub Repository](https://github.com/ishi3012/ClaimFlowEngine)
