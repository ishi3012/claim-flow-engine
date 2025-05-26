# Model Card: Root Cause Clustering (Denial Reason Clustering)

**Component Name:** `cluster.py`  
**Owner:** Shilpa Musale  
**Created:** May 2025  
**Version:** 1.0

---

## Overview

This module groups denied healthcare claims into semantically meaningful **root cause clusters** using NLP embeddings and unsupervised learning. It enables claims with similar denial patterns to be routed efficiently and explained with natural clusters such as **"Coding Error"**, **"Eligibility Issue"**, or **"Missing Prior Authorization."**

---

## Intended Use

**Primary Use Case:**
- Discover patterns in unstructured denial reasons and follow-up notes
- Enable denial-aware claim routing by cluster ID
- Facilitate downstream appeal automation and analytics

**Target Users:**
- Denial management and A/R teams
- ML routing agents
- Auditors and analytics dashboards

**Integration Point:**
- Outputs cluster ID and semantic label per denied claim
- Used by the `policy.py` routing engine to adjust priority or routing team

---

## Clustering Pipeline

| Step                   | Method Used                           |
|------------------------|----------------------------------------|
| Text Input             | `denial_reason + followup_notes`       |
| Embedding              | `Sentence-BERT (all-MiniLM-L6-v2)`     |
| Dimensionality Reduction | `UMAP (n_components=5)`               |
| Clustering Algorithm   | `HDBSCAN (min_cluster_size=15)`        |
| Output                 | Cluster label ID + semantic label      |

**Outlier Handling:** HDBSCAN labels noisy samples as `-1`. These are handled separately in downstream routing.

**Embeddings Shape:**  
- Raw: 384 dimensions (S-BERT)  
- Reduced: 5 dimensions (UMAP)  

---

## Clustering Summary

- Dataset: 1000 claims (denied subset)
- Features used: Unstructured denial text + select structured fields
- Clusters Detected: 2 stable clusters (others labeled as noise)
- Semantic Labels: `"auth_required"`, `"eligibility_mismatch"` (stored in `cluster_labels.json`)

_No hard labels were manually assigned. Semantic labels were derived from cluster centroids + representative examples._

---

## Sample Input & Output

**Input Text:**
```text
"Authorization not found for billed service."
"Service deemed not medically necessary by payer."
```

**Output:**
```json
{
  "cluster_id": 1,
  "semantic_label": "auth_required"
}
```

---

## Observations

- UMAP + HDBSCAN captured non-linear semantic separations in denial patterns
- Clustering quality validated visually via UMAP 2D plots
- Full pipeline logged in [`cluster.log`](cluster.log)

---

## Limitations

- Sensitive to embedding noise if denial reasons are poorly formatted
- `Sentence-BERT` does not account for domain-specific jargon (option to swap with `ClinicalBERT`)
- Cluster cardinality may fluctuate across payers or specialties

---

## Maintenance Strategy

- Update embeddings monthly with new denial text samples
- Tune UMAP & HDBSCAN parameters every 6 months or on drift detection
- Evaluate with Silhouette Score or NMI when labels are available

---

## Author

**Shilpa Musale**  
[LinkedIn](https://www.linkedin.com/in/shilpamusale) • [GitHub](https://github.com/ishi3012) 
<!-- • [Portfolio](https://ishi3012.github.io/ishi-ai/) -->

---

## Repo Link

[ClaimFlowEngine GitHub Repository](https://github.com/ishi3012/ClaimFlowEngine)
