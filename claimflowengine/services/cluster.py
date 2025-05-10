"""
Module: cluster.py

Description:
    Assigns denied claims to latent denial clusters using text embeddings
    and a pre-trained clustering model. This module is config-driven and
    includes logging support for traceability.

Features:
    - Loads vectorizer and clustering model from config-defined paths
    - Predicts cluster ID from denial reason text
    - Maps cluster ID to human-readable label

Intended Use:
    Used by both FastAPI and ADK agent to support root cause clustering.

Inputs:
    - denial_reason (str): Free-text reason from claim or payer

Outputs:
    - dict: {
        "cluster_id": int,
        "cluster_label": str
      }

Author: ClaimFlowEngine Team
"""
