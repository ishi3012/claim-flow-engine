"""
ClaimFlowEngine Streamlit App

This module launches a Streamlit-based UI for uploading claim data,
triggering end-to-end denial prediction, root cause clustering,
and routing prioritization via the `/api/fullroute` FastAPI endpoint.

Features:
- CSV upload interface for batch claim scoring
- Real-time API integration with FastAPI backend
- Visual breakdown of routing queues (bar chart)
- UMAP projection of clustered denial reasons
- Downloadable triaged results as CSV

Intended Use:
    streamlit run streamlit_app.py

Inputs:
- Uploaded CSV file with required claim features

Outputs:
- DataFrame of triaged claims with predictions, clusters, and routing
- Charts + download options for recruiter-ready demos

Author: ClaimFlowEngine Project (2025)
"""

from typing import Any, Optional, cast

import pandas as pd
import plotly.express as px
import requests  # type: ignore
import streamlit as st

# FastAPI backend URL
API_URL: str = "http://localhost:8000/api/fullroute"

st.set_page_config(page_title="ClaimFlowEngine", layout="wide")
st.title("ClaimFlowEngine — Full Triage Dashboard")

# Tabs for layout
tabs = st.tabs(["Upload + Predict", "Routing Breakdown", "UMAP Cluster Plot"])

# --- Tab 1: Upload + Predict ---
with tabs[0]:
    uploaded_file: Optional[Any] = st.file_uploader(
        "Upload a claim CSV for triage", type=["csv"]
    )

    if uploaded_file:
        st.info("Previewing uploaded data...")
        df_input = pd.read_csv(uploaded_file)
        st.dataframe(df_input.head())

        if st.button("Run Prediction + Routing"):
            with st.spinner("Calling FastAPI backend..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv",
                    )
                }
                try:
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        results = response.json()
                        results_df: Optional[pd.DataFrame] = pd.DataFrame(results)

                        if results_df is not None and not results_df.empty:
                            if "recommended_queue" in results_df.columns:
                                st.session_state["results_df"] = results_df
                                st.success("Triage Complete!")
                                st.dataframe(results_df.head())

                                csv = results_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="⬇ Download Triaged Results",
                                    data=csv,
                                    file_name="triaged_claims.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error("Missing expected column: 'recommended_queue'")
                        else:
                            st.warning("Received empty result from API.")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Exception: {e}")

# --- Tab 2: Routing Breakdown ---
with tabs[1]:
    results_df = cast(Optional[pd.DataFrame], st.session_state.get("results_df"))
    if results_df is not None:
        st.write("🔍 Available columns:", results_df.columns.tolist())
        st.dataframe(results_df.head())

    if results_df is not None and "recommended_queue" in results_df.columns:
        st.subheader("Routed Claim Volume by Queue")
        queue_counts = results_df["recommended_queue"].value_counts().reset_index()
        queue_counts.columns = pd.Index(["Queue", "Claim Count"])
        fig = px.bar(
            queue_counts,
            x="Queue",
            y="Claim Count",
            color="Queue",
            title="Routing Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload and predict claims to see routing breakdown.")

# --- Tab 3: UMAP Cluster Plot ---
with tabs[2]:
    if results_df is not None:
        st.write("🔍 Available columns:", results_df.columns.tolist())
        st.dataframe(results_df.head())
    if results_df is not None and {"umap_x", "umap_y", "denial_cluster_id"}.issubset(
        results_df.columns
    ):

        st.subheader("UMAP Projection of Denial Clusters")
        fig_umap = px.scatter(
            results_df,
            x="umap_x",
            y="umap_y",
            color="denial_cluster_id",
            title="UMAP Projection of Clustered Denial Reasons",
            labels={
                "umap_x": "UMAP 1",
                "umap_y": "UMAP 2",
                "denial_cluster_id": "Cluster",
            },
            opacity=0.7,
        )
        st.plotly_chart(fig_umap, use_container_width=True)
    else:
        st.info("UMAP plot unavailable. Run prediction to generate cluster embeddings.")
