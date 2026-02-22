"""
Streamlit dashboard for the DataQuest Insurance Bundle Predictor.

Talks to the FastAPI backend (default: http://localhost:8000).
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DataQuest Predictor", layout="wide")
st.title("Insurance Coverage Bundle Predictor")

# ── Sidebar ──────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["Predict", "Model Info", "Metrics"])

# ── /predict ─────────────────────────────────────────────────────────────────
if page == "Predict":
    st.header("Run Predictions")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="predict")

    if uploaded and st.button("Predict"):
        with st.spinner("Running predictions…"):
            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            )

        if resp.ok:
            result = pd.read_csv(io.StringIO(resp.text))
            latency = resp.headers.get("X-Prediction-Time-Seconds", "?")
            rows = resp.headers.get("X-Rows-Processed", "?")

            st.success(f"Done — {rows} rows predicted in {latency}s")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download predictions.csv",
                data=resp.content,
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")


# ── /info ────────────────────────────────────────────────────────────────────
elif page == "Model Info":
    st.header("Model Information")

    if st.button("Load Info"):
        with st.spinner("Fetching…"):
            resp = requests.get(f"{API_URL}/info")

        if resp.ok:
            info = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Type", info.get("model_type", "N/A"))
            col2.metric("Size (MB)", info.get("model_size_mb", "N/A"))
            col3.metric("Trees", info.get("n_estimators", "N/A"))

            if "feature_names" in info:
                st.subheader("Features")
                st.write(info["feature_names"])

            if "classes" in info:
                st.subheader("Target Classes")
                st.write(info["classes"])

        else:
            st.error(f"Error {resp.status_code}: {resp.text}")

# ── /metrics ─────────────────────────────────────────────────────────────────
elif page == "Metrics":
    st.header("Model Performance Metrics")

    if st.button("Compute Metrics"):
        with st.spinner("Running evaluation on full training set…"):
            resp = requests.get(f"{API_URL}/metrics")

        if resp.ok:
            m = resp.json()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Macro F1", m["macro_f1"])
            col2.metric("Final Score", m["final_score"])
            col3.metric("Size Penalty", m["size_penalty"])
            col4.metric("Time Penalty", m["time_penalty"])

            st.divider()

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", m["accuracy"])
            col2.metric("Model Size (MB)", m["model_size_mb"])
            col3.metric("Prediction Time (s)", m["prediction_time_s"])

            st.subheader("Per-Class Report")
            report = m.get("per_class_report", {})
            # Filter to only class rows
            class_rows = {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")}
            st.dataframe(pd.DataFrame(class_rows).T, use_container_width=True)
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
