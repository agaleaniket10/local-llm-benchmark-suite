"""
app.py — Optional Streamlit dashboard for exploring benchmark results.

Usage (run from project root):
    pip install streamlit
    streamlit run dashboard/app.py
"""

import pandas as pd
import streamlit as st

from scripts.utils import load_config

st.set_page_config(page_title="LLM Benchmark Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 Local LLM Benchmark Suite")
st.caption("Offline comparison of Mistral, Llama 3.1, and Gemma 2")

# ── Load data ──────────────────────────────────────────────────────────────────
config = load_config("config.yaml")
results_path = config["benchmark"]["results_file"]

try:
    df = pd.read_csv(results_path)
except FileNotFoundError:
    st.error(f"Results file not found: `{results_path}`. Run `run_benchmark.py` first.")
    st.stop()

if df.empty:
    st.warning("Results file is empty. Run the benchmark first.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
models = st.sidebar.multiselect(
    "Models", df["model"].unique(), default=list(df["model"].unique())
)
categories = st.sidebar.multiselect(
    "Categories", df["category"].unique(), default=list(df["category"].unique())
)

filtered = df[df["model"].isin(models) & df["category"].isin(categories)]

# ── KPI row ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total Runs", len(filtered))
col2.metric("Avg Latency", f"{filtered['latency_seconds'].mean():.2f}s")
col3.metric("Avg Tokens/s", f"{filtered['tokens_per_second'].mean():.1f}")

st.divider()

# ── Latency chart ──────────────────────────────────────────────────────────────
st.subheader("Average Latency by Model")
latency_summary = filtered.groupby("model")["latency_seconds"].mean().sort_values()
st.bar_chart(latency_summary)

# ── Category breakdown ─────────────────────────────────────────────────────────
st.subheader("Latency by Category × Model")
pivot = filtered.groupby(["category", "model"])["latency_seconds"].mean().unstack()
st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

# ── Raw results ────────────────────────────────────────────────────────────────
with st.expander("Raw Results"):
    st.dataframe(filtered, use_container_width=True)
