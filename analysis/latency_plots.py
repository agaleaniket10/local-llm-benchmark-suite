"""
latency_plots.py — Latency visualization helpers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_latency_by_model(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart of average latency per model."""
    summary = df.groupby("model")["latency_seconds"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary.index, summary.values, color=["#4C72B0", "#DD8452", "#55A868"]
    )
    ax.bar_label(bars, fmt="%.2fs", padding=4)
    ax.set_title("Average Inference Latency by Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (seconds)")
    ax.set_ylim(0, summary.max() * 1.25)
    plt.tight_layout()

    out_path = Path(output_dir) / "latency_by_model.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_latency_by_category(df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart of average latency per model × category."""
    pivot = df.groupby(["category", "model"])["latency_seconds"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_title(
        "Average Latency by Category and Model", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Prompt Category")
    ax.set_ylabel("Latency (seconds)")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = Path(output_dir) / "latency_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_tokens_per_second(df: pd.DataFrame, output_dir: str) -> None:
    """Box plot of tokens/second distribution per model."""
    models = df["model"].unique()
    data = [df[df["model"] == m]["tokens_per_second"].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=models, patch_artist=True)
    ax.set_title(
        "Tokens per Second Distribution by Model", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Tokens / second")
    plt.tight_layout()

    out_path = Path(output_dir) / "tokens_per_second.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
