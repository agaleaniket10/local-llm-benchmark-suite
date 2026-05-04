"""
evaluate_results.py — Compute summary statistics from benchmark results.

Usage:
    python scripts/evaluate_results.py
    python scripts/evaluate_results.py --config config.yaml
"""

import argparse

import pandas as pd

from scripts.utils import get_logger, load_config

logger = get_logger(__name__)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model, per-category summary statistics."""
    return (
        df.groupby(["model", "category"])
        .agg(
            avg_latency=("latency_seconds", "mean"),
            p95_latency=("latency_seconds", lambda x: x.quantile(0.95)),
            avg_tokens_per_sec=("tokens_per_second", "mean"),
            total_runs=("run_id", "count"),
        )
        .round(3)
        .reset_index()
    )


def main(config_path: str) -> None:
    config = load_config(config_path)
    results_file = config["benchmark"]["results_file"]

    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        logger.error(
            "Results file not found: %s — run run_benchmark.py first.", results_file
        )
        return

    if df.empty:
        logger.warning("Results file is empty. Run run_benchmark.py first.")
        return

    logger.info("Loaded %d rows from %s", len(df), results_file)

    summary = summarize(df)
    print("\n📊 Benchmark Summary\n")
    print(summary.to_string(index=False))

    ranking = (
        df.groupby("model")["latency_seconds"]
        .mean()
        .sort_values()
        .reset_index()
        .rename(columns={"latency_seconds": "avg_latency_seconds"})
    )
    print("\n🏆 Model Ranking (fastest → slowest)\n")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
