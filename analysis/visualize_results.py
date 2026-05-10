"""
visualize_results.py — Generate all analysis plots from benchmark results.

Usage (run from project root):
    python -m analysis.visualize_results
    python -m analysis.visualize_results --config config.yaml
"""

import argparse

import pandas as pd

from analysis.latency_plots import (
    plot_latency_by_model,
    plot_latency_by_category,
    plot_tokens_per_second,
)
from scripts.utils import get_logger, load_config, ensure_dir

logger = get_logger(__name__)


def main(config_path: str) -> None:
    config = load_config(config_path)
    results_file = config["benchmark"]["results_file"]
    output_dir = config["analysis"]["output_dir"]

    ensure_dir(output_dir)

    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        logger.error(
            "Results file not found: %s — run run_benchmark.py first.", results_file
        )
        return

    if df.empty:
        logger.warning("Results file is empty.")
        return

    logger.info("Generating plots from %d rows...", len(df))
    plot_latency_by_model(df, output_dir)
    plot_latency_by_category(df, output_dir)
    plot_tokens_per_second(df, output_dir)
    logger.info("All plots saved to %s/", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
