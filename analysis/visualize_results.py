"""
visualize_results.py — Generate all analysis plots from benchmark results.

Usage:
    python analysis/visualize_results.py
    python analysis/visualize_results.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from analysis.latency_plots import (
    plot_latency_by_model,
    plot_latency_by_category,
    plot_tokens_per_second,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    config = load_config(config_path)
    results_file = config["benchmark"]["results_file"]
    output_dir = config["analysis"]["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_file)
    if df.empty:
        print("⚠️  No results found. Run run_benchmark.py first.")
        return

    print("📈 Generating plots...")
    plot_latency_by_model(df, output_dir)
    plot_latency_by_category(df, output_dir)
    plot_tokens_per_second(df, output_dir)
    print(f"✅ Plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
