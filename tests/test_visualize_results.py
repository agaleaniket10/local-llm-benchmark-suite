"""
Tests for analysis/visualize_results.py

All file I/O is handled via tmp_path. No Ollama or display required.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.visualize_results import main


def _write_results_csv(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 5.0,
                "tokens_per_second": 20.0,
            },
            {
                "model": "mistral",
                "category": "reasoning",
                "latency_seconds": 8.0,
                "tokens_per_second": 15.0,
            },
            {
                "model": "llama3",
                "category": "coding",
                "latency_seconds": 7.0,
                "tokens_per_second": 18.0,
            },
            {
                "model": "llama3",
                "category": "reasoning",
                "latency_seconds": 10.0,
                "tokens_per_second": 12.0,
            },
        ]
    )
    df.to_csv(path, index=False)


def _make_config(tmp_path: Path) -> dict:
    results_file = tmp_path / "results.csv"
    output_dir = tmp_path / "plots"
    _write_results_csv(results_file)
    return {
        "benchmark": {"results_file": str(results_file)},
        "analysis": {"output_dir": str(output_dir)},
    }


def test_main_creates_all_plots(tmp_path):
    config = _make_config(tmp_path)
    with patch("analysis.visualize_results.load_config", return_value=config):
        main("config.yaml")

    output_dir = Path(config["analysis"]["output_dir"])
    assert (output_dir / "latency_by_model.png").exists()
    assert (output_dir / "latency_by_category.png").exists()
    assert (output_dir / "tokens_per_second.png").exists()


def test_main_missing_results_file(tmp_path):
    config = {
        "benchmark": {"results_file": str(tmp_path / "nonexistent.csv")},
        "analysis": {"output_dir": str(tmp_path / "plots")},
    }
    with patch("analysis.visualize_results.load_config", return_value=config):
        main("config.yaml")  # should log error and return, not raise


def test_main_empty_results_file(tmp_path):
    results_file = tmp_path / "results.csv"
    pd.DataFrame(
        columns=["model", "category", "latency_seconds", "tokens_per_second"]
    ).to_csv(results_file, index=False)
    config = {
        "benchmark": {"results_file": str(results_file)},
        "analysis": {"output_dir": str(tmp_path / "plots")},
    }
    with patch("analysis.visualize_results.load_config", return_value=config):
        main("config.yaml")  # should log warning and return, not raise
