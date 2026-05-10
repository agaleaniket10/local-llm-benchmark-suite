"""
Tests for scripts/evaluate_results.py — main() function.
(summarize() is already covered in test_evaluate_results.py)
"""

import pandas as pd
from unittest.mock import patch

from scripts.evaluate_results import main


def _make_config(tmp_path, results_file):
    return {"benchmark": {"results_file": str(results_file)}}


def _write_results(path):
    pd.DataFrame(
        [
            {
                "run_id": "a1",
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 5.0,
                "tokens_per_second": 20.0,
            },
            {
                "run_id": "a2",
                "model": "llama3",
                "category": "coding",
                "latency_seconds": 8.0,
                "tokens_per_second": 15.0,
            },
            {
                "run_id": "a3",
                "model": "mistral",
                "category": "reasoning",
                "latency_seconds": 6.0,
                "tokens_per_second": 18.0,
            },
        ]
    ).to_csv(path, index=False)


def test_main_runs_without_error(tmp_path, capsys):
    results_file = tmp_path / "results.csv"
    _write_results(results_file)
    config = _make_config(tmp_path, results_file)
    with patch("scripts.evaluate_results.load_config", return_value=config):
        main("config.yaml")
    captured = capsys.readouterr()
    assert "Benchmark Summary" in captured.out


def test_main_missing_results_file(tmp_path):
    config = {"benchmark": {"results_file": str(tmp_path / "nonexistent.csv")}}
    with patch("scripts.evaluate_results.load_config", return_value=config):
        main("config.yaml")  # should log error and return, not raise


def test_main_empty_results_file(tmp_path):
    results_file = tmp_path / "results.csv"
    pd.DataFrame(
        columns=["run_id", "model", "category", "latency_seconds", "tokens_per_second"]
    ).to_csv(results_file, index=False)
    config = _make_config(tmp_path, results_file)
    with patch("scripts.evaluate_results.load_config", return_value=config):
        main("config.yaml")  # should log warning and return, not raise


def test_main_ranking_output(tmp_path, capsys):
    results_file = tmp_path / "results.csv"
    _write_results(results_file)
    config = _make_config(tmp_path, results_file)
    with patch("scripts.evaluate_results.load_config", return_value=config):
        main("config.yaml")
    captured = capsys.readouterr()
    assert "Ranking" in captured.out
    assert "mistral" in captured.out
