"""
Tests for scripts/run_benchmark.py

All Ollama HTTP calls are mocked — no running Ollama instance required.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.run_benchmark import run_single, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ollama_response(
    tokens: int = 50, response_text: str = "test response"
) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "response": response_text,
        "eval_count": tokens,
    }
    return mock


def _make_config(tmp_path: Path, runs: int = 1) -> dict:
    prompts = [
        {"id": "p1", "category": "coding", "prompt": "Write hello world"},
        {"id": "p2", "category": "reasoning", "prompt": "Solve 2+2"},
    ]
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(prompts))

    results_file = tmp_path / "results.csv"

    return {
        "models": [{"name": "mistral", "tag": "mistral:latest"}],
        "benchmark": {
            "prompt_file": str(prompt_file),
            "results_file": str(results_file),
            "runs_per_prompt": runs,
            "timeout_seconds": 30,
        },
        "ollama": {"host": "http://localhost:11434"},
    }


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------


@patch("scripts.run_benchmark.requests.post")
def test_run_single_returns_expected_keys(mock_post):
    mock_post.return_value = _mock_ollama_response(tokens=40)
    result = run_single("http://localhost:11434", "mistral:latest", "hello", timeout=30)
    assert "latency_seconds" in result
    assert "tokens_generated" in result
    assert "tokens_per_second" in result
    assert "response" in result


@patch("scripts.run_benchmark.requests.post")
def test_run_single_tokens_per_second_calculated(mock_post):
    mock_post.return_value = _mock_ollama_response(tokens=100)
    result = run_single("http://localhost:11434", "mistral:latest", "hello", timeout=30)
    assert result["tokens_generated"] == 100
    assert result["tokens_per_second"] > 0


@patch("scripts.run_benchmark.requests.post")
def test_run_single_strips_newlines_from_response(mock_post):
    mock_post.return_value = _mock_ollama_response(response_text="line1\nline2\nline3")
    result = run_single("http://localhost:11434", "mistral:latest", "hello", timeout=30)
    assert "\n" not in result["response"]


@patch("scripts.run_benchmark.requests.post")
def test_run_single_raises_on_ollama_error(mock_post):
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"error": "model not found"}
    mock_post.return_value = mock
    with pytest.raises(RuntimeError, match="Ollama error"):
        run_single("http://localhost:11434", "bad-model", "hello", timeout=30)


@patch("scripts.run_benchmark.requests.post")
def test_run_single_raises_on_http_error(mock_post):
    mock = MagicMock()
    mock.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    mock_post.return_value = mock
    with pytest.raises(requests.exceptions.HTTPError):
        run_single("http://localhost:11434", "mistral:latest", "hello", timeout=30)


@patch("scripts.run_benchmark.requests.post")
def test_run_single_zero_tokens_does_not_divide_by_zero(mock_post):
    """Edge case: model returns 0 tokens — tokens_per_second should be 0, not crash."""
    mock_post.return_value = _mock_ollama_response(tokens=0)
    result = run_single("http://localhost:11434", "mistral:latest", "hello", timeout=30)
    assert result["tokens_per_second"] == 0


# ---------------------------------------------------------------------------
# main (integration-style, fully mocked)
# ---------------------------------------------------------------------------


@patch("scripts.run_benchmark.requests.post")
def test_main_writes_csv(mock_post, tmp_path):
    mock_post.return_value = _mock_ollama_response(tokens=30)
    config = _make_config(tmp_path, runs=1)

    with patch("scripts.run_benchmark.load_config", return_value=config):
        main("config.yaml")

    results_file = Path(config["benchmark"]["results_file"])
    assert results_file.exists()

    with open(results_file) as f:
        rows = list(csv.DictReader(f))

    # 1 model × 2 prompts × 1 run = 2 rows
    assert len(rows) == 2
    assert rows[0]["model"] == "mistral"
    assert "latency_seconds" in rows[0]


@patch("scripts.run_benchmark.requests.post")
def test_main_multiple_runs_writes_correct_row_count(mock_post, tmp_path):
    mock_post.return_value = _mock_ollama_response(tokens=30)
    config = _make_config(tmp_path, runs=3)

    with patch("scripts.run_benchmark.load_config", return_value=config):
        main("config.yaml")

    results_file = Path(config["benchmark"]["results_file"])
    with open(results_file) as f:
        rows = list(csv.DictReader(f))

    # 1 model × 2 prompts × 3 runs = 6 rows
    assert len(rows) == 6


@patch("scripts.run_benchmark.requests.post")
def test_main_handles_timeout_gracefully(mock_post, tmp_path):
    """Timeouts should be logged and skipped, not crash the benchmark."""
    mock_post.side_effect = requests.exceptions.Timeout
    config = _make_config(tmp_path, runs=1)

    with patch("scripts.run_benchmark.load_config", return_value=config):
        main("config.yaml")  # should not raise

    results_file = Path(config["benchmark"]["results_file"])
    with open(results_file) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0  # all timed out, no rows written


@patch("scripts.run_benchmark.requests.post")
def test_main_csv_has_all_required_columns(mock_post, tmp_path):
    mock_post.return_value = _mock_ollama_response(tokens=20)
    config = _make_config(tmp_path, runs=1)

    with patch("scripts.run_benchmark.load_config", return_value=config):
        main("config.yaml")

    results_file = Path(config["benchmark"]["results_file"])
    with open(results_file) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    expected = {
        "run_id",
        "model",
        "prompt_id",
        "category",
        "latency_seconds",
        "tokens_generated",
        "tokens_per_second",
        "response",
        "timestamp",
    }
    assert expected.issubset(set(fieldnames))
