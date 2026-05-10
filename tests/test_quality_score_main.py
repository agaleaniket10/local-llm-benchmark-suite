"""
Tests for scripts/quality_score.py — main() function.
(call_judge() is already covered in test_quality_score.py)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.quality_score import main


VALID_SCORES = {
    "accuracy": 4,
    "relevance": 5,
    "clarity": 4,
    "completeness": 3,
    "conciseness": 4,
}


def _mock_judge_response(scores: dict) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"response": json.dumps(scores)}
    return mock


def _write_results(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "run_id": "a1",
                "model": "mistral",
                "prompt_id": "p1",
                "category": "coding",
                "response": "def hello(): pass",
            },
            {
                "run_id": "a2",
                "model": "llama3",
                "prompt_id": "p2",
                "category": "reasoning",
                "response": "The answer is 42",
            },
        ]
    ).to_csv(path, index=False)


def _write_prompts(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"id": "p1", "category": "coding", "prompt": "Write hello world"},
                {"id": "p2", "category": "reasoning", "prompt": "What is 6x7?"},
            ]
        )
    )


def _make_config(tmp_path: Path) -> dict:
    results_file = tmp_path / "results.csv"
    prompt_file = tmp_path / "prompts.json"
    _write_results(results_file)
    _write_prompts(prompt_file)
    return {
        "benchmark": {
            "results_file": str(results_file),
            "prompt_file": str(prompt_file),
        },
        "ollama": {"host": "http://localhost:11434"},
    }


@patch("scripts.quality_score.requests.post")
def test_main_writes_quality_scores(mock_post, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    mock_post.return_value = _mock_judge_response(VALID_SCORES)
    config = _make_config(tmp_path)

    with patch("scripts.quality_score.load_config", return_value=config):
        main("config.yaml", "mistral:latest")

    scores_file = tmp_path / "data" / "quality_scores.csv"
    assert scores_file.exists()
    df = pd.read_csv(scores_file)
    assert len(df) == 2
    assert "total" in df.columns
    assert df["total"].iloc[0] == sum(VALID_SCORES.values())


@patch("scripts.quality_score.requests.post")
def test_main_missing_results_file(mock_post, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt_file = tmp_path / "prompts.json"
    _write_prompts(prompt_file)
    config = {
        "benchmark": {
            "results_file": str(tmp_path / "nonexistent.csv"),
            "prompt_file": str(prompt_file),
        },
        "ollama": {"host": "http://localhost:11434"},
    }
    with patch("scripts.quality_score.load_config", return_value=config):
        main("config.yaml", "mistral:latest")  # should log error, not raise


@patch("scripts.quality_score.requests.post")
def test_main_skips_empty_responses(mock_post, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    mock_post.return_value = _mock_judge_response(VALID_SCORES)

    results_file = tmp_path / "results.csv"
    prompt_file = tmp_path / "prompts.json"
    _write_prompts(prompt_file)

    # Include one row with empty response
    pd.DataFrame(
        [
            {
                "run_id": "a1",
                "model": "mistral",
                "prompt_id": "p1",
                "category": "coding",
                "response": "",
            },
            {
                "run_id": "a2",
                "model": "mistral",
                "prompt_id": "p2",
                "category": "reasoning",
                "response": "The answer is 42",
            },
        ]
    ).to_csv(results_file, index=False)

    config = {
        "benchmark": {
            "results_file": str(results_file),
            "prompt_file": str(prompt_file),
        },
        "ollama": {"host": "http://localhost:11434"},
    }
    with patch("scripts.quality_score.load_config", return_value=config):
        main("config.yaml", "mistral:latest")

    scores_file = tmp_path / "data" / "quality_scores.csv"
    df = pd.read_csv(scores_file)
    assert len(df) == 1  # only the non-empty response was scored
