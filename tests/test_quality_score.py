"""Tests for quality scoring logic in scripts/quality_score.py"""

import json
from unittest.mock import MagicMock, patch

from scripts.quality_score import call_judge, SCORE_KEYS


VALID_SCORES = {
    "accuracy": 4,
    "relevance": 5,
    "clarity": 4,
    "completeness": 3,
    "conciseness": 4,
}
VALID_JSON = json.dumps(VALID_SCORES)


def _mock_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"response": text}
    mock.raise_for_status.return_value = None
    return mock


# ── call_judge ─────────────────────────────────────────────────────────────────


@patch("scripts.quality_score.requests.post")
def test_call_judge_valid_response(mock_post):
    mock_post.return_value = _mock_response(VALID_JSON)
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result == VALID_SCORES


@patch("scripts.quality_score.requests.post")
def test_call_judge_with_surrounding_text(mock_post):
    """Judge sometimes wraps JSON in explanation text — should still parse."""
    mock_post.return_value = _mock_response(
        f"Here is my evaluation:\n{VALID_JSON}\nDone."
    )
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result is not None
    assert result["accuracy"] == 4


@patch("scripts.quality_score.requests.post")
def test_call_judge_invalid_json(mock_post):
    mock_post.return_value = _mock_response("I cannot score this response.")
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result is None


@patch("scripts.quality_score.requests.post")
def test_call_judge_out_of_range_score(mock_post):
    bad = json.dumps(
        {
            "accuracy": 6,
            "relevance": 5,
            "clarity": 4,
            "completeness": 3,
            "conciseness": 4,
        }
    )
    mock_post.return_value = _mock_response(bad)
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result is None


@patch("scripts.quality_score.requests.post")
def test_call_judge_missing_key(mock_post):
    incomplete = json.dumps({"accuracy": 4, "relevance": 5})  # missing 3 keys
    mock_post.return_value = _mock_response(incomplete)
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result is None


@patch("scripts.quality_score.requests.post")
def test_call_judge_network_error(mock_post):
    import requests

    mock_post.side_effect = requests.exceptions.ConnectionError("refused")
    result = call_judge(
        "http://localhost:11434", "mistral:latest", "prompt", "response"
    )
    assert result is None


# ── score keys ─────────────────────────────────────────────────────────────────


def test_score_keys_complete():
    assert set(SCORE_KEYS) == {
        "accuracy",
        "relevance",
        "clarity",
        "completeness",
        "conciseness",
    }
