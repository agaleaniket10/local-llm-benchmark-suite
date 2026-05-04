"""Tests for scripts/evaluate_results.py"""

import pandas as pd
import pytest

from scripts.evaluate_results import summarize


def make_df():
    return pd.DataFrame(
        [
            {
                "run_id": "a1",
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 10.0,
                "tokens_per_second": 20.0,
            },
            {
                "run_id": "a2",
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 20.0,
                "tokens_per_second": 25.0,
            },
            {
                "run_id": "b1",
                "model": "llama3",
                "category": "coding",
                "latency_seconds": 15.0,
                "tokens_per_second": 22.0,
            },
            {
                "run_id": "c1",
                "model": "mistral",
                "category": "reasoning",
                "latency_seconds": 8.0,
                "tokens_per_second": 18.0,
            },
        ]
    )


def test_summarize_returns_dataframe():
    df = make_df()
    result = summarize(df)
    assert isinstance(result, pd.DataFrame)


def test_summarize_columns():
    df = make_df()
    result = summarize(df)
    assert "avg_latency" in result.columns
    assert "p95_latency" in result.columns
    assert "avg_tokens_per_sec" in result.columns
    assert "total_runs" in result.columns


def test_summarize_avg_latency():
    df = make_df()
    result = summarize(df)
    mistral_coding = result[
        (result["model"] == "mistral") & (result["category"] == "coding")
    ]
    assert mistral_coding["avg_latency"].values[0] == pytest.approx(15.0)


def test_summarize_total_runs():
    df = make_df()
    result = summarize(df)
    mistral_coding = result[
        (result["model"] == "mistral") & (result["category"] == "coding")
    ]
    assert mistral_coding["total_runs"].values[0] == 2


def test_summarize_empty_df():
    df = pd.DataFrame(
        columns=["run_id", "model", "category", "latency_seconds", "tokens_per_second"]
    )
    result = summarize(df)
    assert result.empty
