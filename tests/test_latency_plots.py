"""
Tests for analysis/latency_plots.py

Matplotlib rendering is mocked — no display or file I/O required for most tests.
File-save tests use tmp_path to verify output files are created.
"""

import pandas as pd
import pytest

from analysis.latency_plots import (
    plot_latency_by_model,
    plot_latency_by_category,
    plot_tokens_per_second,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
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
            {
                "model": "gemma2",
                "category": "coding",
                "latency_seconds": 9.0,
                "tokens_per_second": 14.0,
            },
            {
                "model": "gemma2",
                "category": "reasoning",
                "latency_seconds": 12.0,
                "tokens_per_second": 10.0,
            },
        ]
    )


# ---------------------------------------------------------------------------
# plot_latency_by_model
# ---------------------------------------------------------------------------


def test_plot_latency_by_model_creates_file(sample_df, tmp_path):
    plot_latency_by_model(sample_df, str(tmp_path))
    assert (tmp_path / "latency_by_model.png").exists()


def test_plot_latency_by_model_single_model(tmp_path):
    df = pd.DataFrame(
        [
            {
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 5.0,
                "tokens_per_second": 20.0,
            },
        ]
    )
    plot_latency_by_model(df, str(tmp_path))
    assert (tmp_path / "latency_by_model.png").exists()


def test_plot_latency_by_model_output_dir_created(tmp_path):
    nested = tmp_path / "a" / "b"
    # latency_plots doesn't create dirs — caller (visualize_results) does
    # so we pre-create it and verify the file lands there
    nested.mkdir(parents=True)
    plot_latency_by_model(
        pd.DataFrame(
            [
                {
                    "model": "m",
                    "category": "c",
                    "latency_seconds": 1.0,
                    "tokens_per_second": 10.0,
                }
            ]
        ),
        str(nested),
    )
    assert (nested / "latency_by_model.png").exists()


# ---------------------------------------------------------------------------
# plot_latency_by_category
# ---------------------------------------------------------------------------


def test_plot_latency_by_category_creates_file(sample_df, tmp_path):
    plot_latency_by_category(sample_df, str(tmp_path))
    assert (tmp_path / "latency_by_category.png").exists()


def test_plot_latency_by_category_single_category(tmp_path):
    df = pd.DataFrame(
        [
            {
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 5.0,
                "tokens_per_second": 20.0,
            },
            {
                "model": "llama3",
                "category": "coding",
                "latency_seconds": 7.0,
                "tokens_per_second": 18.0,
            },
        ]
    )
    plot_latency_by_category(df, str(tmp_path))
    assert (tmp_path / "latency_by_category.png").exists()


# ---------------------------------------------------------------------------
# plot_tokens_per_second
# ---------------------------------------------------------------------------


def test_plot_tokens_per_second_creates_file(sample_df, tmp_path):
    plot_tokens_per_second(sample_df, str(tmp_path))
    assert (tmp_path / "tokens_per_second.png").exists()


def test_plot_tokens_per_second_handles_nan(tmp_path):
    """NaN tokens_per_second values should be dropped without crashing."""
    df = pd.DataFrame(
        [
            {
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 5.0,
                "tokens_per_second": float("nan"),
            },
            {
                "model": "mistral",
                "category": "coding",
                "latency_seconds": 6.0,
                "tokens_per_second": 20.0,
            },
        ]
    )
    plot_tokens_per_second(df, str(tmp_path))
    assert (tmp_path / "tokens_per_second.png").exists()


def test_all_three_plots_created(sample_df, tmp_path):
    plot_latency_by_model(sample_df, str(tmp_path))
    plot_latency_by_category(sample_df, str(tmp_path))
    plot_tokens_per_second(sample_df, str(tmp_path))

    assert (tmp_path / "latency_by_model.png").exists()
    assert (tmp_path / "latency_by_category.png").exists()
    assert (tmp_path / "tokens_per_second.png").exists()
