"""Tests for scripts/utils.py"""

import json
from pathlib import Path

import pytest
import yaml

from scripts.utils import (
    load_config,
    load_prompts,
    load_prompts_as_dict,
    timestamp,
    ensure_dir,
    format_duration,
    get_logger,
)


# ── load_config ────────────────────────────────────────────────────────────────


def test_load_config_valid(tmp_path):
    cfg = {
        "models": [{"name": "mistral", "tag": "mistral:latest"}],
        "benchmark": {"runs_per_prompt": 1},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg))
    result = load_config(str(config_file))
    assert result["benchmark"]["runs_per_prompt"] == 1
    assert result["models"][0]["name"] == "mistral"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


# ── load_prompts ───────────────────────────────────────────────────────────────


def test_load_prompts_valid(tmp_path):
    prompts = [{"id": "p1", "category": "coding", "prompt": "Write hello world"}]
    p_file = tmp_path / "prompts.json"
    p_file.write_text(json.dumps(prompts))
    result = load_prompts(str(p_file))
    assert len(result) == 1
    assert result[0]["id"] == "p1"


def test_load_prompts_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prompts("nonexistent_prompts.json")


def test_load_prompts_as_dict(tmp_path):
    prompts = [
        {"id": "p1", "category": "coding", "prompt": "Write hello world"},
        {"id": "p2", "category": "reasoning", "prompt": "Solve 2+2"},
    ]
    p_file = tmp_path / "prompts.json"
    p_file.write_text(json.dumps(prompts))
    result = load_prompts_as_dict(str(p_file))
    assert result["p1"] == "Write hello world"
    assert result["p2"] == "Solve 2+2"


# ── timestamp ──────────────────────────────────────────────────────────────────


def test_timestamp_format():
    ts = timestamp()
    assert "T" in ts  # ISO 8601 format
    assert "+" in ts or "Z" in ts  # timezone info present


# ── ensure_dir ─────────────────────────────────────────────────────────────────


def test_ensure_dir_creates_nested(tmp_path):
    target = str(tmp_path / "a" / "b" / "c")
    ensure_dir(target)
    assert Path(target).is_dir()


def test_ensure_dir_idempotent(tmp_path):
    target = str(tmp_path / "existing")
    ensure_dir(target)
    ensure_dir(target)  # should not raise
    assert Path(target).is_dir()


# ── format_duration ────────────────────────────────────────────────────────────


def test_format_duration_seconds():
    assert format_duration(5.5) == "5.50s"


def test_format_duration_minutes():
    result = format_duration(90.0)
    assert result == "1m 30.0s"


def test_format_duration_zero():
    assert format_duration(0.0) == "0.00s"


# ── get_logger ─────────────────────────────────────────────────────────────────


def test_get_logger_returns_logger():
    import logging

    log = get_logger("test.module")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.module"


def test_get_logger_no_duplicate_handlers():
    get_logger("same.name")
    log2 = get_logger("same.name")
    assert len(log2.handlers) == 1  # should not double-add handlers
