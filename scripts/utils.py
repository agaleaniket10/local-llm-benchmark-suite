"""
utils.py — Shared utility functions for the benchmark suite.
"""

import json
import time
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> list[dict]:
    """Load prompts from a JSON file."""
    with open(prompts_path, "r") as f:
        return json.load(f)


def timestamp() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
