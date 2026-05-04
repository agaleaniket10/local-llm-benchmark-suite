"""
utils.py — Shared utility functions for the benchmark suite.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Logging ────────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Config & data loading ──────────────────────────────────────────────────────


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> list[dict]:
    """Load prompts from a JSON file."""
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    with open(path) as f:
        return json.load(f)


def load_prompts_as_dict(prompts_path: str) -> dict[str, str]:
    """Return a dict of prompt_id -> prompt text."""
    return {p["id"]: p["prompt"] for p in load_prompts(prompts_path)}


# ── Misc helpers ───────────────────────────────────────────────────────────────


def timestamp() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
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
