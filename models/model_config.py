"""
model_config.py — Model metadata and helper utilities.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str  # short identifier used in results (e.g. "mistral")
    tag: str  # Ollama model tag (e.g. "mistral:7b-instruct")
    parameters: str  # human-readable param count (e.g. "7B")
    context_length: int = 4096
    notes: str = ""


# Registry of models used in the benchmark
MODELS: list[ModelConfig] = [
    ModelConfig(
        name="mistral",
        tag="mistral:latest",
        parameters="7B",
        context_length=8192,
        notes="Strong general-purpose instruct model from Mistral AI.",
    ),
    ModelConfig(
        name="llama3",
        tag="llama3.1:latest",
        parameters="8B",
        context_length=128000,
        notes="Meta's Llama 3.1 with a very large context window.",
    ),
    ModelConfig(
        name="gemma2",
        tag="gemma2:latest",
        parameters="9B",
        context_length=8192,
        notes="Google's Gemma 2 — competitive quality at 9B scale.",
    ),
]


def get_model(name: str) -> ModelConfig:
    """Look up a ModelConfig by its short name. Raises ValueError if not found."""
    for m in MODELS:
        if m.name == name:
            return m
    raise ValueError(f"Unknown model: '{name}'. Available: {[m.name for m in MODELS]}")
