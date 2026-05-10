"""
Tests for models/model_config.py
"""

import pytest

from models.model_config import ModelConfig, MODELS, get_model


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


def test_model_config_required_fields():
    m = ModelConfig(name="test", tag="test:latest", parameters="7B")
    assert m.name == "test"
    assert m.tag == "test:latest"
    assert m.parameters == "7B"


def test_model_config_defaults():
    m = ModelConfig(name="test", tag="test:latest", parameters="7B")
    assert m.context_length == 4096
    assert m.notes == ""


def test_model_config_custom_context():
    m = ModelConfig(
        name="llama", tag="llama:latest", parameters="8B", context_length=128000
    )
    assert m.context_length == 128000


# ---------------------------------------------------------------------------
# MODELS registry
# ---------------------------------------------------------------------------


def test_models_registry_not_empty():
    assert len(MODELS) > 0


def test_models_registry_has_required_fields():
    for m in MODELS:
        assert m.name, f"Model missing name: {m}"
        assert m.tag, f"Model missing tag: {m}"
        assert m.parameters, f"Model missing parameters: {m}"
        assert m.context_length > 0, f"Model has invalid context_length: {m}"


def test_models_registry_unique_names():
    names = [m.name for m in MODELS]
    assert len(names) == len(set(names)), "Duplicate model names in registry"


def test_models_registry_unique_tags():
    tags = [m.tag for m in MODELS]
    assert len(tags) == len(set(tags)), "Duplicate model tags in registry"


def test_models_registry_contains_expected_models():
    names = {m.name for m in MODELS}
    assert "mistral" in names
    assert "llama3" in names
    assert "gemma2" in names


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


def test_get_model_returns_correct_model():
    m = get_model("mistral")
    assert m.name == "mistral"
    assert "mistral" in m.tag.lower()


def test_get_model_all_registered_models():
    for model in MODELS:
        result = get_model(model.name)
        assert result.name == model.name


def test_get_model_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent-model")


def test_get_model_error_message_lists_available():
    with pytest.raises(ValueError) as exc_info:
        get_model("gpt-4")
    assert "mistral" in str(exc_info.value) or "Available" in str(exc_info.value)
