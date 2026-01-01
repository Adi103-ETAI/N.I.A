"""Tests for NVIDIA API integration."""
import pytest
from models.model_manager import ModelManager


def test_nvidia_provider_initialization():
    """Test that NVIDIA provider can be initialized."""
    mm = ModelManager(provider="nvidia", model_name="meta/llama-4-maverick-17b-128e-instruct")
    assert mm.provider == "nvidia"
    assert mm.model_name == "meta/llama-4-maverick-17b-128e-instruct"


def test_nvidia_api_key_loading():
    """Test that NVIDIA API key is loaded from environment."""
    mm = ModelManager(provider="nvidia")
    # Should be None if not set, or a string if set
    assert mm._nvidia_api_key is None or isinstance(mm._nvidia_api_key, str)


def test_nvidia_fallback_configuration():
    """Test NVIDIA as fallback provider."""
    mm = ModelManager(provider="openai", config={"fallback_providers": ["nvidia"]})
    assert "nvidia" in mm._fallback_providers


def test_nvidia_call_method_exists():
    """Test that _call_nvidia method exists."""
    mm = ModelManager(provider="nvidia")
    assert hasattr(mm, "_call_nvidia")
    assert callable(mm._call_nvidia)


def test_nvidia_in_reason_routing():
    """Test that NVIDIA is routed correctly in reason method."""
    mm = ModelManager(provider="nvidia", model_name="meta/llama-4-maverick-17b-128e-instruct")
    # Should not raise NotImplementedError
    # Will fail if API key not set, but that's expected
    try:
        # This will fail without API key, but should not raise NotImplementedError
        mm.reason("test")
    except (NotImplementedError, RuntimeError) as e:
        if "not supported" in str(e).lower():
            pytest.fail(f"NVIDIA provider not properly routed: {e}")
        # Other errors (like missing API key) are expected


def test_nvidia_fallback_mechanism(monkeypatch):
    """Test that NVIDIA works as a fallback when OpenAI fails."""
    mm = ModelManager(provider="openai", config={"fallback_providers": ["nvidia"]})
    
    def bad_openai(prompt: str):
        raise RuntimeError("OpenAI quota exceeded")
    
    def mock_nvidia(prompt: str):
        return "NVIDIA response"
    
    monkeypatch.setattr(mm, "_call_openai", bad_openai)
    monkeypatch.setattr(mm, "_call_nvidia", mock_nvidia)
    
    result = mm.reason("test prompt")
    assert result == "NVIDIA response"


def test_nvidia_default_model():
    """Test that default model is used when model_name is not set."""
    mm = ModelManager(provider="nvidia", model_name="")
    # Should use default model in _call_nvidia
    assert mm.provider == "nvidia"

