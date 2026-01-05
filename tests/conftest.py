"""Pytest configuration and fixtures for NIA tests.

Provides:
- Path setup for imports
- Mock fixtures for external APIs
- Temporary directory fixtures
"""
import sys
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_nvidia_api():
    """Mock NVIDIA API calls."""
    with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Mocked NVIDIA response")
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API calls."""
    with patch("langchain_openai.ChatOpenAI") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Mocked OpenAI response")
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def clean_env():
    """Provide a clean environment without API keys."""
    env_vars = ["NVIDIA_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "HUGGINGFACE_API_KEY"]
    original = {k: os.environ.get(k) for k in env_vars}
    
    for key in env_vars:
        if key in os.environ:
            del os.environ[key]
    
    yield
    
    # Restore
    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
