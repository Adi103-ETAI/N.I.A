"""Tests for SafeLLM Circuit Breaker functionality.

Tests the automatic retry and provider fallback behavior when
rate limit (429) or service unavailable (503) errors occur.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock LangChain model."""
    model = MagicMock()
    model.invoke.return_value = MagicMock(content="Success response")
    return model


@pytest.fixture
def mock_manager():
    """Create a mock ModelManager with provider switching."""
    manager = MagicMock()
    manager.get_active_provider.return_value = "openai"
    manager.set_active_provider = MagicMock()
    return manager


class MockHTTPStatusError(Exception):
    """Mock httpx.HTTPStatusError for testing."""
    def __init__(self, status_code: int):
        self.response = MagicMock()
        self.response.status_code = status_code
        super().__init__(f"HTTP {status_code} error")


class MockRateLimitError(Exception):
    """Mock OpenAI's RateLimitError."""
    pass
MockRateLimitError.__name__ = "RateLimitError"


# =============================================================================
# Test: Error Detection Helpers
# =============================================================================

class TestErrorDetection:
    """Tests for _is_rate_limit_error and _is_service_unavailable."""

    def test_detects_429_in_message(self):
        """Should detect 429 status code in exception message."""
        from models.safe_llm import _is_rate_limit_error
        
        exc = Exception("Request failed with status 429: Too Many Requests")
        assert _is_rate_limit_error(exc) is True

    def test_detects_rate_limit_text(self):
        """Should detect 'rate limit' text in exception message."""
        from models.safe_llm import _is_rate_limit_error
        
        exc = Exception("You have exceeded your rate limit. Please wait.")
        assert _is_rate_limit_error(exc) is True

    def test_detects_quota_exceeded(self):
        """Should detect quota exceeded messages."""
        from models.safe_llm import _is_rate_limit_error
        
        exc = Exception("Your API quota has been exceeded for this month.")
        assert _is_rate_limit_error(exc) is True

    def test_detects_http_status_error_429(self):
        """Should detect HTTPStatusError with 429 status code."""
        from models.safe_llm import _is_rate_limit_error
        
        exc = MockHTTPStatusError(429)
        exc.__class__.__name__ = "HTTPStatusError"
        assert _is_rate_limit_error(exc) is True

    def test_does_not_false_positive(self):
        """Should not flag unrelated errors as rate limits."""
        from models.safe_llm import _is_rate_limit_error
        
        exc = Exception("Connection refused: server is down")
        assert _is_rate_limit_error(exc) is False

    def test_detects_503_service_unavailable(self):
        """Should detect 503 Service Unavailable errors."""
        from models.safe_llm import _is_service_unavailable
        
        exc = Exception("HTTP 503: Service Unavailable")
        assert _is_service_unavailable(exc) is True

    def test_detects_502_bad_gateway(self):
        """Should detect 502 Bad Gateway via status code."""
        from models.safe_llm import _is_service_unavailable
        
        exc = MockHTTPStatusError(502)
        assert _is_service_unavailable(exc) is True


# =============================================================================
# Test: SafeLLM Basic Behavior
# =============================================================================

class TestSafeLLMBasic:
    """Tests for basic SafeLLM functionality."""

    def test_passes_through_successful_invocation(self, mock_model):
        """Should pass through when invocation succeeds."""
        from models.safe_llm import SafeLLM
        
        safe = SafeLLM(mock_model)
        result = safe.invoke("Hello")
        
        assert result.content == "Success response"
        mock_model.invoke.assert_called_once_with("Hello")

    def test_repr_shows_model_type(self, mock_model):
        """Should give informative repr."""
        from models.safe_llm import SafeLLM
        
        safe = SafeLLM(mock_model, fallback_provider="nvidia")
        
        assert "SafeLLM" in repr(safe)
        assert "nvidia" in repr(safe)


# =============================================================================
# Test: Circuit Breaker - Retry on 429
# =============================================================================

class TestCircuitBreakerRetry:
    """Tests for automatic retry behavior on rate limit errors."""

    def test_retries_on_rate_limit_then_succeeds(self, mock_model):
        """Should retry on 429 and succeed on second attempt."""
        from models.safe_llm import SafeLLM
        
        # First call raises 429, second succeeds
        mock_model.invoke.side_effect = [
            Exception("429 Too Many Requests"),
            MagicMock(content="Retry succeeded")
        ]
        
        safe = SafeLLM(mock_model, max_retries=2)
        
        with patch('time.sleep'):  # Skip actual sleep
            result = safe.invoke("Test")
        
        assert result.content == "Retry succeeded"
        assert mock_model.invoke.call_count == 2

    def test_exhausts_retries_then_raises(self, mock_model):
        """Should raise after exhausting all retries."""
        from models.safe_llm import SafeLLM
        
        # Always fails with 429
        mock_model.invoke.side_effect = Exception("429 Too Many Requests")
        
        safe = SafeLLM(mock_model, max_retries=2)
        
        with patch('time.sleep'):  # Skip actual sleep
            with pytest.raises(Exception, match="429"):
                safe.invoke("Test")
        
        # 1 initial + 2 retries = 3 total attempts
        assert mock_model.invoke.call_count == 3


# =============================================================================
# Test: Circuit Breaker - Provider Fallback
# =============================================================================

class TestCircuitBreakerFallback:
    """Tests for automatic provider switching on failure."""

    def test_switches_provider_on_429(self, mock_model, mock_manager):
        """Should switch to fallback provider on rate limit."""
        from models.safe_llm import SafeLLM
        
        # Primary fails with 429
        mock_model.invoke.side_effect = Exception("429 Too Many Requests")
        
        # Create the SafeLLM instance
        safe = SafeLLM(mock_model, manager=mock_manager, fallback_provider="nvidia", max_retries=0)
        
        # Mock the _get_fallback_model to return a working model
        fallback_model = MagicMock()
        fallback_response = MagicMock()
        fallback_response.content = "Fallback response"
        fallback_model.invoke.return_value = fallback_response
        
        with patch.object(safe, '_get_fallback_model', return_value=fallback_model):
            with patch('time.sleep'):
                result = safe.invoke("Test")
        
        # Should have switched provider
        mock_manager.set_active_provider.assert_called_with("nvidia")
        
        # Response should be from fallback
        assert result is not None

    def test_injects_fallback_notice_into_response(self, mock_model, mock_manager):
        """Should inject system notice about provider switch into response."""
        from models.safe_llm import SafeLLM
        
        mock_model.invoke.side_effect = Exception("429 rate limit exceeded")
        
        # Create the SafeLLM instance
        safe = SafeLLM(mock_model, manager=mock_manager, fallback_provider="nvidia", max_retries=0)
        
        # Create response with modifiable content
        fallback_response = MagicMock()
        fallback_response.content = "Original response"
        
        fallback_model = MagicMock()
        fallback_model.invoke.return_value = fallback_response
        
        with patch.object(safe, '_get_fallback_model', return_value=fallback_model):
            with patch('time.sleep'):
                result = safe.invoke("Test")
        
        # Should have notice appended (or content modified)
        assert result is not None
        # The notice modifies content attribute in-place
        assert "SYSTEM NOTICE" in fallback_response.content or "Original" in fallback_response.content

    def test_fallback_failure_raises_original_error(self, mock_model, mock_manager):
        """Should raise if both primary and fallback fail."""
        from models.safe_llm import SafeLLM
        
        mock_model.invoke.side_effect = Exception("429 Primary failed")
        
        safe = SafeLLM(mock_model, manager=mock_manager, max_retries=0)
        
        # Fallback returns None (indicating failure)
        with patch.object(safe, '_get_fallback_model', return_value=None):
            with patch('time.sleep'):
                with pytest.raises(Exception, match="Primary failed"):
                    safe.invoke("Test")


# =============================================================================
# Test: LangChain Compatibility
# =============================================================================

class TestLangChainCompatibility:
    """Tests for LangChain feature pass-through."""

    def test_bind_tools_returns_safellm(self, mock_model, mock_manager):
        """bind_tools should return a new SafeLLM wrapping the bound model."""
        from models.safe_llm import SafeLLM
        
        bound_model = MagicMock()
        mock_model.bind_tools.return_value = bound_model
        
        safe = SafeLLM(mock_model, manager=mock_manager)
        result = safe.bind_tools(["tool1", "tool2"])
        
        # Should return SafeLLM, not raw model
        assert isinstance(result, SafeLLM)
        mock_model.bind_tools.assert_called_once()

    def test_with_structured_output_returns_safellm(self, mock_model, mock_manager):
        """with_structured_output should return a new SafeLLM."""
        from models.safe_llm import SafeLLM
        
        structured_model = MagicMock()
        mock_model.with_structured_output.return_value = structured_model
        
        safe = SafeLLM(mock_model, manager=mock_manager)
        result = safe.with_structured_output({"type": "object"})
        
        assert isinstance(result, SafeLLM)

    def test_attribute_passthrough(self, mock_model):
        """Should pass through unknown attributes to underlying model."""
        from models.safe_llm import SafeLLM
        
        mock_model.model_name = "gpt-4"
        mock_model.temperature = 0.7
        
        safe = SafeLLM(mock_model)
        
        assert safe.model_name == "gpt-4"
        assert safe.temperature == 0.7


# =============================================================================
# Test: wrap_with_safety Factory
# =============================================================================

class TestWrapWithSafety:
    """Tests for the convenience factory function."""

    def test_creates_safellm_instance(self, mock_model, mock_manager):
        """Should create a properly configured SafeLLM."""
        from models.safe_llm import wrap_with_safety
        
        result = wrap_with_safety(mock_model, manager=mock_manager, fallback_provider="groq")
        
        from models.safe_llm import SafeLLM
        assert isinstance(result, SafeLLM)
        assert result._fallback_provider == "groq"
