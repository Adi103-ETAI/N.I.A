"""API error types for NiaHarness."""

from __future__ import annotations


class NiaHarnessApiError(RuntimeError):
    """Base class for upstream API failures."""


class AuthenticationFailure(NiaHarnessApiError):
    """Raised when the upstream service rejects the provided credentials."""


class RateLimitFailure(NiaHarnessApiError):
    """Raised when the upstream service rejects the request due to rate limits."""


class RequestFailure(NiaHarnessApiError):
    """Raised for generic request or transport failures."""
