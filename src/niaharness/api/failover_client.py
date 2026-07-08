"""Failover-aware API client — wraps AnthropicApiClient with credential pool rotation.

When a request fails with 401/429/402/403, the client:
  1. Marks the current credential as exhausted in the pool
  2. Selects the next available credential
  3. Retries the request with the new credential
  4. If no credentials remain, raises the original error

This is the "``_try_activate_fallback``" pattern from the reference project,
where a single 401/429 from the only configured provider no longer
terminates the turn — the pool rotates to the next credential (env →
manual → OAuth device-code) and retries.

Usage::

    from niaharness.api.failover_client import FailoverAnthropicClient
    from niaharness.api.credential_pool import load_pool

    pool = load_pool("anthropic")
    client = FailoverAnthropicClient(pool, base_url="https://api.anthropic.com")
    async for event in client.stream_message(request):
        ...

The client transparently swaps the underlying ``AsyncAnthropic`` instance
when the credential changes (each credential gets its own SDK client, cached
by API key + base_url).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional

from anthropic import AsyncAnthropic

from niaharness.api.client import (
    AnthropicApiClient,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiStreamEvent,
    MAX_RETRIES,
)
from niaharness.api.credential_pool import (
    CredentialPool,
    PooledCredential,
    STATUS_OK,
    load_pool,
)
from niaharness.api.errors import (
    AuthenticationFailure,
    NiaHarnessApiError,
    RateLimitFailure,
    RequestFailure,
)

logger = logging.getLogger(__name__)


# Status codes that should trigger credential rotation (vs. just retry).
_ROTATE_STATUS_CODES = frozenset({401, 402, 403, 429})

# Maximum number of credential rotations per request (don't loop forever
# if every credential is exhausted).
MAX_ROTATIONS = 5


class FailoverAnthropicClient:
    """Anthropic API client with credential-pool failover.

    Wraps ``AnthropicApiClient`` and intercepts auth/rate-limit errors to
    rotate to the next credential in the pool before retrying. If the pool
    is exhausted (no available credentials), the original error is raised.

    Each unique (api_key, base_url) pair gets its own cached ``AsyncAnthropic``
    instance to avoid re-creating the SDK client on every rotation.
    """

    def __init__(
        self,
        pool: CredentialPool,
        *,
        base_url: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        provider: str = "anthropic",
    ) -> None:
        self._pool = pool
        self._base_url = base_url
        self._max_retries = max_retries
        self._provider = provider
        self._client_cache: Dict[str, AnthropicApiClient] = {}
        self._client_cache_lock = asyncio.Lock()

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Stream a message with credential-pool failover on auth/rate-limit errors.

        On a 401/402/403/429:
          1. Mark the current credential as exhausted in the pool
          2. Select the next available credential
          3. Retry with the new credential (up to MAX_ROTATIONS times)
          4. If no credentials remain, raise the original error
        """
        rotation_count = 0
        last_error: Optional[Exception] = None

        while rotation_count <= MAX_ROTATIONS:
            # Select a credential.
            credential = self._pool.select()
            if credential is None:
                if last_error is not None:
                    raise last_error
                raise AuthenticationFailure(
                    f"No available credentials for provider '{self._provider}' "
                    f"(pool exhausted)"
                )

            # Get or create the underlying client for this credential.
            client = await self._get_client_for_credential(credential)
            api_key_hint = credential.runtime_api_key

            try:
                # Stream the message using the underlying client.
                async for event in client.stream_message(request):
                    yield event
                return  # Success
            except (AuthenticationFailure, RateLimitFailure) as exc:
                last_error = exc
                status_code = _extract_status_code(exc)
                if status_code not in _ROTATE_STATUS_CODES:
                    raise
                # Attempt credential rotation.
                error_context = _build_error_context(exc)
                next_credential = self._pool.mark_exhausted_and_rotate(
                    status_code=status_code,
                    error_context=error_context,
                    api_key_hint=api_key_hint,
                )
                if next_credential is None:
                    # No more credentials — raise the original error.
                    logger.error(
                        "credential pool[%s]: exhausted all credentials after %d rotations",
                        self._provider, rotation_count,
                    )
                    raise
                rotation_count += 1
                logger.info(
                    "credential pool[%s]: rotated to '%s' (rotation %d/%d)",
                    self._provider,
                    next_credential.label or next_credential.id[:8],
                    rotation_count, MAX_ROTATIONS,
                )
                # Emit a retry event so the UI can show the rotation.
                yield ApiRetryEvent(
                    attempt=rotation_count,
                    max_retries=MAX_ROTATIONS,
                    delay_seconds=0.0,  # No delay on credential rotation
                    error=f"credential rotation (status={status_code})",
                    status_code=status_code,
                )
                continue
            except NiaHarnessApiError:
                raise  # Non-rotatable API errors (context overflow, etc.)
            except Exception as exc:
                # For other exceptions, check if they're auth/rate-limit related.
                status_code = _extract_status_code(exc)
                if status_code in _ROTATE_STATUS_CODES:
                    last_error = exc
                    next_credential = self._pool.mark_exhausted_and_rotate(
                        status_code=status_code,
                        error_context=_build_error_context(exc),
                        api_key_hint=api_key_hint,
                    )
                    if next_credential is None:
                        raise
                    rotation_count += 1
                    continue
                raise

        if last_error is not None:
            raise last_error

    async def _get_client_for_credential(
        self, credential: PooledCredential
    ) -> AnthropicApiClient:
        """Get or create an AnthropicApiClient for a credential.

        Each unique (api_key, base_url) pair gets its own cached client
        to avoid re-creating the SDK instance on every rotation.
        """
        cache_key = f"{credential.runtime_api_key}:{credential.runtime_base_url or 'default'}"
        async with self._client_cache_lock:
            client = self._client_cache.get(cache_key)
            if client is None:
                client = AnthropicApiClient(
                    api_key=credential.runtime_api_key,
                    base_url=credential.runtime_base_url or self._base_url,
                    max_retries=self._max_retries,
                )
                self._client_cache[cache_key] = client
            return client


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Extract an HTTP status code from an exception."""
    # Check common attribute names.
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # Check the underlying __cause__ (for translated errors).
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        for attr in ("status_code", "status", "code"):
            value = getattr(cause, attr, None)
            if isinstance(value, int):
                return value
    return None


def _build_error_context(exc: Exception) -> Dict[str, Any]:
    """Build an error context dict for the credential pool's mark_exhausted."""
    return {
        "message": str(exc),
        "reason": getattr(exc, "reason", None) or type(exc).__name__,
    }


def create_failover_client(
    provider: str,
    *,
    base_url: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> FailoverAnthropicClient:
    """Create a FailoverAnthropicClient for a provider.

    Loads the credential pool for the provider (seeding from env + pool file)
    and wraps it in a failover-aware client.

    Usage::

        client = create_failover_client("anthropic")
        async for event in client.stream_message(request):
            ...
    """
    pool = load_pool(provider)
    return FailoverAnthropicClient(
        pool=pool,
        base_url=base_url,
        max_retries=max_retries,
        provider=provider,
    )


__all__ = [
    "FailoverAnthropicClient",
    "create_failover_client",
    "MAX_ROTATIONS",
]
