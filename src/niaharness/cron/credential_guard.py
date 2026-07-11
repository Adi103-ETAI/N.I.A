"""P1 Cron credential exfiltration guard.

Ported from Hermes Agent's ``cron/scheduler.py`` _guard_job_credential_exfil
function (lines 2340-2383).

Fail closed if a job's stored provider/base_url pair would exfiltrate a
credential (CWE-200/CWE-522).

The model-callable cron tool validates this on create/update, but a job
persisted before that guard — or written directly to the jobs store —
reaches the scheduler's provider-resolution sink unchecked. Re-validate
the EFFECTIVE stored pair with the same guard the tool uses, so a named
provider's stored key is never paired with an off-host base_url at fire
time.

The guard checks:
  1. If the job has a ``base_url`` override, it must NOT point at a
     different host than the provider's official API endpoint.
  2. If the job has a ``provider`` that's a named credential, the
     base_url must be the provider's official endpoint (or empty).

Raises ``RuntimeError`` when the pair is unsafe; returns None otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Official API endpoints for known providers. A base_url override that
# points at a DIFFERENT host is flagged as a potential credential-exfil
# vector — the stored key would be sent to an unverified endpoint.
_PROVIDER_OFFICIAL_ENDPOINTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "openai-compatible": "",  # user-supplied base_url is expected
    "google": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai",
    "xai": "https://api.x.ai",
    "together": "https://api.together.xyz",
    "fireworks": "https://api.fireworks.ai",
    "azure": "",  # user-supplied base_url is expected
    "local": "",  # localhost is always allowed
    "": "",  # no provider = no stored credential to exfil
}

# Hosts that are always safe (localhost variants).
_SAFE_LOCAL_HOSTS = frozenset({
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
})


def _extract_host(url: str) -> str:
    """Extract the hostname from a URL."""
    if not url:
        return ""
    parsed = urlparse(url)
    return (parsed.hostname or "").lower().rstrip("/")


def _is_local_host(host: str) -> bool:
    """Return True if the host is a localhost variant."""
    return host in _SAFE_LOCAL_HOSTS


def validate_cron_base_url(
    provider: Optional[str],
    base_url: Optional[str],
) -> Optional[str]:
    """Validate a (provider, base_url) pair for credential safety.

    Returns:
        None if the pair is safe. An error message string if unsafe.
    """
    provider_lower = (provider or "").lower().strip()
    base_url_clean = (base_url or "").strip()

    # No base_url override → nothing to exfil via this path.
    if not base_url_clean:
        return None

    # Provider with no official endpoint (openai-compatible, azure, local)
    # → user-supplied base_url is expected, no validation possible.
    if provider_lower not in _PROVIDER_OFFICIAL_ENDPOINTS:
        # Unknown provider — can't validate, but also no stored credential.
        return None

    official = _PROVIDER_OFFICIAL_ENDPOINTS.get(provider_lower, "")
    if not official:
        # Provider allows custom base_url (openai-compatible, azure, local).
        return None

    base_host = _extract_host(base_url_clean)
    official_host = _extract_host(official)

    # Localhost base_url is always safe (dev/testing).
    if _is_local_host(base_host):
        return None

    # Same host as the official endpoint → safe.
    if base_host == official_host:
        return None

    # Different host from the official endpoint → potential exfil.
    return (
        f"provider '{provider_lower}' uses '{official_host}' but the job's "
        f"base_url points at '{base_host}' — a stored credential for this "
        f"provider would be sent to an unverified endpoint"
    )


def guard_job_credential_exfil(job: dict[str, Any]) -> None:
    """Fail closed if a job's provider/base_url pair would exfiltrate a credential.

    Called at fire time (in run_job) as the last guard before provider
    resolution. Raises ``RuntimeError`` when the pair is unsafe; returns
    None otherwise.

    Fallback providers come from operator config, not the model-callable
    job, so they are trusted and validated by the caller, not here.
    """
    try:
        err = validate_cron_base_url(job.get("provider"), job.get("base_url"))
    except Exception as exc:
        # Fail CLOSED: this is the last guard before provider resolution,
        # so an unexpected validator error must not silently allow an
        # unvetted pair through. A job that carries no base_url override
        # cannot exfiltrate a stored credential via this path, so it still
        # runs. But any job that DID set a base_url is refused until the
        # validator can actually vet the pair.
        if job.get("base_url"):
            err = (
                f"could not validate provider/base_url pair "
                f"({exc.__class__.__name__}: {exc}); refusing to run a job "
                f"with an unverified base_url override"
            )
        else:
            err = None

    if err:
        job_id = job.get("id")
        logger.error(
            "Job '%s': refusing to run — unsafe provider/base_url pair could "
            "exfiltrate a stored credential: %s",
            job_id, err,
        )
        raise RuntimeError(f"Cron job '{job_id}' blocked for safety: {err}")


__all__ = [
    "guard_job_credential_exfil",
    "validate_cron_base_url",
]
