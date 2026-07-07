"""Cron result delivery — send job output to email, webhook, or both.

The audit (P1 #9) flagged that NIA's cron only runs shell commands with no
way to deliver results to external platforms. This module adds delivery
support:

- **Email** via SMTP (plaintext or HTML).
- **Webhook** via HTTP POST (JSON payload with job metadata + output).
- **Both** simultaneously.

Configuration is per-job: the cron job dict gains a ``delivery`` field::

    {
        "name": "nightly-backup",
        "schedule": "0 2 * * *",
        "command": "pg_dump mydb > /tmp/backup.sql",
        "delivery": {
            "email": {
                "to": ["admin@example.com"],
                "subject": "Nightly backup result",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
                "smtp_password_env": "NIA_SMTP_PASSWORD",
                "use_tls": true
            },
            "webhook": {
                "url": "https://hooks.slack.com/services/...",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "on_success": true,
                "on_failure": true
            }
        }
    }

For security, SMTP passwords are never stored in the job dict — they're
read from the environment variable named in ``smtp_password_env`` at
delivery time.

Reference: Hermes Agent's cron delivery system (gateway/platform delivery).
NIA's version is simpler — email + webhook only, no Telegram/Discord yet.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delivery config validation
# ---------------------------------------------------------------------------


def validate_delivery_config(delivery: dict[str, Any]) -> list[str]:
    """Validate a delivery config. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []
    if not isinstance(delivery, dict):
        return ["delivery must be a dict"]

    email_cfg = delivery.get("email")
    webhook_cfg = delivery.get("webhook")

    if not email_cfg and not webhook_cfg:
        errors.append("delivery must include 'email' or 'webhook' (or both)")
        return errors

    if email_cfg is not None:
        if not isinstance(email_cfg, dict):
            errors.append("delivery.email must be a dict")
        else:
            if not email_cfg.get("to"):
                errors.append("delivery.email.to is required (list of addresses)")
            if not email_cfg.get("smtp_host"):
                errors.append("delivery.email.smtp_host is required")
            if not email_cfg.get("smtp_port"):
                errors.append("delivery.email.smtp_port is required")
            if not email_cfg.get("smtp_user"):
                errors.append("delivery.email.smtp_user is required")
            if not email_cfg.get("smtp_password_env"):
                errors.append(
                    "delivery.email.smtp_password_env is required "
                    "(env var name holding the SMTP password)"
                )

    if webhook_cfg is not None:
        if not isinstance(webhook_cfg, dict):
            errors.append("delivery.webhook must be a dict")
        else:
            # URL can be provided directly OR via env var indirection
            # (security: webhook URLs are bearer tokens — prefer env var).
            url = webhook_cfg.get("url")
            url_env = webhook_cfg.get("url_env")
            if not url and not url_env:
                errors.append(
                    "delivery.webhook requires 'url' or 'url_env' "
                    "(env var name holding the webhook URL — recommended "
                    "for security since webhook URLs are bearer tokens)"
                )
            if url and not url.startswith(("http://", "https://")):
                errors.append("delivery.webhook.url must be http:// or https://")

    return errors


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------


def _build_email(
    *,
    job_name: str,
    subject: str,
    result: dict[str, Any],
    to_addrs: list[str],
    from_addr: str,
) -> MIMEMultipart:
    """Build a MIME multipart email with the job result."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    status = result.get("status", "unknown")
    returncode = result.get("returncode", -1)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    started_at = result.get("started_at", "")
    finished_at = result.get("finished_at", "")

    # Plaintext version.
    text_lines = [
        f"N.I.A Cron Job: {job_name}",
        f"Status: {status} (exit code {returncode})",
        f"Started: {started_at}",
        f"Finished: {finished_at}",
        "",
        "--- stdout ---",
        stdout[:10000] if stdout else "(empty)",
        "",
        "--- stderr ---",
        stderr[:10000] if stderr else "(empty)",
    ]
    text_body = "\n".join(text_lines)

    # HTML version.
    html_lines = [
        "<html><body>",
        f"<h2>N.I.A Cron Job: {job_name}</h2>",
        f"<p><strong>Status:</strong> {status} (exit code {returncode})</p>",
        f"<p><strong>Started:</strong> {started_at}</p>",
        f"<p><strong>Finished:</strong> {finished_at}</p>",
        "<h3>stdout</h3>",
        f"<pre>{stdout[:10000] if stdout else '(empty)'}</pre>",
        "<h3>stderr</h3>",
        f"<pre>{stderr[:10000] if stderr else '(empty)'}</pre>",
        "</body></html>",
    ]
    html_body = "\n".join(html_lines)

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg


async def deliver_email(
    *,
    email_cfg: dict[str, Any],
    job_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Send job result via email. Returns a delivery status dict."""
    to_addrs = email_cfg.get("to", [])
    if isinstance(to_addrs, str):
        to_addrs = [to_addrs]

    smtp_host = email_cfg.get("smtp_host", "")
    smtp_port = int(email_cfg.get("smtp_port", 587))
    smtp_user = email_cfg.get("smtp_user", "")
    smtp_password_env = email_cfg.get("smtp_password_env", "")
    use_tls = email_cfg.get("use_tls", True)
    subject = email_cfg.get("subject", f"N.I.A cron: {job_name}")

    # Resolve password from env.
    smtp_password = os.environ.get(smtp_password_env, "")
    if not smtp_password:
        return {
            "channel": "email",
            "success": False,
            "error": f"SMTP password env var {smtp_password_env!r} is not set",
        }

    try:
        msg = _build_email(
            job_name=job_name,
            subject=subject,
            result=result,
            to_addrs=to_addrs,
            from_addr=smtp_user,
        )

        # SMTP is blocking — run in a thread via httpx's async pattern isn't
        # ideal, but smtplib has no async. Use asyncio.to_thread.
        import asyncio

        def _send_sync():
            if use_tls:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
                server.starttls()
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
            try:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_addrs, msg.as_string())
            finally:
                server.quit()

        await asyncio.to_thread(_send_sync)

        return {
            "channel": "email",
            "success": True,
            "recipients": to_addrs,
            "error": None,
        }
    except Exception as exc:
        logger.warning("Email delivery failed for job %s: %s", job_name, exc)
        return {
            "channel": "email",
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


async def deliver_webhook(
    *,
    webhook_cfg: dict[str, Any],
    job_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Send job result via webhook (HTTP POST). Returns a delivery status dict.

    Security: the webhook URL can be provided directly via 'url' OR via
    env var indirection via 'url_env' (recommended — webhook URLs are
    bearer tokens and should not be persisted in cron_jobs.json).
    """
    # Resolve URL: prefer env var indirection, fall back to direct URL.
    url_env = webhook_cfg.get("url_env")
    if url_env:
        url = os.environ.get(url_env, "")
        if not url:
            return {
                "channel": "webhook",
                "success": False,
                "error": f"Webhook URL env var {url_env!r} is not set",
            }
    else:
        url = webhook_cfg.get("url", "")

    if not url:
        return {
            "channel": "webhook",
            "success": False,
            "error": "No webhook URL configured (set 'url' or 'url_env')",
        }

    method = webhook_cfg.get("method", "POST").upper()
    headers = webhook_cfg.get("headers", {"Content-Type": "application/json"})
    on_success = webhook_cfg.get("on_success", True)
    on_failure = webhook_cfg.get("on_failure", True)

    # Check if we should deliver based on job status.
    status = result.get("status", "")
    if status == "success" and not on_success:
        return {
            "channel": "webhook",
            "success": True,
            "skipped": True,
            "reason": "on_success=False",
        }
    if status != "success" and not on_failure:
        return {
            "channel": "webhook",
            "success": True,
            "skipped": True,
            "reason": "on_failure=False",
        }

    # Build the payload.
    payload = {
        "job_name": job_name,
        "status": status,
        "returncode": result.get("returncode", -1),
        "started_at": result.get("started_at", ""),
        "finished_at": result.get("finished_at", ""),
        "stdout": (result.get("stdout") or "")[:5000],  # cap at 5K chars
        "stderr": (result.get("stderr") or "")[:5000],
        "delivered_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "POST":
                response = await client.post(url, json=payload, headers=headers)
            elif method == "PUT":
                response = await client.put(url, json=payload, headers=headers)
            else:
                response = await client.request(method, url, json=payload, headers=headers)

            response.raise_for_status()
            return {
                "channel": "webhook",
                "success": True,
                "status_code": response.status_code,
                "error": None,
            }
    except httpx.HTTPStatusError as exc:
        logger.warning("Webhook delivery failed (HTTP %s) for job %s", exc.response.status_code, job_name)
        return {
            "channel": "webhook",
            "success": False,
            "status_code": exc.response.status_code,
            "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        }
    except Exception as exc:
        logger.warning("Webhook delivery failed for job %s: %s", job_name, exc)
        return {
            "channel": "webhook",
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Combined delivery
# ---------------------------------------------------------------------------


async def deliver_result(
    *,
    delivery: dict[str, Any],
    job_name: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Deliver a job result via all configured channels.

    Returns a list of delivery status dicts (one per channel).
    """
    statuses: list[dict[str, Any]] = []

    email_cfg = delivery.get("email")
    webhook_cfg = delivery.get("webhook")

    if email_cfg:
        status = await deliver_email(
            email_cfg=email_cfg,
            job_name=job_name,
            result=result,
        )
        statuses.append(status)

    if webhook_cfg:
        status = await deliver_webhook(
            webhook_cfg=webhook_cfg,
            job_name=job_name,
            result=result,
        )
        statuses.append(status)

    return statuses
