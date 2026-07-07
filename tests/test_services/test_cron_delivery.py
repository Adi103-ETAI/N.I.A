"""Tests for cron delivery (email + webhook)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.services.cron_delivery import (
    deliver_email,
    deliver_result,
    deliver_webhook,
    validate_delivery_config,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateDeliveryConfig:
    def test_empty_delivery_returns_error(self):
        errors = validate_delivery_config({})
        assert len(errors) == 1
        assert "email" in errors[0] or "webhook" in errors[0]

    def test_valid_webhook_only(self):
        errors = validate_delivery_config({
            "webhook": {"url": "https://example.com/hook"}
        })
        assert errors == []

    def test_valid_email_only(self):
        errors = validate_delivery_config({
            "email": {
                "to": ["admin@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
                "smtp_password_env": "NIA_SMTP_PASSWORD",
            }
        })
        assert errors == []

    def test_valid_both(self):
        errors = validate_delivery_config({
            "email": {
                "to": ["admin@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
                "smtp_password_env": "NIA_SMTP_PASSWORD",
            },
            "webhook": {"url": "https://example.com/hook"},
        })
        assert errors == []

    def test_webhook_invalid_url(self):
        errors = validate_delivery_config({
            "webhook": {"url": "ftp://bad"}
        })
        assert any("http" in e for e in errors)

    def test_email_missing_to(self):
        errors = validate_delivery_config({
            "email": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
                "smtp_password_env": "NIA_SMTP_PASSWORD",
            }
        })
        assert any("to" in e for e in errors)

    def test_email_missing_smtp_password_env(self):
        errors = validate_delivery_config({
            "email": {
                "to": ["admin@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
            }
        })
        assert any("smtp_password_env" in e for e in errors)


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


class TestWebhookDelivery:
    @pytest.mark.asyncio
    async def test_successful_webhook_delivery(self):
        result = {
            "status": "success",
            "returncode": 0,
            "stdout": "backup complete",
            "stderr": "",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
        }
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            status = await deliver_webhook(
                webhook_cfg={"url": "https://hooks.example.com/test"},
                job_name="nightly-backup",
                result=result,
            )

        assert status["channel"] == "webhook"
        assert status["success"] is True
        assert status["status_code"] == 200

    @pytest.mark.asyncio
    async def test_webhook_http_error(self):
        result = {"status": "failed", "returncode": 1, "stdout": "", "stderr": "err"}
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            import httpx as _httpx

            mock_response.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=mock_response
                )
            )
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            status = await deliver_webhook(
                webhook_cfg={"url": "https://hooks.example.com/test"},
                job_name="test",
                result=result,
            )

        assert status["success"] is False
        assert status["status_code"] == 500

    @pytest.mark.asyncio
    async def test_webhook_skipped_on_success_false(self):
        """When on_success=False and job succeeded, webhook is skipped."""
        result = {"status": "success", "returncode": 0, "stdout": "ok", "stderr": ""}
        status = await deliver_webhook(
            webhook_cfg={"url": "https://example.com", "on_success": False},
            job_name="test",
            result=result,
        )
        assert status["skipped"] is True
        assert "on_success" in status["reason"]

    @pytest.mark.asyncio
    async def test_webhook_skipped_on_failure_false(self):
        """When on_failure=False and job failed, webhook is skipped."""
        result = {"status": "failed", "returncode": 1, "stdout": "", "stderr": "err"}
        status = await deliver_webhook(
            webhook_cfg={"url": "https://example.com", "on_failure": False},
            job_name="test",
            result=result,
        )
        assert status["skipped"] is True


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------


class TestEmailDelivery:
    @pytest.mark.asyncio
    async def test_email_no_password_env(self, monkeypatch: pytest.MonkeyPatch):
        """When the SMTP password env var is not set, return an error."""
        monkeypatch.delenv("NIA_SMTP_PASSWORD", raising=False)
        status = await deliver_email(
            email_cfg={
                "to": ["admin@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "bot@example.com",
                "smtp_password_env": "NIA_SMTP_PASSWORD",
            },
            job_name="test",
            result={"status": "success", "returncode": 0, "stdout": "ok", "stderr": ""},
        )
        assert status["success"] is False
        assert "NIA_SMTP_PASSWORD" in status["error"]

    @pytest.mark.asyncio
    async def test_email_successful_send(self, monkeypatch: pytest.MonkeyPatch):
        """Test successful email send with mocked SMTP."""
        monkeypatch.setenv("NIA_SMTP_PASSWORD", "test-password")
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            status = await deliver_email(
                email_cfg={
                    "to": ["admin@example.com"],
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 587,
                    "smtp_user": "bot@example.com",
                    "smtp_password_env": "NIA_SMTP_PASSWORD",
                    "use_tls": True,
                },
                job_name="nightly-backup",
                result={
                    "status": "success",
                    "returncode": 0,
                    "stdout": "backup complete",
                    "stderr": "",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:01:00+00:00",
                },
            )

        assert status["success"] is True
        assert "admin@example.com" in status["recipients"]
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("bot@example.com", "test-password")
        mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_smtp_error(self, monkeypatch: pytest.MonkeyPatch):
        """When SMTP raises an error, return a failed status."""
        monkeypatch.setenv("NIA_SMTP_PASSWORD", "test-password")
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_server.login.side_effect = Exception("Auth failed")
            mock_smtp.return_value = mock_server

            status = await deliver_email(
                email_cfg={
                    "to": ["admin@example.com"],
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 587,
                    "smtp_user": "bot@example.com",
                    "smtp_password_env": "NIA_SMTP_PASSWORD",
                },
                job_name="test",
                result={"status": "success", "returncode": 0, "stdout": "ok", "stderr": ""},
            )

        assert status["success"] is False
        assert "Auth failed" in status["error"]


# ---------------------------------------------------------------------------
# Combined delivery
# ---------------------------------------------------------------------------


class TestDeliverResult:
    @pytest.mark.asyncio
    async def test_both_channels(self, monkeypatch: pytest.MonkeyPatch):
        """When both email and webhook are configured, deliver to both."""
        monkeypatch.setenv("NIA_SMTP_PASSWORD", "test-password")
        with patch("smtplib.SMTP") as mock_smtp, patch("httpx.AsyncClient") as mock_http:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_http.return_value = mock_client

            statuses = await deliver_result(
                delivery={
                    "email": {
                        "to": ["admin@example.com"],
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": 587,
                        "smtp_user": "bot@example.com",
                        "smtp_password_env": "NIA_SMTP_PASSWORD",
                    },
                    "webhook": {"url": "https://example.com/hook"},
                },
                job_name="test",
                result={"status": "success", "returncode": 0, "stdout": "ok", "stderr": ""},
            )

        assert len(statuses) == 2
        channels = {s["channel"] for s in statuses}
        assert channels == {"email", "webhook"}
        assert all(s["success"] for s in statuses)

    @pytest.mark.asyncio
    async def test_webhook_only(self):
        statuses = await deliver_result(
            delivery={"webhook": {"url": "https://example.com"}},
            job_name="test",
            result={"status": "success", "returncode": 0, "stdout": "ok", "stderr": ""},
        )
        assert len(statuses) == 1
        assert statuses[0]["channel"] == "webhook"
