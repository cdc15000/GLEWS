"""Tests for the alerting integrations module (glews.alerts).

Covers config parsing with env-var expansion, message formatting for
each channel, the no-channels no-op path, mock SMTP/HTTP send paths,
and failure isolation between channels.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from glews.alerts import (
    AlertDispatcher,
    EmailChannel,
    SlackChannel,
    WebhookChannel,
    expand_env_vars,
)


# --------------------------------------------------------------------------
# Environment variable expansion
# --------------------------------------------------------------------------


class TestExpandEnvVars:
    def test_simple_string(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "s3cret")
        assert expand_env_vars("${MY_SECRET}") == "s3cret"

    def test_embedded_in_string(self, monkeypatch):
        monkeypatch.setenv("TOKEN", "abc123")
        assert expand_env_vars("Bearer ${TOKEN}") == "Bearer abc123"

    def test_missing_var_expands_to_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT", raising=False)
        assert expand_env_vars("${NONEXISTENT}") == ""

    def test_dict_expansion(self, monkeypatch):
        monkeypatch.setenv("HOST", "mail.example.com")
        result = expand_env_vars({"smtp_host": "${HOST}", "port": 587})
        assert result == {"smtp_host": "mail.example.com", "port": 587}

    def test_list_expansion(self, monkeypatch):
        monkeypatch.setenv("ADDR", "user@example.com")
        result = expand_env_vars(["${ADDR}", "other@example.com"])
        assert result == ["user@example.com", "other@example.com"]

    def test_nested_expansion(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_TOKEN", "tok_xyz")
        cfg = {
            "headers": {"Authorization": "Bearer ${WEBHOOK_TOKEN}"},
            "url": "https://example.com",
        }
        result = expand_env_vars(cfg)
        assert result["headers"]["Authorization"] == "Bearer tok_xyz"

    def test_non_string_passthrough(self):
        assert expand_env_vars(42) == 42
        assert expand_env_vars(None) is None
        assert expand_env_vars(True) is True

    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("A", "hello")
        monkeypatch.setenv("B", "world")
        assert expand_env_vars("${A} ${B}") == "hello world"


# --------------------------------------------------------------------------
# Email channel
# --------------------------------------------------------------------------


class TestEmailChannel:
    def test_from_config_minimal(self):
        ch = EmailChannel.from_config({
            "smtp_host": "smtp.example.com",
            "from": "glews@example.com",
            "to": ["alice@example.com"],
        })
        assert ch is not None
        assert ch.smtp_host == "smtp.example.com"
        assert ch.smtp_port == 587
        assert ch.to_addrs == ["alice@example.com"]

    def test_from_config_missing_host_returns_none(self):
        assert EmailChannel.from_config({}) is None

    def test_html_format_contains_level_and_site(self):
        html = EmailChannel._format_html(
            "WARNING", "Aletsch Glacier",
            "Anomalous acceleration detected",
            {"max_zscore": 4.2},
        )
        assert "[WARNING]" in html
        assert "Aletsch Glacier" in html
        assert "max_zscore" in html
        assert "4.2" in html

    def test_plain_format(self):
        plain = EmailChannel._format_plain(
            "CRITICAL", "Test Site",
            "Big problem", {"nearest_flag_km": 1.5},
        )
        assert "[CRITICAL]" in plain
        assert "Test Site" in plain
        assert "nearest_flag_km: 1.5" in plain

    @patch("glews.alerts.smtplib.SMTP")
    def test_send_starttls(self, mock_smtp_cls):
        """Port 587 uses SMTP + starttls."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        ch = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
            username="user",
            password="pass",
        )
        ch.send("WARNING", "Site A", "Test message", {"key": "val"})

        mock_smtp_cls.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

    @patch("glews.alerts.smtplib.SMTP_SSL")
    def test_send_ssl(self, mock_smtp_ssl_cls):
        """Port 465 uses SMTP_SSL directly."""
        mock_server = MagicMock()
        mock_smtp_ssl_cls.return_value = mock_server

        ch = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=465,
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
        )
        ch.send("INFO", "Site B", "Info message", {})

        mock_smtp_ssl_cls.assert_called_once()
        # No login when username is empty
        mock_server.login.assert_not_called()
        mock_server.send_message.assert_called_once()

    @patch("glews.alerts.smtplib.SMTP")
    def test_send_no_login_when_no_username(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        ch = EmailChannel(
            smtp_host="smtp.example.com",
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
        )
        ch.send("INFO", "Site", "msg", {})
        mock_server.login.assert_not_called()


# --------------------------------------------------------------------------
# Slack channel
# --------------------------------------------------------------------------


class TestSlackChannel:
    def test_from_config(self):
        ch = SlackChannel.from_config({
            "webhook_url": "https://hooks.slack.com/services/T/B/X",
        })
        assert ch is not None
        assert ch.webhook_url == "https://hooks.slack.com/services/T/B/X"

    def test_from_config_missing_url_returns_none(self):
        assert SlackChannel.from_config({}) is None

    def test_payload_has_text_fallback_and_blocks(self):
        payload = SlackChannel._build_payload(
            "CRITICAL", "Weisshorn",
            "CRITICAL: peak z-score 8.0", {"max_zscore": 8.0},
        )
        assert "text" in payload
        assert "CRITICAL" in payload["text"]
        assert "blocks" in payload
        assert len(payload["blocks"]) == 3

    def test_payload_emoji_mapping(self):
        for level, expected in [
            ("INFO", ":information_source:"),
            ("WARNING", ":warning:"),
            ("CRITICAL", ":rotating_light:"),
        ]:
            payload = SlackChannel._build_payload(level, "S", "m", {})
            assert payload["text"].startswith(expected)

    @patch("glews.alerts.urllib.request.urlopen")
    def test_send_posts_json(self, mock_urlopen):
        ch = SlackChannel(webhook_url="https://hooks.slack.com/test")
        ch.send("WARNING", "Site A", "Alert!", {"z": 3.0})

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data)
        assert "blocks" in body
        assert "text" in body


# --------------------------------------------------------------------------
# Generic webhook channel
# --------------------------------------------------------------------------


class TestWebhookChannel:
    def test_from_config(self):
        ch = WebhookChannel.from_config({
            "url": "https://example.com/api/alerts",
            "headers": {"Authorization": "Bearer tok"},
        })
        assert ch is not None
        assert ch.headers["Authorization"] == "Bearer tok"

    def test_from_config_missing_url_returns_none(self):
        assert WebhookChannel.from_config({}) is None

    def test_payload_structure(self):
        payload = WebhookChannel._build_payload(
            "WARNING", "Site X", "Something happened",
            {"max_zscore": 5.1, "n_flags": 2},
        )
        assert payload["alert_level"] == "WARNING"
        assert payload["site_name"] == "Site X"
        assert payload["message"] == "Something happened"
        assert "timestamp" in payload
        assert payload["details"]["max_zscore"] == 5.1

    @patch("glews.alerts.urllib.request.urlopen")
    def test_send_posts_json_with_custom_headers(self, mock_urlopen):
        ch = WebhookChannel(
            url="https://example.com/hook",
            headers={"X-Custom": "val"},
        )
        ch.send("INFO", "Site", "msg", {"k": "v"})

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-custom") == "val"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data)
        assert body["alert_level"] == "INFO"


# --------------------------------------------------------------------------
# AlertDispatcher
# --------------------------------------------------------------------------


class TestAlertDispatcher:
    def test_no_channels_configured_is_noop(self):
        dispatcher = AlertDispatcher.from_config({})
        results = dispatcher.dispatch("WARNING", "Site", "msg", {})
        assert results == {}

    def test_no_alerts_key_is_noop(self):
        dispatcher = AlertDispatcher.from_config({"site": {"name": "X"}})
        assert len(dispatcher.channels) == 0

    def test_from_config_builds_channels(self, monkeypatch):
        monkeypatch.setenv("SLACK_URL", "https://hooks.slack.com/test")
        config = {
            "alerts": {
                "slack": {"webhook_url": "${SLACK_URL}"},
                "webhook": {
                    "url": "https://example.com/hook",
                    "headers": {"X-Key": "val"},
                },
            },
        }
        dispatcher = AlertDispatcher.from_config(config)
        assert len(dispatcher.channels) == 2
        names = {ch.name for ch in dispatcher.channels}
        assert names == {"slack", "webhook"}

    def test_from_config_skips_unknown_channel(self):
        config = {"alerts": {"carrier_pigeon": {"speed": "fast"}}}
        dispatcher = AlertDispatcher.from_config(config)
        assert len(dispatcher.channels) == 0

    def test_dispatch_returns_per_channel_results(self):
        ch1 = MagicMock(spec=SlackChannel)
        ch1.name = "slack"
        ch2 = MagicMock(spec=WebhookChannel)
        ch2.name = "webhook"

        dispatcher = AlertDispatcher(channels=[ch1, ch2])
        results = dispatcher.dispatch("WARNING", "Site", "msg", {"k": 1})

        assert results == {"slack": True, "webhook": True}
        ch1.send.assert_called_once_with("WARNING", "Site", "msg", {"k": 1})
        ch2.send.assert_called_once_with("WARNING", "Site", "msg", {"k": 1})

    def test_failure_isolation_between_channels(self):
        """Channel A raises; channel B still receives its send call and
        dispatch does not propagate the exception."""
        ch_fail = MagicMock()
        ch_fail.name = "email"
        ch_fail.send.side_effect = ConnectionError("SMTP down")

        ch_ok = MagicMock()
        ch_ok.name = "webhook"

        dispatcher = AlertDispatcher(channels=[ch_fail, ch_ok])
        results = dispatcher.dispatch("CRITICAL", "Site", "alert!", {})

        assert results == {"email": False, "webhook": True}
        ch_ok.send.assert_called_once()

    def test_all_channels_fail_returns_all_false(self):
        ch = MagicMock()
        ch.name = "slack"
        ch.send.side_effect = Exception("network error")

        dispatcher = AlertDispatcher(channels=[ch])
        results = dispatcher.dispatch("WARNING", "Site", "msg", {})

        assert results == {"slack": False}

    def test_env_var_expansion_in_full_config(self, monkeypatch):
        monkeypatch.setenv("GEWS_SMTP_USER", "myuser")
        monkeypatch.setenv("GEWS_SMTP_PASS", "mypass")
        config = {
            "alerts": {
                "email": {
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 587,
                    "from": "glews@example.com",
                    "to": ["analyst@example.com"],
                    "username": "${GEWS_SMTP_USER}",
                    "password": "${GEWS_SMTP_PASS}",
                },
            },
        }
        dispatcher = AlertDispatcher.from_config(config)
        assert len(dispatcher.channels) == 1
        email_ch = dispatcher.channels[0]
        assert isinstance(email_ch, EmailChannel)
        assert email_ch.username == "myuser"
        assert email_ch.password == "mypass"
