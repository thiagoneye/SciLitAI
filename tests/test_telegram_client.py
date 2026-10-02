"""Tests for Telegram transport configuration."""

from __future__ import annotations

from unittest.mock import patch

from sciml_digest.telegram_client import _build_transport


def test_build_transport_forces_ipv4_when_enabled() -> None:
    """Bind the HTTP transport to an IPv4 local address when requested."""

    with patch("sciml_digest.telegram_client.httpx.HTTPTransport") as transport:
        _build_transport(force_ipv4=True)

    transport.assert_called_once_with(
        local_address="0.0.0.0",
        retries=0,
    )


def test_build_transport_uses_default_network_stack_when_disabled() -> None:
    """Use the system network stack when IPv4 forcing is disabled."""

    with patch("sciml_digest.telegram_client.httpx.HTTPTransport") as transport:
        _build_transport(force_ipv4=False)

    transport.assert_called_once_with(retries=0)
