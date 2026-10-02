"""Telegram Bot API client with retry/backoff and IPv4 fallback support."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from sciml_digest.exceptions import TelegramError

LOGGER = logging.getLogger(__name__)

RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class TelegramClient:
    """Small synchronous Telegram Bot API client."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        max_attempts: int = 4,
        timeout_seconds: float = 20.0,
        force_ipv4: bool = False,
    ) -> None:
        """Initialize the Telegram Bot API client."""

        self._endpoint = (
            f"https://api.telegram.org/bot{bot_token}/sendMessage"
        )
        self._chat_id = chat_id
        self._max_attempts = max_attempts

        transport = _build_transport(force_ipv4=force_ipv4)

        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "TelegramClient":
        """Enter the client context manager."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the client when leaving the context manager."""

        self.close()

    def send_messages(self, messages: list[str]) -> None:
        """Send all digest messages sequentially."""

        for index, message in enumerate(messages, start=1):
            LOGGER.info(
                "Sending Telegram message %s/%s.",
                index,
                len(messages),
            )
            self._send_message(message)
            LOGGER.info(
                "Telegram message %s/%s delivered.",
                index,
                len(messages),
            )

    def _send_message(self, text: str) -> None:
        """Send one HTML message with bounded retries."""

        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }

        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(
                    self._endpoint,
                    json=payload,
                )

                if response.status_code in RETRIABLE_STATUS_CODES:
                    if attempt == self._max_attempts:
                        raise TelegramError(
                            "Telegram remained unavailable after retries "
                            f"(HTTP {response.status_code})."
                        )

                    delay = _telegram_retry_delay(response, attempt)
                    LOGGER.warning(
                        "Telegram transient HTTP %s; retrying in %.1fs.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                if response.is_error:
                    raise TelegramError(
                        "Telegram API rejected the message "
                        f"(HTTP {response.status_code})."
                    )

                try:
                    body = response.json()
                except ValueError as exc:
                    raise TelegramError(
                        "Telegram returned a non-JSON success response."
                    ) from exc

                if not body.get("ok", False):
                    description = str(body.get("description", "unknown error"))
                    raise TelegramError(
                        f"Telegram API returned ok=false: {description}."
                    )

                return

            except TelegramError:
                raise
            except httpx.RequestError as exc:
                last_error = exc

                if attempt == self._max_attempts:
                    break

                delay = _retry_delay(attempt)

                # Não registrar a exceção completa: ela pode conter a URL com o token.
                LOGGER.warning(
                    "Telegram network error (%s); retrying in %.1fs.",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        error_name = type(last_error).__name__ if last_error else "unknown"
        raise TelegramError(
            "Telegram delivery failed after all retry attempts "
            f"(last network error: {error_name})."
        ) from last_error


def _build_transport(force_ipv4: bool) -> httpx.HTTPTransport:
    """Build the HTTP transport, optionally binding connections to IPv4."""

    if force_ipv4:
        return httpx.HTTPTransport(
            local_address="0.0.0.0",
            retries=0,
        )

    return httpx.HTTPTransport(retries=0)


def _telegram_retry_delay(
    response: httpx.Response,
    attempt: int,
) -> float:
    """Honor Telegram retry_after when present, otherwise use backoff."""

    try:
        body = response.json()
        retry_after = body.get("parameters", {}).get("retry_after")
        if retry_after is not None:
            return min(float(retry_after), 60.0)
    except (ValueError, TypeError):
        pass

    retry_after_header = response.headers.get("Retry-After")
    if retry_after_header:
        try:
            return min(float(retry_after_header), 60.0)
        except ValueError:
            pass

    return _retry_delay(attempt)


def _retry_delay(
    attempt: int,
    base_seconds: float = 2.0,
    max_seconds: float = 30.0,
) -> float:
    """Calculate bounded exponential backoff with jitter."""

    delay = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
    return delay + random.uniform(0.0, 0.75)
