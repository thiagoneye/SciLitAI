"""Environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from sciml_digest.exceptions import ConfigurationError


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    gemini_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    gemini_model: str = "gemini-3.1-flash-lite"

    arxiv_top_k: int = 5
    arxiv_max_results_per_query: int = 100
    arxiv_terms_per_query: int = 12
    arxiv_timeout_seconds: float = 30.0
    arxiv_max_attempts: int = 4
    arxiv_min_request_interval_seconds: float = 3.0

    gemini_max_attempts: int = 4

    telegram_max_attempts: int = 4
    telegram_timeout_seconds: float = 20.0
    telegram_force_ipv4: bool = True

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate configuration from environment variables."""

        load_dotenv()

        required = {
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "").strip(),
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        settings = cls(
            gemini_api_key=required["GEMINI_API_KEY"],
            telegram_bot_token=required["TELEGRAM_BOT_TOKEN"],
            telegram_chat_id=required["TELEGRAM_CHAT_ID"],
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
            arxiv_top_k=_read_positive_int("ARXIV_TOP_K", 5),
            arxiv_max_results_per_query=_read_positive_int(
                "ARXIV_MAX_RESULTS_PER_QUERY", 100
            ),
            arxiv_terms_per_query=_read_positive_int("ARXIV_TERMS_PER_QUERY", 12),
            arxiv_timeout_seconds=_read_positive_float(
                "ARXIV_TIMEOUT_SECONDS", 30.0
            ),
            arxiv_max_attempts=_read_positive_int("ARXIV_MAX_ATTEMPTS", 4),
            arxiv_min_request_interval_seconds=_read_positive_float(
                "ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 3.0
            ),
            gemini_max_attempts=_read_positive_int("GEMINI_MAX_ATTEMPTS", 4),
            telegram_max_attempts=_read_positive_int(
                "TELEGRAM_MAX_ATTEMPTS",
                4,
            ),
            telegram_timeout_seconds=_read_positive_float(
                "TELEGRAM_TIMEOUT_SECONDS",
                20.0,
            ),
            telegram_force_ipv4=_read_bool(
                "TELEGRAM_FORCE_IPV4",
                True,
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate cross-field and semantic constraints."""

        if self.arxiv_top_k != 5:
            raise ConfigurationError("ARXIV_TOP_K must be 5 for the current digest format.")
        if self.arxiv_max_results_per_query > 2000:
            raise ConfigurationError(
                "ARXIV_MAX_RESULTS_PER_QUERY must be <= 2000."
            )
        if self.arxiv_min_request_interval_seconds < 3.0:
            raise ConfigurationError(
                "ARXIV_MIN_REQUEST_INTERVAL_SECONDS must be >= 3.0."
            )


def _read_positive_int(name: str, default: int) -> int:
    """Read a strictly positive integer environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be > 0.")
    return value


def _read_positive_float(name: str, default: float) -> float:
    """Read a strictly positive float environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be > 0.")
    return value

def _read_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in {"1", "true", "yes", "on"}:
        return True

    if normalized_value in {"0", "false", "no", "off"}:
        return False

    raise ConfigurationError(
        f"{name} must be a boolean value."
    )