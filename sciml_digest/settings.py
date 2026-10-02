"""Environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from sciml_digest.exceptions import ConfigurationError


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    gemini_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    semantic_scholar_api_key: str | None = None
    openalex_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    papers_per_source: int = 3
    arxiv_max_results_per_query: int = 100
    arxiv_terms_per_query: int = 12
    arxiv_timeout_seconds: float = 30.0
    arxiv_max_attempts: int = 4
    arxiv_min_request_interval_seconds: float = 3.0
    semantic_scholar_max_results_per_query: int = 50
    semantic_scholar_terms_per_query: int = 8
    semantic_scholar_timeout_seconds: float = 30.0
    semantic_scholar_max_attempts: int = 4
    semantic_scholar_min_request_interval_seconds: float = 1.0
    openalex_lookback_days: int = 60
    openalex_max_results_per_query: int = 50
    openalex_timeout_seconds: float = 30.0
    openalex_max_attempts: int = 4
    openalex_min_request_interval_seconds: float = 1.0
    openalex_min_relevance_score: float = 10.0
    openalex_validation_max_candidates: int = 25
    openalex_require_validated_date: bool = True
    openalex_require_day_precision: bool = True
    openalex_bibliographic_timeout_seconds: float = 20.0
    openalex_bibliographic_max_attempts: int = 3
    openalex_bibliographic_min_request_interval_seconds: float = 0.5
    gemini_max_attempts: int = 4
    telegram_max_attempts: int = 4
    telegram_timeout_seconds: float = 20.0
    telegram_force_ipv4: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate configuration from environment variables."""

        _load_local_dotenv()

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
            semantic_scholar_api_key=_read_optional_string(
                "SEMANTIC_SCHOLAR_API_KEY"
            ),
            openalex_api_key=_read_optional_string("OPENALEX_API_KEY"),
            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.1-flash-lite",
            ).strip(),
            papers_per_source=_read_positive_int("PAPERS_PER_SOURCE", 3),
            arxiv_max_results_per_query=_read_positive_int(
                "ARXIV_MAX_RESULTS_PER_QUERY",
                100,
            ),
            arxiv_terms_per_query=_read_positive_int(
                "ARXIV_TERMS_PER_QUERY",
                12,
            ),
            arxiv_timeout_seconds=_read_positive_float(
                "ARXIV_TIMEOUT_SECONDS",
                30.0,
            ),
            arxiv_max_attempts=_read_positive_int(
                "ARXIV_MAX_ATTEMPTS",
                4,
            ),
            arxiv_min_request_interval_seconds=_read_positive_float(
                "ARXIV_MIN_REQUEST_INTERVAL_SECONDS",
                3.0,
            ),
            semantic_scholar_max_results_per_query=_read_positive_int(
                "SEMANTIC_SCHOLAR_MAX_RESULTS_PER_QUERY",
                50,
            ),
            semantic_scholar_terms_per_query=_read_positive_int(
                "SEMANTIC_SCHOLAR_TERMS_PER_QUERY",
                8,
            ),
            semantic_scholar_timeout_seconds=_read_positive_float(
                "SEMANTIC_SCHOLAR_TIMEOUT_SECONDS",
                30.0,
            ),
            semantic_scholar_max_attempts=_read_positive_int(
                "SEMANTIC_SCHOLAR_MAX_ATTEMPTS",
                4,
            ),
            semantic_scholar_min_request_interval_seconds=_read_positive_float(
                "SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS",
                1.0,
            ),
            openalex_lookback_days=_read_positive_int(
                "OPENALEX_LOOKBACK_DAYS",
                60,
            ),
            openalex_max_results_per_query=_read_positive_int(
                "OPENALEX_MAX_RESULTS_PER_QUERY",
                50,
            ),
            openalex_timeout_seconds=_read_positive_float(
                "OPENALEX_TIMEOUT_SECONDS",
                30.0,
            ),
            openalex_max_attempts=_read_positive_int(
                "OPENALEX_MAX_ATTEMPTS",
                4,
            ),
            openalex_min_request_interval_seconds=_read_positive_float(
                "OPENALEX_MIN_REQUEST_INTERVAL_SECONDS",
                1.0,
            ),
            openalex_min_relevance_score=_read_non_negative_float(
                "OPENALEX_MIN_RELEVANCE_SCORE",
                10.0,
            ),
            openalex_validation_max_candidates=_read_positive_int(
                "OPENALEX_VALIDATION_MAX_CANDIDATES",
                25,
            ),
            openalex_require_validated_date=_read_bool(
                "OPENALEX_REQUIRE_VALIDATED_DATE",
                True,
            ),
            openalex_require_day_precision=_read_bool(
                "OPENALEX_REQUIRE_DAY_PRECISION",
                True,
            ),
            openalex_bibliographic_timeout_seconds=_read_positive_float(
                "OPENALEX_BIBLIOGRAPHIC_TIMEOUT_SECONDS",
                20.0,
            ),
            openalex_bibliographic_max_attempts=_read_positive_int(
                "OPENALEX_BIBLIOGRAPHIC_MAX_ATTEMPTS",
                3,
            ),
            openalex_bibliographic_min_request_interval_seconds=(
                _read_positive_float(
                    "OPENALEX_BIBLIOGRAPHIC_MIN_REQUEST_INTERVAL_SECONDS",
                    0.5,
                )
            ),
            gemini_max_attempts=_read_positive_int(
                "GEMINI_MAX_ATTEMPTS",
                4,
            ),
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
                False,
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate cross-field and semantic constraints."""

        if self.papers_per_source != 3:
            raise ConfigurationError(
                "PAPERS_PER_SOURCE must be 3 for the current three-source digest."
            )
        if self.arxiv_max_results_per_query > 2000:
            raise ConfigurationError(
                "ARXIV_MAX_RESULTS_PER_QUERY must be <= 2000."
            )
        if self.arxiv_min_request_interval_seconds < 3.0:
            raise ConfigurationError(
                "ARXIV_MIN_REQUEST_INTERVAL_SECONDS must be >= 3.0."
            )
        if self.semantic_scholar_max_results_per_query > 1000:
            raise ConfigurationError(
                "SEMANTIC_SCHOLAR_MAX_RESULTS_PER_QUERY must be <= 1000."
            )
        if self.openalex_max_results_per_query > 50:
            raise ConfigurationError(
                "OPENALEX_MAX_RESULTS_PER_QUERY must be <= 50 for semantic search."
            )
        if self.openalex_min_request_interval_seconds < 1.0:
            raise ConfigurationError(
                "OPENALEX_MIN_REQUEST_INTERVAL_SECONDS must be >= 1.0."
            )
        if self.openalex_validation_max_candidates < self.papers_per_source:
            raise ConfigurationError(
                "OPENALEX_VALIDATION_MAX_CANDIDATES must be >= PAPERS_PER_SOURCE."
            )
        if not self.gemini_model:
            raise ConfigurationError("GEMINI_MODEL cannot be empty.")
        if self.log_level not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ConfigurationError(
                "LOG_LEVEL must be one of CRITICAL, ERROR, WARNING, INFO, or DEBUG."
            )


def _load_local_dotenv() -> None:
    """Load a repository-local .env file when one is available."""

    candidates = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    )

    for dotenv_path in candidates:
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return


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


def _read_non_negative_float(name: str, default: float) -> float:
    """Read a non-negative float environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric.") from exc

    if value < 0:
        raise ConfigurationError(f"{name} must be >= 0.")

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
        f"{name} must be a boolean value: true/false, yes/no, on/off, or 1/0."
    )


def _read_optional_string(name: str) -> str | None:
    """Read an optional non-empty string environment variable."""

    value = os.getenv(name, "").strip()
    return value or None
