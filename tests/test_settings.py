"""Tests for environment-driven settings."""

from __future__ import annotations

import pytest

from sciml_digest.exceptions import ConfigurationError
from sciml_digest.settings import _read_bool


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_read_bool_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    """Parse all supported boolean environment representations."""

    monkeypatch.setenv("TEST_BOOLEAN", raw_value)
    assert _read_bool("TEST_BOOLEAN", False) is expected


def test_read_bool_uses_default_when_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the supplied default when the variable is absent."""

    monkeypatch.delenv("TEST_BOOLEAN", raising=False)
    assert _read_bool("TEST_BOOLEAN", True) is True


def test_read_bool_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported boolean environment values."""

    monkeypatch.setenv("TEST_BOOLEAN", "sometimes")

    with pytest.raises(ConfigurationError):
        _read_bool("TEST_BOOLEAN", False)


def test_openalex_defaults_are_strict() -> None:
    """Keep strict OpenAlex recency and validation defaults."""

    from sciml_digest.settings import Settings

    settings = Settings(
        gemini_api_key="g",
        telegram_bot_token="t",
        telegram_chat_id="c",
    )

    settings.validate()
    assert settings.openalex_lookback_days == 60
    assert settings.openalex_min_relevance_score == 10.0
    assert settings.openalex_require_validated_date is True
    assert settings.openalex_require_day_precision is True
    assert settings.openalex_max_results_per_query == 50
