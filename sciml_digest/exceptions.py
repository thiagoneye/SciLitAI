"""Domain-specific exceptions for the SciML digest pipeline."""


class DigestError(Exception):
    """Base exception for pipeline failures."""


class ConfigurationError(DigestError):
    """Raised when mandatory configuration is missing or invalid."""


class ArxivError(DigestError):
    """Raised when arXiv ingestion fails."""


class SelectionError(DigestError):
    """Raised when the paper selection criteria cannot be satisfied."""


class GeminiError(DigestError):
    """Raised when Gemini enrichment fails."""


class TelegramError(DigestError):
    """Raised when Telegram delivery fails."""
