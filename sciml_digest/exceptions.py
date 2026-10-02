"""Domain-specific exceptions for the SciLitAI pipeline."""


class DigestError(Exception):
    """Base exception for pipeline failures."""


class ConfigurationError(DigestError):
    """Raised when mandatory configuration is missing or invalid."""


class ArxivError(DigestError):
    """Raised when arXiv ingestion fails."""


class SemanticScholarError(DigestError):
    """Raised when Semantic Scholar ingestion fails."""


class OpenAlexError(DigestError):
    """Raised when OpenAlex ingestion fails."""


class BibliographicMetadataError(DigestError):
    """Raised when external bibliographic date validation fails."""


class SelectionError(DigestError):
    """Raised when source-specific selection criteria cannot be satisfied."""


class GeminiError(DigestError):
    """Raised when Gemini enrichment fails."""


class TelegramError(DigestError):
    """Raised when Telegram delivery fails."""
