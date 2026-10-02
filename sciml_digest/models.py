"""Typed data models used across the SciLitAI pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ScientificSource = Literal["arxiv", "semantic_scholar", "openalex"]
SelectionBasis = Literal["relevance", "citations"]
PublicationDateValidationSource = Literal["crossref", "datacite"]
PublicationDatePrecision = Literal["day", "month", "year"]


class ScientificPaper(BaseModel):
    """Normalized metadata for one scientific paper from any configured source."""

    model_config = ConfigDict(frozen=True)

    source: ScientificSource
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    publication_date: datetime
    updated_date: datetime | None = None
    source_publication_date: datetime | None = None
    publication_date_verified: bool = False
    publication_date_validation_source: PublicationDateValidationSource | None = None
    publication_date_precision: PublicationDatePrecision | None = None
    source_url: HttpUrl
    pdf_url: HttpUrl | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    primary_category: str | None = None
    categories: list[str] = Field(default_factory=list)
    source_keywords: list[str] = Field(default_factory=list)
    matched_clusters: list[str] = Field(default_factory=list)


class EnrichedArticle(BaseModel):
    """Structured Gemini enrichment shared by papers from every source."""

    title: str = Field(description="Exact paper title from the source metadata.")
    ai_approach: str = Field(
        description="Primary AI, SciML, numerical, or data-driven approach used by the paper."
    )
    domain_application: str = Field(
        description="Primary physical, engineering, computational, or industrial domain."
    )
    executive_summary: str = Field(
        description=(
            "Objective 2-3 sentence summary describing the problem, methodology, "
            "and scientific or computational impact."
        )
    )
    keywords: list[str] = Field(
        min_length=3,
        max_length=6,
        description="Three to six concise technical keywords grounded in the source metadata.",
    )
    source_url: HttpUrl = Field(description="Direct page URL for the source repository.")


class SelectedPaper(BaseModel):
    """Paper plus deterministic source-specific ranking metadata."""

    model_config = ConfigDict(frozen=True)

    paper: ScientificPaper
    rank: int = Field(ge=1, le=3)
    selection_basis: SelectionBasis
    relevance_score: float = Field(ge=0.0)


class DigestItem(BaseModel):
    """Final digest item combining source metadata and AI enrichment."""

    model_config = ConfigDict(frozen=True)

    selected: SelectedPaper
    enriched: EnrichedArticle
