"""Typed data models used across the SciLitAI pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


AiApproach = Literal[
    "PINNs",
    "FNO",
    "DeepONet",
    "Neural Operator",
    "MLP Surrogate",
    "GNN",
    "Hybrid SciML",
    "Digital Twin",
    "Uncertainty Quantification",
    "Other SciML",
]


class ArxivPaper(BaseModel):
    """Normalized and sanitized metadata for one arXiv paper."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: datetime
    updated_date: datetime
    arxiv_url: HttpUrl
    pdf_url: HttpUrl
    primary_category: str
    categories: list[str] = Field(default_factory=list)
    matched_clusters: list[str] = Field(default_factory=list)


class EnrichedArticle(BaseModel):
    """Structured Gemini output for one paper."""

    title: str = Field(description="Exact paper title.")
    ai_approach: AiApproach = Field(
        description="Primary Scientific Machine Learning approach used by the paper."
    )
    domain_application: str = Field(
        description=(
            "Primary physical or industrial application domain, for example "
            "'Fluid Dynamics / Aerodynamics' or 'General Methodology'."
        )
    )
    executive_summary: str = Field(
        description=(
            "Objective 2-3 sentence summary describing the problem, method, and "
            "computational or scientific impact."
        )
    )
    arxiv_url: HttpUrl = Field(description="Direct arXiv abstract page URL.")


class SelectedPaper(BaseModel):
    """Paper plus deterministic ranking metadata."""

    model_config = ConfigDict(frozen=True)

    paper: ArxivPaper
    rank: int = Field(ge=1)
    relevance_score: float = Field(ge=0.0)


class DigestItem(BaseModel):
    """Final item combining source metadata and AI enrichment."""

    model_config = ConfigDict(frozen=True)

    selected: SelectedPaper
    enriched: EnrichedArticle
