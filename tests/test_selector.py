"""Unit tests for multi-source ranking and deduplication."""

from datetime import date, datetime, timezone

import pytest

from sciml_digest.exceptions import SelectionError
from sciml_digest.models import ScientificPaper
from sciml_digest.selector import (
    paper_identity_keys,
    relevance_score,
    select_arxiv_papers,
    select_openalex_papers,
    select_semantic_scholar_papers,
)


def _paper(
    source: str,
    identifier: str,
    title: str,
    abstract: str,
    publication_date: datetime,
    *,
    citation_count: int | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> ScientificPaper:
    """Create a generic paper fixture."""

    source_urls = {
        "arxiv": f"https://arxiv.org/abs/{identifier}",
        "semantic_scholar": f"https://www.semanticscholar.org/paper/{identifier}",
        "openalex": f"https://openalex.org/{identifier}",
    }
    return ScientificPaper(
        source=source,
        source_id=identifier,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        publication_date=publication_date,
        updated_date=publication_date if source == "arxiv" else None,
        source_url=source_urls[source],
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        primary_category="cs.LG" if source == "arxiv" else None,
        categories=["cs.LG"] if source == "arxiv" else [],
    )


def test_relevance_score_prioritizes_high_weight_clusters() -> None:
    """Give high-priority SciML terms more influence than UQ-only terms."""

    timestamp = datetime(2026, 10, 1, tzinfo=timezone.utc)
    sciml = _paper(
        "semantic_scholar",
        "s1",
        "Physics-Informed PINN method",
        "Scientific Machine Learning with PINNs.",
        timestamp,
    )
    uq = _paper(
        "semantic_scholar",
        "s2",
        "Uncertainty Quantification method",
        "Bayesian uncertainty and MCMC.",
        timestamp,
    )

    assert relevance_score(sciml) > relevance_score(uq)


def test_arxiv_selects_three_relevant_d1_papers() -> None:
    """Select exactly three arXiv papers from the strict D-1 interval."""

    start = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    papers = [
        _paper(
            "arxiv",
            f"2609.0000{index}",
            title,
            abstract,
            datetime(2026, 9, 30, index, tzinfo=timezone.utc),
            arxiv_id=f"2609.0000{index}",
        )
        for index, (title, abstract) in enumerate(
            [
                ("PINN for CFD", "Physics-Informed method for fluid dynamics."),
                ("FNO surrogate", "Fourier Neural Operator surrogate modeling."),
                ("Digital Twin", "Digital Twin predictive maintenance."),
                ("UQ method", "Uncertainty Quantification with MCMC."),
            ],
            start=1,
        )
    ]

    selected = select_arxiv_papers(papers, start, end, limit=3)

    assert len(selected) == 3
    assert [item.rank for item in selected] == [1, 2, 3]
    assert selected[0].selection_basis == "relevance"


def test_semantic_scholar_excludes_prior_source_duplicate() -> None:
    """Backfill a duplicate paper using DOI/title identity keys."""

    publication = datetime(2026, 9, 20, tzinfo=timezone.utc)
    duplicate = _paper(
        "semantic_scholar",
        "s1",
        "Physics-Informed Neural Operator",
        "PINN and FNO surrogate.",
        publication,
        doi="10.1000/example",
    )
    alternatives = [
        _paper(
            "semantic_scholar",
            f"s{index}",
            f"FNO surrogate {index}",
            "Fourier Neural Operator surrogate modeling.",
            publication,
        )
        for index in range(2, 6)
    ]

    selected = select_semantic_scholar_papers(
        [duplicate, *alternatives],
        start_date=date(2026, 9, 2),
        end_date=date(2026, 10, 1),
        limit=3,
        excluded_identity_keys={"doi:10.1000/example"},
    )

    assert len(selected) == 3
    assert all(item.paper.source_id != "s1" for item in selected)


def test_openalex_ranks_by_citations_before_relevance() -> None:
    """Use citation count as the primary OpenAlex ranking criterion."""

    publication = datetime(2026, 6, 1, tzinfo=timezone.utc)
    papers = [
        _paper(
            "openalex",
            "W1",
            "PINN FNO SciML",
            "Physics-Informed Fourier Neural Operator.",
            publication,
            citation_count=5,
        ),
        _paper(
            "openalex",
            "W2",
            "CFD surrogate",
            "Computational Fluid Dynamics surrogate modeling.",
            publication,
            citation_count=120,
        ),
        _paper(
            "openalex",
            "W3",
            "Digital Twin maintenance",
            "Digital Twin predictive maintenance IIoT.",
            publication,
            citation_count=80,
        ),
        _paper(
            "openalex",
            "W4",
            "Uncertainty Quantification",
            "UQ Bayesian method.",
            publication,
            citation_count=60,
        ),
    ]

    selected = select_openalex_papers(
        papers,
        start_date=date(2025, 10, 2),
        end_date=date(2026, 10, 1),
        limit=3,
    )

    assert [item.paper.source_id for item in selected] == ["W2", "W3", "W4"]
    assert all(item.selection_basis == "citations" for item in selected)


def test_selection_fails_when_source_has_fewer_than_three_unique_papers() -> None:
    """Keep the three-per-source requirement strict."""

    publication = datetime(2026, 9, 20, tzinfo=timezone.utc)
    papers = [
        _paper(
            "semantic_scholar",
            "s1",
            "PINN",
            "Physics-Informed Neural Network.",
            publication,
        ),
        _paper(
            "semantic_scholar",
            "s2",
            "FNO",
            "Fourier Neural Operator.",
            publication,
        ),
    ]

    with pytest.raises(SelectionError):
        select_semantic_scholar_papers(
            papers,
            start_date=date(2026, 9, 2),
            end_date=date(2026, 10, 1),
            limit=3,
        )


def test_identity_keys_normalize_doi_and_title() -> None:
    """Generate source-independent DOI and title keys."""

    paper = _paper(
        "openalex",
        "W1",
        "Physics-Informed: Neural Operators!",
        "PINN FNO.",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        doi="https://doi.org/10.1000/ABC",
    )

    keys = paper_identity_keys(paper)

    assert "doi:10.1000/abc" in keys
    assert "title:physics informed neural operators" in keys


def test_openalex_rejects_acronym_only_false_positive() -> None:
    """Reject a citation-rich paper that lacks a high-precision thematic phrase."""

    publication = datetime(2026, 9, 20, tzinfo=timezone.utc)
    false_positive = _paper(
        "openalex",
        "W-false",
        "Hidden Perceptions in Public Libraries",
        "The study reports SVD analysis and ROM storage terminology.",
        publication,
        citation_count=500,
    )
    relevant = _paper(
        "openalex",
        "W-relevant",
        "Digital Twin for Predictive Maintenance",
        "A Digital Twin supports predictive maintenance in smart manufacturing.",
        publication,
        citation_count=20,
    )
    alternatives = [
        _paper(
            "openalex",
            f"W-alt-{index}",
            f"Computational Fluid Dynamics surrogate {index}",
            "Computational Fluid Dynamics with surrogate modeling.",
            publication,
            citation_count=19 - index,
        )
        for index in range(2)
    ]

    selected = select_openalex_papers(
        [false_positive, relevant, *alternatives],
        start_date=date(2026, 8, 4),
        end_date=date(2026, 10, 2),
        limit=3,
        min_relevance_score=10.0,
    )

    assert all(item.paper.source_id != "W-false" for item in selected)
