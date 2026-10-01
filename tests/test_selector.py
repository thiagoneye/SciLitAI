"""Unit tests for strict-window relevance ranking."""

from datetime import datetime, timezone

import pytest

from sciml_digest.exceptions import SelectionError
from sciml_digest.models import ArxivPaper
from sciml_digest.selector import select_top_papers


def _paper(
    identifier: str,
    title: str,
    abstract: str,
    published_hour: int,
) -> ArxivPaper:
    """Create a minimal paper fixture inside the target UTC day."""

    timestamp = datetime(2026, 9, 30, published_hour, 0, tzinfo=timezone.utc)
    return ArxivPaper(
        arxiv_id=identifier,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        published_date=timestamp,
        updated_date=timestamp,
        arxiv_url=f"https://arxiv.org/abs/{identifier}",
        pdf_url=f"https://arxiv.org/pdf/{identifier}",
        primary_category="cs.LG",
        categories=["cs.LG"],
    )


def test_select_top_papers_prefers_dense_topic_matches() -> None:
    """Rank stronger title and abstract term density above weaker matches."""

    papers = [
        _paper(
            "2609.00001",
            "PINN FNO DeepONet Neural Operator SciML for CFD",
            "PINN FNO DeepONet surrogate modeling UQ CFD.",
            1,
        ),
        _paper("2609.00002", "Digital Twin", "A digital twin for maintenance.", 2),
        _paper("2609.00003", "Surrogate Modeling", "A surrogate model for engineering.", 3),
        _paper("2609.00004", "Uncertainty Quantification", "UQ with Gaussian Process.", 4),
        _paper("2609.00005", "Finite Element Surrogate", "FEM surrogate modeling.", 5),
        _paper("2609.00006", "Predictive Maintenance", "RUL prediction for IIoT.", 6),
    ]

    start = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)

    selected = select_top_papers(papers, start, end, limit=5)

    assert len(selected) == 5
    assert selected[0].paper.arxiv_id == "2609.00001"
    assert [item.rank for item in selected] == [1, 2, 3, 4, 5]


def test_select_top_papers_rejects_insufficient_daily_candidates() -> None:
    """Fail instead of backfilling when fewer than five D-1 papers exist."""

    papers = [
        _paper("2609.00001", "PINN", "Physics-Informed method.", 1),
        _paper("2609.00002", "FNO", "Fourier Neural Operator.", 2),
    ]
    start = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(SelectionError):
        select_top_papers(papers, start, end, limit=5)
