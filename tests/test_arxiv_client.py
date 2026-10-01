"""Unit tests for arXiv query construction and normalization."""

from datetime import datetime, timezone

import pytest

pytest.importorskip("feedparser")

from sciml_digest.arxiv_client import _build_query, _canonical_arxiv_id


def test_canonical_arxiv_id_supports_modern_ids() -> None:
    """Remove the version suffix from modern arXiv identifiers."""

    result = _canonical_arxiv_id("https://arxiv.org/abs/2403.12345v2")
    assert result == "2403.12345"


def test_canonical_arxiv_id_supports_legacy_ids() -> None:
    """Preserve the archive prefix used by legacy arXiv identifiers."""

    result = _canonical_arxiv_id("https://arxiv.org/abs/hep-th/9901001v3")
    assert result == "hep-th/9901001"


def test_build_query_combines_category_topic_and_date() -> None:
    """Build an AND-composed query with OR semantics inside each dimension."""

    start = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)

    query = _build_query(
        categories=("cs.LG", "physics.flu-dyn"),
        terms=("PINN", "Fourier Neural Operator"),
        date_field="submittedDate",
        start_utc=start,
        end_utc=end,
    )

    assert "cat:cs.LG OR cat:physics.flu-dyn" in query
    assert 'ti:"PINN" OR abs:"PINN"' in query
    assert 'ti:"Fourier Neural Operator" OR abs:"Fourier Neural Operator"' in query
    assert "submittedDate:[202609300000 TO 202609302359]" in query
