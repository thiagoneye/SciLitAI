"""Unit tests for Semantic Scholar normalization."""

from sciml_digest.semantic_scholar_client import _parse_paper


def test_parse_semantic_scholar_paper() -> None:
    """Normalize API metadata into the shared paper model."""

    raw = {
        "paperId": "abc123",
        "title": "Physics-Informed Neural Operators",
        "abstract": "A neural operator surrogate for CFD.",
        "authors": [{"name": "Ada Lovelace"}],
        "url": "https://www.semanticscholar.org/paper/abc123",
        "publicationDate": "2026-09-20",
        "citationCount": 12,
        "externalIds": {"DOI": "10.1000/test", "ArXiv": "2609.12345"},
        "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        "fieldsOfStudy": ["Computer Science"],
        "s2FieldsOfStudy": [{"category": "Physics"}],
    }

    paper = _parse_paper(raw)

    assert paper is not None
    assert paper.source == "semantic_scholar"
    assert paper.doi == "10.1000/test"
    assert paper.arxiv_id == "2609.12345"
    assert paper.citation_count == 12
    assert "Physics" in paper.categories


def test_parse_semantic_scholar_rejects_unknown_exact_date() -> None:
    """Reject records that cannot prove membership in the 30-day date window."""

    raw = {
        "paperId": "abc123",
        "title": "PINN",
        "abstract": "Physics-Informed model.",
        "publicationDate": None,
        "year": 2026,
    }

    assert _parse_paper(raw) is None
