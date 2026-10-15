"""Unit tests for OpenAlex semantic retrieval helpers and normalization."""

from datetime import date

import httpx

from sciml_digest.openalex_client import (
    OpenAlexClient,
    _build_openalex_filter,
    _parse_work,
    _reconstruct_abstract,
)


def test_build_openalex_filter_uses_semantic_search_compatible_year() -> None:
    """Use a coarse year filter and leave exact date/type checks to local code."""

    result = _build_openalex_filter(
        date(2026, 8, 4),
        date(2026, 10, 2),
    )

    assert result == "publication_year:2026"


def test_build_openalex_filter_supports_year_boundary() -> None:
    """Include both publication years when the exact window crosses New Year."""

    result = _build_openalex_filter(
        date(2025, 12, 15),
        date(2026, 1, 15),
    )

    assert result == "publication_year:2025|2026"


def test_reconstruct_abstract_orders_inverted_index_positions() -> None:
    """Rebuild OpenAlex abstract text in token-position order."""

    result = _reconstruct_abstract(
        {
            "operator": [3],
            "A": [0],
            "neural": [2],
            "physics-informed": [1],
        }
    )

    assert result == "A physics-informed neural operator"


def test_parse_openalex_work_preserves_source_date_and_metadata() -> None:
    """Normalize source date, citation count, keywords, DOI, and abstract metadata."""

    raw = {
        "id": "https://openalex.org/W123",
        "display_name": "Neural Operator for CFD",
        "publication_date": "2026-09-15",
        "publication_year": 2026,
        "type": "article",
        "abstract_inverted_index": {
            "Neural": [0],
            "operator": [1],
            "for": [2],
            "CFD": [3],
        },
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "doi": "https://doi.org/10.1000/openalex",
        "cited_by_count": 42,
        "keywords": [
            {"display_name": "neural operators", "score": 0.99},
            {"display_name": "computational fluid dynamics", "score": 0.95},
        ],
        "topics": [{"display_name": "Scientific Machine Learning"}],
        "locations": [],
    }

    paper = _parse_work(raw)

    assert paper is not None
    assert paper.source == "openalex"
    assert paper.citation_count == 42
    assert paper.source_publication_date == paper.publication_date
    assert paper.publication_date_verified is False
    assert paper.source_keywords == [
        "neural operators",
        "computational fluid dynamics",
    ]
    assert paper.doi == "https://doi.org/10.1000/openalex"


def test_openalex_api_key_is_sent_as_query_parameter(monkeypatch) -> None:
    """Send configured API keys using OpenAlex's api_key query parameter."""

    client = OpenAlexClient(api_key="test-api-key")
    captured: dict[str, object] = {}

    def fake_get(url: str, *, params: dict[str, object]) -> httpx.Response:
        captured.update(params)
        return httpx.Response(
            200,
            json={},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(client._client, "get", fake_get)
    try:
        client._request({"search.semantic": "scientific machine learning"})
    finally:
        client.close()

    assert "Authorization" not in client._client.headers
    assert captured == {
        "search.semantic": "scientific machine learning",
        "api_key": "test-api-key",
    }
