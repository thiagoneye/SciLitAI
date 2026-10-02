"""Unit tests for DOI publication-date resolution and OpenAlex validation."""

from datetime import date, datetime, timezone

import pytest

from sciml_digest.bibliographic_date_client import (
    PublicationDateEvidence,
    _parse_crossref_publication_date,
    _parse_datacite_publication_date,
    validate_openalex_publication_dates,
)
from sciml_digest.exceptions import SelectionError
from sciml_digest.models import ScientificPaper


class StubBibliographicClient:
    """Return deterministic DOI publication-date evidence for tests."""

    def __init__(self, mapping: dict[str, PublicationDateEvidence | None]) -> None:
        self._mapping = mapping

    def resolve_publication_date(self, doi: str) -> PublicationDateEvidence | None:
        """Return configured evidence for one DOI."""

        return self._mapping.get(doi)


def _paper(identifier: str, doi: str, citations: int) -> ScientificPaper:
    """Create one OpenAlex paper fixture."""

    publication = datetime(2026, 9, 20, tzinfo=timezone.utc)
    return ScientificPaper(
        source="openalex",
        source_id=identifier,
        title="Digital Twin for Predictive Maintenance",
        authors=["Ada Lovelace"],
        abstract="Digital Twin predictive maintenance for smart manufacturing.",
        publication_date=publication,
        source_publication_date=publication,
        source_url=f"https://openalex.org/{identifier}",
        doi=doi,
        citation_count=citations,
    )


def test_crossref_parser_chooses_earliest_publication_date() -> None:
    """Prefer the earliest publication-like Crossref date."""

    payload = {
        "message": {
            "published-online": {"date-parts": [[2026, 9, 20]]},
            "published-print": {"date-parts": [[2026, 10, 1]]},
            "issued": {"date-parts": [[2026, 9, 20]]},
        }
    }

    evidence = _parse_crossref_publication_date(payload)

    assert evidence is not None
    assert evidence.publication_date == date(2026, 9, 20)
    assert evidence.precision == "day"
    assert evidence.source == "crossref"


def test_datacite_parser_uses_issued_date_over_registration_metadata() -> None:
    """Use publication/issued evidence rather than DOI creation or registration dates."""

    payload = {
        "data": {
            "attributes": {
                "published": "1997-04-01",
                "publicationYear": 1997,
                "created": "2026-09-26T00:00:00Z",
                "registered": "2026-09-26T00:00:00Z",
                "dates": [
                    {"date": "1997-04-01", "dateType": "Issued"},
                    {"date": "2026-09-26", "dateType": "Updated"},
                ],
            }
        }
    }

    evidence = _parse_datacite_publication_date(payload)

    assert evidence is not None
    assert evidence.publication_date == date(1997, 4, 1)
    assert evidence.precision == "day"
    assert evidence.source == "datacite"


def test_date_evidence_requires_entire_coarse_interval_inside_window() -> None:
    """Treat month/year precision conservatively for short recency windows."""

    month = PublicationDateEvidence(
        publication_date=date(2026, 9, 1),
        precision="month",
        source="datacite",
    )
    year = PublicationDateEvidence(
        publication_date=date(2026, 1, 1),
        precision="year",
        source="datacite",
    )

    assert month.is_definitely_within(date(2026, 8, 4), date(2026, 10, 2))
    assert not year.is_definitely_within(date(2026, 8, 4), date(2026, 10, 2))


def test_validation_rejects_old_republication_and_backfills() -> None:
    """Reject an old DOI even when OpenAlex reports a recent source date."""

    papers = [
        _paper("W-old", "10.1000/old", 100),
        _paper("W-new-1", "10.1000/new1", 90),
        _paper("W-new-2", "10.1000/new2", 80),
        _paper("W-new-3", "10.1000/new3", 70),
    ]
    client = StubBibliographicClient(
        {
            "10.1000/old": PublicationDateEvidence(
                publication_date=date(1997, 4, 1),
                precision="day",
                source="datacite",
            ),
            "10.1000/new1": PublicationDateEvidence(
                publication_date=date(2026, 9, 1),
                precision="day",
                source="crossref",
            ),
            "10.1000/new2": PublicationDateEvidence(
                publication_date=date(2026, 9, 10),
                precision="day",
                source="crossref",
            ),
            "10.1000/new3": PublicationDateEvidence(
                publication_date=date(2026, 9, 15),
                precision="day",
                source="datacite",
            ),
        }
    )

    validated = validate_openalex_publication_dates(
        papers,
        start_date=date(2026, 8, 4),
        end_date=date(2026, 10, 2),
        required_count=3,
        max_candidates=4,
        client=client,  # type: ignore[arg-type]
        require_validated_date=True,
        require_day_precision=True,
    )

    assert [paper.source_id for paper in validated] == [
        "W-new-1",
        "W-new-2",
        "W-new-3",
    ]
    assert all(paper.publication_date_verified for paper in validated)
    assert validated[0].publication_date.date() == date(2026, 9, 1)


def test_validation_fails_closed_when_not_enough_dates_are_verified() -> None:
    """Fail instead of widening the window or accepting unverifiable dates."""

    papers = [_paper("W1", "10.1000/unknown", 10)]
    client = StubBibliographicClient({"10.1000/unknown": None})

    with pytest.raises(SelectionError):
        validate_openalex_publication_dates(
            papers,
            start_date=date(2026, 8, 4),
            end_date=date(2026, 10, 2),
            required_count=1,
            max_candidates=1,
            client=client,  # type: ignore[arg-type]
        )
