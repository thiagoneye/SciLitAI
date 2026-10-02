"""External DOI metadata client for conservative publication-date validation."""

from __future__ import annotations

import calendar
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as day_time, timezone
from typing import Any
from urllib.parse import quote

import httpx

from sciml_digest.exceptions import BibliographicMetadataError, SelectionError
from sciml_digest.models import (
    PublicationDatePrecision,
    PublicationDateValidationSource,
    ScientificPaper,
)
from sciml_digest.source_utils import retry_delay

LOGGER = logging.getLogger(__name__)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
DATACITE_DOIS_URL = "https://api.datacite.org/dois"
RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class PublicationDateEvidence:
    """Publication-date evidence retrieved from a DOI registration agency."""

    publication_date: date
    precision: PublicationDatePrecision
    source: PublicationDateValidationSource

    @property
    def interval_start(self) -> date:
        """Return the earliest calendar date represented by this evidence."""

        return self.publication_date

    @property
    def interval_end(self) -> date:
        """Return the latest calendar date represented by this evidence."""

        if self.precision == "day":
            return self.publication_date
        if self.precision == "month":
            last_day = calendar.monthrange(
                self.publication_date.year,
                self.publication_date.month,
            )[1]
            return date(
                self.publication_date.year,
                self.publication_date.month,
                last_day,
            )
        return date(self.publication_date.year, 12, 31)

    def is_definitely_within(self, start_date: date, end_date: date) -> bool:
        """Return whether the entire represented date interval is inside the window."""

        return start_date <= self.interval_start and self.interval_end <= end_date


class BibliographicDateClient:
    """Resolve DOI publication dates using Crossref first and DataCite second."""

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        min_request_interval_seconds: float = 0.5,
        user_agent: str = "SciLitAI/4.0",
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0.")
        if min_request_interval_seconds <= 0:
            raise ValueError("min_request_interval_seconds must be > 0.")

        self._max_attempts = max_attempts
        self._min_request_interval_seconds = min_request_interval_seconds
        self._last_request_started_at: float | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "BibliographicDateClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def resolve_publication_date(self, doi: str) -> PublicationDateEvidence | None:
        """Resolve the earliest publication-like DOI date from Crossref or DataCite."""

        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            return None

        encoded_doi = quote(normalized_doi, safe="")

        crossref_payload = self._request_json(
            f"{CROSSREF_WORKS_URL}/{encoded_doi}",
            provider="Crossref",
        )
        if crossref_payload is not None:
            evidence = _parse_crossref_publication_date(crossref_payload)
            if evidence is not None:
                return evidence

        datacite_payload = self._request_json(
            f"{DATACITE_DOIS_URL}/{encoded_doi}",
            provider="DataCite",
        )
        if datacite_payload is not None:
            return _parse_datacite_publication_date(datacite_payload)

        return None

    def _request_json(
        self,
        url: str,
        provider: str,
    ) -> dict[str, Any] | None:
        """Retrieve one metadata record with bounded retries and 404 fall-through."""

        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._respect_request_interval()

            try:
                self._last_request_started_at = time.monotonic()
                response = self._client.get(url)

                if response.status_code == 404:
                    return None

                if response.status_code in RETRIABLE_STATUS_CODES:
                    if attempt == self._max_attempts:
                        raise BibliographicMetadataError(
                            f"{provider} remained unavailable after retries "
                            f"(HTTP {response.status_code})."
                        )

                    delay = retry_delay(
                        attempt,
                        retry_after=response.headers.get("Retry-After"),
                        minimum_seconds=self._min_request_interval_seconds,
                    )
                    LOGGER.warning(
                        "%s transient HTTP %s; retrying in %.1fs.",
                        provider,
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise BibliographicMetadataError(
                        f"{provider} returned a non-object JSON response."
                    )
                return payload

            except httpx.HTTPStatusError as exc:
                raise BibliographicMetadataError(
                    f"{provider} metadata request failed with HTTP "
                    f"{exc.response.status_code}."
                ) from exc
            except (httpx.RequestError, ValueError) as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break

                delay = retry_delay(
                    attempt,
                    minimum_seconds=self._min_request_interval_seconds,
                )
                LOGGER.warning(
                    "%s metadata request error (%s); retrying in %.1fs.",
                    provider,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise BibliographicMetadataError(
            f"{provider} metadata request failed after all retry attempts."
        ) from last_error

    def _respect_request_interval(self) -> None:
        """Apply a minimum interval between DOI metadata requests."""

        if self._last_request_started_at is None:
            return

        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def validate_openalex_publication_dates(
    papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
    required_count: int,
    max_candidates: int,
    client: BibliographicDateClient,
    require_validated_date: bool = True,
    require_day_precision: bool = True,
) -> list[ScientificPaper]:
    """Validate ranked OpenAlex candidates until enough eligible papers are found."""

    if start_date > end_date:
        raise ValueError("start_date must not be later than end_date.")
    if required_count <= 0:
        raise ValueError("required_count must be > 0.")
    if max_candidates < required_count:
        raise ValueError("max_candidates must be >= required_count.")

    validated: list[ScientificPaper] = []

    for paper in papers[:max_candidates]:
        if paper.source != "openalex":
            continue
        if not start_date <= paper.publication_date.date() <= end_date:
            continue

        evidence = (
            client.resolve_publication_date(paper.doi)
            if paper.doi is not None
            else None
        )

        if evidence is None:
            if require_validated_date:
                LOGGER.info(
                    "Rejecting OpenAlex work=%s because no DOI publication-date "
                    "evidence was found.",
                    paper.source_id,
                )
                continue

            validated.append(paper)
            if len(validated) == required_count:
                break
            continue

        if require_day_precision and evidence.precision != "day":
            LOGGER.info(
                "Rejecting OpenAlex work=%s because DOI date precision=%s is "
                "insufficient for a strict short recency window.",
                paper.source_id,
                evidence.precision,
            )
            continue

        if not evidence.is_definitely_within(start_date, end_date):
            LOGGER.info(
                "Rejecting OpenAlex work=%s because validated publication date "
                "%s (%s, %s precision) is outside [%s, %s].",
                paper.source_id,
                evidence.publication_date.isoformat(),
                evidence.source,
                evidence.precision,
                start_date.isoformat(),
                end_date.isoformat(),
            )
            continue

        validated_date = datetime.combine(
            evidence.publication_date,
            day_time.min,
            tzinfo=timezone.utc,
        )
        source_publication_date = (
            paper.source_publication_date or paper.publication_date
        )

        if source_publication_date.date() != validated_date.date():
            LOGGER.warning(
                "OpenAlex work=%s source date=%s differs from validated %s "
                "date=%s.",
                paper.source_id,
                source_publication_date.date().isoformat(),
                evidence.source,
                validated_date.date().isoformat(),
            )

        validated.append(
            paper.model_copy(
                update={
                    "publication_date": validated_date,
                    "source_publication_date": source_publication_date,
                    "publication_date_verified": True,
                    "publication_date_validation_source": evidence.source,
                    "publication_date_precision": evidence.precision,
                }
            )
        )

        if len(validated) == required_count:
            break

    if len(validated) < required_count:
        raise SelectionError(
            f"Only {len(validated)} OpenAlex papers passed bibliographic date "
            f"validation; {required_count} are required."
        )

    return validated


def normalize_doi(doi: str) -> str:
    """Normalize DOI URL and prefix variants to a canonical identifier."""

    normalized = doi.strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
        "doi:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.strip()


def _parse_crossref_publication_date(
    payload: dict[str, Any],
) -> PublicationDateEvidence | None:
    """Extract the earliest publication-like date from a Crossref work record."""

    message = payload.get("message")
    if not isinstance(message, dict):
        return None

    candidates: list[PublicationDateEvidence] = []
    for field in (
        "published-print",
        "published-online",
        "issued",
        "published",
    ):
        evidence = _parse_crossref_date_parts(message.get(field))
        if evidence is not None:
            candidates.append(evidence)

    return _earliest_evidence(candidates)


def _parse_crossref_date_parts(value: Any) -> PublicationDateEvidence | None:
    """Parse Crossref date-parts metadata into publication-date evidence."""

    if not isinstance(value, dict):
        return None
    date_parts = value.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts:
        return None
    first = date_parts[0]
    if not isinstance(first, list) or not first:
        return None

    return _build_date_evidence(first, source="crossref")


def _parse_datacite_publication_date(
    payload: dict[str, Any],
) -> PublicationDateEvidence | None:
    """Extract the earliest issued/publication date from a DataCite DOI record."""

    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    attributes = data.get("attributes")
    if not isinstance(attributes, dict):
        return None

    candidates: list[PublicationDateEvidence] = []

    published = attributes.get("published")
    evidence = _parse_flexible_date(published, source="datacite")
    if evidence is not None:
        candidates.append(evidence)

    for date_entry in attributes.get("dates") or []:
        if not isinstance(date_entry, dict):
            continue
        date_type = str(date_entry.get("dateType") or "").casefold()
        if date_type != "issued":
            continue
        evidence = _parse_flexible_date(
            date_entry.get("date"),
            source="datacite",
        )
        if evidence is not None:
            candidates.append(evidence)

    if not candidates:
        publication_year = attributes.get("publicationYear")
        if isinstance(publication_year, int):
            candidates.append(
                PublicationDateEvidence(
                    publication_date=date(publication_year, 1, 1),
                    precision="year",
                    source="datacite",
                )
            )

    return _earliest_evidence(candidates)


def _parse_flexible_date(
    value: Any,
    source: PublicationDateValidationSource,
) -> PublicationDateEvidence | None:
    """Parse a year, month, date, or ISO datetime into date evidence."""

    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    if len(normalized) == 4 and normalized.isdigit():
        return PublicationDateEvidence(
            publication_date=date(int(normalized), 1, 1),
            precision="year",
            source=source,
        )

    if len(normalized) == 7 and normalized[4] == "-":
        try:
            year, month = (int(part) for part in normalized.split("-"))
            parsed = date(year, month, 1)
        except ValueError:
            return None
        return PublicationDateEvidence(
            publication_date=parsed,
            precision="month",
            source=source,
        )

    try:
        parsed = date.fromisoformat(normalized[:10])
    except ValueError:
        return None

    return PublicationDateEvidence(
        publication_date=parsed,
        precision="day",
        source=source,
    )


def _build_date_evidence(
    parts: list[Any],
    source: PublicationDateValidationSource,
) -> PublicationDateEvidence | None:
    """Build date evidence from Crossref-style numeric date parts."""

    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else 1
        day = int(parts[2]) if len(parts) >= 3 else 1
        parsed = date(year, month, day)
    except (TypeError, ValueError):
        return None

    precision: PublicationDatePrecision
    if len(parts) >= 3:
        precision = "day"
    elif len(parts) == 2:
        precision = "month"
    else:
        precision = "year"

    return PublicationDateEvidence(
        publication_date=parsed,
        precision=precision,
        source=source,
    )


def _earliest_evidence(
    candidates: list[PublicationDateEvidence],
) -> PublicationDateEvidence | None:
    """Return the earliest available publication-date evidence."""

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda evidence: (
            evidence.interval_start,
            evidence.interval_end,
        ),
    )
