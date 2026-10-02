"""arXiv Atom API client with clustered queries, pacing, and retries."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

from sciml_digest.exceptions import ArxivError
from sciml_digest.models import ScientificPaper
from sciml_digest.source_utils import chunked, clean_text, retry_delay
from sciml_digest.taxonomy import SEARCH_CLUSTERS, TARGET_CATEGORIES

LOGGER = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DATE_FIELDS = ("submittedDate", "lastUpdatedDate")


class ArxivClient:
    """HTTP client for clustered retrieval from the public arXiv Atom API."""

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        min_request_interval_seconds: float = 3.0,
        user_agent: str = "SciLitAI/3.0",
    ) -> None:
        if min_request_interval_seconds < 3.0:
            raise ValueError("min_request_interval_seconds must be >= 3.0.")

        self._max_attempts = max_attempts
        self._min_request_interval_seconds = min_request_interval_seconds
        self._last_request_started_at: float | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/atom+xml",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_candidates(
        self,
        start_utc: datetime,
        end_utc: datetime,
        max_results_per_query: int = 100,
        terms_per_query: int = 12,
    ) -> list[ScientificPaper]:
        """Fetch papers submitted or updated inside one strict UTC interval."""

        start_utc = _as_utc(start_utc)
        end_utc = _as_utc(end_utc)

        if start_utc >= end_utc:
            raise ValueError("start_utc must be earlier than end_utc.")
        if max_results_per_query <= 0:
            raise ValueError("max_results_per_query must be > 0.")

        papers_by_id: dict[str, ScientificPaper] = {}

        for cluster_name, terms in SEARCH_CLUSTERS.items():
            for term_chunk in chunked(terms, terms_per_query):
                for date_field in DATE_FIELDS:
                    query = _build_query(
                        categories=TARGET_CATEGORIES,
                        terms=term_chunk,
                        date_field=date_field,
                        start_utc=start_utc,
                        end_utc=end_utc,
                    )
                    params = {
                        "search_query": query,
                        "start": 0,
                        "max_results": max_results_per_query,
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    }

                    LOGGER.info(
                        "Querying arXiv cluster=%s date_field=%s terms=%s.",
                        cluster_name,
                        date_field,
                        len(term_chunk),
                    )
                    response = self._request(params)
                    parsed = self._parse_feed(response.text)

                    for paper in parsed:
                        if paper.primary_category not in TARGET_CATEGORIES:
                            continue
                        if not _is_in_window(paper, start_utc, end_utc):
                            continue

                        paper_with_cluster = paper.model_copy(
                            update={"matched_clusters": [cluster_name]}
                        )
                        existing = papers_by_id.get(paper.source_id)
                        if existing is None:
                            papers_by_id[paper.source_id] = paper_with_cluster
                        else:
                            papers_by_id[paper.source_id] = _merge_papers(
                                existing,
                                paper_with_cluster,
                            )

        papers = list(papers_by_id.values())
        papers.sort(
            key=lambda paper: (
                paper.updated_date or paper.publication_date,
                paper.publication_date,
            ),
            reverse=True,
        )
        return papers

    def _request(self, params: dict[str, Any]) -> httpx.Response:
        """Execute one arXiv request with pacing and bounded retries."""

        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._respect_request_interval()

            try:
                self._last_request_started_at = time.monotonic()
                response = self._client.get(ARXIV_API_URL, params=params)

                if response.status_code in RETRIABLE_STATUS_CODES:
                    if attempt == self._max_attempts:
                        raise ArxivError(
                            "arXiv API remained unavailable after retries "
                            f"(HTTP {response.status_code})."
                        )

                    delay = retry_delay(
                        attempt=attempt,
                        retry_after=response.headers.get("Retry-After"),
                        minimum_seconds=self._min_request_interval_seconds,
                    )
                    LOGGER.warning(
                        "arXiv transient HTTP %s; retrying in %.1fs.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as exc:
                raise ArxivError(
                    f"arXiv request failed with HTTP {exc.response.status_code}."
                ) from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break

                delay = retry_delay(
                    attempt=attempt,
                    minimum_seconds=self._min_request_interval_seconds,
                )
                LOGGER.warning(
                    "arXiv network error (%s); retrying in %.1fs.",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise ArxivError("arXiv request failed after all retry attempts.") from last_error

    def _respect_request_interval(self) -> None:
        """Ensure consecutive arXiv requests start at least three seconds apart."""

        if self._last_request_started_at is None:
            return

        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _parse_feed(payload: str) -> list[ScientificPaper]:
        """Parse and normalize an Atom feed into generic paper records."""

        feed = feedparser.parse(payload)
        if getattr(feed, "bozo", False) and not getattr(feed, "entries", []):
            raise ArxivError("arXiv returned an invalid or unreadable Atom feed.")

        papers: list[ScientificPaper] = []
        for entry in feed.entries:
            try:
                papers.append(_parse_entry(entry))
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Skipping malformed arXiv entry due to %s.",
                    type(exc).__name__,
                )

        return papers


def _build_query(
    categories: Sequence[str],
    terms: Sequence[str],
    date_field: str,
    start_utc: datetime,
    end_utc: datetime,
) -> str:
    """Build one category AND topic AND date query for the arXiv API."""

    if date_field not in DATE_FIELDS:
        raise ValueError(f"Unsupported date field: {date_field}")

    category_clause = " OR ".join(f"cat:{category}" for category in categories)

    topic_clauses: list[str] = []
    for term in terms:
        escaped_term = term.replace('"', r'\"')
        topic_clauses.append(f'(ti:"{escaped_term}" OR abs:"{escaped_term}")')

    start_token = start_utc.strftime("%Y%m%d%H%M")
    # A API usa resolução de minuto; o minuto final cobre integralmente D-1.
    end_token = (end_utc - timedelta(minutes=1)).strftime("%Y%m%d%H%M")

    return (
        f"({category_clause}) AND "
        f"({' OR '.join(topic_clauses)}) AND "
        f"{date_field}:[{start_token} TO {end_token}]"
    )


def _parse_entry(entry: Any) -> ScientificPaper:
    """Normalize one feedparser arXiv entry."""

    raw_id = str(entry["id"]).strip().rstrip("/")
    arxiv_id = _canonical_arxiv_id(raw_id)

    title = clean_text(str(entry["title"]))
    abstract = clean_text(str(entry["summary"]))
    authors = [
        clean_text(str(author["name"]))
        for author in entry.get("authors", [])
        if author.get("name")
    ]
    categories = [
        str(tag.get("term", "")).strip()
        for tag in entry.get("tags", [])
        if tag.get("term")
    ]
    primary_category = _extract_primary_category(entry, categories)

    publication_date = _parse_datetime(str(entry["published"]))
    updated_date = _parse_datetime(str(entry.get("updated", entry["published"])))
    source_url = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = _extract_pdf_url(entry, arxiv_id)
    doi = _extract_doi(entry)

    return ScientificPaper(
        source="arxiv",
        source_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        publication_date=publication_date,
        updated_date=updated_date,
        source_url=source_url,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        primary_category=primary_category,
        categories=categories,
    )


def _canonical_arxiv_id(raw_id: str) -> str:
    """Extract a canonical arXiv identifier, including legacy archive prefixes."""

    identifier = re.sub(
        r"^https?://(?:export\.)?arxiv\.org/abs/",
        "",
        raw_id,
        flags=re.IGNORECASE,
    )
    return re.sub(r"v\d+$", "", identifier)


def _extract_primary_category(entry: Any, categories: list[str]) -> str:
    """Extract the arXiv primary category with a deterministic fallback."""

    primary = entry.get("arxiv_primary_category", {})
    if isinstance(primary, dict):
        term = str(primary.get("term", "")).strip()
        if term:
            return term

    return categories[0] if categories else ""


def _extract_pdf_url(entry: Any, arxiv_id: str) -> str:
    """Extract the PDF URL or construct the canonical fallback."""

    for link in entry.get("links", []):
        if link.get("type") == "application/pdf" and link.get("href"):
            return str(link["href"]).replace("http://", "https://")
    return f"https://arxiv.org/pdf/{arxiv_id}"


def _extract_doi(entry: Any) -> str | None:
    """Extract DOI metadata when arXiv exposes it."""

    value = entry.get("arxiv_doi")
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("href")
    normalized = str(value).strip()
    return normalized or None


def _parse_datetime(value: str) -> datetime:
    """Parse an Atom timestamp and normalize it to UTC."""

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Validate timezone awareness and normalize a datetime to UTC."""

    if value.tzinfo is None:
        raise ValueError("Datetime values must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _is_in_window(
    paper: ScientificPaper,
    start_utc: datetime,
    end_utc: datetime,
) -> bool:
    """Return whether publication or last update falls in the target interval."""

    published = start_utc <= paper.publication_date < end_utc
    updated = (
        paper.updated_date is not None
        and start_utc <= paper.updated_date < end_utc
    )
    return published or updated


def _merge_papers(
    left: ScientificPaper,
    right: ScientificPaper,
) -> ScientificPaper:
    """Merge duplicate arXiv records while preserving matched clusters."""

    merged_clusters = sorted(set(left.matched_clusters) | set(right.matched_clusters))
    merged_categories = list(dict.fromkeys([*left.categories, *right.categories]))
    return left.model_copy(
        update={
            "matched_clusters": merged_clusters,
            "categories": merged_categories,
        }
    )
