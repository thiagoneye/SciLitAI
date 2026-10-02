"""OpenAlex Works API client for semantically precise recent-literature retrieval."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as day_time, timezone
from typing import Any

import httpx

from sciml_digest.exceptions import OpenAlexError
from sciml_digest.models import ScientificPaper
from sciml_digest.source_utils import clean_text, retry_delay
from sciml_digest.taxonomy import (
    OPENALEX_ALLOWED_WORK_TYPES,
    OPENALEX_SEMANTIC_QUERIES,
)

LOGGER = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
OPENALEX_SELECT_FIELDS = ",".join(
    (
        "id",
        "display_name",
        "title",
        "publication_date",
        "publication_year",
        "type",
        "abstract_inverted_index",
        "authorships",
        "doi",
        "cited_by_count",
        "keywords",
        "topics",
        "locations",
        "best_oa_location",
        "primary_location",
    )
)


class OpenAlexClient:
    """Retrieve recent OpenAlex works using cluster-specific semantic search."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        min_request_interval_seconds: float = 1.0,
        user_agent: str = "SciLitAI/4.0",
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0.")
        if min_request_interval_seconds < 1.0:
            raise ValueError(
                "min_request_interval_seconds must be >= 1.0 for semantic search."
            )

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._max_attempts = max_attempts
        self._min_request_interval_seconds = min_request_interval_seconds
        self._last_request_started_at: float | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers=headers,
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "OpenAlexClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_recent_cited_candidates(
        self,
        start_date: date,
        end_date: date,
        max_results_per_query: int = 50,
    ) -> list[ScientificPaper]:
        """Fetch semantically relevant works from the requested date window."""

        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date.")
        if max_results_per_query <= 0 or max_results_per_query > 50:
            raise ValueError(
                "max_results_per_query must be between 1 and 50 for semantic search."
            )

        papers_by_id: dict[str, ScientificPaper] = {}
        work_filter = _build_openalex_filter(start_date, end_date)

        for cluster_name, semantic_query in OPENALEX_SEMANTIC_QUERIES.items():
            params: dict[str, Any] = {
                "search.semantic": semantic_query,
                "filter": work_filter,
                "per_page": max_results_per_query,
                "select": OPENALEX_SELECT_FIELDS,
            }
            LOGGER.info("Querying OpenAlex semantic cluster=%s.", cluster_name)
            payload = self._request(params).json()

            for raw_work in payload.get("results", []):
                if not isinstance(raw_work, dict):
                    continue

                work_type = str(raw_work.get("type") or "").strip().casefold()
                if work_type not in OPENALEX_ALLOWED_WORK_TYPES:
                    continue

                paper = _parse_work(raw_work)
                if paper is None:
                    continue
                if not start_date <= paper.publication_date.date() <= end_date:
                    continue

                publication_year = raw_work.get("publication_year")
                if (
                    isinstance(publication_year, int)
                    and publication_year != paper.publication_date.year
                ):
                    LOGGER.warning(
                        "Rejecting OpenAlex work=%s due internal date mismatch: "
                        "publication_date=%s publication_year=%s.",
                        paper.source_id,
                        paper.publication_date.date().isoformat(),
                        publication_year,
                    )
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
                paper.citation_count or 0,
                paper.publication_date,
            ),
            reverse=True,
        )
        return papers

    def _request(self, params: dict[str, Any]) -> httpx.Response:
        """Execute one OpenAlex request with pacing and bounded retries."""

        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._respect_request_interval()

            try:
                self._last_request_started_at = time.monotonic()
                response = self._client.get(OPENALEX_WORKS_URL, params=params)

                if response.status_code in RETRIABLE_STATUS_CODES:
                    if attempt == self._max_attempts:
                        raise OpenAlexError(
                            "OpenAlex remained unavailable after retries "
                            f"(HTTP {response.status_code})."
                        )
                    delay = retry_delay(
                        attempt,
                        retry_after=response.headers.get("Retry-After"),
                        minimum_seconds=self._min_request_interval_seconds,
                    )
                    LOGGER.warning(
                        "OpenAlex transient HTTP %s; retrying in %.1fs.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as exc:
                detail = _extract_openalex_error_detail(exc.response)
                raise OpenAlexError(
                    "OpenAlex request failed with HTTP "
                    f"{exc.response.status_code}: {detail}"
                ) from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break

                delay = retry_delay(
                    attempt,
                    minimum_seconds=self._min_request_interval_seconds,
                )
                LOGGER.warning(
                    "OpenAlex network error (%s); retrying in %.1fs.",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise OpenAlexError(
            "OpenAlex request failed after all retry attempts."
        ) from last_error

    def _respect_request_interval(self) -> None:
        """Apply the semantic-search minimum interval between API calls."""

        if self._last_request_started_at is None:
            return

        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _extract_openalex_error_detail(response: httpx.Response) -> str:
    """Extract a concise OpenAlex API error without exposing request secrets."""

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = str(payload.get("error") or "").strip()
        message = str(payload.get("message") or "").strip()
        detail = ": ".join(part for part in (error, message) if part)
        if detail:
            return detail[:500]

    text = response.text.strip().replace("\n", " ")
    return text[:500] or "No error detail returned by OpenAlex."


def _build_openalex_filter(start_date: date, end_date: date) -> str:
    """Build a semantic-search-compatible coarse publication-year filter."""

    if start_date > end_date:
        raise ValueError("start_date must not be later than end_date.")

    years = "|".join(
        str(year) for year in range(start_date.year, end_date.year + 1)
    )
    return f"publication_year:{years}"


def _parse_work(raw: dict[str, Any]) -> ScientificPaper | None:
    """Normalize one OpenAlex work response."""

    raw_id = str(raw.get("id") or "").strip()
    source_id = raw_id.rsplit("/", 1)[-1]
    title = clean_text(str(raw.get("display_name") or raw.get("title") or ""))
    publication_date_raw = raw.get("publication_date")
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))

    if not source_id or not title or not abstract or not publication_date_raw:
        return None

    try:
        publication_date = datetime.combine(
            date.fromisoformat(str(publication_date_raw)),
            day_time.min,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None

    authors: list[str] = []
    for authorship in raw.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = clean_text(str(author.get("display_name") or ""))
        if name:
            authors.append(name)

    doi = _optional_string(raw.get("doi"))
    source_url = raw_id or (doi or "https://openalex.org")

    best_oa = raw.get("best_oa_location") or {}
    primary_location = raw.get("primary_location") or {}
    pdf_url = (
        _optional_string(best_oa.get("pdf_url"))
        or _optional_string(primary_location.get("pdf_url"))
    )

    citation_count = raw.get("cited_by_count")
    if not isinstance(citation_count, int) or citation_count < 0:
        citation_count = 0

    source_keywords: list[str] = []
    for keyword in raw.get("keywords") or []:
        if not isinstance(keyword, dict):
            continue
        display_name = clean_text(str(keyword.get("display_name") or ""))
        if display_name and display_name not in source_keywords:
            source_keywords.append(display_name)
        if len(source_keywords) >= 6:
            break

    categories: list[str] = []
    for topic in raw.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        display_name = clean_text(str(topic.get("display_name") or ""))
        if display_name and display_name not in categories:
            categories.append(display_name)

    arxiv_id = _extract_arxiv_id(raw)

    return ScientificPaper(
        source="openalex",
        source_id=source_id,
        title=title,
        authors=authors,
        abstract=abstract,
        publication_date=publication_date,
        source_publication_date=publication_date,
        source_url=source_url,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        citation_count=citation_count,
        categories=categories,
        source_keywords=source_keywords,
    )


def _reconstruct_abstract(inverted_index: Any) -> str:
    """Reconstruct plain abstract text from OpenAlex's inverted index."""

    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""

    positioned_tokens: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned_tokens.append((position, str(token)))

    positioned_tokens.sort(key=lambda item: item[0])
    return clean_text(" ".join(token for _, token in positioned_tokens))


def _extract_arxiv_id(raw: dict[str, Any]) -> str | None:
    """Extract an arXiv identifier from OpenAlex locations when available."""

    import re

    for location in raw.get("locations") or []:
        if not isinstance(location, dict):
            continue
        for field in ("landing_page_url", "pdf_url"):
            value = _optional_string(location.get(field))
            if not value or "arxiv.org" not in value.casefold():
                continue
            normalized = value.rstrip("/")
            if "/abs/" in normalized:
                identifier = normalized.split("/abs/", 1)[1]
            elif "/pdf/" in normalized:
                identifier = normalized.split("/pdf/", 1)[1]
            else:
                continue
            if identifier.endswith(".pdf"):
                identifier = identifier[:-4]
            return re.sub(r"v\d+$", "", identifier)
    return None


def _merge_papers(
    left: ScientificPaper,
    right: ScientificPaper,
) -> ScientificPaper:
    """Merge duplicate OpenAlex results from multiple semantic cluster queries."""

    return left.model_copy(
        update={
            "matched_clusters": sorted(
                set(left.matched_clusters) | set(right.matched_clusters)
            ),
            "categories": list(dict.fromkeys([*left.categories, *right.categories])),
            "source_keywords": list(
                dict.fromkeys([*left.source_keywords, *right.source_keywords])
            )[:6],
        }
    )


def _optional_string(value: Any) -> str | None:
    """Normalize an optional scalar value to a non-empty string."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
