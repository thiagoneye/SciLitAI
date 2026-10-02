"""Semantic Scholar Academic Graph client for recent relevance retrieval."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as day_time, timezone
from typing import Any

import httpx

from sciml_digest.exceptions import SemanticScholarError
from sciml_digest.models import ScientificPaper
from sciml_digest.source_utils import (
    chunked,
    clean_text,
    normalize_plain_query_term,
    retry_delay,
)
from sciml_digest.taxonomy import SEARCH_CLUSTERS

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETURN_FIELDS = ",".join(
    (
        "title",
        "abstract",
        "authors",
        "url",
        "publicationDate",
        "year",
        "citationCount",
        "externalIds",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
    )
)


class SemanticScholarClient:
    """Retrieve recent relevance-ranked papers from Semantic Scholar."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        min_request_interval_seconds: float = 1.0,
        user_agent: str = "SciLitAI/3.0",
    ) -> None:
        if min_request_interval_seconds <= 0:
            raise ValueError("min_request_interval_seconds must be > 0.")

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key

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

    def __enter__(self) -> "SemanticScholarClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_recent_candidates(
        self,
        start_date: date,
        end_date: date,
        max_results_per_query: int = 50,
        terms_per_query: int = 8,
    ) -> list[ScientificPaper]:
        """Fetch papers published in an inclusive calendar-date interval."""

        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date.")
        if max_results_per_query <= 0 or max_results_per_query > 1000:
            raise ValueError("max_results_per_query must be between 1 and 1000.")

        papers_by_id: dict[str, ScientificPaper] = {}
        publication_range = f"{start_date.isoformat()}:{end_date.isoformat()}"

        for cluster_name, terms in SEARCH_CLUSTERS.items():
            for term_chunk in chunked(terms, terms_per_query):
                query = " ".join(
                    normalize_plain_query_term(term) for term in term_chunk
                )
                params = {
                    "query": query,
                    "publicationDateOrYear": publication_range,
                    "limit": max_results_per_query,
                    "offset": 0,
                    "fields": RETURN_FIELDS,
                }

                LOGGER.info(
                    "Querying Semantic Scholar cluster=%s terms=%s.",
                    cluster_name,
                    len(term_chunk),
                )
                payload = self._request(params).json()

                for raw_paper in payload.get("data", []):
                    paper = _parse_paper(raw_paper)
                    if paper is None:
                        continue
                    if not start_date <= paper.publication_date.date() <= end_date:
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
            key=lambda paper: (paper.publication_date, paper.citation_count or 0),
            reverse=True,
        )
        return papers

    def _request(self, params: dict[str, Any]) -> httpx.Response:
        """Execute one Semantic Scholar request with pacing and retries."""

        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._respect_request_interval()

            try:
                self._last_request_started_at = time.monotonic()
                response = self._client.get(
                    SEMANTIC_SCHOLAR_SEARCH_URL,
                    params=params,
                )

                if response.status_code in RETRIABLE_STATUS_CODES:
                    if attempt == self._max_attempts:
                        raise SemanticScholarError(
                            "Semantic Scholar remained unavailable after retries "
                            f"(HTTP {response.status_code})."
                        )
                    delay = retry_delay(
                        attempt,
                        retry_after=response.headers.get("Retry-After"),
                        minimum_seconds=self._min_request_interval_seconds,
                    )
                    LOGGER.warning(
                        "Semantic Scholar transient HTTP %s; retrying in %.1fs.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as exc:
                raise SemanticScholarError(
                    "Semantic Scholar request failed with HTTP "
                    f"{exc.response.status_code}."
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
                    "Semantic Scholar network error (%s); retrying in %.1fs.",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise SemanticScholarError(
            "Semantic Scholar request failed after all retry attempts."
        ) from last_error

    def _respect_request_interval(self) -> None:
        """Apply a conservative minimum interval between API calls."""

        if self._last_request_started_at is None:
            return

        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _parse_paper(raw: dict[str, Any]) -> ScientificPaper | None:
    """Normalize one Semantic Scholar paper record."""

    source_id = str(raw.get("paperId") or "").strip()
    title = clean_text(str(raw.get("title") or ""))
    abstract = clean_text(str(raw.get("abstract") or ""))
    publication_date_raw = raw.get("publicationDate")

    # Para uma janela estrita de 30 dias, datas apenas anuais são insuficientes.
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

    authors = [
        clean_text(str(author.get("name") or ""))
        for author in raw.get("authors") or []
        if author.get("name")
    ]

    external_ids = raw.get("externalIds") or {}
    doi = _optional_string(external_ids.get("DOI"))
    arxiv_id = _optional_string(external_ids.get("ArXiv"))
    source_url = _optional_string(raw.get("url")) or (
        f"https://www.semanticscholar.org/paper/{source_id}"
    )

    open_access_pdf = raw.get("openAccessPdf") or {}
    pdf_url = _optional_string(open_access_pdf.get("url"))

    categories: list[str] = []
    for field in raw.get("fieldsOfStudy") or []:
        value = clean_text(str(field))
        if value:
            categories.append(value)
    for field in raw.get("s2FieldsOfStudy") or []:
        if isinstance(field, dict):
            value = clean_text(str(field.get("category") or ""))
            if value and value not in categories:
                categories.append(value)

    citation_count = raw.get("citationCount")
    if not isinstance(citation_count, int) or citation_count < 0:
        citation_count = None

    return ScientificPaper(
        source="semantic_scholar",
        source_id=source_id,
        title=title,
        authors=authors,
        abstract=abstract,
        publication_date=publication_date,
        source_url=source_url,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        citation_count=citation_count,
        categories=categories,
    )


def _merge_papers(
    left: ScientificPaper,
    right: ScientificPaper,
) -> ScientificPaper:
    """Merge duplicate Semantic Scholar results from multiple cluster queries."""

    return left.model_copy(
        update={
            "matched_clusters": sorted(
                set(left.matched_clusters) | set(right.matched_clusters)
            ),
            "categories": list(dict.fromkeys([*left.categories, *right.categories])),
        }
    )


def _optional_string(value: Any) -> str | None:
    """Normalize an optional scalar value to a non-empty string."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
