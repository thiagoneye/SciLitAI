"""Deterministic relevance ranking for the daily arXiv candidate set."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from sciml_digest.exceptions import SelectionError
from sciml_digest.models import ArxivPaper, SelectedPaper
from sciml_digest.taxonomy import CLUSTER_WEIGHTS, SEARCH_CLUSTERS

WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def select_top_papers(
    papers: list[ArxivPaper],
    window_start_utc: datetime,
    window_end_utc: datetime,
    limit: int = 5,
) -> list[SelectedPaper]:
    """Rank and select the top papers strictly inside the D-1 UTC window."""

    if limit <= 0:
        raise ValueError("limit must be > 0.")

    window_start_utc = _as_utc(window_start_utc)
    window_end_utc = _as_utc(window_end_utc)

    eligible = [
        paper
        for paper in papers
        if _is_in_window(paper, window_start_utc, window_end_utc)
    ]

    if len(eligible) < limit:
        raise SelectionError(
            f"Only {len(eligible)} eligible arXiv papers were found in the strict "
            f"D-1 UTC window; {limit} are required."
        )

    ranked = sorted(
        eligible,
        key=lambda paper: (
            _relevance_score(paper),
            _event_date_in_window(paper, window_start_utc, window_end_utc),
            paper.arxiv_id,
        ),
        reverse=True,
    )

    return [
        SelectedPaper(
            paper=paper,
            rank=rank,
            relevance_score=_relevance_score(paper),
        )
        for rank, paper in enumerate(ranked[:limit], start=1)
    ]


def _relevance_score(paper: ArxivPaper) -> float:
    """Score term density using thematic cluster priorities."""

    title = paper.title.casefold()
    abstract = paper.abstract.casefold()

    title_token_count = max(
        len(WORD_PATTERN.findall(title)),
        1,
    )
    abstract_token_count = max(
        len(WORD_PATTERN.findall(abstract)),
        1,
    )

    weighted_title_hits = 0.0
    weighted_abstract_hits = 0.0

    matched_terms: set[str] = set()
    matched_clusters: set[str] = set()

    for cluster_name, terms in SEARCH_CLUSTERS.items():
        cluster_weight = CLUSTER_WEIGHTS[cluster_name]
        cluster_matched = False

        for term in terms:
            normalized_term = term.casefold()

            term_weight = _term_weight(term)
            weighted_term = term_weight * cluster_weight

            title_count = _term_count(
                title,
                normalized_term,
            )
            abstract_count = _term_count(
                abstract,
                normalized_term,
            )

            if title_count or abstract_count:
                cluster_matched = True
                matched_terms.add(normalized_term)

                weighted_title_hits += (
                    weighted_term * title_count
                )

                weighted_abstract_hits += (
                    weighted_term * abstract_count
                )

        if cluster_matched:
            matched_clusters.add(cluster_name)

    title_density = (
        weighted_title_hits / title_token_count
    )

    abstract_density = (
        weighted_abstract_hits / abstract_token_count
    )

    # O título recebe maior peso por representar
    # um sinal temático mais forte.
    density_score = 100.0 * (
        (3.0 * title_density)
        + abstract_density
    )

    diversity_bonus = (
        0.35 * len(matched_terms)
    )

    weighted_cluster_breadth = sum(
        CLUSTER_WEIGHTS[cluster_name]
        for cluster_name in matched_clusters
    )

    cluster_bonus = (
        0.75 * weighted_cluster_breadth
    )

    return round(
        density_score
        + diversity_bonus
        + cluster_bonus,
        6,
    )


def _term_weight(term: str) -> float:
    """Give multi-word technical phrases slightly more weight than acronyms."""

    token_count = max(len(WORD_PATTERN.findall(term)), 1)
    return 1.0 + min(0.15 * (token_count - 1), 0.75)


def _term_count(text: str, term: str) -> int:
    """Count exact word or phrase occurrences case-insensitively."""

    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _event_date_in_window(
    paper: ArxivPaper,
    start_utc: datetime,
    end_utc: datetime,
) -> datetime:
    """Return the latest qualifying submission/update timestamp for tie-breaking."""

    candidates = [
        value
        for value in (paper.published_date, paper.updated_date)
        if start_utc <= value < end_utc
    ]
    return max(candidates)


def _is_in_window(
    paper: ArxivPaper,
    start_utc: datetime,
    end_utc: datetime,
) -> bool:
    """Return whether submission or last update falls in the target interval."""

    return (
        start_utc <= paper.published_date < end_utc
        or start_utc <= paper.updated_date < end_utc
    )


def _as_utc(value: datetime) -> datetime:
    """Validate timezone awareness and normalize a datetime to UTC."""

    if value.tzinfo is None:
        raise ValueError("Datetime values must be timezone-aware.")
    return value.astimezone(timezone.utc)
