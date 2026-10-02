"""Deterministic ranking and cross-source deduplication for SciLitAI."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from sciml_digest.exceptions import SelectionError
from sciml_digest.models import ScientificPaper, SelectedPaper, SelectionBasis
from sciml_digest.taxonomy import (
    CLUSTER_WEIGHTS,
    OPENALEX_STRONG_TERMS,
    SEARCH_CLUSTERS,
)

WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")

# O OpenAlex usa citações como critério principal somente após um filtro temático
# mais estrito que o empregado nas demais fontes.
DEFAULT_OPENALEX_MIN_RELEVANCE_SCORE = 10.0


def select_arxiv_papers(
    papers: list[ScientificPaper],
    window_start_utc: datetime,
    window_end_utc: datetime,
    limit: int = 3,
    excluded_identity_keys: set[str] | None = None,
) -> list[SelectedPaper]:
    """Select the most thematically relevant arXiv papers in strict D-1 UTC."""

    start_utc = _as_utc(window_start_utc)
    end_utc = _as_utc(window_end_utc)
    excluded = excluded_identity_keys or set()

    eligible = [
        paper
        for paper in papers
        if paper.source == "arxiv"
        and _arxiv_in_window(paper, start_utc, end_utc)
        and relevance_score(paper) > 0.0
    ]

    ranked = sorted(
        eligible,
        key=lambda paper: (
            relevance_score(paper),
            _arxiv_event_date(paper, start_utc, end_utc),
            paper.source_id,
        ),
        reverse=True,
    )

    return _take_unique(
        ranked,
        limit=limit,
        selection_basis="relevance",
        excluded_identity_keys=excluded,
        source_label="arXiv",
    )


def select_semantic_scholar_papers(
    papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
    limit: int = 3,
    excluded_identity_keys: set[str] | None = None,
) -> list[SelectedPaper]:
    """Select the most thematically relevant Semantic Scholar papers in the window."""

    excluded = excluded_identity_keys or set()
    eligible = [
        paper
        for paper in papers
        if paper.source == "semantic_scholar"
        and start_date <= paper.publication_date.date() <= end_date
        and relevance_score(paper) > 0.0
    ]

    ranked = sorted(
        eligible,
        key=lambda paper: (
            relevance_score(paper),
            paper.publication_date,
            paper.citation_count or 0,
            paper.source_id,
        ),
        reverse=True,
    )

    return _take_unique(
        ranked,
        limit=limit,
        selection_basis="relevance",
        excluded_identity_keys=excluded,
        source_label="Semantic Scholar",
    )


def rank_openalex_candidates(
    papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
    excluded_identity_keys: set[str] | None = None,
    min_relevance_score: float = DEFAULT_OPENALEX_MIN_RELEVANCE_SCORE,
) -> list[ScientificPaper]:
    """Rank unique OpenAlex candidates after strict thematic eligibility checks."""

    if start_date > end_date:
        raise ValueError("start_date must not be later than end_date.")
    if min_relevance_score < 0.0:
        raise ValueError("min_relevance_score must be >= 0.0.")

    excluded = excluded_identity_keys or set()
    eligible = [
        paper
        for paper in papers
        if paper.source == "openalex"
        and start_date <= paper.publication_date.date() <= end_date
        and is_openalex_thematically_eligible(
            paper,
            min_relevance_score=min_relevance_score,
        )
    ]

    ranked = sorted(
        eligible,
        key=lambda paper: (
            paper.citation_count or 0,
            relevance_score(paper),
            paper.publication_date,
            paper.source_id,
        ),
        reverse=True,
    )

    return _unique_ranked_papers(
        ranked,
        excluded_identity_keys=excluded,
    )


def select_openalex_papers(
    papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
    limit: int = 3,
    excluded_identity_keys: set[str] | None = None,
    min_relevance_score: float = DEFAULT_OPENALEX_MIN_RELEVANCE_SCORE,
) -> list[SelectedPaper]:
    """Select the most-cited thematically eligible OpenAlex papers in the window."""

    ranked = rank_openalex_candidates(
        papers,
        start_date=start_date,
        end_date=end_date,
        excluded_identity_keys=excluded_identity_keys,
        min_relevance_score=min_relevance_score,
    )

    return _take_unique(
        ranked,
        limit=limit,
        selection_basis="citations",
        excluded_identity_keys=set(),
        source_label="OpenAlex",
    )


def is_openalex_thematically_eligible(
    paper: ScientificPaper,
    min_relevance_score: float = DEFAULT_OPENALEX_MIN_RELEVANCE_SCORE,
) -> bool:
    """Return whether an OpenAlex paper satisfies strict thematic eligibility."""

    if min_relevance_score < 0.0:
        raise ValueError("min_relevance_score must be >= 0.0.")
    if paper.source != "openalex":
        return False

    return (
        relevance_score(paper) >= min_relevance_score
        and has_openalex_strong_thematic_match(paper)
    )


def has_openalex_strong_thematic_match(paper: ScientificPaper) -> bool:
    """Return whether title or abstract contains a high-precision thematic phrase."""

    text = f"{paper.title}\n{paper.abstract}".casefold()
    return any(
        _term_count(text, term.casefold()) > 0
        for term in OPENALEX_STRONG_TERMS
    )


def relevance_score(paper: ScientificPaper) -> float:
    """Score title and abstract density using configured cluster priorities."""

    title = paper.title.casefold()
    abstract = paper.abstract.casefold()
    title_token_count = max(len(WORD_PATTERN.findall(title)), 1)
    abstract_token_count = max(len(WORD_PATTERN.findall(abstract)), 1)

    weighted_title_hits = 0.0
    weighted_abstract_hits = 0.0
    matched_terms: set[str] = set()
    matched_clusters: set[str] = set()

    for cluster_name, terms in SEARCH_CLUSTERS.items():
        cluster_weight = CLUSTER_WEIGHTS[cluster_name]
        cluster_matched = False

        for term in terms:
            normalized_term = term.casefold()
            weighted_term = _term_weight(term) * cluster_weight
            title_count = _term_count(title, normalized_term)
            abstract_count = _term_count(abstract, normalized_term)

            if title_count or abstract_count:
                cluster_matched = True
                matched_terms.add(normalized_term)
                weighted_title_hits += weighted_term * title_count
                weighted_abstract_hits += weighted_term * abstract_count

        if cluster_matched:
            matched_clusters.add(cluster_name)

    title_density = weighted_title_hits / title_token_count
    abstract_density = weighted_abstract_hits / abstract_token_count

    # O título recebe maior peso por representar um sinal temático mais forte.
    density_score = 100.0 * ((3.0 * title_density) + abstract_density)
    diversity_bonus = 0.35 * len(matched_terms)
    weighted_cluster_breadth = sum(
        CLUSTER_WEIGHTS[cluster_name] for cluster_name in matched_clusters
    )
    cluster_bonus = 0.75 * weighted_cluster_breadth

    return round(density_score + diversity_bonus + cluster_bonus, 6)


def paper_identity_keys(paper: ScientificPaper) -> set[str]:
    """Build stable identity keys for cross-repository deduplication."""

    keys = {f"title:{_normalize_title(paper.title)}"}

    if paper.doi:
        keys.add(f"doi:{_normalize_doi(paper.doi)}")
    if paper.arxiv_id:
        keys.add(f"arxiv:{paper.arxiv_id.casefold()}")

    return keys


def register_identity_keys(
    selected: list[SelectedPaper],
    destination: set[str],
) -> None:
    """Register selected papers so later repositories can backfill duplicates."""

    for item in selected:
        destination.update(paper_identity_keys(item.paper))


def _take_unique(
    ranked: list[ScientificPaper],
    limit: int,
    selection_basis: SelectionBasis,
    excluded_identity_keys: set[str],
    source_label: str,
) -> list[SelectedPaper]:
    """Take the requested number of papers while excluding prior-source duplicates."""

    if limit <= 0:
        raise ValueError("limit must be > 0.")

    chosen = _unique_ranked_papers(
        ranked,
        excluded_identity_keys=excluded_identity_keys,
    )[:limit]

    if len(chosen) < limit:
        raise SelectionError(
            f"Only {len(chosen)} unique eligible {source_label} papers were found; "
            f"{limit} are required."
        )

    return [
        SelectedPaper(
            paper=paper,
            rank=rank,
            selection_basis=selection_basis,
            relevance_score=relevance_score(paper),
        )
        for rank, paper in enumerate(chosen, start=1)
    ]


def _unique_ranked_papers(
    ranked: list[ScientificPaper],
    excluded_identity_keys: set[str],
) -> list[ScientificPaper]:
    """Return ranked papers after cross-source and within-source deduplication."""

    chosen: list[ScientificPaper] = []
    local_keys: set[str] = set()

    for paper in ranked:
        keys = paper_identity_keys(paper)
        if keys & excluded_identity_keys:
            continue
        if keys & local_keys:
            continue

        chosen.append(paper)
        local_keys.update(keys)

    return chosen


def _term_weight(term: str) -> float:
    """Give multi-word technical phrases slightly more weight than acronyms."""

    token_count = max(len(WORD_PATTERN.findall(term)), 1)
    return 1.0 + min(0.15 * (token_count - 1), 0.75)


def _term_count(text: str, term: str) -> int:
    """Count exact word or phrase occurrences case-insensitively."""

    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _arxiv_event_date(
    paper: ScientificPaper,
    start_utc: datetime,
    end_utc: datetime,
) -> datetime:
    """Return the latest qualifying arXiv publication or update timestamp."""

    values = [paper.publication_date]
    if paper.updated_date is not None:
        values.append(paper.updated_date)

    candidates = [value for value in values if start_utc <= value < end_utc]
    return max(candidates)


def _arxiv_in_window(
    paper: ScientificPaper,
    start_utc: datetime,
    end_utc: datetime,
) -> bool:
    """Return whether arXiv publication or update falls in the target interval."""

    published = start_utc <= paper.publication_date < end_utc
    updated = (
        paper.updated_date is not None
        and start_utc <= paper.updated_date < end_utc
    )
    return published or updated


def _normalize_title(title: str) -> str:
    """Normalize a title for conservative cross-source duplicate detection."""

    return NON_ALNUM_PATTERN.sub(" ", title.casefold()).strip()


def _normalize_doi(doi: str) -> str:
    """Normalize DOI URL and prefix variants to a canonical identifier."""

    normalized = doi.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized.strip()


def _as_utc(value: datetime) -> datetime:
    """Validate timezone awareness and normalize a datetime to UTC."""

    if value.tzinfo is None:
        raise ValueError("Datetime values must be timezone-aware.")
    return value.astimezone(timezone.utc)
