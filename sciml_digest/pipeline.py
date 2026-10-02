"""End-to-end orchestration for the multi-source SciLitAI digest."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone

from sciml_digest.arxiv_client import ArxivClient
from sciml_digest.bibliographic_date_client import (
    BibliographicDateClient,
    validate_openalex_publication_dates,
)
from sciml_digest.formatter import build_telegram_messages
from sciml_digest.gemini_service import GeminiService
from sciml_digest.models import DigestItem, ScientificPaper, SelectedPaper
from sciml_digest.openalex_client import OpenAlexClient
from sciml_digest.selector import (
    rank_openalex_candidates,
    register_identity_keys,
    select_arxiv_papers,
    select_openalex_papers,
    select_semantic_scholar_papers,
)
from sciml_digest.semantic_scholar_client import SemanticScholarClient
from sciml_digest.settings import Settings
from sciml_digest.telegram_client import TelegramClient

LOGGER = logging.getLogger(__name__)


def run_pipeline(settings: Settings) -> None:
    """Execute multi-source ingestion, selection, enrichment, and delivery."""

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()

    arxiv_target_date = today_utc - timedelta(days=1)
    arxiv_start_utc = datetime.combine(
        arxiv_target_date,
        time.min,
        tzinfo=timezone.utc,
    )
    arxiv_end_utc = arxiv_start_utc + timedelta(days=1)

    # As janelas são inclusivas em datas de calendário.
    semantic_end_date = today_utc
    semantic_start_date = today_utc - timedelta(days=29)
    openalex_end_date = today_utc
    openalex_start_date = today_utc - timedelta(
        days=settings.openalex_lookback_days - 1
    )

    arxiv_candidates = _fetch_arxiv(
        settings,
        arxiv_start_utc,
        arxiv_end_utc,
    )
    semantic_candidates = _fetch_semantic_scholar(
        settings,
        semantic_start_date,
        semantic_end_date,
    )
    openalex_candidates = _fetch_openalex(
        settings,
        openalex_start_date,
        openalex_end_date,
    )

    seen_identity_keys: set[str] = set()

    arxiv_selected = select_arxiv_papers(
        arxiv_candidates,
        window_start_utc=arxiv_start_utc,
        window_end_utc=arxiv_end_utc,
        limit=settings.papers_per_source,
    )
    register_identity_keys(arxiv_selected, seen_identity_keys)

    semantic_selected = select_semantic_scholar_papers(
        semantic_candidates,
        start_date=semantic_start_date,
        end_date=semantic_end_date,
        limit=settings.papers_per_source,
        excluded_identity_keys=seen_identity_keys,
    )
    register_identity_keys(semantic_selected, seen_identity_keys)

    openalex_ranked = rank_openalex_candidates(
        openalex_candidates,
        start_date=openalex_start_date,
        end_date=openalex_end_date,
        excluded_identity_keys=seen_identity_keys,
        min_relevance_score=settings.openalex_min_relevance_score,
    )
    LOGGER.info(
        "OpenAlex thematic gate retained %s unique ranked candidates.",
        len(openalex_ranked),
    )

    openalex_validated = _validate_openalex_dates(
        settings,
        openalex_ranked,
        openalex_start_date,
        openalex_end_date,
    )

    openalex_selected = select_openalex_papers(
        openalex_validated,
        start_date=openalex_start_date,
        end_date=openalex_end_date,
        limit=settings.papers_per_source,
        excluded_identity_keys=seen_identity_keys,
        min_relevance_score=settings.openalex_min_relevance_score,
    )
    register_identity_keys(openalex_selected, seen_identity_keys)

    selected = [
        *arxiv_selected,
        *semantic_selected,
        *openalex_selected,
    ]

    _log_selection(selected)

    gemini = GeminiService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_attempts=settings.gemini_max_attempts,
    )

    digest_items: list[DigestItem] = []
    for position, selected_paper in enumerate(selected, start=1):
        LOGGER.info(
            "Enriching paper %s/%s source=%s id=%s.",
            position,
            len(selected),
            selected_paper.paper.source,
            selected_paper.paper.source_id,
        )
        enriched = gemini.enrich(selected_paper.paper)
        digest_items.append(
            DigestItem(
                selected=selected_paper,
                enriched=enriched,
            )
        )

    messages = build_telegram_messages(
        digest_items,
        run_date=today_utc,
    )

    LOGGER.info("Formatted digest into %s Telegram message(s).", len(messages))

    with TelegramClient(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        max_attempts=settings.telegram_max_attempts,
        timeout_seconds=settings.telegram_timeout_seconds,
        force_ipv4=settings.telegram_force_ipv4,
    ) as telegram:
        telegram.send_messages(messages)

    LOGGER.info("SciLitAI multi-source digest completed successfully.")


def _fetch_arxiv(
    settings: Settings,
    start_utc: datetime,
    end_utc: datetime,
) -> list[ScientificPaper]:
    """Fetch arXiv candidates for strict D-1 publication/update activity."""

    LOGGER.info(
        "Fetching arXiv candidates in strict UTC window [%s, %s).",
        start_utc.isoformat(),
        end_utc.isoformat(),
    )

    with ArxivClient(
        timeout_seconds=settings.arxiv_timeout_seconds,
        max_attempts=settings.arxiv_max_attempts,
        min_request_interval_seconds=settings.arxiv_min_request_interval_seconds,
    ) as client:
        papers = client.fetch_daily_candidates(
            start_utc=start_utc,
            end_utc=end_utc,
            max_results_per_query=settings.arxiv_max_results_per_query,
            terms_per_query=settings.arxiv_terms_per_query,
        )

    LOGGER.info("Received %s unique eligible arXiv candidates.", len(papers))
    return papers


def _fetch_semantic_scholar(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Fetch Semantic Scholar candidates from the last 30 calendar days."""

    LOGGER.info(
        "Fetching Semantic Scholar candidates for [%s, %s].",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    with SemanticScholarClient(
        api_key=settings.semantic_scholar_api_key,
        timeout_seconds=settings.semantic_scholar_timeout_seconds,
        max_attempts=settings.semantic_scholar_max_attempts,
        min_request_interval_seconds=(
            settings.semantic_scholar_min_request_interval_seconds
        ),
    ) as client:
        papers = client.fetch_recent_candidates(
            start_date=start_date,
            end_date=end_date,
            max_results_per_query=settings.semantic_scholar_max_results_per_query,
            terms_per_query=settings.semantic_scholar_terms_per_query,
        )

    LOGGER.info(
        "Received %s unique eligible Semantic Scholar candidates.",
        len(papers),
    )
    return papers


def _fetch_openalex(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Fetch semantically relevant OpenAlex candidates from the configured window."""

    LOGGER.info(
        "Fetching OpenAlex candidates for [%s, %s].",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    with OpenAlexClient(
        api_key=settings.openalex_api_key,
        timeout_seconds=settings.openalex_timeout_seconds,
        max_attempts=settings.openalex_max_attempts,
        min_request_interval_seconds=settings.openalex_min_request_interval_seconds,
    ) as client:
        papers = client.fetch_recent_cited_candidates(
            start_date=start_date,
            end_date=end_date,
            max_results_per_query=settings.openalex_max_results_per_query,
        )

    LOGGER.info("Received %s unique OpenAlex semantic candidates.", len(papers))
    return papers


def _validate_openalex_dates(
    settings: Settings,
    ranked_papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Validate OpenAlex publication dates against DOI registration metadata."""

    with BibliographicDateClient(
        timeout_seconds=settings.openalex_bibliographic_timeout_seconds,
        max_attempts=settings.openalex_bibliographic_max_attempts,
        min_request_interval_seconds=(
            settings.openalex_bibliographic_min_request_interval_seconds
        ),
    ) as client:
        validated = validate_openalex_publication_dates(
            ranked_papers,
            start_date=start_date,
            end_date=end_date,
            required_count=settings.papers_per_source,
            max_candidates=settings.openalex_validation_max_candidates,
            client=client,
            require_validated_date=settings.openalex_require_validated_date,
            require_day_precision=settings.openalex_require_day_precision,
        )

    LOGGER.info(
        "Validated %s OpenAlex publication dates for final selection.",
        len(validated),
    )
    return validated


def _log_selection(selected: list[SelectedPaper]) -> None:
    """Log the final source-specific selections without exposing secrets."""

    for item in selected:
        paper = item.paper
        if item.selection_basis == "citations":
            metric = f"citations={paper.citation_count or 0}"
        else:
            metric = f"relevance={item.relevance_score:.3f}"

        LOGGER.info(
            "Selected source=%s rank=%s id=%s %s.",
            paper.source,
            item.rank,
            paper.source_id,
            metric,
        )
