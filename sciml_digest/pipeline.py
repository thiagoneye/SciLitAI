"""End-to-end orchestration for the daily SciLitAI digest."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone

from sciml_digest.arxiv_client import ArxivClient
from sciml_digest.formatter import build_telegram_messages
from sciml_digest.gemini_service import GeminiService
from sciml_digest.models import DigestItem
from sciml_digest.selector import select_top_papers
from sciml_digest.settings import Settings
from sciml_digest.telegram_client import TelegramClient

LOGGER = logging.getLogger(__name__)


def run_pipeline(settings: Settings) -> None:
    """Execute ingestion, ranking, enrichment, formatting, and delivery."""

    now_utc = datetime.now(timezone.utc)
    target_date = now_utc.date() - timedelta(days=1)
    window_start_utc = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    window_end_utc = window_start_utc + timedelta(days=1)

    LOGGER.info(
        "Fetching arXiv candidates for strict D-1 UTC window [%s, %s).",
        window_start_utc.isoformat(),
        window_end_utc.isoformat(),
    )

    with ArxivClient(
        timeout_seconds=settings.arxiv_timeout_seconds,
        max_attempts=settings.arxiv_max_attempts,
        min_request_interval_seconds=settings.arxiv_min_request_interval_seconds,
    ) as arxiv:
        papers = arxiv.fetch_daily_candidates(
            start_utc=window_start_utc,
            end_utc=window_end_utc,
            max_results_per_query=settings.arxiv_max_results_per_query,
            terms_per_query=settings.arxiv_terms_per_query,
        )

    LOGGER.info("Received %s unique eligible arXiv candidates.", len(papers))

    selected = select_top_papers(
        papers=papers,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        limit=settings.arxiv_top_k,
    )

    LOGGER.info(
        "Selected Top %s arXiv IDs: %s",
        settings.arxiv_top_k,
        ", ".join(item.paper.arxiv_id for item in selected),
    )

    gemini = GeminiService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_attempts=settings.gemini_max_attempts,
    )

    digest_items: list[DigestItem] = []
    for position, selected_paper in enumerate(selected, start=1):
        LOGGER.info(
            "Enriching paper %s/%s: %s",
            position,
            settings.arxiv_top_k,
            selected_paper.paper.arxiv_id,
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
        digest_date=target_date,
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

    LOGGER.info("SciLitAI daily digest completed successfully.")
