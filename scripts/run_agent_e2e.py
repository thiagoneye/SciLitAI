"""Live end-to-end diagnostic test for the SciLitAI agent."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from collections.abc import Callable
from datetime import date, datetime, time as day_time, timedelta, timezone
from typing import TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sciml_digest.arxiv_client import ArxivClient
from sciml_digest.bibliographic_date_client import (
    BibliographicDateClient,
    validate_openalex_publication_dates,
)
from sciml_digest.exceptions import DigestError, SelectionError
from sciml_digest.formatter import build_telegram_messages
from sciml_digest.gemini_service import GeminiService
from sciml_digest.models import (
    DigestItem,
    ScientificPaper,
    SelectedPaper,
    ScientificSource,
)
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

LOGGER = logging.getLogger("scilitai.e2e")
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the live end-to-end test."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a live SciLitAI end-to-end diagnostic across arXiv, "
            "Semantic Scholar, OpenAlex, DOI validation, Gemini, formatter, "
            "and optionally Telegram delivery."
        )
    )
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Send the generated digest to the configured Telegram chat.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override LOG_LEVEL for this test run.",
    )
    return parser.parse_args()


def run_stage(name: str, operation: Callable[[], T]) -> T:
    """Execute one test stage and report elapsed time and failures."""

    print(f"\n{'=' * 88}")
    print(f"STAGE: {name}")
    print("=" * 88)
    started_at = time.monotonic()

    try:
        result = operation()
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        print(f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        raise

    elapsed = time.monotonic() - started_at
    print(f"PASSED in {elapsed:.1f}s")
    return result


def run_optional_source_stage(
    name: str,
    source: ScientificSource,
    operation: Callable[[], T],
    fallback: T,
    source_statuses: dict[ScientificSource, str],
) -> T:
    """Run one source stage without preventing other repositories from running."""

    try:
        return run_stage(name, operation)
    except SelectionError as exc:
        source_statuses[source] = (
            "Não foram encontrados artigos únicos elegíveis em quantidade suficiente."
        )
        print(
            f"CONTINUING without {source}: insufficient eligible papers "
            f"({type(exc).__name__})."
        )
    except DigestError as exc:
        source_statuses[source] = (
            "Falha na consulta ou validação dos dados; consulte os logs."
        )
        print(
            f"CONTINUING without {source}: source failure "
            f"({type(exc).__name__})."
        )

    return fallback


def print_settings(settings: Settings) -> None:
    """Print non-secret runtime settings used by the test."""

    print("Runtime configuration:")
    print(f"  Gemini model: {settings.gemini_model}")
    print(f"  Papers per source: {settings.papers_per_source}")
    print(f"  Semantic Scholar API key configured: {bool(settings.semantic_scholar_api_key)}")
    print(f"  OpenAlex API key configured: {bool(settings.openalex_api_key)}")
    print(f"  OpenAlex lookback days: {settings.openalex_lookback_days}")
    print(f"  OpenAlex min relevance score: {settings.openalex_min_relevance_score}")
    print(f"  OpenAlex require validated date: {settings.openalex_require_validated_date}")
    print(f"  OpenAlex require day precision: {settings.openalex_require_day_precision}")
    print(f"  Telegram force IPv4: {settings.telegram_force_ipv4}")


def fetch_arxiv(
    settings: Settings,
    start_utc: datetime,
    end_utc: datetime,
) -> list[ScientificPaper]:
    """Fetch live arXiv candidates for the strict D-1 UTC window."""

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

    print(f"arXiv candidates: {len(papers)}")
    return papers


def fetch_semantic_scholar(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Fetch live Semantic Scholar candidates for the configured 30-day window."""

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

    print(f"Semantic Scholar candidates: {len(papers)}")
    return papers


def fetch_openalex(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Fetch live semantically discovered OpenAlex candidates."""

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

    outside_window = [
        paper
        for paper in papers
        if not start_date <= paper.publication_date.date() <= end_date
    ]
    if outside_window:
        raise AssertionError(
            f"OpenAlex returned {len(outside_window)} locally accepted papers "
            "outside the requested calendar window."
        )

    print(f"OpenAlex semantic candidates: {len(papers)}")
    print("OpenAlex local temporal check: OK")
    return papers


def validate_openalex_dates(
    settings: Settings,
    ranked_papers: list[ScientificPaper],
    start_date: date,
    end_date: date,
) -> list[ScientificPaper]:
    """Validate OpenAlex publication dates against DOI metadata."""

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

    for paper in validated:
        if not start_date <= paper.publication_date.date() <= end_date:
            raise AssertionError(
                f"Validated OpenAlex paper {paper.source_id} is outside the "
                "requested date window."
            )
        if settings.openalex_require_validated_date and not paper.publication_date_verified:
            raise AssertionError(
                f"OpenAlex paper {paper.source_id} was not bibliographically validated."
            )
        if (
            settings.openalex_require_day_precision
            and paper.publication_date_precision != "day"
        ):
            raise AssertionError(
                f"OpenAlex paper {paper.source_id} does not have day-level precision."
            )

    print(f"OpenAlex DOI-validated candidates retained: {len(validated)}")
    return validated


def print_selection(selected: list[SelectedPaper]) -> None:
    """Print the final source-specific selections and ranking metrics."""

    print(f"Final selected papers: {len(selected)}")
    for item in selected:
        paper = item.paper
        metric = (
            f"citations={paper.citation_count or 0}"
            if item.selection_basis == "citations"
            else f"score={item.relevance_score:.3f}"
        )
        print(
            f"  [{paper.source}] #{item.rank} {metric} | "
            f"{paper.publication_date.date()} | {paper.title}"
        )
        if paper.source == "openalex":
            print(
                "      "
                f"verified={paper.publication_date_verified} "
                f"source={paper.publication_date_validation_source} "
                f"precision={paper.publication_date_precision}"
            )


def enrich_selected(
    settings: Settings,
    selected: list[SelectedPaper],
) -> list[DigestItem]:
    """Run live Gemini enrichment for every selected paper."""

    gemini = GeminiService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_attempts=settings.gemini_max_attempts,
    )

    digest_items: list[DigestItem] = []
    for index, selected_paper in enumerate(selected, start=1):
        print(
            f"Gemini {index}/{len(selected)}: "
            f"[{selected_paper.paper.source}] {selected_paper.paper.title}"
        )
        enriched = gemini.enrich(selected_paper.paper)
        if len(enriched.keywords) < 3:
            raise AssertionError("Gemini returned fewer than three keywords.")
        digest_items.append(
            DigestItem(
                selected=selected_paper,
                enriched=enriched,
            )
        )

    return digest_items


def send_telegram(settings: Settings, messages: list[str]) -> None:
    """Send the formatted live test digest through Telegram."""

    with TelegramClient(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        max_attempts=settings.telegram_max_attempts,
        timeout_seconds=settings.telegram_timeout_seconds,
        force_ipv4=settings.telegram_force_ipv4,
    ) as client:
        client.send_messages(messages)


def main() -> int:
    """Run the complete live SciLitAI diagnostic test."""

    args = parse_args()

    try:
        settings = Settings.from_env()
        log_level = args.log_level or settings.log_level
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        print("SciLitAI live end-to-end test")
        print_settings(settings)

        if not settings.semantic_scholar_api_key:
            print(
                "\nWARNING: SEMANTIC_SCHOLAR_API_KEY is not configured. "
                "The live test may fail with HTTP 429 before later stages."
            )

        now_utc = datetime.now(timezone.utc)
        today_utc = now_utc.date()

        arxiv_target_date = today_utc - timedelta(days=1)
        arxiv_start_utc = datetime.combine(
            arxiv_target_date,
            day_time.min,
            tzinfo=timezone.utc,
        )
        arxiv_end_utc = arxiv_start_utc + timedelta(days=1)

        semantic_end_date = today_utc
        semantic_start_date = today_utc - timedelta(days=29)

        openalex_end_date = today_utc
        openalex_start_date = today_utc - timedelta(
            days=settings.openalex_lookback_days - 1
        )

        print("\nTest windows:")
        print(
            "  arXiv: "
            f"[{arxiv_start_utc.isoformat()}, {arxiv_end_utc.isoformat()})"
        )
        print(
            "  Semantic Scholar: "
            f"[{semantic_start_date}, {semantic_end_date}]"
        )
        print(
            "  OpenAlex: "
            f"[{openalex_start_date}, {openalex_end_date}]"
        )

        source_statuses: dict[ScientificSource, str] = {}

        arxiv_candidates = run_optional_source_stage(
            "1/10 arXiv live retrieval",
            "arxiv",
            lambda: fetch_arxiv(settings, arxiv_start_utc, arxiv_end_utc),
            [],
            source_statuses,
        )
        semantic_candidates = run_optional_source_stage(
            "2/10 Semantic Scholar live retrieval",
            "semantic_scholar",
            lambda: fetch_semantic_scholar(
                settings,
                semantic_start_date,
                semantic_end_date,
            ),
            [],
            source_statuses,
        )
        openalex_candidates = run_optional_source_stage(
            "3/10 OpenAlex semantic live retrieval",
            "openalex",
            lambda: fetch_openalex(
                settings,
                openalex_start_date,
                openalex_end_date,
            ),
            [],
            source_statuses,
        )

        seen_identity_keys: set[str] = set()

        if "arxiv" in source_statuses:
            arxiv_selected: list[SelectedPaper] = []
        else:
            arxiv_selected = run_optional_source_stage(
                "4/10 arXiv selection",
                "arxiv",
                lambda: select_arxiv_papers(
                    arxiv_candidates,
                    window_start_utc=arxiv_start_utc,
                    window_end_utc=arxiv_end_utc,
                    limit=settings.papers_per_source,
                ),
                [],
                source_statuses,
            )
        register_identity_keys(arxiv_selected, seen_identity_keys)

        if "semantic_scholar" in source_statuses:
            semantic_selected: list[SelectedPaper] = []
        else:
            semantic_selected = run_optional_source_stage(
                "5/10 Semantic Scholar selection and cross-source deduplication",
                "semantic_scholar",
                lambda: select_semantic_scholar_papers(
                    semantic_candidates,
                    start_date=semantic_start_date,
                    end_date=semantic_end_date,
                    limit=settings.papers_per_source,
                    excluded_identity_keys=seen_identity_keys,
                ),
                [],
                source_statuses,
            )
        register_identity_keys(semantic_selected, seen_identity_keys)

        if "openalex" in source_statuses:
            openalex_selected: list[SelectedPaper] = []
        else:
            openalex_ranked = run_optional_source_stage(
                "6/10 OpenAlex thematic gate and citation ranking",
                "openalex",
                lambda: rank_openalex_candidates(
                    openalex_candidates,
                    start_date=openalex_start_date,
                    end_date=openalex_end_date,
                    excluded_identity_keys=seen_identity_keys,
                    min_relevance_score=settings.openalex_min_relevance_score,
                ),
                [],
                source_statuses,
            )
            print(
                "OpenAlex thematically eligible ranked papers: "
                f"{len(openalex_ranked)}"
            )

            openalex_validated = run_optional_source_stage(
                "7/10 OpenAlex DOI publication-date validation",
                "openalex",
                lambda: validate_openalex_dates(
                    settings,
                    openalex_ranked,
                    openalex_start_date,
                    openalex_end_date,
                ),
                [],
                source_statuses,
            )

            openalex_selected = run_optional_source_stage(
                "8/10 OpenAlex final Top 3 selection",
                "openalex",
                lambda: select_openalex_papers(
                    openalex_validated,
                    start_date=openalex_start_date,
                    end_date=openalex_end_date,
                    limit=settings.papers_per_source,
                    excluded_identity_keys=seen_identity_keys,
                    min_relevance_score=settings.openalex_min_relevance_score,
                ),
                [],
                source_statuses,
            )
        register_identity_keys(openalex_selected, seen_identity_keys)

        selected = [
            *arxiv_selected,
            *semantic_selected,
            *openalex_selected,
        ]
        if not selected:
            raise SelectionError(
                "No papers could be selected from any repository."
            )
        if len(selected) > settings.papers_per_source * 3:
            raise AssertionError("The selected paper count exceeds source limits.")
        print_selection(selected)
        for source, status in source_statuses.items():
            print(f"Source omitted ({source}): {status}")

        digest_items = run_stage(
            "9/10 Gemini enrichment for all selected papers",
            lambda: enrich_selected(settings, selected),
        )

        messages = run_stage(
            "10/10 Telegram formatting",
            lambda: build_telegram_messages(
                digest_items,
                run_date=today_utc,
                source_statuses=source_statuses,
            ),
        )

        print(f"Telegram messages generated: {len(messages)}")
        for index, message in enumerate(messages, start=1):
            print(f"  Message {index}: {len(message)} characters")

        if args.send_telegram:
            run_stage(
                "Telegram live delivery",
                lambda: send_telegram(settings, messages),
            )
        else:
            print(
                "\nTelegram delivery was not executed. "
                "Run again with --send-telegram to test the complete external delivery path."
            )

        print("\n" + "=" * 88)
        print(
            "SciLitAI END-TO-END TEST PASSED WITH SOURCE WARNINGS"
            if source_statuses
            else "SciLitAI END-TO-END TEST PASSED"
        )
        if not args.send_telegram:
            print(
                "Status: live Telegram delivery was skipped; "
                f"omitted sources: {len(source_statuses)}."
            )
        else:
            print(
                "Status: all selected papers were processed and Telegram delivery "
                f"passed; omitted sources: {len(source_statuses)}."
            )
        print("=" * 88)
        return 0

    except Exception:
        LOGGER.exception("SciLitAI end-to-end test failed.")
        print("\nSciLitAI END-TO-END TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
