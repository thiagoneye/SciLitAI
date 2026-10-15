"""Unit tests for the multi-source Telegram formatter."""

from datetime import date, datetime, timezone
import html
import re

from sciml_digest.formatter import TELEGRAM_SAFE_LIMIT, build_telegram_messages
from sciml_digest.models import DigestItem, EnrichedArticle, ScientificPaper, SelectedPaper


def _item(source: str, rank: int) -> DigestItem:
    """Create one digest item fixture with visible keywords."""

    paper = ScientificPaper(
        source=source,
        source_id=f"{source}-{rank}",
        title=f"PINN paper {rank}",
        authors=["Ada Lovelace"],
        abstract="Physics-Informed Neural Network for CFD.",
        publication_date=datetime(2026, 9, 20, tzinfo=timezone.utc),
        updated_date=(
            datetime(2026, 9, 30, tzinfo=timezone.utc)
            if source == "arxiv"
            else None
        ),
        source_url=(
            f"https://arxiv.org/abs/2609.0000{rank}"
            if source == "arxiv"
            else (
                f"https://www.semanticscholar.org/paper/s{rank}"
                if source == "semantic_scholar"
                else f"https://openalex.org/W{rank}"
            )
        ),
        citation_count=25 if source != "arxiv" else None,
        arxiv_id=f"2609.0000{rank}" if source == "arxiv" else None,
    )
    selected = SelectedPaper(
        paper=paper,
        rank=rank,
        selection_basis="citations" if source == "openalex" else "relevance",
        relevance_score=12.5,
    )
    enriched = EnrichedArticle(
        title=paper.title,
        ai_approach="PINNs",
        domain_application="Computational Fluid Dynamics",
        executive_summary="Resumo técnico objetivo em português.",
        keywords=["PINNs", "CFD", "surrogate modeling"],
        source_url=paper.source_url,
    )
    return DigestItem(selected=selected, enriched=enriched)


def test_formatter_builds_three_sections_and_keywords() -> None:
    """Render three papers per source and include keywords in Telegram output."""

    items = [
        _item(source, rank)
        for source in ("arxiv", "semantic_scholar", "openalex")
        for rank in range(1, 4)
    ]

    messages = build_telegram_messages(items, run_date=date(2026, 10, 1))
    rendered = "\n".join(messages)

    assert "arXiv — Top 3" in rendered
    assert "Semantic Scholar — Top 3" in rendered
    assert "OpenAlex — Top 3" in rendered
    assert "Palavras-chave:" in rendered
    assert "PINNs, CFD, surrogate modeling" in rendered


def test_formatter_renders_partial_digest_with_unavailable_source_note() -> None:
    """Render successful sources when one repository has no selected papers."""

    items = [
        _item(source, rank)
        for source in ("arxiv", "semantic_scholar")
        for rank in range(1, 4)
    ]

    messages = build_telegram_messages(items, run_date=date(2026, 10, 1))
    rendered = "\n".join(messages)

    assert "Nenhum artigo disponível para esta fonte nesta execução." in rendered
    assert "OpenAlex — Top 3 mais citados" in rendered
    assert "OpenAlex-1" not in rendered


def test_formatter_splits_oversized_paper_without_losing_content() -> None:
    """Keep every message within the limit and preserve a long summary."""

    items = [
        _item(source, rank)
        for source in ("arxiv", "semantic_scholar", "openalex")
        for rank in range(1, 4)
    ]
    long_title = "long-title-" * 400
    long_summary = "summary<&>-" * 500
    items[1] = items[1].model_copy(
        update={
            "enriched": items[1].enriched.model_copy(
                update={"title": long_title}
            )
        }
    )
    items[0] = items[0].model_copy(
        update={
            "enriched": items[0].enriched.model_copy(
                update={"executive_summary": long_summary}
            )
        }
    )

    messages = build_telegram_messages(items, run_date=date(2026, 10, 1))
    visible_text = "".join(
        html.unescape(re.sub(r"<[^>]+>", "", message))
        for message in messages
    )

    assert len(messages) > 1
    assert all(len(message) <= TELEGRAM_SAFE_LIMIT for message in messages)
    assert all(message.count("<b>") == message.count("</b>") for message in messages)
    assert long_title in visible_text
    assert long_summary in visible_text
