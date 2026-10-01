"""Telegram HTML formatter with safe escaping and message chunking."""

from __future__ import annotations

import html
import re
from datetime import date

from sciml_digest.models import DigestItem

TELEGRAM_SAFE_LIMIT = 3800
EXPECTED_ARTICLE_COUNT = 5


def build_telegram_messages(
    items: list[DigestItem],
    digest_date: date,
) -> list[str]:
    """Build Telegram HTML messages below the platform text limit."""

    if len(items) != EXPECTED_ARTICLE_COUNT:
        raise ValueError(
            f"Digest must contain exactly {EXPECTED_ARTICLE_COUNT} ranked items."
        )

    ordered_items = sorted(items, key=lambda item: item.selected.rank)
    header = (
        "<b>SciLitAI — Daily Scientific Literature Digest</b>\n"
        f"<i>D-1 UTC: {html.escape(digest_date.isoformat())}</i>\n\n"
        "<b>Top 5 por relevância temática</b>"
    )

    blocks = [header]
    blocks.extend(_format_item(item) for item in ordered_items)

    messages: list[str] = []
    current = ""

    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= TELEGRAM_SAFE_LIMIT:
            current = candidate
            continue

        if current:
            messages.append(current)
        current = block

    if current:
        messages.append(current)

    return messages


def _format_item(item: DigestItem) -> str:
    """Format one ranked digest item using escaped Telegram HTML."""

    paper = item.selected.paper
    enriched = item.enriched

    title = html.escape(enriched.title)
    approach = html.escape(enriched.ai_approach)
    domain = html.escape(enriched.domain_application)
    summary = html.escape(enriched.executive_summary)
    authors = html.escape(_format_authors(paper.authors))
    published = paper.published_date.date().isoformat()
    updated = paper.updated_date.date().isoformat()
    arxiv_url = html.escape(str(paper.arxiv_url), quote=True)
    pdf_url = html.escape(str(paper.pdf_url), quote=True)

    tags = " ".join(
        [
            _to_hashtag(enriched.ai_approach),
            _to_hashtag(enriched.domain_application),
        ]
    )

    return (
        f"<b>{item.selected.rank}. {title}</b>\n"
        f"<b>Abordagem:</b> {approach}\n"
        f"<b>Domínio:</b> {domain}\n"
        f"<b>Autores:</b> {authors}\n"
        f"<b>Publicado:</b> {published} | <b>Atualizado:</b> {updated}\n"
        f"<b>Score:</b> {item.selected.relevance_score:.3f}\n"
        f"{summary}\n"
        f"{tags}\n"
        f'<a href="{arxiv_url}">arXiv</a> · '
        f'<a href="{pdf_url}">PDF</a>'
    )


def _format_authors(authors: list[str]) -> str:
    """Limit author-line size while retaining useful attribution."""

    if not authors:
        return "Não informado"
    if len(authors) <= 4:
        return ", ".join(authors)
    return f"{', '.join(authors[:4])} et al."


def _to_hashtag(value: str) -> str:
    """Normalize a label to a compact Telegram hashtag."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return f"#{normalized}" if normalized else "#SciML"
