"""Telegram HTML formatter for the three-source scientific literature digest."""

from __future__ import annotations

import html
import re
from collections import Counter
from datetime import date

from sciml_digest.models import DigestItem, ScientificPaper, ScientificSource

TELEGRAM_SAFE_LIMIT = 3800
EXPECTED_PER_SOURCE = 3
SOURCE_ORDER: tuple[ScientificSource, ...] = (
    "arxiv",
    "semantic_scholar",
    "openalex",
)
SOURCE_NAMES: dict[ScientificSource, str] = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
}
SECTION_TITLES: dict[ScientificSource, str] = {
    "arxiv": "arXiv — Top 3 relevantes publicados/atualizados em D-1",
    "semantic_scholar": "Semantic Scholar — Top 3 relevantes dos últimos 30 dias",
    "openalex": "OpenAlex — Top 3 mais citados dos últimos 60 dias",
}


def build_telegram_messages(
    items: list[DigestItem],
    run_date: date,
) -> list[str]:
    """Build Telegram HTML messages below the platform text limit."""

    _validate_source_counts(items)

    blocks = [
        (
            "<b>SciLitAI — Scientific Literature Digest</b>\n"
            f"<i>Execução UTC: {html.escape(run_date.isoformat())}</i>"
        )
    ]

    for source in SOURCE_ORDER:
        blocks.append(f"<b>{html.escape(SECTION_TITLES[source])}</b>")
        source_items = sorted(
            (
                item
                for item in items
                if item.selected.paper.source == source
            ),
            key=lambda item: item.selected.rank,
        )
        blocks.extend(_format_item(item) for item in source_items)

    return _chunk_blocks(blocks)


def _format_item(item: DigestItem) -> str:
    """Format one ranked digest item using escaped Telegram HTML."""

    paper = item.selected.paper
    enriched = item.enriched

    title = html.escape(enriched.title)
    approach = html.escape(enriched.ai_approach)
    domain = html.escape(enriched.domain_application)
    summary = html.escape(enriched.executive_summary)
    authors = html.escape(_format_authors(paper.authors))
    keywords = html.escape(", ".join(enriched.keywords))
    published = _format_publication_date(paper)
    source_url = html.escape(str(paper.source_url), quote=True)
    source_name = SOURCE_NAMES[paper.source]

    metadata_lines = [
        f"<b>Fonte:</b> {html.escape(source_name)}",
        f"<b>Abordagem:</b> {approach}",
        f"<b>Domínio:</b> {domain}",
        f"<b>Autores:</b> {authors}",
        f"<b>Publicado:</b> {published}",
    ]

    if (
        paper.source == "openalex"
        and paper.publication_date_verified
        and paper.publication_date_validation_source is not None
    ):
        metadata_lines[-1] += (
            " | <b>Data validada:</b> "
            f"{html.escape(paper.publication_date_validation_source.title())}"
        )

    if paper.source == "arxiv" and paper.updated_date is not None:
        metadata_lines[-1] += (
            f" | <b>Atualizado:</b> {paper.updated_date.date().isoformat()}"
        )

    if item.selected.selection_basis == "relevance":
        metadata_lines.append(
            f"<b>Score temático:</b> {item.selected.relevance_score:.3f}"
        )

    if paper.citation_count is not None:
        metadata_lines.append(f"<b>Citações:</b> {paper.citation_count}")

    metadata_lines.append(f"<b>Palavras-chave:</b> {keywords}")

    tags = " ".join(
        [
            _to_hashtag(enriched.ai_approach),
            _to_hashtag(enriched.domain_application),
        ]
    )

    links = [f'<a href="{source_url}">{html.escape(source_name)}</a>']
    if paper.pdf_url is not None:
        pdf_url = html.escape(str(paper.pdf_url), quote=True)
        links.append(f'<a href="{pdf_url}">PDF</a>')
    if paper.doi:
        doi_url = html.escape(_doi_url(paper.doi), quote=True)
        links.append(f'<a href="{doi_url}">DOI</a>')

    metadata = "\n".join(metadata_lines)
    rendered_links = " · ".join(links)

    return (
        f"<b>{item.selected.rank}. {title}</b>\n"
        f"{metadata}\n"
        f"{summary}\n"
        f"{tags}\n"
        f"{rendered_links}"
    )


def _format_publication_date(paper: ScientificPaper) -> str:
    """Format publication date according to available bibliographic precision."""

    value = paper.publication_date
    if paper.publication_date_precision == "year":
        return f"{value.year:04d}"
    if paper.publication_date_precision == "month":
        return f"{value.year:04d}-{value.month:02d}"
    return value.date().isoformat()


def _validate_source_counts(items: list[DigestItem]) -> None:
    """Require exactly three selected papers from each configured source."""

    counts = Counter(item.selected.paper.source for item in items)
    invalid = {
        SOURCE_NAMES[source]: counts[source]
        for source in SOURCE_ORDER
        if counts[source] != EXPECTED_PER_SOURCE
    }
    if invalid:
        details = ", ".join(f"{name}={count}" for name, count in invalid.items())
        raise ValueError(
            "Digest must contain exactly three papers from each source; "
            f"received {details}."
        )


def _chunk_blocks(blocks: list[str]) -> list[str]:
    """Chunk formatted blocks without breaking Telegram's safe text limit."""

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


def _doi_url(doi: str) -> str:
    """Build a canonical HTTPS DOI URL."""

    normalized = doi.strip()
    lowered = normalized.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return f"https://doi.org/{normalized}"
