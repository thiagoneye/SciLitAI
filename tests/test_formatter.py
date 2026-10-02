"""Unit tests for the multi-source Telegram formatter."""

from datetime import date, datetime, timezone

from sciml_digest.formatter import build_telegram_messages
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
