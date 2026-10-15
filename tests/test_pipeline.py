"""Tests for source-level failure isolation in the digest pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sciml_digest import pipeline
from sciml_digest.exceptions import (
    ArxivError,
    BibliographicMetadataError,
    OpenAlexError,
    SelectionError,
    SemanticScholarError,
)
from sciml_digest.models import (
    EnrichedArticle,
    ScientificPaper,
    SelectedPaper,
    ScientificSource,
)
from scripts.run_agent_e2e import run_optional_source_stage
from sciml_digest.settings import Settings


SOURCE_ERRORS = {
    "arxiv": ArxivError,
    "semantic_scholar": SemanticScholarError,
    "openalex": OpenAlexError,
}


def _selected_papers(source: str) -> list[SelectedPaper]:
    """Create three selected papers for one repository."""

    source_urls = {
        "arxiv": "https://arxiv.org/abs/2609.00001",
        "semantic_scholar": "https://www.semanticscholar.org/paper/s1",
        "openalex": "https://openalex.org/W1",
    }
    return [
        SelectedPaper(
            paper=ScientificPaper(
                source=source,
                source_id=f"{source}-{rank}",
                title=f"Paper {source} {rank}",
                authors=["Ada Lovelace"],
                abstract="Scientific Machine Learning.",
                publication_date=datetime(2026, 9, 20, tzinfo=timezone.utc),
                source_url=source_urls[source],
            ),
            rank=rank,
            selection_basis="citations" if source == "openalex" else "relevance",
            relevance_score=1.0,
        )
        for rank in range(1, 4)
    ]


class _FakeGemini:
    def __init__(self, **_: object) -> None:
        pass

    def enrich(self, paper: ScientificPaper) -> EnrichedArticle:
        return EnrichedArticle(
            title=paper.title,
            ai_approach="SciML",
            domain_application="Physics",
            executive_summary="Test summary.",
            keywords=["SciML", "PINN", "CFD"],
            source_url=paper.source_url,
        )


class _FakeTelegram:
    sent_messages: list[str] = []

    def __init__(self, **_: object) -> None:
        pass

    def __enter__(self) -> _FakeTelegram:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def send_messages(self, messages: list[str]) -> None:
        type(self).sent_messages = messages


@pytest.mark.parametrize(
    ("failed_source", "failure_kind"),
    [
        *((source, "api") for source in SOURCE_ERRORS),
        *((source, "selection") for source in SOURCE_ERRORS),
        ("openalex", "validation"),
    ],
)
def test_pipeline_continues_when_one_repository_fails(
    monkeypatch: pytest.MonkeyPatch,
    failed_source: str,
    failure_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Send a partial digest for API, selection, or DOI-validation failures."""

    def fetch(source: str):
        def _fetch(*_: object, **__: object) -> list[ScientificPaper]:
            del __
            if source == failed_source and failure_kind == "api":
                raise SOURCE_ERRORS[source]("repository unavailable")
            return []

        return _fetch

    def select(source: str):
        def _select(*args: object, **kwargs: object) -> list[SelectedPaper]:
            del args, kwargs
            if source == failed_source and failure_kind == "selection":
                raise SelectionError("not enough unique eligible papers")
            return _selected_papers(source)

        return _select

    def no_candidates(*args: object, **kwargs: object) -> list[ScientificPaper]:
        del args, kwargs
        return []

    def validate_openalex(*args: object, **kwargs: object) -> list[ScientificPaper]:
        del args, kwargs
        if failed_source == "openalex" and failure_kind == "validation":
            raise BibliographicMetadataError("date metadata unavailable")
        return []

    monkeypatch.setattr(pipeline, "_fetch_arxiv", fetch("arxiv"))
    monkeypatch.setattr(
        pipeline,
        "_fetch_semantic_scholar",
        fetch("semantic_scholar"),
    )
    monkeypatch.setattr(pipeline, "_fetch_openalex", fetch("openalex"))
    monkeypatch.setattr(
        pipeline,
        "select_arxiv_papers",
        select("arxiv"),
    )
    monkeypatch.setattr(
        pipeline,
        "select_semantic_scholar_papers",
        select("semantic_scholar"),
    )
    monkeypatch.setattr(
        pipeline,
        "rank_openalex_candidates",
        no_candidates,
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_openalex_dates",
        validate_openalex,
    )
    monkeypatch.setattr(
        pipeline,
        "select_openalex_papers",
        select("openalex"),
    )
    monkeypatch.setattr(pipeline, "GeminiService", _FakeGemini)
    _FakeTelegram.sent_messages = []
    monkeypatch.setattr(pipeline, "TelegramClient", _FakeTelegram)

    settings = Settings(
        gemini_api_key="test",
        telegram_bot_token="test",
        telegram_chat_id="test",
    )

    pipeline.run_pipeline(settings)

    sent = "\n".join(_FakeTelegram.sent_messages)
    if failure_kind == "selection":
        assert "Não foram encontrados artigos únicos elegíveis" in sent
        assert "insufficient eligible papers" in caplog.text
    else:
        assert "Falha na consulta ou validação dos dados" in sent
        expected_error = (
            SOURCE_ERRORS[failed_source]
            if failure_kind == "api"
            else BibliographicMetadataError
        )
        assert expected_error.__name__ in caplog.text
    for source in SOURCE_ERRORS:
        if source != failed_source:
            assert source in sent


def test_pipeline_fails_explicitly_when_all_repositories_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not send an empty digest if every repository is unavailable."""

    def fail(*_: object, **__: object) -> list[ScientificPaper]:
        del __
        raise ArxivError("repository unavailable")

    monkeypatch.setattr(pipeline, "_fetch_arxiv", fail)
    monkeypatch.setattr(pipeline, "_fetch_semantic_scholar", fail)
    monkeypatch.setattr(pipeline, "_fetch_openalex", fail)

    settings = Settings(
        gemini_api_key="test",
        telegram_bot_token="test",
        telegram_chat_id="test",
    )

    with pytest.raises(SelectionError, match="No papers could be selected"):
        pipeline.run_pipeline(settings)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            SelectionError("not enough eligible papers"),
            "insufficient eligible papers",
        ),
        (
            ArxivError("repository unavailable"),
            "source failure",
        ),
    ],
)
def test_e2e_optional_source_stage_continues_on_domain_errors(
    error: Exception,
    expected_status: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the live diagnostic aligned with partial-source pipeline behavior."""

    statuses: dict[ScientificSource, str] = {}

    def fail() -> list[ScientificPaper]:
        raise error

    result = run_optional_source_stage(
        "source stage",
        "arxiv",
        fail,
        [],
        statuses,
    )

    output = capsys.readouterr().out
    assert result == []
    assert statuses["arxiv"]
    assert expected_status in output


def test_e2e_optional_source_stage_does_not_hide_programming_errors() -> None:
    """Let unexpected exceptions remain visible instead of treating them as outages."""

    statuses: dict[ScientificSource, str] = {}

    def fail() -> list[ScientificPaper]:
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        run_optional_source_stage(
            "source stage",
            "arxiv",
            fail,
            [],
            statuses,
        )

    assert statuses == {}
