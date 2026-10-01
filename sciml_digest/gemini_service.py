"""Gemini structured-output enrichment service."""

from __future__ import annotations

import logging
import random
import time

from google import genai
from google.genai import errors, types

from sciml_digest.exceptions import GeminiError
from sciml_digest.models import ArxivPaper, EnrichedArticle

LOGGER = logging.getLogger(__name__)

RETRIABLE_CODES = {408, 429, 500, 502, 503, 504}


class GeminiService:
    """Enrich paper metadata using Gemini and a Pydantic response schema."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_attempts: int = 4,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_attempts = max_attempts

    def enrich(self, paper: ArxivPaper) -> EnrichedArticle:
        """Classify and summarize one paper with structured output."""

        prompt = _build_prompt(paper)
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EnrichedArticle,
                        temperature=0.1,
                    ),
                )

                parsed = response.parsed
                if isinstance(parsed, EnrichedArticle):
                    result = parsed
                elif response.text:
                    result = EnrichedArticle.model_validate_json(response.text)
                else:
                    raise GeminiError("Gemini returned an empty structured response.")

                # Título e URL são dados de origem; nunca devem ser alterados pelo LLM.
                return result.model_copy(
                    update={
                        "title": paper.title,
                        "arxiv_url": paper.arxiv_url,
                    }
                )

            except errors.APIError as exc:
                last_error = exc
                if exc.code not in RETRIABLE_CODES or attempt == self._max_attempts:
                    raise GeminiError(
                        f"Gemini API failed with HTTP/API code {exc.code}."
                    ) from exc
                delay = _retry_delay(attempt)
                LOGGER.warning(
                    "Gemini transient API error %s; retrying in %.1fs.",
                    exc.code,
                    delay,
                )
                time.sleep(delay)
            except GeminiError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break
                delay = _retry_delay(attempt)
                LOGGER.warning(
                    "Gemini transient client error (%s); retrying in %.1fs.",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise GeminiError("Gemini enrichment failed after all retries.") from last_error


def _build_prompt(paper: ArxivPaper) -> str:
    """Build a constrained extraction prompt from source metadata."""

    authors = ", ".join(paper.authors) if paper.authors else "Not provided"
    categories = ", ".join(paper.categories) if paper.categories else "Not provided"

    return f"""
You are a Scientific Machine Learning research analyst.

Analyze ONLY the metadata below. Do not invent numerical speedups, accuracy gains,
datasets, equations, or claims that are not explicitly supported by the title or
abstract.

Required behavior:
- Keep `title` exactly equal to the source title.
- Choose the closest `ai_approach` value allowed by the response schema.
- Use a concise, technically precise `domain_application`.
- Write `executive_summary` in Brazilian Portuguese, in 2 to 3 objective sentences.
- The summary must state: (1) the problem, (2) the proposed methodology, and
  (3) the scientific/computational impact. If the abstract does not quantify a
  gain, describe it qualitatively instead of inventing a number.
- Keep `arxiv_url` exactly equal to the source arXiv URL.

SOURCE METADATA
arXiv ID: {paper.arxiv_id}
Title: {paper.title}
Authors: {authors}
Categories: {categories}
Published: {paper.published_date.isoformat()}
arXiv URL: {paper.arxiv_url}
Abstract:
{paper.abstract}
""".strip()


def _retry_delay(
    attempt: int,
    base_seconds: float = 2.0,
    max_seconds: float = 30.0,
) -> float:
    """Calculate bounded exponential backoff with jitter."""

    delay = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
    return delay + random.uniform(0.0, 0.75)
