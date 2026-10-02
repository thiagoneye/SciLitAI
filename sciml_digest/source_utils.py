"""Shared helpers for scientific repository clients."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence


def chunked(values: Sequence[str], chunk_size: int) -> Iterable[tuple[str, ...]]:
    """Yield fixed-size chunks from a sequence."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    for start in range(0, len(values), chunk_size):
        yield tuple(values[start : start + chunk_size])


def clean_text(value: str) -> str:
    """Collapse line breaks and duplicate whitespace."""

    return re.sub(r"\s+", " ", value).strip()


def retry_delay(
    attempt: int,
    retry_after: str | None = None,
    minimum_seconds: float = 1.0,
    max_seconds: float = 60.0,
) -> float:
    """Calculate bounded exponential backoff with jitter."""

    if retry_after:
        try:
            return max(float(retry_after), minimum_seconds)
        except ValueError:
            pass

    delay = min(minimum_seconds * (2 ** (attempt - 1)), max_seconds)
    return delay + random.uniform(0.0, 0.75)


def normalize_plain_query_term(term: str) -> str:
    """Normalize a taxonomy term for APIs that accept plain-text queries only."""

    return re.sub(r"[-/]", " ", term).strip()
