"""CLI entry point for the automated SciML digest."""

from __future__ import annotations

import logging
import sys

from sciml_digest.exceptions import DigestError
from sciml_digest.pipeline import run_pipeline
from sciml_digest.settings import Settings


def main() -> int:
    """Run the application and return a process exit code."""

    try:
        settings = Settings.from_env()
        logging.basicConfig(
            level=getattr(logging, settings.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        run_pipeline(settings)
        return 0
    except DigestError as exc:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.getLogger(__name__).error(
            "Pipeline failed: %s",
            exc,
        )
        return 1
    except Exception:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.getLogger(__name__).exception(
            "Unexpected fatal pipeline error."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
