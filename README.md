# SciLitAI

SciLitAI is an automated scientific-literature intelligence pipeline that discovers, ranks, summarizes, and delivers a daily digest of research papers related to Scientific Machine Learning, Physics AI, neural operators, surrogate modeling, uncertainty quantification, computational mechanics, and industrial digitalization.

## Pipeline

```text
GitHub Actions (daily 10:00 UTC)
        |
        v
arXiv Atom API
        |
        +-- target primary categories
        +-- five thematic search clusters
        +-- submittedDate D-1 UTC
        +-- lastUpdatedDate D-1 UTC
        |
        v
Normalization + primary-category validation + deduplication
        |
        v
Deterministic relevance ranking
        |
        +-- title term density
        +-- abstract term density
        +-- thematic diversity
        |
        v
Top 5 papers
        |
        v
Gemini structured enrichment
        |
        v
Telegram HTML digest
```

## 1. Target arXiv categories

SciLitAI restricts candidates to papers whose **primary arXiv category** is one of:

- `cs.LG`, `stat.ML`
- `physics.comp-ph`, `physics.flu-dyn`, `cond-mat.soft`
- `math.NA`
- `eess.SY`, `cs.SE`, `cs.DC`

The query itself uses `cat:` clauses, and the returned Atom metadata is subsequently validated against the paper's primary category. This second validation prevents a paper from being accepted only because one of its secondary categories is in scope.

## 2. Search clusters

The taxonomy is defined in `sciml_digest/taxonomy.py` and contains five clusters:

1. SciML & PINNs
2. Neural Operators & ROM/Surrogates
3. UQ & probabilistic methods
4. CFD, rheology, FEM & generative fluid flow
5. Industry 4.0, UNS & predictive maintenance

Each cluster uses OR semantics internally. Categories, topic terms, and the temporal condition are combined with AND semantics.

To keep request URLs bounded, large clusters are split into configurable chunks controlled by:

```env
ARXIV_TERMS_PER_QUERY=12
```

Every chunk preserves the same category and temporal restrictions. Results from all chunks are merged and deduplicated by canonical arXiv ID.

## 3. Strict D-1 UTC window

The pipeline processes exactly the previous UTC calendar day:

```text
00:00:00 UTC <= event timestamp < 00:00:00 UTC of the next day
```

For example, a run on `2026-10-01` processes the window:

```text
2026-09-30T00:00:00Z <= timestamp < 2026-10-01T00:00:00Z
```

SciLitAI retrieves both:

- papers whose initial submission is in D-1 (`submittedDate`);
- papers whose last update is in D-1 (`lastUpdatedDate`).

The two result streams are unioned and deduplicated. A local timestamp check is applied after parsing to enforce the interval exactly.

There is **no temporal backfill**. If fewer than five eligible papers exist, the pipeline raises `SelectionError` instead of silently selecting papers from earlier days.

## 4. Ranking

The arXiv requests are sent with:

```text
sortBy=relevance
sortOrder=descending
```

The final Top 5 is nevertheless computed locally so that ranking is deterministic across the merged candidate set.

The score uses term density in title and abstract, with a stronger title contribution:

```text
score = 100 * (3 * title_density + abstract_density)
        + matched_term_diversity_bonus
        + matched_cluster_coverage_bonus
```

Multi-word technical phrases receive a small weight increase relative to short acronyms. Ties are broken by the most recent qualifying submission/update timestamp and then by arXiv ID.

## 5. Output metadata

Each normalized `ArxivPaper` is a Pydantic model containing at least:

```text
arxiv_id
 title
 authors: list[str]
 abstract
 published_date
 updated_date
 arxiv_url
 pdf_url
 primary_category
 categories
 matched_clusters
```

Text fields are sanitized by collapsing line breaks and duplicate whitespace. Timestamps are timezone-aware UTC datetimes and serialize to ISO-8601.

## 6. arXiv request policy

SciLitAI enforces a minimum interval of three seconds between consecutive arXiv requests:

```env
ARXIV_MIN_REQUEST_INTERVAL_SECONDS=3
```

Transient HTTP errors (`408`, `429`, `500`, `502`, `503`, `504`) and transport failures use bounded exponential backoff. `Retry-After` is honored when provided.

Other ingestion controls:

```env
ARXIV_MAX_RESULTS_PER_QUERY=100
ARXIV_TERMS_PER_QUERY=12
ARXIV_TIMEOUT_SECONDS=30
ARXIV_MAX_ATTEMPTS=4
```

## 7. Gemini enrichment

The selected Top 5 papers are sent to the Google GenAI SDK using structured JSON output backed by a Pydantic schema.

For each paper Gemini produces:

- `title`
- `ai_approach`
- `domain_application`
- `executive_summary`
- `arxiv_url`

The source title and arXiv URL are overwritten with the original arXiv metadata after inference so the LLM cannot modify them.

## 8. Telegram delivery

The final digest is rendered using Telegram HTML with escaping of dynamic content. Each item includes its rank, title, AI approach, application domain, authors, publication/update dates, relevance score, executive summary, tags, arXiv link, and direct PDF link.

Long digests are split into messages below a safe character threshold.

## Project structure

```text
SciLitAI/
├── .github/
│   └── workflows/
│       └── daily_sciml_digest.yml
├── sciml_digest/
│   ├── __init__.py
│   ├── arxiv_client.py
│   ├── exceptions.py
│   ├── formatter.py
│   ├── gemini_service.py
│   ├── main.py
│   ├── models.py
│   ├── pipeline.py
│   ├── selector.py
│   ├── settings.py
│   ├── taxonomy.py
│   └── telegram_client.py
├── tests/
│   ├── test_arxiv_client.py
│   └── test_selector.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Configuration

Required GitHub Actions secrets:

```text
GEMINI_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Optional configuration is documented in `.env.example`.

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
python -m pytest -q
python -m sciml_digest.main
```

## GitHub Actions

The workflow runs daily at:

```yaml
schedule:
  - cron: "0 10 * * *"
```

This is 10:00 UTC, corresponding to 07:00 in `America/Sao_Paulo` while the timezone is UTC-03. Manual execution is also available through `workflow_dispatch`.

Before running the digest, CI compiles the Python sources and executes the unit tests.
