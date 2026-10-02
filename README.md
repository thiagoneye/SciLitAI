# SciLitAI

SciLitAI is an automated scientific-literature intelligence pipeline that discovers, ranks, summarizes, and delivers research updates from **arXiv**, **Semantic Scholar**, and **OpenAlex**.

The thematic scope covers Scientific Machine Learning, physics-informed methods, neural operators, surrogate modeling, uncertainty quantification, CFD, computational mechanics, and industrial digitalization.

## Multi-source strategy

Every daily execution builds three independent literature views:

```text
arXiv
  -> papers published OR updated in D-1 UTC
  -> thematic filtering
  -> deterministic relevance ranking
  -> Top 3

Semantic Scholar
  -> papers published in the last 30 calendar days
  -> thematic filtering
  -> deterministic relevance ranking
  -> Top 3

OpenAlex
  -> papers published in the last 60 calendar days
  -> thematic filtering
  -> citation-count ranking
  -> Top 3

3 + 3 + 3 = 9 papers
  -> cross-source deduplication
  -> Gemini enrichment + keyword extraction
  -> Telegram HTML digest
```

Cross-source deduplication uses DOI, arXiv ID when available, and normalized title. Source precedence is arXiv -> Semantic Scholar -> OpenAlex. If a later source returns a paper already selected from an earlier source, SciLitAI backfills it with the next eligible candidate from that source.

## Search taxonomy

The taxonomy is centralized in `sciml_digest/taxonomy.py` and contains five thematic clusters.

The current cluster priorities are:

```text
sciml_pinns                         2.0
neural_operators_rom_surrogates    2.0
cfd_rheology_fem_generative_flow   1.0
industry40_uns_predictive_maintenance 1.0
uq_probabilistic_methods           0.5
```

The relevance score weights title matches more strongly than abstract matches and multiplies term contributions by these cluster priorities.

## arXiv selection

arXiv remains category-constrained to the configured primary categories:

```text
cs.LG
stat.ML
physics.comp-ph
physics.flu-dyn
cond-mat.soft
math.NA
eess.SY
cs.SE
cs.DC
```

The target interval is the complete previous UTC calendar day:

```text
D-1 00:00:00 UTC <= submitted/updated < D 00:00:00 UTC
```

Both `submittedDate` and `lastUpdatedDate` are queried. Results are merged by canonical arXiv ID, the primary category is validated locally, and the final Top 3 is ranked by the weighted thematic relevance score.

arXiv requests preserve a minimum interval of three seconds between calls and use exponential backoff for transient failures.

## Semantic Scholar selection

Semantic Scholar uses the Academic Graph paper relevance search endpoint. SciLitAI searches the same thematic clusters and restricts results to the last 30 calendar dates, including the run date.

Records without an exact `publicationDate` are excluded because a year-only record cannot prove membership in the strict 30-day window.

The API returns relevance-ranked candidates, which are then merged and reranked locally with SciLitAI's weighted thematic score. The final result is the Top 3 relevant papers.

`SEMANTIC_SCHOLAR_API_KEY` is optional in configuration. When supplied, it is sent through the `x-api-key` header.

## OpenAlex selection

OpenAlex searches the same thematic taxonomy over works published in the last 60 calendar dates, including the run date.

Queries use Boolean OR search expressions and request:

```text
sort=cited_by_count:desc
```

The final local ordering uses:

```text
1. citation_count descending
2. thematic relevance score descending
3. publication date descending
```

Therefore the selected OpenAlex section is explicitly the **Top 3 most-cited eligible papers in the last 60 days**.

When OpenAlex provides work keywords, they are preserved in normalized source metadata and supplied to Gemini as additional evidence.

`OPENALEX_API_KEY` is optional.

## Relevance ranking

For relevance-ranked sources, SciLitAI uses approximately:

```text
score = 100 * (3 * weighted_title_density + weighted_abstract_density)
        + term_diversity_bonus
        + weighted_cluster_coverage_bonus
```

Each matched term receives:

```text
term_weight * cluster_weight
```

This means SciML/PINN and Neural Operator/ROM/Surrogate terms have twice the base contribution of the intermediate CFD/Industry clusters and four times the contribution of the low-priority UQ cluster.

## Normalized paper model

All repositories are converted into one `ScientificPaper` model containing fields such as:

```text
source
source_id
title
authors
abstract
publication_date
updated_date
source_url
pdf_url
doi
arxiv_id
citation_count
primary_category
categories
source_keywords
matched_clusters
```

This prevents source-specific metadata shapes from leaking into ranking, Gemini enrichment, and Telegram formatting.

## Gemini enrichment

The nine selected papers are sent individually to Gemini using Pydantic-backed structured output.

For every paper Gemini returns:

```text
title
ai_approach
domain_application
executive_summary
keywords
source_url
```

The summary is generated in Brazilian Portuguese. `keywords` contains 3-6 concise technical keywords grounded in the source metadata. Repository-provided keywords are preferred when available; otherwise keywords are extracted from the title and abstract.

The original title and source URL overwrite the model output after inference, preventing the LLM from changing source references.

## Telegram digest

The Telegram digest is organized into three sections:

```text
arXiv — Top 3 relevantes publicados/atualizados em D-1
Semantic Scholar — Top 3 relevantes dos últimos 30 dias
OpenAlex — Top 3 mais citados dos últimos 60 dias
```

Each paper includes:

```text
rank
title
source
AI/methodological approach
domain
authors
publication/update date
relevance score when applicable
citation count when available
keywords
executive summary
source/PDF/DOI links when available
```

Messages are escaped as Telegram HTML and split below a safe text-length threshold.

## Project structure

```text
SciLitAI/
├── .github/
│   └── workflows/
│       └── daily_sciml_digest.yml
├── sciml_digest/
│   ├── __init__.py
│   ├── arxiv_client.py
│   ├── semantic_scholar_client.py
│   ├── openalex_client.py
│   ├── source_utils.py
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
│   ├── test_semantic_scholar_client.py
│   ├── test_openalex_client.py
│   ├── test_formatter.py
│   ├── test_selector.py
│   ├── test_settings.py
│   └── test_telegram_client.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Required credentials

Required:

```text
GEMINI_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Optional repository credentials:

```text
SEMANTIC_SCHOLAR_API_KEY
OPENALEX_API_KEY
```

For GitHub Actions, configure the required values as Repository Secrets. The two repository API keys can also be created as secrets when available; an unset optional secret is treated as an empty value.

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

If the local host has broken IPv6 routing to Telegram, set:

```env
TELEGRAM_FORCE_IPV4=true
```

The portable default remains `false`.

## GitHub Actions

The workflow runs daily at:

```yaml
schedule:
  - cron: "0 10 * * *"
```

It also supports manual execution through `workflow_dispatch`.

The runner compiles the Python source, executes the unit tests, and only then runs the live multi-source pipeline. The job timeout is 30 minutes to accommodate repository pacing, transient API retries, and nine Gemini enrichment calls.

## Operational note

The arXiv rule remains strict D-1. On days when fewer than three eligible arXiv papers exist, the pipeline raises `SelectionError` rather than silently extending the time window. The same strict three-paper requirement applies after cross-source deduplication for Semantic Scholar and OpenAlex.
