# Direct Ratings

This package gives cached EigenBench responses 1--10 judge ratings. It does not
generate new target responses.

The Kindness configuration has 1,000 scenarios, eight targets, and eight judges:

```text
whole constitution: 1,000 x 8 x 8 = 64,000 ratings
criterion-wise:     1,000 x 8 x 8 = 64,000 calls, 512,000 ratings
```

Whole-constitution calls return one score and rationale. Criterion-wise calls
return one for each of eight criteria.

## Setup

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r numerical_rating/requirements-test.txt
.venv/bin/python -m pytest numerical_rating/tests
```

Tests use fixtures and need no experiment data. Generated data is not committed;
see [DATA.md](DATA.md) for required paths and reference hashes.

## Collection

Provide the cached responses listed in [DATA.md](DATA.md), set
`OPENROUTER_API_KEY`, then run:

```bash
.venv/bin/python -m numerical_rating.collection.run_round_robin
.venv/bin/python -m numerical_rating.collection.run_criterion_round_robin
```

Runs are resumable. Use repeated `--judge` arguments for a subset and `--limit`
for a smoke test. Configuration lives in
`numerical_rating/configs/kindness_1000_round_robin.yaml`.

## Analysis

```bash
.venv/bin/python -m numerical_rating.analysis.reports.round_robin_report
.venv/bin/python -m numerical_rating.analysis.reports.criterion_report
.venv/bin/python -m numerical_rating.analysis.experiments.scenario_uncertainty
.venv/bin/python -m numerical_rating.analysis.experiments.direct_vs_pairwise
.venv/bin/python -m numerical_rating.analysis.experiments.criterion_direct_vs_btd
```

Reports produce whole and criterion EigenTrust rankings. Experiments compare
direct-rating SVD `U,V` with Davidson BTD and measure scenario uncertainty.
Outputs go to `data/output/numerical_rating`; pairwise comparisons also need the
original EigenBench logs.

## Inspect Viewer

```bash
.venv/bin/python -m numerical_rating.publishing.publish_viewer
.venv/bin/python -m numerical_rating.publishing.publish_viewer \
  --rating-mode criterion_wise
```

This bundles one complete log per judge under `docs` and redacts home paths.

## Layout

- `collection/`: response loading and Inspect tasks.
- `analysis/data_loading/`, `analysis/model_fitting/`: shared data and model code.
- `analysis/evaluation/`, `analysis/experiments/`: comparisons and uncertainty.
- `analysis/aggregation/`, `analysis/reports/`: EigenTrust outputs.
- `publishing/`: static Inspect viewer.
- `configs/`, `prompts/`, `tests/`: run inputs and tests.
