# Numerical Rating

Inspect tasks for rating cached ValueArena responses against a constitution.
The target responses are reused; only the judge calls are generated.

## Full Kindness Round Robin

```text
1,000 AskReddit scenarios
x 8 cached target responses
x 8 judge models
= 64,000 ratings
```

Each judge receives one scenario, one response, and the full Kindness
constitution. It returns one 1--10 score and a short rationale.

Run or resume every configured judge:

```bash
.venv/bin/python -m numerical_rating.run_round_robin
```

Run or resume selected judges in the same log directory:

```bash
.venv/bin/python -m numerical_rating.run_round_robin \
  --judge "Claude 4 Sonnet" \
  --judge "GPT 4.1"
```

Gemini 2.5 Pro has a 4,096-token generation budget in the run manifest because
its reasoning tokens count against the output limit. It can be resumed alone:

```bash
.venv/bin/python -m numerical_rating.run_round_robin \
  --judge "Gemini 2.5 Pro" \
  --log-dir runs/numerical_rating/kindness_1000_round_robin_gemini4096
```

Inspect reuses completed samples when a cancelled log is resumed with the same
task arguments. Connection count and HTTP retry settings can change between
resumes without changing task identity.

### Response repairs

The cache builder rejects provider-error strings and truncated reserved-token
artifacts. It writes the repaired cache atomically and records every changed
cell in a hash-bound manifest:

```bash
.venv/bin/python -m numerical_rating.complete_response_cache \
  --cache data/output/valuearena/processed/full8_kindness/askreddit_1000_responses_completed.json \
  --evaluations data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl \
  --repairs data/output/valuearena/processed/full8_kindness/askreddit_1000_invalid_response_repairs.jsonl \
  --output data/output/valuearena/processed/full8_kindness/askreddit_1000_responses_completed.json \
  --repaired-cells data/output/valuearena/processed/full8_kindness/askreddit_1000_repaired_cells.json \
  --all-cache-scenarios
```

Only those changed response cells are then rerated by all eight judges:

```bash
.venv/bin/python -m numerical_rating.run_round_robin \
  --cell-manifest data/output/valuearena/processed/full8_kindness/askreddit_1000_repaired_cells.json \
  --log-dir runs/numerical_rating/kindness_1000_round_robin_repairs
```

## Analysis

After all eight judge runs complete:

```bash
.venv/bin/python -m numerical_rating.round_robin_analysis
```

The analysis validates the complete `8 x 1,000 x 8` tensor, standardizes each
judge's ratings, constructs the judge-to-target trust matrix, applies
EigenTrust, and compares the matched 871-scenario ranking with the published
pairwise run. This slice uses the scenarios that survive the
published run's own comparison extraction and consistency handling, not all
923 scenario IDs present in its raw file. The comparison is descriptive:
published pairwise extraction excluded failed response cells, whereas the
numerical run replaces them before scoring. Grok 4.3 is also the configured
judge-side proxy for the unavailable Grok 4 endpoint.

By default the analysis reads both the main run directory and Gemini's
4,096-token resume directory. It requires the repair manifest and all eight
repair logs, then replaces only stale response hashes. Both the raw and
corrected rankings are retained in the result artifact.

## Inspect Viewer

After all eight judge logs are complete, publish the native Inspect bundle:

```bash
.venv/bin/python -m numerical_rating.publish_viewer
```

The publisher selects one complete log per configured judge across the main,
Gemini, and repair log directories; validates the corrected 64,000-cell tensor
and its provenance; checks GitHub file and Pages size limits; then replaces
`docs/numerical_rating`.

## Files

- `configs/kindness_1000_round_robin.yaml`: dataset, model roster, and run
  settings.
- `data.py`: response-cache and constitution loading.
- `prompts/whole_constitution_pointwise.md`: judge prompt.
- `run_pointwise.py`: one Inspect rating task.
- `run_round_robin.py`: parallel, resumable eval-set runner.
- `round_robin_analysis.py`: validation, whitening, EigenTrust, and comparison.
- `publish_viewer.py`: validated native Inspect viewer publisher.
