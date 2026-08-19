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

## Criterion-Wise Round Robin

The criterion-wise task reuses the same response cache, judge roster, Inspect
runner, batching, retry, and resume behavior. Each judge call returns one score
and rationale for every Kindness criterion:

```text
1,000 scenarios x 8 targets x 8 judges = 64,000 calls
64,000 calls x 8 criteria = 512,000 criterion ratings
```

Run or resume it separately:

```bash
.venv/bin/python -m numerical_rating.run_criterion_round_robin
```

After all eight judge logs complete, validate and analyze them:

```bash
.venv/bin/python -m numerical_rating.criterion_analysis
.venv/bin/python -m numerical_rating.publish_viewer \
  --rating-mode criterion_wise
```

The criterion task uses the same log selection, rating records, tensor
validation, CSV writing, trust computation, and viewer publisher as the
whole-constitution task. Exact criterion text and hashes live in task metadata;
each rating carries its criterion ID and hash. Analysis requires all 512,000
ratings and computes one independent EigenTrust result per criterion. The
criterion output budget is configured under `criterion_generation` in the
shared run manifest.

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

Scenario-level uncertainty on the shared 871-scenario support is computed
offline for both methods:

```bash
.venv/bin/python -m numerical_rating.scenario_uncertainty
```

The default run uses 2,000 paired bootstrap draws. Each draw selects the same
scenario IDs for both methods and retains their complete rating or comparison
blocks. Pairwise BTD refits use deterministic multi-start optimization and fail
if the point or jackknife fits have incompatible near-optima. Bootstrap draws
retain the deterministic lowest-loss fit and record any such instability rather
than dropping draws. Progress is checkpointed every 50 draws.

Ten fixed, near-equal scenario groups provide delete-group jackknife estimates.
Only the 783/784-scenario complements are fitted; the small groups are not
ranked in isolation. Pairwise records are exact-deduplicated, then repeated
collection passes are identified by response and reflection hashes and
reconciled independently. The published cleaner is retained only as a point
sensitivity because it drops groups containing multiple collection passes.

These intervals measure scenario-selection uncertainty. Stored target and judge
responses remain fixed, so generation and judge-sampling uncertainty are not
included.

Direct-rating judge and model embeddings are fitted with:

```bash
.venv/bin/python -m numerical_rating.direct_embedding_comparison
```

The whole-constitution fit factorizes the `8 x 8` standardized mean-rating
matrix. The criterion fit reshapes the complete `8 x 8 x 1,000 x 8` tensor into
a `64 x 8` matrix, giving each criterion-judge pair its own vector while sharing
one model-vector table. The output includes embeddings, reconstructed score
matrices, scenario-held-out rank CV, scenario bootstrap draws, and the existing
comparison with pairwise BTD.

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
- `prompts/criterion_pointwise.md`: criterion-wise judge prompt.
- `rating_task.py`: shared cached-response Inspect task construction.
- `run_pointwise.py`: whole-constitution Inspect task.
- `run_criteria.py`: criterion-wise Inspect task and strict parser.
- `run_round_robin.py`: parallel, resumable eval-set runner.
- `run_criterion_round_robin.py`: criterion-wise round-robin entrypoint.
- `criterion_analysis.py`: per-criterion analysis through the shared rating
  pipeline.
- `round_robin_analysis.py`: validation, whitening, EigenTrust, and comparison.
- `scenario_uncertainty.py`: paired scenario bootstrap, delete-group jackknife,
  and pairwise-cleaning sensitivity.
- `direct_embedding_comparison.py`: whole-constitution and criterion-wise
  low-rank embeddings, stability checks, and pairwise BTD comparison.
- `trust.py`: shared numerical-rating standardization and EigenTrust math.
- `publish_viewer.py`: validated native Inspect viewer publisher.
