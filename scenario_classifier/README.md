# Scenario Classifier

Retrieves constitution criteria relevant to natural scenarios using BM25 or
embedding similarity. Scenario generation and deduplication are not implemented.

## Layout

- `core/`: shared repo paths and JSON loaders.
- `retrieval/`: retrieval classes. BM25 is lexical; embeddings use cosine similarity over cached vectors.
- `audits/`: retrieval runners and optional comparison utilities for fresh annotation sets.
- `corpora/`: scripts that build retrieval corpora, currently raw criteria by default.
- `data/`: ignored local scenarios, corpora, and outputs for this task.

## Local Data

The runnable defaults look under `scenario_classifier/data/`:

- `scenarios/`: local scenario JSON files.
- `corpora/retrieval_units/`: raw-criteria units plus optional criteria-card artifacts.
- `private_anchors/`: optional ignored local anchor JSON files included by the raw-criteria builder.
- `outputs/embedding_retrieval/`: embedding vectors, manifests, retrieval rows, and comparison tables.
- `outputs/bm25_retrieval/`: BM25 retrieval rows and summaries.

## Common Commands

Run BM25 retrieval on the full AIRisk scenarios:

```bash
python -m scenario_classifier.audits.bm25_audit \
  --records scenario_classifier/data/scenarios/airiskdilemmas.json \
  --output-name airiskdilemmas_full_2999 \
  --corpus raw
```

Run embedding retrieval on the full AIRisk scenarios:

```bash
python -m scenario_classifier.audits.embedding_retrieval \
  --records scenario_classifier/data/scenarios/airiskdilemmas.json \
  --output-name airiskdilemmas_full_2999 \
  --model google/gemini-embedding-2
```

Swap `--records` and `--output-name` to run either retriever on another JSON/JSONL scenario file.

Run both embedding models used in the comparison table:

```bash
python -m scenario_classifier.audits.embedding_retrieval \
  --records scenario_classifier/data/scenarios/airiskdilemmas.json \
  --output-name airiskdilemmas_full_2999 \
  --model google/gemini-embedding-2 \
  --model qwen/qwen3-embedding-8b
```

Use `--corpus cards` or `--corpus both` only for criteria-card comparisons.

Build a retrieval comparison table only when you have a current annotation file:

```bash
python -m scenario_classifier.audits.retrieval_comparison \
  --annotations path/to/current_annotations.jsonl \
  --output-name matching_retrieval_output_name \
  --corpus raw
```

Rebuild raw criteria units:

```bash
python -m scenario_classifier.corpora.raw_criteria_units
```

Run the offline unit tests:

```bash
python -m unittest discover -s scenario_classifier/tests -v
```
