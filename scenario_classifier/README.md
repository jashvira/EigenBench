# Scenario Classifier

Project 1 candidate-generation code for routing natural scenarios to relevant constitution criteria.

## Layout

- `core/`: shared repo paths, JSON loaders, and recall metrics.
- `retrieval/`: retrieval classes. BM25 is lexical; embeddings use cosine similarity over cached vectors.
- `audits/`: retrieval runner plus labelled comparison scripts.
- `labels/`: utilities for merging reviewed AIRisk constitution assignments.
- `corpora/`: scripts that build retrieval corpora, currently raw criteria by default.
- `data/`: ignored local datasets, corpora, labels, caches, and outputs for this task.

## Local Data

The runnable defaults look under `scenario_classifier/data/`:

- `scenarios/`: local scenario JSON files; current labelled work uses AIRisk.
- `corpora/retrieval_units/`: raw-criteria units plus optional criteria-card artifacts.
- `private_anchors/`: optional ignored local anchor JSON files included by the raw-criteria builder.
- `valuearena/processed/`: processed ValueArena tie-rate tables.
- `labels/constitution_matches/`: AIRisk review labels and batch annotations.
- `outputs/embedding_retrieval/`: embedding vectors, manifests, retrieval rows, and comparison tables.
- `outputs/bm25_retrieval/`: BM25 retrieval rows and summaries.

## Common Commands

Run BM25 retrieval on the AIRisk labelled rows:

```bash
python -m scenario_classifier.audits.bm25_audit \
  --records scenario_classifier/data/labels/constitution_matches/airisk_623_constitution_matches.jsonl \
  --output-name airisk_623 \
  --corpus raw
```

Run embedding retrieval on the AIRisk labelled rows:

```bash
python -m scenario_classifier.audits.embedding_retrieval \
  --records scenario_classifier/data/labels/constitution_matches/airisk_623_constitution_matches.jsonl \
  --output-name airisk_623 \
  --model google/gemini-embedding-2
```

Swap `--records` and `--output-name` to run either retriever on another JSON/JSONL scenario file.

Run both embedding models used in the comparison table:

```bash
python -m scenario_classifier.audits.embedding_retrieval \
  --records scenario_classifier/data/labels/constitution_matches/airisk_623_constitution_matches.jsonl \
  --output-name airisk_623 \
  --model google/gemini-embedding-2 \
  --model qwen/qwen3-embedding-8b
```

Use `--corpus cards` or `--corpus both` only for criteria-card comparisons.

Build a labelled retrieval comparison table:

```bash
python -m scenario_classifier.audits.retrieval_comparison \
  --annotations scenario_classifier/data/labels/constitution_matches/airisk_623_constitution_matches.jsonl \
  --output-name airisk_623 \
  --corpus both
```

Rebuild raw criteria units:

```bash
python -m scenario_classifier.corpora.raw_criteria_units
```
