# Scenario Classifier

Project 1 candidate-generation code for routing natural scenarios to relevant constitution criteria.

## Layout

- `core/`: shared repo paths, JSON loaders, and recall metrics.
- `retrieval/`: retrieval classes. BM25 is lexical; embeddings use cosine similarity over cached vectors.
- `audits/`: runnable analyses for OASST and AIRisk.
- `labels/`: utilities for merging reviewed AIRisk constitution assignments.
- `corpora/`: scripts that build retrieval corpora, such as the raw-criteria baseline.
- `data/`: ignored local datasets, corpora, labels, caches, and outputs for this task.

## Local Data

The runnable defaults look under `scenario_classifier/data/`:

- `scenarios/`: `oasst_questions.json`, `airiskdilemmas.json`, optional Reddit/WildChat samples.
- `corpora/criteria_cards/`: criteria cards plus raw-criteria units.
- `private_anchors/`: optional ignored local anchor JSON files included by the raw-criteria builder.
- `valuearena/processed/`: processed ValueArena tie-rate tables.
- `labels/constitution_matches/`: AIRisk review labels and batch annotations.
- `outputs/embedding_audit/`: embedding vectors, manifests, retrieval rows, and comparison tables.
- `outputs/router_audit/`: BM25 audit outputs.

## Common Commands

Run the OASST BM25 audit:

```bash
python -m scenario_classifier.audits.bm25_audit
```

Run OASST embedding retrieval:

```bash
python -m scenario_classifier.audits.embedding_audit --model google/gemini-embedding-2
```

Run AIRisk embedding retrieval:

```bash
python -m scenario_classifier.audits.airisk_embedding_retrieval --model google/gemini-embedding-2
```

Build the AIRisk retrieval comparison table:

```bash
python -m scenario_classifier.audits.airisk_retrieval_table
```

Rebuild raw criteria units:

```bash
python -m scenario_classifier.corpora.raw_criteria_units
```
