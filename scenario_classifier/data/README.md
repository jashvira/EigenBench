# Scenario Classifier Local Data

This folder is the local, ignored data home for Project 1 scenario-classifier work.

Expected structure:

- `scenarios/`: local scenario JSON files; current labelled work uses AIRisk.
- `corpora/retrieval_units/`: raw-criteria retrieval corpora plus optional criteria-card files.
- `private_anchors/`: optional ignored local anchor JSON files included by the raw-criteria builder.
- `valuearena/processed/`: processed ValueArena tie-rate tables.
- `labels/constitution_matches/`: AIRisk scenario-to-constitution review labels and batch inputs.
- `outputs/embedding_retrieval/`: embedding vectors, manifests, and retrieval rows.
- `outputs/bm25_retrieval/`: BM25 retrieval rows and summaries.
- `outputs/airiskdilemmas_623_stats/`: AIRisk subset statistics.

Payload files are intentionally ignored. Keep source/code here, not private datasets or generated caches.
