# Scenario Classifier Local Data

This folder is the local, ignored data home for Project 1 scenario-classifier work.

Expected structure:

- `scenarios/`: OASST, AIRisk, Reddit, and WildChat scenario JSON files.
- `corpora/criteria_cards/`: criteria-card and raw-criteria retrieval corpora.
- `private_anchors/`: optional ignored local anchor JSON files included by the raw-criteria builder.
- `valuearena/processed/`: processed ValueArena tie-rate tables.
- `labels/constitution_matches/`: AIRisk scenario-to-constitution review labels and batch inputs.
- `outputs/embedding_audit/`: embedding vectors, manifests, and retrieval rows.
- `outputs/router_audit/`: BM25 audit outputs.
- `outputs/airiskdilemmas_623_stats/`: AIRisk subset statistics.

Payload files are intentionally ignored. Keep source/code here, not private datasets or generated caches.
