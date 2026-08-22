import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scenario_classifier.audits.embedding_retrieval import load_or_embed
from scenario_classifier.core.io import load_records
from scenario_classifier.core.ranking_metrics import (
    collapse_to_constitutions,
    recall_at_k,
)
from scenario_classifier.corpora.raw_criteria_units import embedding_text_from_criterion
from scenario_classifier.retrieval.embedding_texts import (
    format_raw_document,
    format_scenario_query,
)
from scenario_classifier.retrieval.retrievers import BM25Retriever, EmbeddingRetriever


class RecordLoadingTests(unittest.TestCase):
    def test_loads_string_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.json"
            records_path.write_text(json.dumps(["first", "second"]), encoding="utf-8")

            self.assertEqual(
                load_records(records_path),
                [
                    {"scenario_index": 0, "scenario": "first"},
                    {"scenario_index": 1, "scenario": "second"},
                ],
            )

    def test_maps_configurable_record_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "id": 7,
                        "prompt": "scenario",
                        "label": "kindness",
                        "certainty": "high",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_records(
                    records_path,
                    scenario_field="prompt",
                    index_field="id",
                    label_field="label",
                    confidence_field="certainty",
                ),
                [
                    {
                        "scenario_index": 7,
                        "scenario": "scenario",
                        "annotation_primary": "kindness",
                        "annotation_confidence": "high",
                    }
                ],
            )


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            {
                "criterion_id": "kindness.1",
                "constitution": "kindness",
                "embedding_text": "show compassion and goodwill",
            },
            {
                "criterion_id": "honesty.1",
                "constitution": "honesty",
                "embedding_text": "avoid deception and tell the truth",
            },
        ]

    def test_bm25_ranks_lexical_match_first(self) -> None:
        result = BM25Retriever(self.docs).retrieve("Please tell the truth", top_k=1)

        self.assertEqual(result[0]["criterion_id"], "honesty.1")

    def test_embedding_retriever_uses_cosine_similarity(self) -> None:
        retriever = EmbeddingRetriever(
            self.docs,
            np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        )

        result = retriever.retrieve_many(np.array([[0.0, 3.0]], dtype=np.float32), top_k=1)

        self.assertEqual(result[0][0]["criterion_id"], "honesty.1")

    def test_embedding_cache_is_reused_for_unchanged_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch(
                "scenario_classifier.audits.embedding_retrieval.embed_batch",
                return_value=[[1.0, 0.0], [0.0, 1.0]],
            ) as embed_batch:
                first = load_or_embed(
                    model="test/model",
                    corpus_name="examples",
                    texts=["first", "second"],
                    api_key="unused",
                    batch_size=8,
                    force=False,
                    output_dir=output_dir,
                )
                second = load_or_embed(
                    model="test/model",
                    corpus_name="examples",
                    texts=["first", "second"],
                    api_key="unused",
                    batch_size=8,
                    force=False,
                    output_dir=output_dir,
                )

            np.testing.assert_array_equal(first, second)
            self.assertEqual(embed_batch.call_count, 1)


class FormattingAndMetricTests(unittest.TestCase):
    def test_embedding_formats_are_asymmetric(self) -> None:
        scenario = format_scenario_query("google/gemini-embedding-2", "A user asks for help")
        criterion = format_raw_document(
            "google/gemini-embedding-2",
            {"embedding_text": "show compassion"},
        )

        self.assertIn("query:", scenario)
        self.assertIn("Scenario: A user asks for help", scenario)
        self.assertEqual(criterion, "title: none | text: show compassion")

    def test_criterion_cleanup_removes_evaluation_wrappers(self) -> None:
        text = "Criterion 4 for Kindness: prefer the response that demonstrates actual caring"

        self.assertEqual(embedding_text_from_criterion(text), "demonstrates actual caring")

    def test_collapsed_recall_deduplicates_constitutions(self) -> None:
        ranking = [
            {"constitution": "kindness"},
            {"constitution": "kindness"},
            {"constitution": "honesty"},
        ]
        rows = [{"annotation_primary": "honesty", "ranking": ranking}]

        self.assertEqual(collapse_to_constitutions(ranking), ["kindness", "honesty"])
        self.assertEqual(
            recall_at_k(rows, "ranking", [2], hit_mode="unit"),
            {"recall@2": 0.0},
        )
        self.assertEqual(
            recall_at_k(rows, "ranking", [2], hit_mode="collapsed"),
            {"recall@2": 1.0},
        )
