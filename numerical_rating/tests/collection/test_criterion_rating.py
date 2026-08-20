"""Contract tests for criterion-wise numerical ratings."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from inspect_ai.scorer import Score

from numerical_rating.analysis.reports.criterion_report import analyze
from numerical_rating.analysis.reports.round_robin_report import (
    Rating,
    eval_log_path,
    load_ratings,
    rating_tensor,
)
from numerical_rating.collection.data import (
    load_config,
    load_constitution_criteria,
    load_constitution_text,
    load_response_cells,
)
from numerical_rating.collection.run_criteria import (
    criterion_scores,
    parse_criterion_scores,
    parse_one_missing_score,
    pointwise_criterion_rating,
    repair_one_missing_score,
)

CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"


class CriterionRatingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)
        cls.criteria = load_constitution_criteria(cls.config.constitution)

    def valid_completion(self) -> str:
        return json.dumps(
            {
                "scores": [
                    {
                        "criterion_id": criterion.criterion_id,
                        "score": index + 1,
                        "rationale": f"Reason {index + 1}",
                    }
                    for index, criterion in enumerate(self.criteria)
                ]
            }
        )

    def test_criteria_have_stable_ids_and_text_hashes(self) -> None:
        self.assertEqual(
            [criterion.criterion_id for criterion in self.criteria],
            [f"criterion_{index:02d}" for index in range(1, 9)],
        )
        self.assertTrue(
            all(len(criterion.text_hash) == 16 for criterion in self.criteria)
        )

    def test_criteria_require_explicit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constitution.json"
            path.write_text('["Prefer kindness."]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no explicit criterion number"):
                load_constitution_criteria(path)

    def test_holistic_text_accepts_unnumbered_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constitution.json"
            path.write_text('["Prefer kindness."]\n', encoding="utf-8")
            self.assertEqual(load_constitution_text(path), "Prefer kindness.")

    def test_local_eval_uri_becomes_a_path(self) -> None:
        info = SimpleNamespace(name="file:///tmp/rating%20log.eval")
        self.assertEqual(eval_log_path(info), Path("/tmp/rating log.eval"))

    def test_parser_returns_all_eight_scores(self) -> None:
        values, rationales = parse_criterion_scores(
            self.valid_completion(),
            criteria=self.criteria,
            score_min=1,
            score_max=10,
        )

        self.assertEqual(
            list(values),
            [criterion.criterion_id for criterion in self.criteria],
        )
        self.assertEqual(values["criterion_08"], 8.0)
        self.assertEqual(rationales["criterion_01"], "Reason 1")

    def test_parser_normalizes_reordered_criteria(self) -> None:
        payload = json.loads(self.valid_completion())
        payload["scores"] = list(reversed(payload["scores"]))

        values, _ = parse_criterion_scores(
            json.dumps(payload),
            criteria=self.criteria,
            score_min=1,
            score_max=10,
        )

        self.assertEqual(
            list(values),
            [criterion.criterion_id for criterion in self.criteria],
        )

    def test_parser_rejects_missing_or_duplicate_criteria(self) -> None:
        for mutate in (
            lambda scores: scores.pop(),
            lambda scores: scores.__setitem__(1, scores[0]),
        ):
            with self.subTest(mutate=mutate):
                payload = json.loads(self.valid_completion())
                mutate(payload["scores"])
                with self.assertRaisesRegex(
                    ValueError, "exactly once|duplicate criterion ID"
                ):
                    parse_criterion_scores(
                        json.dumps(payload),
                        criteria=self.criteria,
                        score_min=1,
                        score_max=10,
                    )

    def test_parser_normalizes_numeric_strings(self) -> None:
        payload = json.loads(self.valid_completion())
        payload["scores"][0]["score"] = "7"

        values, _ = parse_criterion_scores(
            json.dumps(payload),
            criteria=self.criteria,
            score_min=1,
            score_max=10,
        )

        self.assertEqual(values["criterion_01"], 7.0)

    def test_parser_rejects_non_numeric_scores(self) -> None:
        for invalid in (True, "high"):
            with self.subTest(invalid=invalid):
                payload = json.loads(self.valid_completion())
                payload["scores"][0]["score"] = invalid
                with self.assertRaisesRegex(ValueError, "must be a number"):
                    parse_criterion_scores(
                        json.dumps(payload),
                        criteria=self.criteria,
                        score_min=1,
                        score_max=10,
                    )

    def test_one_missing_score_parser_preserves_existing_judgment(self) -> None:
        payload = json.loads(self.valid_completion())
        del payload["scores"][6]["score"]

        values, rationales, missing = parse_one_missing_score(
            json.dumps(payload),
            criteria=self.criteria,
            score_min=1,
            score_max=10,
        )

        self.assertEqual(missing, "criterion_07")
        self.assertNotIn(missing, values)
        self.assertEqual(values["criterion_06"], 6)
        self.assertEqual(rationales[missing], "Reason 7")

    def test_missing_score_repair_uses_same_judge_followup(self) -> None:
        model = SimpleNamespace()

        async def generate(messages, config):
            self.assertIn("criterion_07", messages[-1].content)
            self.assertEqual(config.temperature, 0)
            return SimpleNamespace(completion='{"score": 7}')

        model.generate = generate
        state = SimpleNamespace(messages=[])
        with patch(
            "numerical_rating.collection.run_criteria.get_model",
            return_value=model,
        ):
            value = asyncio.run(
                repair_one_missing_score(
                    state,
                    criterion_id="criterion_07",
                    score_min=1,
                    score_max=10,
                )
            )

        self.assertEqual(value, 7)

    def test_scorer_repairs_exactly_one_missing_score(self) -> None:
        payload = json.loads(self.valid_completion())
        del payload["scores"][6]["score"]
        model = SimpleNamespace()

        async def generate(messages, config):
            return SimpleNamespace(completion='{"score": 7}')

        model.generate = generate
        state = SimpleNamespace(
            messages=[],
            output=SimpleNamespace(completion=json.dumps(payload)),
        )
        score = criterion_scores(criteria=self.criteria)
        with patch(
            "numerical_rating.collection.run_criteria.get_model",
            return_value=model,
        ):
            result = asyncio.run(score(state, SimpleNamespace()))

        self.assertEqual(result.value["criterion_07"], 7)
        self.assertEqual(
            result.metadata,
            {
                "missing_score_repaired": "criterion_07",
                "repair_method": "same_judge_followup",
            },
        )

    def test_task_reuses_cached_response_and_records_provenance(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}):
            task = pointwise_criterion_rating(
                config=CONFIG,
                limit=1,
                judge_model=self.config.judges[0].model,
            )
        sample = task.dataset[0]
        prompt = sample.input[1].content

        self.assertEqual(task.name, "pointwise_criterion_rating")
        self.assertNotEqual(task.version, 0)
        self.assertEqual(task.metadata["rating_mode"], "criterion_wise")
        self.assertEqual(
            task.metadata["generation_max_tokens"],
            self.config.criterion_generation_max_tokens,
        )
        self.assertEqual(sample.metadata["criterion_count"], 8)
        self.assertNotIn("criteria", sample.metadata)
        self.assertEqual(len(task.metadata["criteria"]), 8)
        self.assertEqual(
            task.metadata["response_schema"]["name"],
            "criterion_scores",
        )
        self.assertTrue(task.metadata["response_schema"]["strict"])
        self.assertIsInstance(task.metadata["response_schema"]["description"], str)
        self.assertIn("criterion_01: Criterion 1 for Kindness", prompt)
        self.assertIn(sample.metadata["scenario"], prompt)
        self.assertEqual(len(task.dataset), 1)

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}):
            smaller_budget = pointwise_criterion_rating(
                config=CONFIG,
                limit=1,
                judge_model=self.config.judges[0].model,
                generation_max_tokens=2_048,
            )
        self.assertNotEqual(task.version, smaller_budget.version)

    def test_task_rejects_unsafe_limits(self) -> None:
        for limit in (-1, 0, 8_001):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit"):
                    pointwise_criterion_rating(
                        config=CONFIG,
                        limit=limit,
                        judge_model=self.config.judges[0].model,
                    )

    def test_shared_loader_expands_criterion_scores(self) -> None:
        cell = load_response_cells(self.config)[0]
        judge = self.config.judges[0]
        values, rationales = parse_criterion_scores(
            self.valid_completion(),
            criteria=self.criteria,
            score_min=1,
            score_max=10,
        )
        sample = SimpleNamespace(
            id="sample",
            scores={
                "criterion_scores": Score(
                    value=values,
                    explanation=json.dumps(rationales),
                )
            },
            metadata={
                "judge_id": 0,
                "judge_name": judge.name,
                "judge_model": judge.model,
                "scenario_index": cell.scenario_index,
                "model_id": cell.model_id,
                "model_name": cell.model_name,
                "response_hash": cell.response_hash,
                "criterion_ids": [
                    criterion.criterion_id for criterion in self.criteria
                ],
            },
        )
        with patch(
            "numerical_rating.analysis.reports.round_robin_report.read_eval_log_samples",
            return_value=[sample],
        ):
            ratings = load_ratings(
                [SimpleNamespace(name="test.eval")],
                self.config,
                scorer_name="criterion_scores",
                dimensions=self.criteria,
            )

        self.assertEqual(len(ratings), 8)
        self.assertEqual(
            [rating.dimension_id for rating in ratings],
            [criterion.criterion_id for criterion in self.criteria],
        )

    def test_analysis_keeps_criteria_separate(self) -> None:
        ratings = (
            np.random.default_rng(7)
            .integers(
                1,
                11,
                size=(8, 8, 5, 8),
            )
            .astype(float)
        )

        result = analyze(
            ratings,
            config=self.config,
            scenario_ids=list(range(5)),
        )

        self.assertEqual(result["criterion_count"], 8)
        self.assertEqual(len(result["criteria"]), 8)
        self.assertEqual(
            result["aggregation"],
            "independent EigenTrust fit per criterion",
        )
        for criterion in result["criteria"]:
            self.assertAlmostEqual(sum(criterion["trust"].values()), 1.0)

    def test_shared_tensor_builder_rejects_negative_indices(self) -> None:
        rating = Rating(
            judge_id=-1,
            judge_name="judge",
            judge_model="judge-model",
            scenario_index=0,
            model_id=0,
            model_name="target",
            score=5,
            rationale="reason",
            response_hash="hash",
            dimension_id="criterion_01",
            dimension_hash=self.criteria[0].text_hash,
        )
        with self.assertRaisesRegex(ValueError, "Invalid judge_id"):
            rating_tensor(
                [rating],
                self.config,
                dimension_id="criterion_01",
            )


if __name__ == "__main__":
    unittest.main()
