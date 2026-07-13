"""Contract tests for the numerical round-robin run and aggregation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from numerical_rating.data import (
    ResponseCell,
    is_valid_response,
    load_config,
    load_response_cells,
)
from numerical_rating.run_pointwise import extract_json_object
from numerical_rating.round_robin_analysis import (
    Rating,
    bootstrap_intervals,
    comparison_intervals,
    compute_trust,
    parsed_rationale,
    published_scenarios,
    published_trust,
    reconcile_ratings,
)


CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"


class RoundRobinDataTest(unittest.TestCase):
    def test_rationale_accepts_dict_and_serialized_metadata(self) -> None:
        expected = "The response follows the constitution."
        self.assertEqual(parsed_rationale({"rationale": expected}, None), expected)
        self.assertEqual(
            parsed_rationale('{"rationale": "' + expected + '"}', None),
            expected,
        )

    def test_provider_failures_are_not_responses(self) -> None:
        self.assertFalse(is_valid_response(""))
        self.assertFalse(
            is_valid_response("Error in Gemini API call: 503 UNAVAILABLE")
        )
        self.assertFalse(is_valid_response("answer<|reserved_token_123|>"))
        self.assertTrue(is_valid_response("A substantive model response."))

    def test_non_finite_json_constants_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JSON constant"):
            extract_json_object('{"score": NaN, "rationale": "bad"}')

    def test_response_cache_is_complete_rectangle(self) -> None:
        config = load_config(CONFIG)
        cells = load_response_cells(config)

        self.assertEqual(len(config.judges), 8)
        self.assertEqual(
            [judge.target_model_id for judge in config.judges],
            list(range(8)),
        )
        self.assertEqual(config.judges[3].proxy_for, "Grok 4 (grok-4-0709)")
        self.assertEqual(len(cells), 8_000)
        self.assertEqual(len({cell.scenario_index for cell in cells}), 1_000)
        self.assertEqual(len({cell.model_id for cell in cells}), 8)
        self.assertTrue(all(cell.response.strip() for cell in cells))

        judge_limits = {judge.name: judge.max_tokens for judge in config.judges}
        self.assertEqual(judge_limits["Gemini 2.5 Pro"], 4_096)
        self.assertTrue(
            all(
                limit is None
                for name, limit in judge_limits.items()
                if name != "Gemini 2.5 Pro"
            )
        )

    def test_published_comparison_slice_has_871_ids(self) -> None:
        config = load_config(CONFIG)
        indices = published_scenarios(config)
        self.assertEqual(len(indices), 871)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(set(indices) <= set(range(1_000)))

    def test_published_trust_matches_configured_model_order(self) -> None:
        config = load_config(CONFIG)
        trust = published_trust(config)

        self.assertEqual(trust.shape, (8,))
        self.assertAlmostEqual(float(trust.sum()), 1.0)


class TrustMathTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.ratings = rng.integers(1, 11, size=(8, 50, 8)).astype(float)

    def test_primary_invariants(self) -> None:
        result = compute_trust(self.ratings)

        np.testing.assert_allclose(result.affinity.sum(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.trust_matrix.sum(axis=1), 1.0)
        self.assertAlmostEqual(float(result.trust.sum()), 1.0)
        self.assertLess(
            float(np.linalg.norm(result.trust @ result.trust_matrix - result.trust, 1)),
            1e-10,
        )

    def test_zero_diagonal_is_renormalized(self) -> None:
        result = compute_trust(self.ratings, zero_diagonal=True)

        np.testing.assert_allclose(np.diag(result.trust_matrix), 0.0)
        np.testing.assert_allclose(result.trust_matrix.sum(axis=1), 1.0)

    def test_self_score_exclusion_removes_self_score_influence(self) -> None:
        changed = self.ratings.copy()
        changed[0, :, 0] = 1.0
        base = compute_trust(self.ratings, zero_diagonal=True)
        modified = compute_trust(changed, zero_diagonal=True)

        np.testing.assert_allclose(modified.sigma[0], base.sigma[0])
        np.testing.assert_allclose(
            modified.trust_matrix[0],
            base.trust_matrix[0],
        )

    def test_flat_judge_fails_explicitly(self) -> None:
        flat = np.full((8, 10, 8), 5.0)
        with self.assertRaisesRegex(ValueError, "zero residual score variance"):
            compute_trust(flat)

    def test_target_permutation_preserves_result(self) -> None:
        permutation = np.array([2, 7, 1, 5, 0, 6, 4, 3])
        base = compute_trust(self.ratings)
        relabelled = self.ratings[permutation, :, :][:, :, permutation]
        permuted = compute_trust(relabelled)

        np.testing.assert_allclose(
            permuted.affinity,
            base.affinity[permutation, :][:, permutation],
        )
        np.testing.assert_allclose(
            permuted.trust_matrix,
            base.trust_matrix[permutation, :][:, permutation],
        )
        np.testing.assert_allclose(permuted.trust, base.trust[permutation])

    def test_bootstrap_comparison_intervals_are_finite(self) -> None:
        *_, draws = bootstrap_intervals(self.ratings, samples=10, seed=7)
        reference = compute_trust(self.ratings).trust
        intervals = comparison_intervals(draws, reference)

        self.assertEqual(set(intervals), {"l1", "spearman", "kendall"})
        for lower, upper in intervals.values():
            self.assertTrue(np.isfinite([lower, upper]).all())
            self.assertLessEqual(lower, upper)


class RepairOverlayTest(unittest.TestCase):
    def test_repair_replaces_only_a_stale_response_hash(self) -> None:
        config = SimpleNamespace(judges=(object(),), expected_response_cells=1)
        cell = ResponseCell(
            scenario_index=3,
            scenario="scenario",
            model_id=0,
            model_name="target",
            model_api_id="target-id",
            response="new response",
            response_hash="new-hash",
        )
        base = Rating(0, "judge", "judge-id", 3, 0, "target", 1, "old", "old-hash")
        repair = Rating(0, "judge", "judge-id", 3, 0, "target", 7, "new", "new-hash")

        with patch(
            "numerical_rating.round_robin_analysis.load_response_cells",
            return_value=[cell],
        ):
            ratings, count = reconcile_ratings([base], [repair], config)

        self.assertEqual(count, 1)
        self.assertEqual(ratings, [repair])

    def test_repair_with_wrong_hash_fails(self) -> None:
        config = SimpleNamespace(judges=(object(),), expected_response_cells=1)
        cell = ResponseCell(3, "scenario", 0, "target", "target-id", "new", "new-hash")
        base = Rating(0, "judge", "judge-id", 3, 0, "target", 1, "old", "old-hash")
        bad = Rating(0, "judge", "judge-id", 3, 0, "target", 7, "bad", "wrong")

        with patch(
            "numerical_rating.round_robin_analysis.load_response_cells",
            return_value=[cell],
        ):
            with self.assertRaisesRegex(ValueError, "does not match"):
                reconcile_ratings([base], [bad], config)


if __name__ == "__main__":
    unittest.main()
