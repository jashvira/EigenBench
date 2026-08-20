"""Contract tests for the numerical round-robin run and aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from numerical_rating.analysis.data_loading.pairwise_judgments import (
    load_pairwise_data,
)
from numerical_rating.analysis.data_loading.pairwise_judgments import (
    reconcile_pass_comparisons as _reconcile_pass_comparisons,
)
from numerical_rating.analysis.experiments.scenario_uncertainty.resampling import (
    jackknife_interval,
)
from numerical_rating.analysis.experiments.scenario_uncertainty.resampling import (
    load_bootstrap_checkpoint as _load_bootstrap_checkpoint,
)
from numerical_rating.analysis.experiments.scenario_uncertainty.resampling import (
    save_bootstrap_checkpoint as _save_bootstrap_checkpoint,
)
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    PairwiseFit,
    fit_pairwise_btd,
    fit_pairwise_btd_multistart,
)
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    btd_loss_gradient as _btd_loss_gradient,
)
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    pairwise_trust as _pairwise_trust,
)
from numerical_rating.analysis.reports.round_robin_report import (
    bootstrap_intervals,
    comparison_intervals,
    compute_trust,
    published_scenarios,
    published_trust,
)
from numerical_rating.collection.data import (
    is_valid_response,
    load_config,
    load_response_cells,
)
from numerical_rating.collection.run_pointwise import extract_json_object

CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"


class RoundRobinDataTest(unittest.TestCase):
    def test_provider_failures_are_not_responses(self) -> None:
        self.assertFalse(is_valid_response(""))
        self.assertFalse(is_valid_response("Error in Gemini API call: 503 UNAVAILABLE"))
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

    def test_btd_gradient_matches_finite_difference(self) -> None:
        rows = np.asarray([[0, 1, 2, 1], [1, 2, 0, 0], [2, 0, 1, 2]], dtype=np.int64)
        parameters = np.random.default_rng(5).normal(size=15) * 0.1
        _, analytic = _btd_loss_gradient(parameters, rows, num_models=3, dimension=2)
        numerical = np.empty_like(analytic)
        step = 1e-6
        for index in range(len(parameters)):
            upper = parameters.copy()
            lower = parameters.copy()
            upper[index] += step
            lower[index] -= step
            upper_loss, _ = _btd_loss_gradient(upper, rows, num_models=3, dimension=2)
            lower_loss, _ = _btd_loss_gradient(lower, rows, num_models=3, dimension=2)
            numerical[index] = (upper_loss - lower_loss) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, atol=1e-7)

    def test_pairwise_trust_matches_published_aggregation(self) -> None:
        parameters = np.linspace(-0.3, 0.4, 15)

        trust = _pairwise_trust(parameters, num_models=3, dimension=2)

        np.testing.assert_allclose(
            trust,
            [0.33998720558859563, 0.3332748295185366, 0.3267379648928681],
            rtol=0.0,
            atol=1e-12,
        )

    def test_btd_fit_rejects_optimizer_failure(self) -> None:
        failed = SimpleNamespace(
            fun=1.0,
            x=np.zeros(40),
            success=False,
            nit=300,
            message="iteration limit",
        )
        with patch(
            "numerical_rating.analysis.utilities.optimization.minimize",
            return_value=failed,
        ):
            with self.assertRaisesRegex(RuntimeError, "iteration limit"):
                fit_pairwise_btd(
                    np.asarray([[0, 1, 2, 1]], dtype=np.int64),
                    initial=np.zeros(40),
                )

    def test_transpose_pair_is_reconciled_within_collection_pass(self) -> None:
        rows = [
            [0, 4, 3, 1, 2, 1],
            [0, 4, 3, 2, 1, 1],
        ]

        cleaned = _reconcile_pass_comparisons(rows)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual([row[-1] for row in cleaned], [0, 0])

    def test_corrected_pass_rejects_unidentified_repeats(self) -> None:
        rows = [
            [0, 4, 3, 1, 2, 1],
            [0, 4, 3, 2, 1, 2],
            [0, 4, 3, 1, 2, 1],
            [0, 4, 3, 2, 1, 2],
        ]

        with self.assertRaisesRegex(ValueError, "Ambiguous rows"):
            _reconcile_pass_comparisons(rows)

    def test_corrected_cleaning_exactly_deduplicates_raw_records(self) -> None:
        evaluation = {
            "judge response": "<criterion_1_choice>1</criterion_1_choice>",
            "eval1 response": "left",
            "eval2 response": "right",
            "eval1 reflection": "left reflection",
            "eval2 reflection": "right reflection",
            "constitution": "criterion",
            "scenario": "scenario",
            "scenario_index": 7,
            "judge": 0,
            "eval1": 1,
            "eval2": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_text(
                json.dumps(evaluation) + "\n" + json.dumps(evaluation) + "\n",
                encoding="utf-8",
            )
            data = load_pairwise_data(path, num_criteria=1, cleaning="corrected")

        self.assertEqual(data.diagnostics["exact_duplicates_removed"], 1)
        self.assertEqual(data.diagnostics["retained_criterion_rows"], 1)
        self.assertEqual(data.blocks[7].shape, (1, 4))

    def test_pairwise_cleaning_defaults_to_repeat_preserving(self) -> None:
        evaluation = {
            "judge response": "<criterion_1_choice>1</criterion_1_choice>",
            "eval1 response": "left",
            "eval2 response": "right",
            "eval1 reflection": "left reflection",
            "eval2 reflection": "right reflection",
            "constitution": "criterion",
            "scenario": "scenario",
            "scenario_index": 7,
            "judge": 0,
            "eval1": 1,
            "eval2": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
            data = load_pairwise_data(path, num_criteria=1)

        self.assertEqual(data.diagnostics["cleaning"], "corrected")

    def test_corrected_cleaning_preserves_distinct_collection_passes(self) -> None:
        def evaluation(
            first: int,
            second: int,
            *,
            suffix: str,
            choice: int,
        ) -> dict:
            return {
                "judge response": (
                    f"<criterion_1_choice>{choice}</criterion_1_choice>"
                ),
                "eval1 response": f"response-{first}-{suffix}",
                "eval2 response": f"response-{second}-{suffix}",
                "eval1 reflection": f"reflection-{first}-{suffix}",
                "eval2 reflection": f"reflection-{second}-{suffix}",
                "constitution": "criterion",
                "scenario": "scenario",
                "scenario_index": 7,
                "judge": 0,
                "eval1": first,
                "eval2": second,
            }

        records = []
        for suffix in ("first-pass", "second-pass"):
            records.extend(
                (
                    evaluation(1, 2, suffix=suffix, choice=1),
                    evaluation(2, 1, suffix=suffix, choice=2),
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            data = load_pairwise_data(path, num_criteria=1, cleaning="corrected")

        self.assertEqual(data.diagnostics["identified_passes"], 2)
        self.assertEqual(data.diagnostics["retained_criterion_rows"], 4)
        self.assertEqual(data.blocks[7].shape, (4, 4))
        self.assertEqual(data.blocks[7][:, -1].tolist(), [1, 2, 1, 2])

    def test_multistart_chooses_lowest_loss_stable_fit(self) -> None:
        first = PairwiseFit(np.zeros(40), np.full(8, 0.125), 1.1, 4)
        second = PairwiseFit(np.ones(40), np.full(8, 0.125), 1.0, 5)
        with patch(
            "numerical_rating.analysis.model_fitting.pairwise_btd.fit_pairwise_btd",
            side_effect=[first, second],
        ):
            fit = fit_pairwise_btd_multistart(
                np.asarray([[0, 1, 2, 1]], dtype=np.int64),
                initials=[np.zeros(40), np.ones(40)],
            )

        self.assertEqual(fit.loss, 1.0)
        self.assertEqual(fit.best_start, 1)
        self.assertEqual(fit.successful_starts, 2)

    def test_multistart_rejects_disagreeing_near_optima(self) -> None:
        trust_a = np.full(8, 0.125)
        trust_b = trust_a.copy()
        trust_b[:2] += [0.01, -0.01]
        first = PairwiseFit(np.zeros(40), trust_a, 1.0, 4)
        second = PairwiseFit(np.ones(40), trust_b, 1.0 + 1e-9, 5)
        with patch(
            "numerical_rating.analysis.model_fitting.pairwise_btd.fit_pairwise_btd",
            side_effect=[first, second],
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible trust"):
                fit_pairwise_btd_multistart(
                    np.asarray([[0, 1, 2, 1]], dtype=np.int64),
                    initials=[np.zeros(40), np.ones(40)],
                )

    def test_multistart_can_flag_disagreement_without_dropping_draw(self) -> None:
        trust_a = np.full(8, 0.125)
        trust_b = trust_a.copy()
        trust_b[:2] += [0.01, -0.01]
        first = PairwiseFit(np.zeros(40), trust_a, 1.0, 4)
        second = PairwiseFit(np.ones(40), trust_b, 1.0 + 1e-9, 5)
        with patch(
            "numerical_rating.analysis.model_fitting.pairwise_btd.fit_pairwise_btd",
            side_effect=[first, second],
        ):
            fit = fit_pairwise_btd_multistart(
                np.asarray([[0, 1, 2, 1]], dtype=np.int64),
                initials=[np.zeros(40), np.ones(40)],
                reject_unstable=False,
            )

        self.assertFalse(fit.stable_near_optima)
        self.assertAlmostEqual(fit.max_near_optimal_trust_l1, 0.02)

    def test_bootstrap_checkpoint_is_bound_to_input_fingerprint(self) -> None:
        numerical = np.full((2, 8), np.nan)
        pairwise = np.full((2, 8), np.nan)
        numerical[0] = 0.125
        pairwise[0] = 0.125
        completed = np.asarray([True, False])
        diagnostics = [{"draw": 0}, None]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _save_bootstrap_checkpoint(
                output,
                numerical_draws=numerical,
                pairwise_draws=pairwise,
                completed=completed,
                diagnostics=diagnostics,
                seed=42,
                input_fingerprint="expected",
            )
            loaded = _load_bootstrap_checkpoint(
                output,
                bootstrap_samples=2,
                seed=42,
                input_fingerprint="expected",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_bootstrap_checkpoint(
                    output,
                    bootstrap_samples=2,
                    seed=42,
                    input_fingerprint="changed",
                )

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[2], completed)

    def test_unequal_delete_group_jackknife_uses_group_sizes(self) -> None:
        point = np.asarray([0.5])
        leave_out = np.asarray([[0.4], [0.6]])

        estimate, standard_error, lower, upper = jackknife_interval(
            point,
            leave_out,
            omitted_sizes=np.asarray([1, 2]),
            total_size=3,
        )

        # h=(3, 1.5), so pseudovalues are 0.7 and 0.45.
        np.testing.assert_allclose(estimate, [0.5333333333333333])
        np.testing.assert_allclose(standard_error, [0.1178511301977579])
        np.testing.assert_allclose(lower, estimate - 1.96 * standard_error)
        np.testing.assert_allclose(upper, estimate + 1.96 * standard_error)


if __name__ == "__main__":
    unittest.main()
