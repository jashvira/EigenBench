"""Score-matrix and aligned-factor comparisons."""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    fit_rank_svd,
)


def geometry_metrics(direct: np.ndarray, pairwise: np.ndarray) -> dict[str, float]:
    """Compare row-centred score matrices and their pairwise margins."""
    direct = direct - direct.mean(axis=1, keepdims=True)
    pairwise = pairwise - pairwise.mean(axis=1, keepdims=True)
    direct_margins = []
    pairwise_margins = []
    for judge in range(direct.shape[0]):
        for left in range(direct.shape[1]):
            for right in range(left + 1, direct.shape[1]):
                direct_margins.append(direct[judge, left] - direct[judge, right])
                pairwise_margins.append(pairwise[judge, left] - pairwise[judge, right])
    direct_margins = np.asarray(direct_margins)
    pairwise_margins = np.asarray(pairwise_margins)
    slope = float(np.sum(direct * pairwise) / np.sum(np.square(direct)))
    return {
        "score_pearson": float(pearsonr(direct.ravel(), pairwise.ravel()).statistic),
        "score_spearman": float(spearmanr(direct.ravel(), pairwise.ravel()).statistic),
        "margin_pearson": float(pearsonr(direct_margins, pairwise_margins).statistic),
        "margin_spearman": float(spearmanr(direct_margins, pairwise_margins).statistic),
        "margin_sign_agreement": float(
            np.mean(np.sign(direct_margins) == np.sign(pairwise_margins))
        ),
        "best_scale_direct_to_pairwise": slope,
        "scaled_relative_frobenius_error": float(
            np.linalg.norm(slope * direct - pairwise) / np.linalg.norm(pairwise)
        ),
    }


def aligned_embedding_metrics(
    direct_scores: np.ndarray, pairwise_scores: np.ndarray, rank: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize score matrices, align factors, and compare distances."""
    direct = direct_scores - direct_scores.mean(axis=1, keepdims=True)
    pairwise = pairwise_scores - pairwise_scores.mean(axis=1, keepdims=True)
    direct /= np.linalg.norm(direct)
    pairwise /= np.linalg.norm(pairwise)
    direct_fit = fit_rank_svd(direct, rank)
    pairwise_fit = fit_rank_svd(pairwise, rank)
    direct_stack = np.vstack((direct_fit.judges, direct_fit.models))
    pairwise_stack = np.vstack((pairwise_fit.judges, pairwise_fit.models))
    rotation, _ = orthogonal_procrustes(direct_stack, pairwise_stack)
    aligned_judges = direct_fit.judges @ rotation
    aligned_models = direct_fit.models @ rotation
    aligned_stack = np.vstack((aligned_judges, aligned_models))
    difference = aligned_stack - pairwise_stack
    metrics = {
        "aligned_relative_error": float(
            np.linalg.norm(difference) / np.linalg.norm(pairwise_stack)
        ),
        "judge_distance_spearman": float(
            spearmanr(pdist(aligned_judges), pdist(pairwise_fit.judges)).statistic
        ),
        "model_distance_spearman": float(
            spearmanr(pdist(aligned_models), pdist(pairwise_fit.models)).statistic
        ),
    }
    return (
        metrics,
        aligned_judges,
        aligned_models,
        pairwise_fit.judges,
        pairwise_fit.models,
    )


def shared_model_metrics(
    reference_scores: np.ndarray, candidate_scores: np.ndarray, rank: int
) -> dict[str, float]:
    """Compare canonical model vectors from score matrices with different rows."""
    reference = reference_scores / np.linalg.norm(reference_scores)
    candidate = candidate_scores / np.linalg.norm(candidate_scores)
    reference_models = fit_rank_svd(reference, rank).models
    candidate_models = fit_rank_svd(candidate, rank).models
    rotation, _ = orthogonal_procrustes(candidate_models, reference_models)
    candidate_aligned = candidate_models @ rotation
    return {
        "aligned_relative_error": float(
            np.linalg.norm(candidate_aligned - reference_models)
            / np.linalg.norm(reference_models)
        ),
        "distance_spearman": float(
            spearmanr(pdist(candidate_aligned), pdist(reference_models)).statistic
        ),
    }
