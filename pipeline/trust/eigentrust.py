"""Trust-matrix and EigenTrust utilities."""

from __future__ import annotations

import torch
from tqdm import tqdm


def compute_trust_matrix(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    S = U @ V.t()
    S = torch.exp(S)
    return S


def compute_trust_matrix_ties(model, device: str = "cpu"):
    """Build Davidson trust from a trained PyTorch model.

    This is the legacy model-facing wrapper. It extracts the fitted score
    vectors and tie parameters, then delegates to the representation-agnostic
    helper below.
    """
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    log_lambda = model.log_lambda.weight.data.to(device)

    return compute_trust_matrix_ties_from_logits(U @ V.t(), log_lambda)


def compute_trust_matrix_ties_from_logits(
    logits: torch.Tensor,
    log_lambda: torch.Tensor,
    *,
    logit_clip: float | None = None,
):
    """Build the Davidson trust matrix from fitted logits and tie strengths.

    This is the shared math used by both paths: the original BTD path passes
    values from a PyTorch model, while the numerical comparison passes arrays
    fitted by SciPy. They produce the same score matrix, tie contribution,
    row normalization, and EigenTrust input; only the fitting framework differs.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Davidson logits must be a square matrix")
    if log_lambda.numel() != logits.shape[0]:
        raise ValueError("Expected one Davidson tie parameter per judge")
    log_lambda = log_lambda.reshape(-1, 1)
    if logit_clip is not None:
        logits = torch.clamp(logits, -logit_clip, logit_clip)
        log_lambda = torch.clamp(log_lambda, -logit_clip, logit_clip)

    s = torch.exp(logits)
    lambda_i = torch.exp(log_lambda)

    sqrt_s = torch.sqrt(s)
    sqrt_s_sum = sqrt_s.sum(dim=1, keepdim=True)
    tie_terms = sqrt_s * (sqrt_s_sum - sqrt_s)
    tie_contribution = 0.5 * lambda_i * tie_terms

    S = s + tie_contribution
    Z_i = S.sum(dim=1, keepdim=True)
    T = S / Z_i
    return T


def row_normalize(S):
    row_sums = S.sum(dim=1, keepdim=True)
    C = S / row_sums
    return C


def damp_matrix(C, alpha: float = 0.0):
    M = C.size(0)
    E = torch.full_like(C, 1.0 / M)
    return (1 - alpha) * C + alpha * E


def eigentrust(
    C,
    alpha: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = True,
    raise_on_nonconvergence: bool = False,
):
    T = damp_matrix(C, alpha)
    t = torch.full(
        (T.size(0),),
        1.0 / T.size(0),
        device=T.device,
        dtype=T.dtype,
    )

    iterations = tqdm(range(max_iter)) if verbose else range(max_iter)
    for _ in iterations:
        t_next = t @ T
        if torch.norm(t_next - t, p=1) < tol:
            return t_next
        t = t_next

    if raise_on_nonconvergence:
        raise RuntimeError("EigenTrust did not converge")
    return t_next
