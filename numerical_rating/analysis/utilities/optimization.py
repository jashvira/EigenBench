"""Shared deterministic optimizer call used by the fitted factor models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

Objective = Callable[..., tuple[float, np.ndarray]]


def minimize_lbfgs(
    objective: Objective,
    initial: np.ndarray,
    *,
    args: tuple,
    max_iterations: int,
    record_trace: bool,
) -> tuple[OptimizeResult, tuple[dict[str, float | int], ...]]:
    """Run the common full-batch L-BFGS configuration."""
    trace: list[dict[str, float | int]] = []

    def record(parameters: np.ndarray) -> None:
        loss, gradient = objective(parameters, *args)
        trace.append(
            {
                "iteration": len(trace),
                "loss": float(loss),
                "gradient_l2": float(np.linalg.norm(gradient)),
            }
        )

    callback = record if record_trace else None
    if record_trace:
        record(initial)
    result = minimize(
        objective,
        initial,
        args=args,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 30,
        },
    )
    if (
        record_trace
        and result.success
        and np.isfinite(result.fun)
        and (
            not trace
            or not np.isclose(
                trace[-1]["loss"],
                result.fun,
                rtol=1e-10,
                atol=1e-12,
            )
        )
    ):
        record(result.x)
    return result, tuple(trace)
