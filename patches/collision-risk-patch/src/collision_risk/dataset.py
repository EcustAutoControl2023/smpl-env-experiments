"""Dataset augmentation utilities for collision risk."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .model import CollisionRiskModel


def augment_dataset_with_risk(
    observations: np.ndarray,
    *,
    model: CollisionRiskModel,
) -> np.ndarray:
    """Append risk predictions as an additional observation dimension.

    Parameters
    ----------
    observations:
        Array of shape ``(T, obs_dim)`` containing the base observations.
    model:
        Collision risk estimator used to produce risk predictions.

    Returns
    -------
    np.ndarray
        Array with shape ``(T, obs_dim + 1)`` where the last column holds the
        predicted risk values.
    """

    obs = np.asarray(observations, dtype=float)
    risks = model.batch_predict(obs).astype(float).reshape(-1)

    if risks.size != obs.shape[0]:
        raise ValueError(
            "Collision risk predictor returned a mismatched number of samples."
        )

    finite_mask = np.isfinite(risks)
    if not finite_mask.all():
        fallback = model._clip(model.default_risk)  # type: ignore[attr-defined]
        risks = np.where(finite_mask, risks, fallback)

    risk_min = float(np.min(risks)) if risks.size else model.clip_min
    risk_max = float(np.max(risks)) if risks.size else model.clip_max

    if not np.isfinite(risk_min) or not np.isfinite(risk_max):
        risk_min = risk_max = model._clip(model.default_risk)  # type: ignore[attr-defined]

    if risk_max - risk_min <= np.finfo(np.float32).eps:
        # When the risk signal is (almost) constant we add a tiny deterministic
        # jitter so downstream min-max scalers avoid zero division. The offsets
        # keep the signal within the configured clip range.
        clip_span = max(model.clip_max - model.clip_min, 1.0)
        if risks.size > 1:
            offsets = np.linspace(-0.5, 0.5, risks.size, dtype=np.float32)
            risks = risks + offsets * (clip_span * 1e-3)
        else:
            risks = np.full_like(risks, model._clip(model.default_risk))  # type: ignore[attr-defined]

    risks = np.clip(risks, model.clip_min, model.clip_max).reshape(-1, 1)
    return np.concatenate([obs, risks], axis=-1)
