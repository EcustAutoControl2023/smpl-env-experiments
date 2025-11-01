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
    risks = model.batch_predict(obs).reshape(-1, 1)
    return np.concatenate([obs, risks], axis=-1)
