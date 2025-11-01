"""Collision risk model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
from numpy.typing import ArrayLike

try:  # pragma: no cover - optional dependency guard
    from sklearn.gaussian_process import GaussianProcessRegressor
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[misc]


@dataclass
class CollisionRiskModel:
    """Abstract interface for collision risk estimators."""

    default_risk: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 1.0

    def predict_risk(self, features: ArrayLike) -> float:
        """Predict risk for a single feature vector."""

        raise NotImplementedError

    def batch_predict(self, features: ArrayLike) -> np.ndarray:
        """Vectorised risk prediction."""

        feature_array = np.asarray(features)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        results = np.array([self.predict_risk(row) for row in feature_array])
        return results

    def _clip(self, risk: float) -> float:
        return float(np.clip(risk, self.clip_min, self.clip_max))


@dataclass
class GaussianProcessCollisionModel(CollisionRiskModel):
    """Gaussian-process based collision risk model.

    Parameters
    ----------
    gp:
        Trained :class:`~sklearn.gaussian_process.GaussianProcessRegressor` instance.
    default_risk:
        Fallback risk value when the GP model is not available.
    clip_min / clip_max:
        Bounds applied to the predicted risk.
    """

    gp: Optional[GaussianProcessRegressor] = None

    def predict_risk(self, features: ArrayLike) -> float:
        feature_vector = np.asarray(features, dtype=float).reshape(1, -1)
        if self.gp is None:
            return self._clip(self.default_risk)
        try:
            scaler = getattr(self.gp, "feature_scaler_", None)
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            prediction = float(self.gp.predict(feature_vector, return_std=False)[0])
        except Exception:
            return self._clip(self.default_risk)
        return self._clip(prediction)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        default_risk: float = 0.0,
        clip: Sequence[float] = (0.0, 1.0),
    ) -> "GaussianProcessCollisionModel":
        """Load a GP model from a joblib checkpoint."""

        state = joblib.load(checkpoint)
        gp = state.get("model") if isinstance(state, dict) else state
        clip_min, clip_max = clip
        return cls(
            gp=gp if GaussianProcessRegressor is not None else None,
            default_risk=default_risk,
            clip_min=clip_min,
            clip_max=clip_max,
        )


def load_collision_model(
    checkpoint: Optional[str],
    *,
    default_risk: float = 0.0,
    clip: Sequence[float] = (0.0, 1.0),
) -> GaussianProcessCollisionModel:
    """Utility for loading a :class:`GaussianProcessCollisionModel`.

    When no checkpoint is provided, a model that always returns ``default_risk``
    is returned. This keeps the rest of the integration logic simple while
    allowing experiments to proceed even without a trained GP checkpoint.
    """

    if checkpoint is None:
        return GaussianProcessCollisionModel(
            gp=None,
            default_risk=default_risk,
            clip_min=clip[0],
            clip_max=clip[1],
        )
    return GaussianProcessCollisionModel.from_checkpoint(
        checkpoint, default_risk=default_risk, clip=clip
    )
