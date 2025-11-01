"""Collision risk model implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
from numpy.typing import ArrayLike
import logging

try:  # pragma: no cover - optional dependency guard
    from sklearn.gaussian_process import GaussianProcessRegressor
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency guard
    import torch
    import gpytorch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    gpytorch = None  # type: ignore[assignment]

from .gpytorch_backend import GPyTorchModelBundle, ensure_gpytorch_available, load_model_bundle
from .trajectory import TrajectoryWindow


LOGGER = logging.getLogger(__name__)


@dataclass
class CollisionRiskModel:
    """Abstract interface for collision risk estimators."""

    default_risk: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    history: int = 1
    include_actions: bool = False
    feature_dim: Optional[int] = None
    metadata: Optional[Dict[str, object]] = None
    _warned_dim_mismatch: bool = field(default=False, init=False, repr=False)

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

    def make_feature_window(self) -> TrajectoryWindow:
        return TrajectoryWindow(window_size=max(1, int(self.history)), include_actions=self.include_actions)

    def build_feature_vector(
        self,
        window: TrajectoryWindow,
        observation: ArrayLike,
        action: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        action_array = action if (self.include_actions and action is not None) else None
        window.append(observation, action_array)
        return window.as_feature_vector()

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
    backend: str = "sklearn"
    gpytorch_bundle: Optional[GPyTorchModelBundle] = None
    gpytorch_model: Optional["gpytorch.models.ExactGP"] = None
    gpytorch_likelihood: Optional["gpytorch.likelihoods.GaussianLikelihood"] = None
    feature_scaler: Optional[object] = None
    device: str = "cpu"

    def predict_risk(self, features: ArrayLike) -> float:
        feature_vector = np.asarray(features, dtype=float).reshape(1, -1)
        expected = self.feature_dim
        if expected is not None and feature_vector.shape[1] != expected:
            if not self._warned_dim_mismatch:
                LOGGER.warning(
                    "Collision risk feature dimension mismatch: expected %s, received %s. Returning default risk.",
                    expected,
                    feature_vector.shape[1],
                )
                self._warned_dim_mismatch = True
            return self._clip(self.default_risk)
        scaler = self.feature_scaler
        if scaler is None and self.backend == "sklearn" and self.gp is not None:
            scaler = getattr(self.gp, "feature_scaler_", None)
        if scaler is not None:
            try:
                feature_vector = scaler.transform(feature_vector)
            except Exception:
                return self._clip(self.default_risk)

        if self.backend == "gpytorch":
            if self.gpytorch_model is None or self.gpytorch_likelihood is None:
                return self._clip(self.default_risk)
            try:
                ensure_gpytorch_available()
                target_device = self.device
                if target_device != "cpu" and not torch.cuda.is_available():
                    target_device = "cpu"
                device = torch.device(target_device)
                model = self.gpytorch_model.to(device)
                likelihood = self.gpytorch_likelihood.to(device)
                self.gpytorch_model = model
                self.gpytorch_likelihood = likelihood
                self.device = target_device
                x_tensor = torch.as_tensor(feature_vector, dtype=torch.float32, device=device)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = likelihood(model(x_tensor))
                prediction = float(posterior.mean.squeeze().cpu().numpy())
            except Exception:
                return self._clip(self.default_risk)
            return self._clip(prediction)

        if self.gp is None:
            return self._clip(self.default_risk)
        try:
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
        clip_min, clip_max = clip
        metadata = state.get("metadata", {}) if isinstance(state, dict) else {}

        if isinstance(state, dict) and state.get("backend") == "gpytorch":
            bundle_obj = state.get("bundle")
            bundle = bundle_obj if isinstance(bundle_obj, GPyTorchModelBundle) else None
            scaler = state.get("feature_scaler")
            device = metadata.get("device", "cpu") if isinstance(metadata, dict) else "cpu"
            try:
                ensure_gpytorch_available()
                model, likelihood = load_model_bundle(bundle) if bundle is not None else (None, None)
            except Exception:
                model = None
                likelihood = None
            return cls(
                gp=None,
                backend="gpytorch",
                gpytorch_bundle=bundle,
                gpytorch_model=model,
                gpytorch_likelihood=likelihood,
                feature_scaler=scaler,
                device=device,
                default_risk=default_risk,
                clip_min=clip_min,
                clip_max=clip_max,
                history=int(metadata.get("history", 1) or 1),
                include_actions=bool(metadata.get("include_actions", False)),
                feature_dim=int(metadata.get("feature_dim", 0) or 0) or None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )

        gp = state.get("model") if isinstance(state, dict) else state
        scaler = None
        if isinstance(state, dict):
            scaler = getattr(gp, "feature_scaler_", None)
            if scaler is None:
                scaler = state.get("feature_scaler")
        return cls(
            gp=gp if GaussianProcessRegressor is not None else None,
            backend="sklearn",
            gpytorch_bundle=None,
            gpytorch_model=None,
            gpytorch_likelihood=None,
            feature_scaler=scaler,
            device="cpu",
            default_risk=default_risk,
            clip_min=clip_min,
            clip_max=clip_max,
            history=int(metadata.get("history", 1) or 1),
            include_actions=bool(metadata.get("include_actions", False)),
            feature_dim=int(metadata.get("feature_dim", 0) or 0) or None,
            metadata=metadata if isinstance(metadata, dict) else None,
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
