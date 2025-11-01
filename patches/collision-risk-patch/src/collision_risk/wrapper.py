"""Gym/Gymnasium wrapper that injects collision risk predictions."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .model import CollisionRiskModel
from .trajectory import TrajectoryWindow

try:  # pragma: no cover - soft dependency for Gymnasium
    import gymnasium as _gymnasium
except Exception:  # pragma: no cover - gymnasium might be unavailable
    _gymnasium = None

try:  # pragma: no cover - support legacy Gym environments
    import gym as _gym
except Exception:  # pragma: no cover - gym might be unavailable
    _gym = None


class _CollisionWrapperMixin:
    """Shared logic for risk-aware observation augmentation."""

    model: CollisionRiskModel
    _risk_window: TrajectoryWindow

    def _configure_observation_space(self) -> None:
        space = getattr(self.env, "observation_space", None)
        if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("Environment must define a Box observation space with low/high bounds")

        low = np.concatenate([np.asarray(space.low, dtype=np.float32), [self.model.clip_min]])
        high = np.concatenate([np.asarray(space.high, dtype=np.float32), [self.model.clip_max]])

        # Recreate a Box using the same class as the wrapped environment to keep dtype semantics.
        self.observation_space = space.__class__(low=low, high=high, dtype=np.float32)
        self._risk_window = self.model.make_feature_window()
        self._last_action: Optional[np.ndarray] = None

    def _augment_observation(
        self,
        observation: np.ndarray,
        *,
        action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        feature = self.model.build_feature_vector(self._risk_window, observation, action)
        risk = (
            self.model.predict_risk(feature)
            if feature.size
            else self.model._clip(self.model.default_risk)  # type: ignore[attr-defined]
        )
        return np.concatenate([np.asarray(observation, dtype=np.float32), [risk]], dtype=np.float32)

    # The methods below intentionally avoid super() calls so they work for both Gym and Gymnasium.
    def observation(self, observation: np.ndarray) -> np.ndarray:
        action = self._last_action if self.model.include_actions else None
        return self._augment_observation(observation, action=action)

    def reset(self, **kwargs: Any):  # type: ignore[override]
        self._risk_window.reset()
        self._last_action = None
        observation = self.env.reset(**kwargs)
        if isinstance(observation, tuple) and len(observation) == 2:
            obs, info = observation
            return self.observation(obs), info
        return self.observation(observation)

    def step(self, action: np.ndarray):  # type: ignore[override]
        outcome = self.env.step(action)
        if isinstance(outcome, tuple):
            if len(outcome) == 5:  # Gymnasium API
                obs, reward, terminated, truncated, info = outcome
                augmented = self._augment_observation(obs, action=action if self.model.include_actions else None)
                if terminated or truncated:
                    self._risk_window.reset()
                    self._last_action = None
                else:
                    self._last_action = action if self.model.include_actions else None
                return augmented, reward, terminated, truncated, info
            if len(outcome) == 4:  # Legacy Gym API
                obs, reward, done, info = outcome
                augmented = self._augment_observation(obs, action=action if self.model.include_actions else None)
                if done:
                    self._risk_window.reset()
                    self._last_action = None
                else:
                    self._last_action = action if self.model.include_actions else None
                return augmented, reward, done, info
        raise TypeError("Unexpected environment step return signature")


if _gymnasium is not None:  # pragma: no cover - environment-dependent definitions

    class _GymnasiumCollisionRiskObservationWrapper(_CollisionWrapperMixin, _gymnasium.ObservationWrapper):  # type: ignore[misc]
        """Wrapper implementation for Gymnasium environments."""

        def __init__(self, env: "_gymnasium.Env", *, model: CollisionRiskModel) -> None:
            _gymnasium.ObservationWrapper.__init__(self, env)
            self.model = model
            self._configure_observation_space()


else:  # pragma: no cover - Gymnasium not installed
    _GymnasiumCollisionRiskObservationWrapper = None  # type: ignore[assignment]


if _gym is not None:  # pragma: no cover - environment-dependent definitions

    class _GymCollisionRiskObservationWrapper(_CollisionWrapperMixin, _gym.ObservationWrapper):  # type: ignore[misc]
        """Wrapper implementation for classic Gym environments."""

        def __init__(self, env: "_gym.Env", *, model: CollisionRiskModel) -> None:
            _gym.ObservationWrapper.__init__(self, env)
            self.model = model
            self._configure_observation_space()


else:  # pragma: no cover - Gym not installed
    _GymCollisionRiskObservationWrapper = None  # type: ignore[assignment]


class _GenericCollisionRiskObservationWrapper(_CollisionWrapperMixin):
    """Fallback wrapper that relies on duck typing instead of Gym base classes."""

    def __init__(self, env, *, model: CollisionRiskModel) -> None:
        self.env = env
        self.model = model
        # Propagate common attributes that downstream consumers expect on gym-like envs.
        for attr in ("action_space", "reward_range", "metadata", "spec"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))
        self._configure_observation_space()

    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped environment for methods such
        # as render/close/etc. Using __getattr__ keeps the wrapper minimal while
        # still behaving like the original environment.
        return getattr(self.env, name)


class CollisionRiskObservationWrapper:
    """Factory wrapper that supports Gym, Gymnasium, and generic envs."""

    def __new__(cls, env, *, model: CollisionRiskModel):  # noqa: D401 - behaviour described above
        if _gymnasium is not None and isinstance(env, _gymnasium.Env):
            return _GymnasiumCollisionRiskObservationWrapper(env, model=model)
        if _gym is not None and isinstance(env, _gym.Env):
            return _GymCollisionRiskObservationWrapper(env, model=model)

        # Fallback to a generic duck-typed wrapper when the environment does not
        # inherit from gymnasium/gym base classes (or when those optional dependencies
        # are unavailable). This keeps the risk augmentation functional for custom
        # environments such as PenSimEnvGym.
        return _GenericCollisionRiskObservationWrapper(env, model=model)


__all__ = ["CollisionRiskObservationWrapper"]
