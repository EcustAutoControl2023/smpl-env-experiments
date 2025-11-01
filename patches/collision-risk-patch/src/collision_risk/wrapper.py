"""Gym/Gymnasium wrapper that injects collision risk predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import CollisionRiskModel

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

    def _configure_observation_space(self) -> None:
        space = getattr(self.env, "observation_space", None)
        if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("Environment must define a Box observation space with low/high bounds")

        low = np.concatenate([np.asarray(space.low, dtype=np.float32), [self.model.clip_min]])
        high = np.concatenate([np.asarray(space.high, dtype=np.float32), [self.model.clip_max]])

        # Recreate a Box using the same class as the wrapped environment to keep dtype semantics.
        self.observation_space = space.__class__(low=low, high=high, dtype=np.float32)

    # The methods below intentionally avoid super() calls so they work for both Gym and Gymnasium.
    def observation(self, observation: np.ndarray) -> np.ndarray:
        risk = self.model.predict_risk(observation)
        return np.concatenate([np.asarray(observation, dtype=np.float32), [risk]])

    def reset(self, **kwargs: Any):  # type: ignore[override]
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
                return self.observation(obs), reward, terminated, truncated, info
            if len(outcome) == 4:  # Legacy Gym API
                obs, reward, done, info = outcome
                return self.observation(obs), reward, done, info
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


class CollisionRiskObservationWrapper:
    """Factory wrapper that supports both Gymnasium and classic Gym envs."""

    def __new__(cls, env, *, model: CollisionRiskModel):  # noqa: D401 - behaviour described above
        if _gymnasium is not None and isinstance(env, _gymnasium.Env):
            return _GymnasiumCollisionRiskObservationWrapper(env, model=model)
        if _gym is not None and isinstance(env, _gym.Env):
            return _GymCollisionRiskObservationWrapper(env, model=model)

        available = [name for name, mod in {"gymnasium": _gymnasium, "gym": _gym}.items() if mod is not None]
        raise TypeError(
            "Unsupported environment type for collision risk wrapper. "
            f"Available integrations: {', '.join(available) if available else 'none'}."
        )


__all__ = ["CollisionRiskObservationWrapper"]
