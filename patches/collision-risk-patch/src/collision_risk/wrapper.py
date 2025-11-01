"""Gym wrapper that injects collision risk predictions into observations."""

from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - compatibility shim
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym
import numpy as np

from .model import CollisionRiskModel


class CollisionRiskObservationWrapper(gym.ObservationWrapper):
    """Augment environment observations with risk predictions."""

    def __init__(
        self,
        env: gym.Env,
        *,
        model: CollisionRiskModel,
    ) -> None:
        super().__init__(env)
        self.model = model
        assert isinstance(env.observation_space, gym.spaces.Box)
        low = np.concatenate([env.observation_space.low, [model.clip_min]])
        high = np.concatenate([env.observation_space.high, [model.clip_max]])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        risk = self.model.predict_risk(observation)
        return np.concatenate([np.asarray(observation, dtype=np.float32), [risk]])

    def reset(self, **kwargs: Any) -> Any:
        observation = self.env.reset(**kwargs)
        if isinstance(observation, tuple):
            obs, info = observation
            return self.observation(obs), info
        return self.observation(observation)

    def step(self, action: np.ndarray):
        outcome = self.env.step(action)
        obs, reward, terminated, truncated, info = outcome
        aug_obs = self.observation(obs)
        return aug_obs, reward, terminated, truncated, info
