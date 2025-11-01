"""Utilities for maintaining short trajectory histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from collections import deque


@dataclass
class TrajectoryWindow:
    """Fixed-size FIFO buffer storing recent observations and actions."""

    window_size: int
    include_actions: bool = False
    _observations: Deque[np.ndarray] = field(init=False, repr=False)
    _actions: Deque[np.ndarray] = field(init=False, repr=False)
    _obs_dim: Optional[int] = field(init=False, default=None, repr=False)
    _act_dim: Optional[int] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._observations = deque(maxlen=self.window_size)
        self._actions = deque(maxlen=self.window_size) if self.include_actions else deque()

    def reset(self) -> None:
        self._observations.clear()
        self._actions.clear()

    def append(self, observation: ArrayLike, action: Optional[ArrayLike] = None) -> None:
        obs_arr = np.asarray(observation, dtype=float).ravel()
        if self._obs_dim is None:
            self._obs_dim = obs_arr.size
        elif obs_arr.size != self._obs_dim:
            raise ValueError(
                f"Observation size {obs_arr.size} does not match expected {self._obs_dim}."
            )
        self._observations.append(obs_arr)

        if self.include_actions and action is not None:
            act_arr = np.asarray(action, dtype=float).ravel()
            if self._act_dim is None:
                self._act_dim = act_arr.size
            elif act_arr.size != self._act_dim:
                raise ValueError(
                    f"Action size {act_arr.size} does not match expected {self._act_dim}."
                )
            self._actions.append(act_arr)

    def as_feature_vector(self) -> np.ndarray:
        """Flatten buffered content into a fixed-size feature vector."""

        if self._obs_dim is None:
            return np.zeros(0, dtype=float)

        feature = np.zeros(self.feature_dim, dtype=float)

        obs_offset = self.window_size - len(self._observations)
        for idx, obs in enumerate(self._observations):
            start = (obs_offset + idx) * self._obs_dim
            feature[start : start + self._obs_dim] = obs

        if self.include_actions and self._act_dim is not None:
            base = self.window_size * self._obs_dim
            act_offset = self.window_size - len(self._actions)
            for idx, act in enumerate(self._actions):
                start = base + (act_offset + idx) * self._act_dim
                feature[start : start + self._act_dim] = act

        return feature

    @property
    def feature_dim(self) -> int:
        if self._obs_dim is None:
            return 0
        total_dim = self.window_size * self._obs_dim
        if self.include_actions and self._act_dim is not None:
            total_dim += self.window_size * self._act_dim
        return total_dim

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack(self._observations) if self._observations else np.empty((0,))
        if self.include_actions and self._actions:
            act = np.stack(self._actions)
        else:
            act = np.empty((0,))
        return obs, act
