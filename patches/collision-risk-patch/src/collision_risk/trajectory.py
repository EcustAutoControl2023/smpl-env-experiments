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

    def __post_init__(self) -> None:
        self._observations = deque(maxlen=self.window_size)
        self._actions = deque(maxlen=self.window_size) if self.include_actions else deque()

    def reset(self) -> None:
        self._observations.clear()
        self._actions.clear()

    def append(self, observation: ArrayLike, action: Optional[ArrayLike] = None) -> None:
        self._observations.append(np.asarray(observation, dtype=float))
        if self.include_actions and action is not None:
            self._actions.append(np.asarray(action, dtype=float))

    def as_feature_vector(self) -> np.ndarray:
        """Flatten buffered content into a feature vector."""

        if not self._observations:
            return np.zeros(self.feature_dim, dtype=float)
        obs_stack = np.concatenate(list(self._observations))
        if not self.include_actions or not self._actions:
            return obs_stack
        act_stack = np.concatenate(list(self._actions))
        return np.concatenate([obs_stack, act_stack])

    @property
    def feature_dim(self) -> int:
        obs_dim = self._observations[0].size if self._observations else 0
        act_dim = self._actions[0].size if self._actions else 0
        total_dim = obs_dim * len(self._observations)
        if self.include_actions:
            total_dim += act_dim * len(self._actions)
        return total_dim if total_dim > 0 else 0

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack(self._observations) if self._observations else np.empty((0,))
        if self.include_actions and self._actions:
            act = np.stack(self._actions)
        else:
            act = np.empty((0,))
        return obs, act
