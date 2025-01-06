from typing import List, cast

import numpy as np

from ..dataset import Transition
from d3rlpy.iterators.base import TransitionIterator


class RoundIterator(TransitionIterator):

    _shuffle: bool
    _indices: np.ndarray
    _index: int

    def __init__(
        self,
        transitions: List[Transition],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
        shuffle: bool = True,
        time_length: int = 1,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._shuffle = shuffle
        self._indices = np.arange(len(self._transitions))
        self._index = 0
        self._time_length = time_length

    def _reset(self) -> None:
        self._indices = np.arange(len(self._transitions))
        if self._shuffle:
            self._indices = self._shuffle_in_chunks(self._indices, self._time_length)
        self._index = 0

    def _shuffle_in_chunks(self, indices, chunk_size):
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        np.random.shuffle(chunks)
        return np.concatenate(chunks)

    def _next(self) -> Transition:
        transition = self._transitions[cast(int, self._indices[self._index])]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions) // self._real_batch_size
