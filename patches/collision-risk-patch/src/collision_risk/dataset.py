"""Dataset utilities for collision risk modelling."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .model import CollisionRiskModel
from .trajectory import TrajectoryWindow

try:  # pragma: no cover - optional dependency guard
    from d3rlpy.dataset import MDPDataset
except Exception:  # pragma: no cover
    MDPDataset = None  # type: ignore[misc]


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset supporting the minimal fields required for GP tooling."""

    if path.suffix == ".npz":
        archive = np.load(path)
        observations = archive["observations"]
        actions = archive["actions"]
        terminals = archive.get("terminals", np.zeros(len(observations), dtype=bool))
        return observations, actions, terminals
    if path.suffix == ".pkl":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            observations = np.asarray(payload["observations"])
            actions = np.asarray(payload["actions"])
            terminal_key = next(
                (key for key in ("terminals", "dones", "timeouts") if key in payload),
                None,
            )
            if terminal_key is not None:
                terminals = np.asarray(payload[terminal_key])
            else:
                terminals = np.zeros(len(observations), dtype=bool)
            return observations, actions, terminals
        raise ValueError(
            "Pickle datasets must store a mapping with 'observations' and 'actions' arrays."
        )
    if path.suffix in {".h5", ".hdf5"}:
        if MDPDataset is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("d3rlpy is required to load HDF5 datasets. Install d3rlpy first.")
        dataset = MDPDataset.load(str(path))
        return (dataset.observations, dataset.actions, dataset.terminals)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def iter_episode_indices(terminals: np.ndarray) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, end)`` indices for each episode in the flat arrays."""

    start = 0
    for idx, terminal in enumerate(terminals):
        if terminal:
            yield start, idx + 1
            start = idx + 1
    if start < len(terminals):
        yield start, len(terminals)


def future_collision_targets(terminals: np.ndarray, horizon: int) -> np.ndarray:
    """Label each timestep with whether a collision occurs within ``horizon``."""

    targets = np.zeros_like(terminals, dtype=float)
    lookahead = 0
    for idx in range(len(terminals) - 1, -1, -1):
        if terminals[idx]:
            lookahead = horizon
            targets[idx] = 1.0
        else:
            lookahead = max(lookahead - 1, 0)
            targets[idx] = 1.0 if lookahead > 0 else 0.0
    return targets


def build_training_matrix(
    observations: np.ndarray,
    actions: np.ndarray,
    terminals: np.ndarray,
    *,
    history: int,
    include_actions: bool,
    horizon: int,
    max_samples: int | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Construct feature/target arrays for GP fitting and evaluation."""

    window = TrajectoryWindow(window_size=history, include_actions=include_actions)
    features: list[np.ndarray] = []
    targets: list[float] = []
    rng = np.random.default_rng(seed) if max_samples is not None else None
    total_samples = 0
    collision_targets = future_collision_targets(terminals.astype(bool), horizon)

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    progress = None
    if tqdm is not None:
        progress = tqdm(total=int(len(terminals)), desc="Preparing GP dataset", leave=False)

    try:
        for start, end in iter_episode_indices(terminals.astype(bool)):
            window.reset()
            for idx in range(start, end):
                if progress is not None:
                    progress.update(1)
                window.append(observations[idx], actions[idx] if include_actions else None)
                feature = window.as_feature_vector()
                if feature.size == 0:
                    continue
                total_samples += 1
                if max_samples is None or len(features) < max_samples:
                    features.append(feature)
                    targets.append(collision_targets[idx])
                else:
                    assert rng is not None
                    replacement_index = int(rng.integers(0, total_samples))
                    if replacement_index < max_samples:
                        features[replacement_index] = feature
                        targets[replacement_index] = collision_targets[idx]
    finally:
        if progress is not None:
            progress.close()

    if not features:
        raise RuntimeError("No training samples were generated from the dataset.")

    X = np.stack(features)
    y = np.asarray(targets, dtype=float)
    return X, y, total_samples, len(features)


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


def save_evaluation_summary(path: Path, payload: dict) -> None:
    """Persist a JSON summary of evaluation artefacts alongside the figures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
