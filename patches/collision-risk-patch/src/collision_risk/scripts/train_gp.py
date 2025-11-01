"""Command-line helpers for training collision risk GP models."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np

try:  # pragma: no cover - optional dependency guard
    from d3rlpy.dataset import MDPDataset
except Exception:  # pragma: no cover
    MDPDataset = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency guard
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[misc]
    ConstantKernel = RBF = WhiteKernel = None  # type: ignore[misc]
    StandardScaler = None  # type: ignore[misc]

from ..trajectory import TrajectoryWindow
from ..gpytorch_backend import (
    GPyTorchModelBundle,
    ensure_gpytorch_available,
    train_exact_gp,
)


def _load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset supporting the minimal fields required for training."""

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
            raise RuntimeError(
                "d3rlpy is required to load HDF5 datasets. Install d3rlpy first."
            )
        dataset = MDPDataset.load(str(path))
        return (dataset.observations, dataset.actions, dataset.terminals)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _iter_episode_indices(terminals: np.ndarray) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) indices for each episode in the flat arrays."""

    start = 0
    for idx, terminal in enumerate(terminals):
        if terminal:
            yield start, idx + 1
            start = idx + 1
    if start < len(terminals):
        yield start, len(terminals)


def _future_collision_targets(terminals: np.ndarray, horizon: int) -> np.ndarray:
    """Label each timestep with whether a collision occurs within the horizon."""

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
    """Construct feature/target arrays for GP fitting.

    Returns the stacked features/targets along with counters indicating the
    total number of candidate samples seen and the number retained after any
    subsampling.
    """

    window = TrajectoryWindow(window_size=history, include_actions=include_actions)
    features: list[np.ndarray] = []
    targets: list[float] = []
    rng = np.random.default_rng(seed) if max_samples is not None else None
    total_samples = 0
    collision_targets = _future_collision_targets(terminals.astype(bool), horizon)

    for start, end in _iter_episode_indices(terminals.astype(bool)):
        window.reset()
        for idx in range(start, end):
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

    if not features:
        raise RuntimeError("No training samples were generated from the dataset.")

    X = np.stack(features)
    y = np.asarray(targets, dtype=float)
    return X, y, total_samples, len(features)


def fit_gaussian_process(
    X: np.ndarray,
    y: np.ndarray,
    *,
    length_scale: float,
    noise_level: float,

) -> GaussianProcessRegressor:
    """Train a Gaussian-process regressor on the prepared dataset."""

    if GaussianProcessRegressor is None or StandardScaler is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required to train the collision GP model.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale, (1e-2, 1e2)) + WhiteKernel(
        noise_level=noise_level
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled, y)
    gp.feature_scaler_ = scaler  # type: ignore[attr-defined]
    return gp


def fit_gpytorch_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
) -> GPyTorchModelBundle:
    """Train an exact GP using GPyTorch with optional GPU acceleration."""

    ensure_gpytorch_available()
    return train_exact_gp(X, y, device=device, epochs=epochs, learning_rate=learning_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a collision risk GP model.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to an offline dataset (.npz, .pkl, or .h5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/collision_gp.joblib"),
        help="Where to save the trained GP checkpoint.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of past steps to include when forming the feature vector.",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include action history alongside observations in the feature vector.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="How many steps ahead to look when labelling collisions.",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=10.0,
        help="Initial length-scale for the RBF kernel.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=1e-3,
        help="Noise level for the WhiteKernel component.",
    )
    parser.add_argument(
        "--backend",
        choices=("sklearn", "gpytorch"),
        default="sklearn",
        help=(
            "Select the GP backend. Use 'gpytorch' to enable GPU-accelerated training "
            "(requires torch+gpytorch)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "Torch device to use when --backend=gpytorch. Defaults to 'cuda' so GPUs are "
            "used when available."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of optimisation epochs when training with the gpytorch backend.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for the Adam optimiser when using gpytorch.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help=(
            "Upper bound on the number of training samples retained via reservoir "
            "sampling. Set to 0 to keep all samples."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for subsampling when --max-samples is set.",
    )
    args = parser.parse_args()

    observations, actions, terminals = _load_dataset(args.dataset)
    max_cap = max(args.max_samples, 0)
    max_samples = None if max_cap == 0 else max_cap
    X, y, total_samples, retained_samples = build_training_matrix(
        observations,
        actions,
        terminals,
        history=args.history,
        include_actions=args.include_actions,
        horizon=args.horizon,
        max_samples=max_samples,
        seed=args.seed,
    )
    # Ensure metadata records the effective sampling cap used during training.
    args.max_samples = int(max_cap)

    payload: dict
    feature_scaler = None
    if args.backend == "gpytorch":
        if StandardScaler is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("scikit-learn is required to scale features for gpytorch training.")
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        bundle = fit_gpytorch_model(
            X_scaled,
            y,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        payload = {
            "backend": "gpytorch",
            "bundle": bundle,
            "feature_scaler": feature_scaler,
            "metadata": vars(args),
        }
    else:
        gp = fit_gaussian_process(
            X,
            y,
            length_scale=args.length_scale,
            noise_level=args.noise_level,
        )
        payload = {
            "backend": "sklearn",
            "model": gp,
            "metadata": vars(args),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output)

    summary = {
        "samples_total": int(total_samples),
        "samples_retained": int(retained_samples),
        "features": int(X.shape[1]),
        "positive_fraction": float(y.mean()),
        "checkpoint": str(args.output),
        "backend": args.backend,
        "device": args.device if args.backend == "gpytorch" else "cpu",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
