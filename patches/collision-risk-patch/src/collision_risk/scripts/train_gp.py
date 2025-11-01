"""Command-line helpers for training collision risk GP models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

try:  # pragma: no cover - optional dependency guard
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[misc]
    ConstantKernel = RBF = WhiteKernel = None  # type: ignore[misc]
    StandardScaler = None  # type: ignore[misc]

from ..dataset import build_training_matrix, load_dataset
from ..gpytorch_backend import (
    GPyTorchModelBundle,
    ensure_gpytorch_available,
    train_exact_gp,
)


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
    bundle, loss_history = train_exact_gp(
        X,
        y,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    bundle.training_loss_history = loss_history  # type: ignore[attr-defined]
    return bundle


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

    observations, actions, terminals = load_dataset(args.dataset)
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
    metadata = vars(args).copy()
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
        loss_history = getattr(bundle, "training_loss_history", None)
        if loss_history is not None:
            metadata["loss_history"] = list(loss_history)
        payload = {
            "backend": "gpytorch",
            "bundle": bundle,
            "feature_scaler": feature_scaler,
            "metadata": metadata,
        }
    else:
        if tqdm is not None:
            progress = tqdm(total=1, desc="Fitting GP (sklearn)", leave=False)
        else:
            progress = None
        try:
            gp = fit_gaussian_process(
                X,
                y,
                length_scale=args.length_scale,
                noise_level=args.noise_level,
            )
        finally:
            if progress is not None:
                progress.update(1)
                progress.close()
        if hasattr(gp, "log_marginal_likelihood_value_"):
            metadata["log_marginal_likelihood"] = float(gp.log_marginal_likelihood_value_)
        payload = {
            "backend": "sklearn",
            "model": gp,
            "metadata": metadata,
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
