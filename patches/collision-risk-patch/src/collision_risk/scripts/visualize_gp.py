"""Visualisation helpers for analysing trained collision GP models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np

try:  # pragma: no cover - optional dependency guard
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required to plot GP diagnostics. Install it before running this script."
    ) from exc

try:  # pragma: no cover - optional dependency guard
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required to evaluate GP performance. Install it before running this script."
    ) from exc

from ..dataset import build_training_matrix, load_dataset, save_evaluation_summary
from ..model import load_collision_model


def _resolve_metadata(checkpoint: Path) -> dict:
    state = joblib.load(checkpoint)
    if isinstance(state, dict):
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
    return {}


def _determine_bool_override(include_actions_flag: Optional[bool], metadata: dict) -> bool:
    if include_actions_flag is not None:
        return include_actions_flag
    return bool(metadata.get("include_actions", False))


def _select_history(metadata: dict, override: Optional[int]) -> int:
    if override is not None:
        return override
    history = metadata.get("history")
    if history is None:
        raise ValueError(
            "Unable to infer the history window. Please provide --history when the checkpoint "
            "metadata is missing this field."
        )
    return int(history)


def _select_horizon(metadata: dict, override: Optional[int]) -> int:
    if override is not None:
        return override
    horizon = metadata.get("horizon")
    if horizon is None:
        raise ValueError(
            "Unable to infer the labelling horizon. Please provide --horizon when the checkpoint "
            "metadata is missing this field."
        )
    return int(horizon)


def _compute_learning_curve(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sample_sizes: Iterable[int],
    test_features: np.ndarray,
    test_targets: np.ndarray,
    seed: int,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)

    sizes = sorted({size for size in sample_sizes if size <= len(features) and size > 0})
    if not sizes:
        return results

    for size in sizes:
        subset = indices[:size]
        train_x = features[subset]
        train_y = targets[subset]
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
        kernel += WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(train_x_scaled, train_y)
        eval_x = scaler.transform(test_features)
        predictions = gp.predict(eval_x)
        brier = float(brier_score_loss(test_targets, predictions))
        roc_auc = float(roc_auc_score(test_targets, predictions)) if 0 < test_targets.mean() < 1 else float("nan")
        results.append({
            "samples": float(size),
            "roc_auc": roc_auc,
            "brier": brier,
        })
    return results


def _plot_metrics(
    output_path: Path,
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    dataset_label: str,
) -> dict[str, float]:
    has_both_classes = 0 < targets.mean() < 1
    roc_auc = float(roc_auc_score(targets, predictions)) if has_both_classes else float("nan")
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = float(average_precision_score(targets, predictions)) if has_both_classes else float("nan")
    if has_both_classes:
        fpr, tpr, _ = roc_curve(targets, predictions)
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    brier = float(brier_score_loss(targets, predictions))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_roc, ax_pr, ax_hist = axes

    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")

    ax_pr.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="lower left")

    positives = predictions[targets == 1]
    negatives = predictions[targets == 0]
    bins = np.linspace(0.0, 1.0, 30)
    ax_hist.hist(
        negatives,
        bins=bins,
        alpha=0.7,
        label="Non-collision",
        density=True,
        color="#2ca02c",
    )
    ax_hist.hist(
        positives,
        bins=bins,
        alpha=0.7,
        label="Collision",
        density=True,
        color="#d62728",
    )
    ax_hist.set_xlabel("Predicted collision risk")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Risk distribution")
    ax_hist.legend()

    fig.suptitle(f"Collision GP diagnostics – {dataset_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "brier_score": brier,
        "positive_rate": float(targets.mean()),
        "samples": int(targets.size),
    }


def _plot_convergence(
    output_path: Path,
    *,
    loss_history: Optional[Iterable[float]],
    learning_curve: list[dict[str, float]],
    dataset_label: str,
) -> dict[str, object]:
    fig, ax_primary = plt.subplots(figsize=(7, 4.5))

    summary: dict[str, object] = {}

    loss_values = list(loss_history) if loss_history else []

    if loss_values:
        epochs = np.arange(1, len(loss_values) + 1)
        losses = np.asarray(loss_values, dtype=float)
        ax_primary.plot(epochs, losses, label="Training NLL", color="#1f77b4")
        ax_primary.set_xlabel("Epoch")
        ax_primary.set_ylabel("Negative log likelihood")
        ax_primary.set_title("GP convergence (training backend)")
        summary["loss_history_length"] = int(len(losses))
        summary["final_loss"] = float(losses[-1]) if losses.size else float("nan")
    else:
        ax_primary.set_xlabel("Training samples")
        ax_primary.set_ylabel("Evaluation metric")
        ax_primary.set_title("Learning curve (scikit-learn retrain)")

    if learning_curve:
        curve_array = np.asarray([[entry["samples"], entry["roc_auc"], entry["brier"]] for entry in learning_curve])
        ax_secondary = ax_primary.twinx() if loss_values else ax_primary
        ax_secondary.plot(
            curve_array[:, 0],
            curve_array[:, 1],
            marker="o",
            color="#ff7f0e",
            label="ROC-AUC",
        )
        ax_secondary.set_ylabel("ROC-AUC", color="#ff7f0e")
        ax_secondary.tick_params(axis="y", labelcolor="#ff7f0e")

        if loss_values:
            ax_primary.legend(loc="upper left")
            ax_secondary.legend(loc="lower right")
        else:
            ax_primary.legend(["ROC-AUC"], loc="lower right")

        summary["learning_curve"] = learning_curve
    else:
        summary.setdefault("learning_curve", [])

    fig.suptitle(f"Collision GP convergence – {dataset_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise collision GP performance metrics.")
    parser.add_argument("dataset", type=Path, help="Offline dataset used for evaluation (pkl/npz/h5).")
    parser.add_argument("checkpoint", type=Path, help="Trained collision GP checkpoint (.joblib).")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for generated figures.")
    parser.add_argument("--history", type=int, default=None, help="Override the history window size.")
    parser.add_argument("--horizon", type=int, default=None, help="Override the collision labelling horizon.")
    parser.add_argument(
        "--include-actions",
        dest="include_actions",
        action="store_true",
        help="Force inclusion of action history when building features.",
    )
    parser.add_argument(
        "--exclude-actions",
        dest="include_actions",
        action="store_false",
        help="Force exclusion of actions from the feature vector.",
    )
    parser.set_defaults(include_actions=None)
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of samples reserved for evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and train/test split.")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=50000,
        help="Cap on the number of evaluation samples (0 keeps all).",
    )
    parser.add_argument(
        "--learning-curve-points",
        type=int,
        default=6,
        help="Number of points to evaluate in the synthetic learning curve.",
    )
    parser.add_argument(
        "--max-learning-train",
        type=int,
        default=4000,
        help="Maximum number of samples used when fitting learning-curve models.",
    )
    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
        help="Disable the synthetic learning-curve computation.",
    )

    args = parser.parse_args()

    metadata = _resolve_metadata(args.checkpoint)
    history = _select_history(metadata, args.history)
    horizon = _select_horizon(metadata, args.horizon)
    include_actions = _determine_bool_override(args.include_actions, metadata)

    observations, actions, terminals = load_dataset(args.dataset)
    max_cap = args.max_eval_samples if args.max_eval_samples > 0 else None
    features, targets, _, _ = build_training_matrix(
        observations,
        actions,
        terminals,
        history=history,
        include_actions=include_actions,
        horizon=horizon,
        max_samples=max_cap,
        seed=args.seed,
    )

    split_kwargs = {"random_state": args.seed}
    if 0 < targets.mean() < 1:
        split_kwargs["stratify"] = targets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=args.test_fraction,
        **split_kwargs,
    )

    model = load_collision_model(str(args.checkpoint))
    predictions = model.batch_predict(X_test).astype(float)
    predictions = np.clip(predictions, model.clip_min, model.clip_max)

    dataset_label = f"PenSim dataset (history={history}, horizon={horizon})"
    metrics_path = args.output_dir / "gp_performance_metrics.png"
    metrics_summary = _plot_metrics(metrics_path, predictions=predictions, targets=y_test, dataset_label=dataset_label)

    loss_history = metadata.get("loss_history") if isinstance(metadata.get("loss_history"), list) else None
    learning_curve_summary: list[dict[str, float]] = []
    if not args.skip_learning_curve:
        train_cap = min(args.max_learning_train, len(X_train))
        rng = np.random.default_rng(args.seed)
        if train_cap < len(X_train):
            subset_indices = rng.choice(len(X_train), size=train_cap, replace=False)
        else:
            subset_indices = np.arange(len(X_train))
        X_subset = X_train[subset_indices]
        y_subset = y_train[subset_indices]
        start_size = min(train_cap, max(50, X_subset.shape[1] * 2))
        base_sizes = np.linspace(
            start_size,
            train_cap,
            num=max(2, args.learning_curve_points),
            dtype=int,
        )
        learning_curve_summary = _compute_learning_curve(
            X_subset,
            y_subset,
            sample_sizes=base_sizes,
            test_features=X_test,
            test_targets=y_test,
            seed=args.seed,
        )

    convergence_path = args.output_dir / "gp_convergence.png"
    convergence_summary = _plot_convergence(
        convergence_path,
        loss_history=loss_history,
        learning_curve=learning_curve_summary,
        dataset_label=dataset_label,
    )

    summary_path = args.output_dir / "gp_performance_summary.json"
    summary_payload = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "history": history,
        "horizon": horizon,
        "include_actions": include_actions,
        "metrics": metrics_summary,
        "convergence": convergence_summary,
    }
    save_evaluation_summary(summary_path, summary_payload)

    print(f"Saved diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
