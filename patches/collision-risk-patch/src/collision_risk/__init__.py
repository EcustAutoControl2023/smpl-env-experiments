"""Collision risk utilities package."""

from .model import CollisionRiskModel, GaussianProcessCollisionModel, load_collision_model
from .trajectory import TrajectoryWindow
from .wrapper import CollisionRiskObservationWrapper
from .dataset import (
    augment_dataset_with_risk,
    build_training_matrix,
    future_collision_targets,
    iter_episode_indices,
    load_dataset,
    save_evaluation_summary,
)
from .d3rlpy_adapter import expand_observation_scaler, safe_initialize_from_offline

__all__ = [
    "CollisionRiskModel",
    "GaussianProcessCollisionModel",
    "TrajectoryWindow",
    "CollisionRiskObservationWrapper",
    "augment_dataset_with_risk",
    "load_dataset",
    "iter_episode_indices",
    "future_collision_targets",
    "build_training_matrix",
    "save_evaluation_summary",
    "load_collision_model",
    "expand_observation_scaler",
    "safe_initialize_from_offline",
]
