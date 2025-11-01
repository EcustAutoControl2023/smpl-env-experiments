"""Utilities for training and loading GPyTorch-based GP models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import torch
    import gpytorch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    gpytorch = None  # type: ignore[assignment]


@dataclass
class GPyTorchModelBundle:
    """Container for trained GPyTorch artefacts used at inference time."""

    model_state_dict: dict
    likelihood_state_dict: dict
    train_inputs: np.ndarray
    train_targets: np.ndarray
    feature_dim: int


def ensure_gpytorch_available() -> None:
    """Raise an informative error when GPyTorch is missing."""

    if torch is None or gpytorch is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "GPyTorch and PyTorch are required for the requested GPU-enabled backend."
        )


def _prepare_training_tensors(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    ensure_gpytorch_available()
    device_obj = torch.device(device)
    train_x = torch.as_tensor(features, dtype=torch.float32, device=device_obj)
    train_y = torch.as_tensor(targets, dtype=torch.float32, device=device_obj)
    return train_x, train_y


def _initialise_model(
    train_x: "torch.Tensor",
    train_y: "torch.Tensor",
) -> tuple["gpytorch.models.ExactGP", "gpytorch.likelihoods.GaussianLikelihood"]:
    ensure_gpytorch_available()

    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_inputs, train_targets, likelihood):
            super().__init__(train_inputs, train_targets, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):  # type: ignore[override]
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(train_x.device)
    model = _ExactGPModel(train_x, train_y, likelihood).to(train_x.device)
    return model, likelihood


def train_exact_gp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
) -> GPyTorchModelBundle:
    """Fit an exact GP model using GPyTorch."""

    train_x, train_y = _prepare_training_tensors(features, targets, device=device)
    model, likelihood = _initialise_model(train_x, train_y)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(max(epochs, 1)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    model_cpu = model.cpu()
    likelihood_cpu = likelihood.cpu()
    train_inputs_cpu = model_cpu.train_inputs[0].detach().cpu().numpy().astype(np.float32)
    train_targets_cpu = model_cpu.train_targets.detach().cpu().numpy().astype(np.float32)

    return GPyTorchModelBundle(
        model_state_dict=model_cpu.state_dict(),
        likelihood_state_dict=likelihood_cpu.state_dict(),
        train_inputs=train_inputs_cpu,
        train_targets=train_targets_cpu,
        feature_dim=train_inputs_cpu.shape[1],
    )


def load_model_bundle(bundle: GPyTorchModelBundle) -> tuple["gpytorch.models.ExactGP", "gpytorch.likelihoods.GaussianLikelihood"]:
    """Reconstruct the GPyTorch model from a persisted :class:`GPyTorchModelBundle`."""

    ensure_gpytorch_available()

    train_x = torch.as_tensor(bundle.train_inputs, dtype=torch.float32)
    train_y = torch.as_tensor(bundle.train_targets, dtype=torch.float32)
    model, likelihood = _initialise_model(train_x, train_y)
    model.load_state_dict(bundle.model_state_dict)
    likelihood.load_state_dict(bundle.likelihood_state_dict)
    model.eval()
    likelihood.eval()
    return model, likelihood
