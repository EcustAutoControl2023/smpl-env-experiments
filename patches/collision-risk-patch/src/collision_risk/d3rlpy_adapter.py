"""Helpers for reconciling d3rlpy models with augmented risk features."""

from __future__ import annotations

import copy
import logging
from typing import Tuple

import numpy as np

try:  # pragma: no cover - torch is an optional dependency at import time
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def expand_observation_scaler(
    scaler,
    *,
    target_dim: int,
    risk_default: float,
    clip: Tuple[float, float],
):
    """Return a copy of ``scaler`` whose statistics span ``target_dim`` features."""

    if scaler is None:
        return None

    padded = copy.deepcopy(scaler)
    clip_min, clip_max = clip

    def _pad_attribute(name: str, value: float) -> None:
        if not hasattr(padded, name):
            return
        arr = getattr(padded, name)
        if arr is None:
            return
        np_arr = np.asarray(arr)
        if np_arr.shape == ():
            return
        if np_arr.shape[-1] >= target_dim:
            return
        pad_width = target_dim - np_arr.shape[-1]
        pad_shape = np_arr.shape[:-1] + (pad_width,)
        pad_values = np.full(pad_shape, value, dtype=np_arr.dtype)
        expanded = np.concatenate([np_arr, pad_values], axis=-1)
        setattr(padded, name, expanded)

    _pad_attribute("minimum", clip_min)
    _pad_attribute("maximum", clip_max)
    _pad_attribute("mean", risk_default)
    _pad_attribute("std", 1.0)
    _pad_attribute("var", 1.0)
    _pad_attribute("running_mean", risk_default)
    _pad_attribute("running_var", 1.0)

    if hasattr(padded, "n_features_in_"):
        try:
            padded.n_features_in_ = target_dim  # type: ignore[attr-defined]
        except Exception:
            pass

    return padded


def safe_initialize_from_offline(online_algo, offline_algo) -> None:
    """Initialise ``online_algo`` weights from ``offline_algo`` tolerantly."""

    if offline_algo is None:
        return
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to copy model parameters")

    try:
        online_algo.copy_policy_from(offline_algo)
        online_algo.copy_q_function_from(offline_algo)
        return
    except RuntimeError as err:
        if "must match" not in str(err):
            raise
        LOGGER.info("Falling back to partial weight transfer: %s", err)

    _copy_impl_modules(getattr(online_algo, "impl", None), getattr(offline_algo, "impl", None))


def _copy_impl_modules(target_impl, source_impl) -> None:
    if target_impl is None or source_impl is None:
        return

    for name in dir(source_impl):
        if name.startswith("_"):
            continue
        source_attr = getattr(source_impl, name)
        target_attr = getattr(target_impl, name, None)
        if isinstance(source_attr, nn.Module) and isinstance(target_attr, nn.Module):
            _copy_module_state(target_attr, source_attr)


def _copy_module_state(target: nn.Module, source: nn.Module) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    patched_state = {}

    for key, target_tensor in target_state.items():
        if key not in source_state:
            continue
        source_tensor = source_state[key]
        patched_state[key] = _pad_tensor_like(target_tensor, source_tensor)

    target.load_state_dict(patched_state, strict=False)


def _pad_tensor_like(target_tensor: torch.Tensor, source_tensor: torch.Tensor) -> torch.Tensor:
    device = target_tensor.device
    result = target_tensor.detach().clone()
    source = source_tensor.to(device=device, dtype=target_tensor.dtype)
    if target_tensor.shape == source.shape:
        return source

    slices = tuple(slice(0, min(t_dim, s_dim)) for t_dim, s_dim in zip(target_tensor.shape, source.shape))
    if slices:
        result[slices] = source[slices]
    return result
