# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unified accelerator backend for CUDA (NVIDIA) and SDAA (Teco).

Background
----------
The Teco SDAA PyTorch build exposes a ``torch.sdaa`` namespace (analogous to
``torch.cuda``) while *also* keeping a ``torch.cuda`` namespace that exists but
reports ``is_available() == False``. On NVIDIA builds the opposite holds. Any
code that used to assume "cuda" (``torch.cuda.*``, device string ``"cuda"`` or
distributed backend ``"nccl"``) therefore breaks or silently falls back to CPU
on SDAA.

This module centralises accelerator detection so the rest of the codebase can
work on both CUDA and SDAA without hardcoding either name. On CPU-only machines
every accessor degrades to a safe no-op / ``"cpu"``.

Notes on SDAA specifics (verified against Torch-SDAA 3.2.1 / torch 2.12):
- ``torch.sdaa.device_count`` / ``set_device`` / ``empty_cache`` /
  ``manual_seed_all`` / ``synchronize`` / ``current_device`` /
  ``get_device_name`` / ``get_device_capability`` are all present.
- ``torch.backends.sdaa.matmul`` and ``torch.backends.sdaa.sdp_kernel`` do
  **not** exist, so ``allow_tf32`` and the SDPA kernel-context manager must be
  guarded to CUDA only.
- The distributed collective backend is ``"tccl"`` rather than ``"nccl"``.
"""

import torch

__all__ = [
    "sdaa_available",
    "cuda_available",
    "accel_available",
    "device_type",
    "device_count",
    "set_device",
    "current_device",
    "empty_cache",
    "manual_seed_all",
    "synchronize",
    "get_device_capability",
    "get_device_name",
    "distributed_backend",
    "to_accel",
    "autocast_device",
    "supports_tf32",
    "supports_sdp_kernel",
]


def sdaa_available() -> bool:
    """Whether a Teco SDAA accelerator is usable."""
    try:
        return hasattr(torch, "sdaa") and bool(torch.sdaa.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    """Whether an NVIDIA CUDA accelerator is usable."""
    try:
        return hasattr(torch, "cuda") and bool(torch.cuda.is_available())
    except Exception:
        return False


def accel_available() -> bool:
    """Whether any accelerator (CUDA or SDAA) is usable."""
    return sdaa_available() or cuda_available()


def device_type() -> str:
    """Return ``"sdaa"``, ``"cuda"`` or ``"cpu"``.

    SDAA takes priority because on Teco builds ``torch.cuda`` exists as a stub
    (with ``is_available() == False``); the real accelerator is SDAA.
    """
    if sdaa_available():
        return "sdaa"
    if cuda_available():
        return "cuda"
    return "cpu"


def _accel_module():
    if sdaa_available():
        return torch.sdaa
    if cuda_available():
        return torch.cuda
    return None


def device_count() -> int:
    mod = _accel_module()
    if mod is None:
        return 0
    return int(mod.device_count())


def set_device(idx) -> None:
    mod = _accel_module()
    if mod is not None:
        mod.set_device(idx)


def current_device() -> int:
    mod = _accel_module()
    return int(mod.current_device()) if mod is not None else -1


def empty_cache() -> None:
    mod = _accel_module()
    if mod is not None:
        mod.empty_cache()


def manual_seed_all(seed: int) -> None:
    mod = _accel_module()
    if mod is not None:
        mod.manual_seed_all(seed)


def synchronize() -> None:
    mod = _accel_module()
    if mod is not None:
        mod.synchronize()


def get_device_capability():
    """Return ``(major, minor)``, or ``None`` when unknown/unsupported.

    Both CUDA and SDAA expose ``get_device_capability``; this simply routes to
    the active backend and tolerates failure.
    """
    mod = _accel_module()
    if mod is not None and hasattr(mod, "get_device_capability"):
        try:
            return mod.get_device_capability()
        except Exception:
            return None
    return None


def get_device_name() -> str:
    mod = _accel_module()
    if mod is not None and hasattr(mod, "get_device_name"):
        try:
            return str(mod.get_device_name())
        except Exception:
            pass
    return device_type()


def distributed_backend() -> str:
    """Collective backend string: ``"tccl"`` on SDAA, else ``"nccl"``."""
    return "tccl" if sdaa_available() else "nccl"


def to_accel(obj):
    """Move a tensor/module to the available accelerator (``.sdaa()`` or
    ``.cuda()``). On CPU-only machines this returns ``obj`` unchanged."""
    if sdaa_available() and hasattr(obj, "sdaa"):
        return obj.sdaa()
    if cuda_available() and hasattr(obj, "cuda"):
        return obj.cuda()
    return obj


def autocast_device() -> str:
    """Device string to pass to ``torch.autocast`` / ``torch.amp.autocast``."""
    return device_type()


def supports_tf32() -> bool:
    """``torch.backends.*.matmul.allow_tf32`` only exists on CUDA."""
    return cuda_available()


def supports_sdp_kernel() -> bool:
    """``torch.backends.cuda.sdp_kernel`` context manager only exists on CUDA."""
    return cuda_available() and hasattr(torch.backends.cuda, "sdp_kernel")