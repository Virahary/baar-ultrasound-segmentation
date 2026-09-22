"""Binary morphology shared by the BoundaryAttention gate, loss, and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def as_4d(mask: torch.Tensor) -> torch.Tensor:
    """Return a BCHW tensor while preserving batch and spatial dimensions."""
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D mask, got shape={tuple(mask.shape)}")
    return mask


def binary_morphological_gradient(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Square-kernel morphological gradient using max-pooling.

    Only in-grid neighbors contribute at image borders. Input values are
    expected to be binary; the result has the same BCHW shape.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    binary = as_4d(mask).float()
    if radius == 0:
        return torch.zeros_like(binary)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(binary, kernel_size, stride=1, padding=radius)
    eroded = -F.max_pool2d(-binary, kernel_size, stride=1, padding=radius)
    return (dilated - eroded).clamp_(0.0, 1.0)


def hard_boundary_gate(
    pooled_logits: torch.Tensor,
    radius: int = 2,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Build the detached hard gate from pooled coarse logits."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
    binary = (torch.sigmoid(pooled_logits.detach()) > threshold).float()
    return binary_morphological_gradient(binary, radius)
