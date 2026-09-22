"""Hybrid final-segmentation objective used to train the baseline and BAAR."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .morphology import as_4d, binary_morphological_gradient


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(as_4d(logits))
    ground_truth = as_4d(target).float()
    intersection = (probability * ground_truth).sum(dim=(1, 2, 3))
    numerator = 2.0 * intersection + 1.0
    denominator = (
        probability.sum(dim=(1, 2, 3)) + ground_truth.sum(dim=(1, 2, 3)) + 1.0
    )
    return (1.0 - numerator / denominator).mean()


def gt_band_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: int = 3,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(as_4d(logits))
    ground_truth = as_4d(target).float()
    band = binary_morphological_gradient(ground_truth, radius)
    pred_band = probability * band
    target_band = ground_truth * band
    intersection = (pred_band * target_band).sum(dim=(1, 2, 3))
    numerator = 2.0 * intersection + epsilon
    denominator = (
        pred_band.sum(dim=(1, 2, 3)) + target_band.sum(dim=(1, 2, 3)) + epsilon
    )
    return (1.0 - numerator / denominator).mean()


def truncated_grid_distance(binary_target: torch.Tensor, max_distance: int = 10) -> torch.Tensor:
    """Four-neighbour distance to the nearest positive cell, truncated at max_distance."""
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    target = as_4d(binary_target).squeeze(1).bool()
    distance = torch.where(
        target,
        torch.zeros_like(target, dtype=torch.float32),
        torch.full_like(target, float(max_distance), dtype=torch.float32),
    )
    fill = float(max_distance)
    for _ in range(max_distance):
        up = F.pad(distance[:, 1:, :], (0, 0, 0, 1), value=fill)
        down = F.pad(distance[:, :-1, :], (0, 0, 1, 0), value=fill)
        left = F.pad(distance[:, :, 1:], (0, 1, 0, 0), value=fill)
        right = F.pad(distance[:, :, :-1], (1, 0, 0, 0), value=fill)
        neighbor = torch.minimum(torch.minimum(up, down), torch.minimum(left, right))
        distance = torch.minimum(distance, neighbor + 1.0)
        distance = torch.where(target, torch.zeros_like(distance), distance)
    return distance.clamp_max_(fill)


def signed_distance_field(target: torch.Tensor, max_distance: int = 10) -> torch.Tensor:
    """Normalized signed field: negative inside and positive outside."""
    ground_truth = as_4d(target).squeeze(1).float().detach()
    distance_outside = truncated_grid_distance(ground_truth, max_distance)
    distance_inside = truncated_grid_distance(1.0 - ground_truth, max_distance)
    signed = (distance_outside - distance_inside).clamp(-max_distance, max_distance)
    return signed / float(max_distance)


def distance_field_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    max_distance: int = 10,
) -> torch.Tensor:
    probability = torch.sigmoid(as_4d(logits)).squeeze(1)
    signed = signed_distance_field(target, max_distance)
    soft_target = torch.sigmoid(-3.0 * signed)
    weight = 1.0 + 4.0 * torch.exp(-2.0 * signed.abs())
    return (weight * (probability - soft_target).abs()).mean()


class HybridBoundaryLoss(nn.Module):
    """0.4 Dice + 0.3 GT-band Dice + 0.2 BCE + 0.1 distance-field loss."""

    def __init__(
        self,
        *,
        dice: float = 0.4,
        gt_band_dice: float = 0.3,
        bce: float = 0.2,
        distance_field: float = 0.1,
        gt_band_radius: int = 3,
        max_distance: int = 10,
    ) -> None:
        super().__init__()
        self.weights = {
            "dice": float(dice),
            "gt_band_dice": float(gt_band_dice),
            "bce": float(bce),
            "distance_field": float(distance_field),
        }
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("loss weights must be finite, non-negative, and have a positive sum")
        if gt_band_radius < 1 or max_distance < 1:
            raise ValueError("band radius and maximum distance must be positive")
        self.gt_band_radius = int(gt_band_radius)
        self.max_distance = int(max_distance)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        logits_4d = as_4d(logits)
        target_4d = as_4d(target).float()
        components = {
            "dice": soft_dice_loss(logits_4d, target_4d),
            "gt_band_dice": gt_band_dice_loss(
                logits_4d, target_4d, radius=self.gt_band_radius
            ),
            "bce": F.binary_cross_entropy_with_logits(logits_4d, target_4d),
            "distance_field": distance_field_loss(
                logits_4d, target_4d, max_distance=self.max_distance
            ),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components}
