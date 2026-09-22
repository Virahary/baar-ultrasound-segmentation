"""The six paper metrics, evaluated per image on the resized pixel grid."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

METRIC_NAMES = ("bf1_at_2", "boundary_dice_w3", "hd95", "assd", "dice", "iou")


def boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Square morphological gradient with zero exterior background."""
    mask = mask.float()[None, None]
    kernel = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1)
    count = F.conv2d(mask, kernel, padding=radius)
    return ((count > 0) & (count < kernel.numel()))[0, 0]


def _nearest_distances(points: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # Chunking limits memory for large or fragmented contours.
    return torch.cat([
        torch.cdist(points[i:i + 256], reference).amin(dim=1)
        for i in range(0, len(points), 256)
    ])


def segmentation_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Accept two binary HxW masks; distance metrics are in resized-image pixels."""
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Metrics require two HxW masks of equal shape")
    prediction = (prediction > 0).float()
    target = (target > 0).float()
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    result = {
        "dice": float(2 * intersection / (union + 1e-8)),
        "iou": float(intersection / (union - intersection + 1e-8)),
    }
    points_p = boundary_band(prediction, 1).nonzero().float()
    points_g = boundary_band(target, 1).nonzero().float()
    if len(points_p) == 0 or len(points_g) == 0:
        diagonal = math.hypot(*prediction.shape)
        return {**result, "bf1_at_2": 0.0, "boundary_dice_w3": 0.0,
                "hd95": diagonal, "assd": diagonal}
    distance_p = _nearest_distances(points_p, points_g)
    distance_g = _nearest_distances(points_g, points_p)
    precision = float((distance_p <= 2).float().mean())
    recall = float((distance_g <= 2).float().mean())
    band_p = boundary_band(prediction, 3)
    band_g = boundary_band(target, 3)
    result.update(
        bf1_at_2=2 * precision * recall / (precision + recall + 1e-8),
        boundary_dice_w3=float(2 * (band_p & band_g).sum() / (band_p.sum() + band_g.sum() + 1e-8)),
        hd95=max(float(torch.quantile(distance_p, 0.95)), float(torch.quantile(distance_g, 0.95))),
        assd=(float(distance_p.mean()) + float(distance_g.mean())) / 2,
    )
    return result
