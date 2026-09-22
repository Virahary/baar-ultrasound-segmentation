"""Boundary-Aware Attention Refinement with bounded, spatially supported corrections."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .morphology import binary_morphological_gradient
from .attention import BoundaryAttention, AttentionConfig


@dataclass(frozen=True)
class BoundaryRepairConfig:
    hidden_channels: int = 16
    candidate_radius: int = 4
    action_temperature: float = 1.0
    crossing_margin: float = 0.25
    max_logit_delta: float = 6.0
    attention_context: bool = True
    attention_grid_size: int = 16
    attention_projection_dim: int = 32
    attention_num_heads: int = 4
    attention_boundary_radius: int = 2
    attention_threshold: float = 0.5
    attention_dropout: float = 0.1
    attention_branch_channels: int = 4
    attention_alpha_init: float = 0.1
    hard_local_support: bool = True
    attention_self_attention_enabled: bool = True
    attention_cross_attention_enabled: bool = True

    def validate(self) -> None:
        if self.hidden_channels <= 0 or self.candidate_radius <= 0:
            raise ValueError("hidden channels and candidate radius must be positive")
        for name in ("action_temperature", "crossing_margin", "max_logit_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("attention_context", "hard_local_support",
                     "attention_self_attention_enabled", "attention_cross_attention_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        self.attention_config().validate()

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            grid_size=self.attention_grid_size,
            projection_dim=self.attention_projection_dim,
            num_heads=self.attention_num_heads,
            boundary_radius=self.attention_boundary_radius,
            threshold=self.attention_threshold,
            dropout=self.attention_dropout,
            branch_channels=self.attention_branch_channels,
            alpha_init=self.attention_alpha_init,
            self_attention_enabled=self.attention_self_attention_enabled,
            cross_attention_enabled=self.attention_cross_attention_enabled,
        )


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class LocalEncoder(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


def _image_gradient(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    scale = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return magnitude / scale


class BAAR(nn.Module):
    """Full-resolution repair whose output is identical outside a hard envelope."""

    def __init__(
        self,
        feature_channels: int,
        config: BoundaryRepairConfig | None = None,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.feature_channels = int(feature_channels)
        self.config = config or BoundaryRepairConfig()
        self.config.validate()
        hidden = self.config.hidden_channels
        self.feature_encoder = LocalEncoder(feature_channels, hidden)
        self.cue_encoder = LocalEncoder(6, hidden)
        if self.config.attention_context:
            self.attention = BoundaryAttention(
                feature_channels,
                self.config.attention_config(),
            )
            self.attention_encoder = LocalEncoder(feature_channels, hidden)
            fusion_channels = 3 * hidden
        else:
            self.attention = None
            self.attention_encoder = None
            fusion_channels = 2 * hidden
        self.fusion = LocalEncoder(fusion_channels, hidden)
        self.action_head = nn.Conv2d(hidden, 3, 1)
        self.magnitude_head = nn.Conv2d(hidden, 2, 1)

        nn.init.zeros_(self.action_head.weight)
        nn.init.constant_(self.action_head.bias[0:1], 2.0)
        nn.init.constant_(self.action_head.bias[1:], -2.0)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)

    @staticmethod
    def _finite_scalar(value: float, name: str) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real scalar") from exc
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar

    def forward(
        self,
        image: torch.Tensor,
        feature: torch.Tensor,
        coarse_logits: torch.Tensor,
        *,
        strength: float = 1.0,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 4 or feature.ndim != 4 or coarse_logits.ndim != 4:
            raise ValueError("image, feature, and coarse_logits must be BCHW tensors")
        if image.shape[1] != 1:
            raise ValueError("image must have exactly one channel")
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {feature.shape[1]}"
            )
        if coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have exactly one channel")
        if image.shape[0] != feature.shape[0] or image.shape[0] != coarse_logits.shape[0]:
            raise ValueError("image, feature, and coarse_logits batch sizes must match")
        if image.shape[0] <= 0 or min(
            image.shape[-2:] + feature.shape[-2:] + coarse_logits.shape[-2:]
        ) <= 0:
            raise ValueError("image, feature, and coarse_logits must have non-empty dimensions")
        if image.device != feature.device or image.device != coarse_logits.device:
            raise ValueError("image, feature, and coarse_logits must be on the same device")
        if not all(torch.is_floating_point(value) for value in (image, feature, coarse_logits)):
            raise TypeError("image, feature, and coarse_logits must be floating-point tensors")
        repair_strength = self._finite_scalar(strength, "strength")
        if not 0.0 <= repair_strength <= 2.0:
            raise ValueError("strength must lie in [0, 2]")
        spatial_size = coarse_logits.shape[-2:]
        image_full = F.interpolate(image, spatial_size, mode="bilinear", align_corners=False)
        feature_full = F.interpolate(
            feature.detach(), spatial_size, mode="bilinear", align_corners=False
        )
        detached_logits = coarse_logits.detach()
        probability = torch.sigmoid(detached_logits)
        coarse_binary = (probability > 0.5).float()
        candidate = binary_morphological_gradient(
            coarse_binary, self.config.candidate_radius
        ).detach()
        add_support = remove_support = candidate
        uncertainty = (4.0 * probability * (1.0 - probability)).clamp(0.0, 1.0)
        cue_values = [
            image_full,
            probability,
            torch.tanh(detached_logits),
            uncertainty,
            candidate,
        ]
        cue_values.append(_image_gradient(image_full))
        cues = torch.cat(cue_values, dim=1)
        encoded_feature = self.feature_encoder(feature_full)
        encoded_cues = self.cue_encoder(cues)
        fusion_inputs = [encoded_feature, encoded_cues]
        attention_debug: dict[str, torch.Tensor] = {}
        if self.attention is not None and self.attention_encoder is not None:
            detached_feature = feature.detach()
            attention_delta, raw_attention_debug = self.attention(
                detached_feature,
                detached_logits,
                return_debug=True,
            )
            attention_full = F.interpolate(
                attention_delta,
                spatial_size,
                mode="bilinear",
                align_corners=False,
            )
            encoded_attention = self.attention_encoder(attention_full)
            fusion_inputs.append(encoded_attention)
            attention_debug = {
                "attention_gate": raw_attention_debug["gate"],
                "attention_modulation": raw_attention_debug["modulation"],
                "attention_context": attention_full,
                "attention_encoded_context": encoded_attention,
                "self_attention_active": raw_attention_debug["self_attention_active"],
                "cross_attention_active": raw_attention_debug["cross_attention_active"],
            }
        fused = self.fusion(torch.cat(fusion_inputs, dim=1))

        action_logits = self.action_head(fused)
        action_probability = torch.softmax(
            action_logits / self.config.action_temperature, dim=1
        )
        keep, add, remove = action_probability.chunk(3, dim=1)
        magnitude = 0.5 + torch.sigmoid(self.magnitude_head(fused))
        add_factor, remove_factor = magnitude.chunk(2, dim=1)
        add_needed = ((-detached_logits).clamp_min(0.0) + self.config.crossing_margin)
        remove_needed = (detached_logits.clamp_min(0.0) + self.config.crossing_margin)
        add_delta = (add_needed * add_factor).clamp_max(self.config.max_logit_delta)
        remove_delta = (remove_needed * remove_factor).clamp_max(
            self.config.max_logit_delta
        )
        if self.config.hard_local_support:
            residual_add_support = add_support
            residual_remove_support = remove_support
        else:
            # Hard-support ablation keeps the candidate band among the cues.
            residual_add_support = torch.ones_like(add_support)
            residual_remove_support = torch.ones_like(remove_support)
        add_residual = residual_add_support * add * add_delta
        remove_residual = residual_remove_support * remove * remove_delta
        unit_residual = add_residual - remove_residual
        applied_residual = repair_strength * unit_residual
        refined_logits = detached_logits + applied_residual

        if not return_debug:
            return refined_logits
        debug = {
            "candidate": candidate,
            "add_support": add_support,
            "remove_support": remove_support,
            "residual_add_support": residual_add_support,
            "residual_remove_support": residual_remove_support,
            "action_logits": action_logits,
            "action_probability": action_probability,
            "keep_probability": keep,
            "add_probability": add,
            "remove_probability": remove,
            "unit_logit_residual": unit_residual,
            "applied_logit_residual": applied_residual,
            "add_logit_residual": add_residual,
            "remove_logit_residual": remove_residual,
        }
        debug.update(attention_debug)
        return refined_logits, debug
