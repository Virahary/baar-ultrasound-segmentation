"""Multi-scale boundary attention supplying context to the local repair head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .morphology import hard_boundary_gate


@dataclass(frozen=True)
class AttentionConfig:
    grid_size: int = 16
    projection_dim: int = 32
    num_heads: int = 4
    boundary_radius: int = 2
    threshold: float = 0.5
    dropout: float = 0.1
    branch_channels: int = 4
    alpha_init: float = 0.1
    self_attention_enabled: bool = True
    cross_attention_enabled: bool = True

    def validate(self) -> None:
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.projection_dim <= 0 or self.projection_dim % self.num_heads:
            raise ValueError("projection_dim must be positive and divisible by num_heads")
        if not isinstance(self.self_attention_enabled, bool):
            raise TypeError("self_attention_enabled must be boolean")
        if not isinstance(self.cross_attention_enabled, bool):
            raise TypeError("cross_attention_enabled must be boolean")
        if self.boundary_radius < 0:
            raise ValueError("boundary_radius must be non-negative")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if self.branch_channels <= 0:
            raise ValueError("branch_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableAtrousConv(nn.Sequential):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__(
            ConvBNReLU(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            ConvBNReLU(channels, channels, 1),
        )


class LightweightASPP(nn.Module):
    """Four-branch multi-scale aggregation with a residual connection."""

    def __init__(self, channels: int, branch_channels: int = 4) -> None:
        super().__init__()
        self.branch1 = ConvBNReLU(channels, branch_channels, 1)
        self.branch2 = nn.Sequential(
            DepthwiseSeparableAtrousConv(channels, dilation=3),
            ConvBNReLU(channels, branch_channels, 1),
        )
        self.branch3 = nn.Sequential(
            DepthwiseSeparableAtrousConv(channels, dilation=6),
            ConvBNReLU(channels, branch_channels, 1),
        )
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        # BatchNorm is intentionally omitted after 1x1 context projection: it
        # is undefined for a B=1, 1x1 training tensor.
        self.context_proj = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBNReLU(branch_channels * 4, channels, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        spatial_size = feature.shape[-2:]
        context = self.context_proj(self.context_pool(feature))
        context = F.interpolate(context, spatial_size, mode="bilinear", align_corners=False)
        merged = torch.cat(
            (self.branch1(feature), self.branch2(feature), self.branch3(feature), context),
            dim=1,
        )
        return feature + self.fuse(merged)


class BoundaryAttention(nn.Module):
    """Boundary-aware attention refinement over a fixed dense token grid.

    Boundary gating changes token values but never gathers or removes token
    positions. Both attention stages therefore operate on N=grid_size**2
    queries and keys.
    """

    def __init__(self, feature_channels: int, config: AttentionConfig | None = None) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.feature_channels = feature_channels
        self.config = config or AttentionConfig()
        self.config.validate()

        dim = self.config.projection_dim
        self.aspp = LightweightASPP(feature_channels, self.config.branch_channels)
        self.token_projection = nn.Sequential(
            nn.Conv2d(feature_channels, dim, 1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.self_attention = nn.MultiheadAttention(
            dim,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
            bias=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
            bias=True,
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Sequential(
            nn.Conv2d(dim, feature_channels, 1, bias=True),
            nn.BatchNorm2d(feature_channels),
        )
        self.alpha = nn.Parameter(torch.tensor(float(self.config.alpha_init)))

    @property
    def token_count(self) -> int:
        return self.config.grid_size**2

    def forward(
        self,
        feature: torch.Tensor,
        coarse_logits: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if feature.ndim != 4 or coarse_logits.ndim != 4:
            raise ValueError("feature and coarse_logits must both be BCHW tensors")
        if feature.shape[0] != coarse_logits.shape[0]:
            raise ValueError("feature and coarse_logits batch sizes must match")
        if coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have exactly one channel")
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {feature.shape[1]}"
            )
        if feature.shape[0] <= 0 or min(
            feature.shape[-2:] + coarse_logits.shape[-2:]
        ) <= 0:
            raise ValueError("feature and coarse_logits must have non-empty dimensions")
        if feature.device != coarse_logits.device:
            raise ValueError("feature and coarse_logits must be on the same device")
        if not torch.is_floating_point(feature) or not torch.is_floating_point(coarse_logits):
            raise TypeError("feature and coarse_logits must be floating-point tensors")

        batch, _, height, width = feature.shape
        grid = self.config.grid_size
        pooled_feature = F.adaptive_avg_pool2d(feature, (grid, grid))
        pooled_logits = F.adaptive_avg_pool2d(coarse_logits.detach(), (grid, grid))

        gate = hard_boundary_gate(
            pooled_logits,
            radius=self.config.boundary_radius,
            threshold=self.config.threshold,
        )
        enhanced = self.aspp(pooled_feature)
        projected = self.token_projection(enhanced)
        dense_tokens = projected.flatten(2).transpose(1, 2)
        flat_gate = gate.flatten(2).transpose(1, 2)
        gated_tokens = dense_tokens * flat_gate

        if self.config.self_attention_enabled:
            self_attended, _ = self.self_attention(
                gated_tokens,
                gated_tokens,
                gated_tokens,
                need_weights=False,
            )
            boundary_conditioned = self.self_norm(self_attended + gated_tokens)
        else:
            # Explicitly bypass only self-attention.  Keeping the gated-token
            # residual and normalization gives the retained cross-attention a
            # real boundary-conditioned key/value path.
            boundary_conditioned = self.self_norm(gated_tokens)

        if self.config.cross_attention_enabled:
            cross_attended, _ = self.cross_attention(
                dense_tokens,
                boundary_conditioned,
                boundary_conditioned,
                need_weights=False,
            )
            output_tokens = self.cross_norm(cross_attended + dense_tokens)
        else:
            # Replace the cross-attention output with the self-attended
            # boundary context while preserving the dense residual path.
            output_tokens = self.cross_norm(boundary_conditioned + dense_tokens)

        modulation = output_tokens.transpose(1, 2).reshape(
            batch, self.config.projection_dim, grid, grid
        )
        modulation = self.output_projection(modulation)
        modulation = F.interpolate(
            modulation,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        increment = self.alpha * modulation

        if not return_debug:
            return increment
        debug = {
            "pooled_logits": pooled_logits,
            "gate": gate,
            "dense_tokens": dense_tokens,
            "gated_tokens": gated_tokens,
            "boundary_conditioned_tokens": boundary_conditioned,
            "output_tokens": output_tokens,
            "modulation": modulation,
            "self_attention_active": dense_tokens.new_tensor(
                float(self.config.self_attention_enabled)
            ),
            "cross_attention_active": dense_tokens.new_tensor(
                float(self.config.cross_attention_enabled)
            ),
        }
        return increment, debug
