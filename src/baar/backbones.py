"""Grayscale segmentation backbones and frozen BAAR integration."""
from __future__ import annotations

import torch
from torch import nn

from .refinement import BAAR, BoundaryRepairConfig

BACKBONES = ("espnet", "unext_s", "unext", "cmunext")


class _FeatureAdapter(nn.Module):
    """Expose the feature entering the final classifier of an upstream model."""

    def __init__(self, model: nn.Module, head_name: str, feature_channels: int,
                 binary_difference: bool = False) -> None:
        super().__init__()
        self.model = model
        self.head_name = head_name
        self.feature_channels = feature_channels
        self.binary_difference = binary_difference

    @property
    def head(self) -> nn.Module:
        return getattr(self.model, self.head_name)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        handle = self.head.register_forward_pre_hook(
            lambda _module, inputs: features.append(inputs[0])
        )
        try:
            logits = self.model(image.repeat(1, 3, 1, 1))
        finally:
            handle.remove()
        if len(features) != 1:
            raise RuntimeError("Expected exactly one final decoder feature map")
        if self.binary_difference:
            logits = logits[:, 1:2] - logits[:, 0:1]
        return logits, features[0]


def create_backbone(name: str, image_size: int = 256) -> nn.Module:
    """Build a paper backbone with one logit channel and exposed decoder features."""
    if image_size < 32 or image_size % 32:
        raise ValueError("image_size must be a positive multiple of 32")
    if name in ("unext", "unext_s"):
        from ._backbones.unext import UNext, UNext_S

        model = (UNext if name == "unext" else UNext_S)(
            num_classes=1, img_size=image_size, input_channels=3, in_chans=3
        )
        return _FeatureAdapter(model, "final", 16 if name == "unext" else 8)
    if name == "espnet":
        from ._backbones.espnet import ESPNet

        # The upstream decoder uses classes / 5. Preserve its 20-channel
        # classifier and form a binary logit from foreground minus background.
        return _FeatureAdapter(ESPNet(classes=20, p=2, q=3), "classifier", 20, True)
    if name == "cmunext":
        from ._backbones.cmunext import CMUNeXt

        return CMUNeXt(num_classes=1)
    raise ValueError(f"Unknown backbone {name!r}; choose one of {BACKBONES}")


class SegmentationModel(nn.Module):
    """A baseline, optionally followed by BAAR with a fully frozen backbone."""

    def __init__(self, backbone: nn.Module, config: BoundaryRepairConfig | None = None,
                 *, refine: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.refiner = BAAR(backbone.feature_channels, config) if refine else None
        if self.refiner is not None:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "SegmentationModel":
        super().train(mode)
        if self.refiner is not None:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor, *, strength: float = 1.0,
                return_debug: bool = False):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected BCHW grayscale input")
        if self.refiner is None:
            logits, _ = self.backbone(image)
            return (logits, {}) if return_debug else logits
        with torch.no_grad():
            coarse, feature = self.backbone(image)
        return self.refiner(image, feature.detach(), coarse.detach(),
                            strength=strength, return_debug=return_debug)
