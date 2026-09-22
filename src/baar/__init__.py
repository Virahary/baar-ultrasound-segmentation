"""Boundary-Aware Attention Refinement for binary ultrasound segmentation."""
from .refinement import BAAR, BoundaryRepairConfig
from .losses import HybridBoundaryLoss
from .backbones import SegmentationModel, create_backbone

__all__ = ["BAAR", "BoundaryRepairConfig", "HybridBoundaryLoss",
           "SegmentationModel", "create_backbone"]
