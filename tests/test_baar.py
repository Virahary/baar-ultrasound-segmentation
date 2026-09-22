"""Behavioral checks using synthetic tensors and generated geometric masks."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from baar import BAAR, BoundaryRepairConfig, HybridBoundaryLoss, SegmentationModel, create_backbone
from baar.cli import main
from baar.data import ImageMaskDataset
from baar.engine import correction_strength, model_from_checkpoint, select_checkpoint
from baar.losses import signed_distance_field, truncated_grid_distance
from baar.metrics import boundary_band, segmentation_metrics
from baar.morphology import binary_morphological_gradient


class BAARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def inputs(self):
        image = torch.rand(2, 1, 32, 32)
        feature = torch.randn(2, 8, 16, 16, requires_grad=True)
        logits = torch.full((2, 1, 32, 32), -3.0)
        logits[:, :, 10:22, 10:22] = 3.0
        return image, feature, logits.requires_grad_()

    def test_paper_parameter_counts_and_output_shapes(self):
        expected = {"espnet": (203189, 26490), "unext_s": (253561, 20838),
                    "unext": (1471921, 24542), "cmunext": (3149185, 24542)}
        for name, (base_count, repair_count) in expected.items():
            with self.subTest(backbone=name):
                model = SegmentationModel(create_backbone(name)).eval()
                self.assertEqual(sum(p.numel() for p in model.backbone.parameters()), base_count)
                self.assertEqual(sum(p.numel() for p in model.refiner.parameters()), repair_count)
                with torch.no_grad():
                    output = model(torch.rand(1, 1, 64, 64))
                self.assertEqual(output.shape, (1, 1, 64, 64))
                self.assertTrue(torch.isfinite(output).all())

    def test_outside_support_identity_and_bounded_update(self):
        module = BAAR(8).eval()
        image, features, logits = self.inputs()
        with torch.no_grad():
            module.action_head.weight.normal_(0, 0.2)
            module.magnitude_head.weight.normal_(0, 0.2)
        result, debug = module(image, features, logits, strength=1.05, return_debug=True)
        outside = debug["candidate"] == 0
        self.assertTrue(outside.any())
        self.assertTrue(torch.equal(result[outside], logits[outside]))
        self.assertLessEqual(float((result - logits).detach().abs().max()), 1.05 * 6 + 1e-6)
        self.assertTrue(torch.allclose(debug["action_probability"].sum(1), torch.ones(2, 32, 32)))

    def test_zero_strength_and_constant_masks(self):
        module = BAAR(8).eval()
        image, features, logits = self.inputs()
        self.assertTrue(torch.equal(module(image, features, logits, strength=0), logits))
        for value in (-4.0, 4.0):
            constant = torch.full_like(logits, value)
            result, debug = module(image, features, constant, return_debug=True)
            self.assertEqual(float(debug["candidate"].sum()), 0)
            self.assertTrue(torch.equal(result, constant))

    def test_detached_inputs_and_trainable_head(self):
        module = BAAR(8)
        image, feature, logits = self.inputs()
        target = torch.zeros_like(logits)
        target[:, :, 8:24, 8:24] = 1
        loss = HybridBoundaryLoss()(module(image, feature, logits), target)["total"]
        loss.backward()
        self.assertIsNone(feature.grad)
        self.assertIsNone(logits.grad)
        self.assertGreater(float(module.action_head.weight.grad.abs().sum()), 0)

    def test_backbone_parameters_and_batchnorm_remain_frozen(self):
        model = SegmentationModel(create_backbone("unext_s"))
        before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        model.train()
        self.assertFalse(model.backbone.training)
        image = torch.rand(2, 1, 64, 64)
        optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3)
        HybridBoundaryLoss()(model(image), (image > 0.5).float())["total"].backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.backbone.state_dict().items()))

    def test_attention_returns_direct_scaled_modulation(self):
        module = BAAR(8).eval()
        _, features, logits = self.inputs()
        with torch.no_grad():
            increment, debug = module.attention(features, logits, return_debug=True)
        self.assertTrue(torch.equal(increment, module.attention.alpha * debug["modulation"]))
        self.assertEqual(debug["dense_tokens"].shape, (2, 256, 32))

    def test_paper_ablations_remain_executable(self):
        image, features, logits = self.inputs()
        for option in ("hard_local_support", "attention_context",
                       "attention_self_attention_enabled", "attention_cross_attention_enabled"):
            with self.subTest(option=option):
                model = BAAR(8, replace(BoundaryRepairConfig(), **{option: False})).eval()
                self.assertTrue(torch.isfinite(model(image, features, logits)).all())

    def test_model_and_metric_border_conventions(self):
        full = torch.ones(12, 12)
        self.assertEqual(float(binary_morphological_gradient(full, 3).sum()), 0)
        self.assertGreater(int(boundary_band(full, 3).sum()), 0)

    def test_distance_field_matches_four_neighbor_distances(self):
        mask = torch.zeros(7, 7)
        mask[3, 3] = 1
        expected = torch.tensor([[min(abs(r - 3) + abs(c - 3), 4)
                                  for c in range(7)] for r in range(7)], dtype=torch.float32)
        torch.testing.assert_close(truncated_grid_distance(mask, 4)[0], expected)
        self.assertTrue(torch.equal(signed_distance_field(torch.zeros(7, 7)), torch.ones(1, 7, 7)))
        self.assertTrue(torch.equal(signed_distance_field(torch.ones(7, 7)), -torch.ones(1, 7, 7)))

    def test_hybrid_loss_weights_and_finite_empty_targets(self):
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        for value in (0, 1):
            terms = HybridBoundaryLoss()(logits, torch.full_like(logits, value))
            expected = sum(weight * terms[name] for name, weight in
                           {"dice": 0.4, "gt_band_dice": 0.3, "bce": 0.2, "distance_field": 0.1}.items())
            torch.testing.assert_close(terms["total"], expected)
            self.assertTrue(torch.isfinite(terms["total"]))
            terms["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_metrics_identical_and_shifted_contours(self):
        mask = torch.zeros(32, 32)
        mask[8:20, 8:20] = 1
        perfect = segmentation_metrics(mask, mask)
        for key in ("dice", "iou", "bf1_at_2", "boundary_dice_w3"):
            self.assertAlmostEqual(perfect[key], 1.0, places=6)
        self.assertAlmostEqual(perfect["hd95"], 0.0)
        shifted = segmentation_metrics(torch.roll(mask, 5, dims=1), mask)
        self.assertLess(shifted["dice"], 1)
        self.assertGreater(shifted["hd95"], 0)

    def test_empty_metric_policy(self):
        result = segmentation_metrics(torch.zeros(12, 16), torch.zeros(12, 16))
        for name in ("dice", "iou", "bf1_at_2", "boundary_dice_w3"):
            self.assertEqual(result[name], 0)
        self.assertEqual(result["hd95"], 20)
        self.assertEqual(result["assd"], 20)

    def test_selection_and_ramp(self):
        self.assertEqual(correction_strength(0), 0)
        self.assertAlmostEqual(correction_strength(3), 0.5)
        self.assertEqual(correction_strength(6), 1)
        self.assertEqual(correction_strength(1, 0), 1)
        m = {"dice": 0.8, "hd95": 3, "bf1_at_2": 0.5, "boundary_dice_w3": 0.6}
        best = select_checkpoint("baar", m, 1, None, 0.8)
        self.assertIs(select_checkpoint("baar", m, 2, best, 0.8), best)
        self.assertIsNone(select_checkpoint("baar", {**m, "dice": 0.79}, 1, None, 0.8))
        base = select_checkpoint("baseline", m, 1, None)
        self.assertEqual(select_checkpoint("baseline", {**m, "hd95": 2}, 2, base)["epoch"], 2)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            BAAR(8, replace(BoundaryRepairConfig(), candidate_radius=0))
        with self.assertRaises(ValueError):
            HybridBoundaryLoss(dice=-1)
        with self.assertRaises(ValueError):
            create_backbone("unext", image_size=63)


class WorkflowTests(unittest.TestCase):
    def make_data(self, root):
        for split in ("train", "val", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "masks").mkdir()
            for index in range(2):
                mask = np.zeros((64, 64), np.uint8)
                cv2.circle(mask, (28 + index * 4, 32), 14, 255, -1)
                image = (mask // 2 + np.full_like(mask, 32)).astype(np.uint8)
                cv2.imwrite(str(root / split / "images" / f"sample_{index}.png"), image)
                cv2.imwrite(str(root / split / "masks" / f"sample_{index}.png"), mask)

    def test_missing_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            (root / "train" / "masks" / "sample_0.png").unlink()
            with self.assertRaises(ValueError):
                ImageMaskDataset(root, "train")

    def test_two_stage_cli_checkpoint_evaluation_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "data"
            self.make_data(data)
            base, repair = root / "baseline", root / "refinement"
            common = ["--data-root", str(data), "--device", "cpu", "--epochs", "1", "--batch-size", "2"]
            main(["train", "--stage", "baseline", "--backbone", "unext_s", "--image-size", "64",
                  "--output-dir", str(base), *common])
            main(["train", "--stage", "baar", "--baseline", str(base / "best.pt"),
                  "--output-dir", str(repair), *common])
            checkpoint = repair / "best.pt"
            first, metadata = model_from_checkpoint(checkpoint, torch.device("cpu"))
            second, _ = model_from_checkpoint(checkpoint, torch.device("cpu"))
            with torch.no_grad():
                image = torch.rand(1, 1, 64, 64)
                self.assertTrue(torch.equal(first(image), second(image)))
            self.assertEqual(metadata["stage"], "baar")
            serialized_metadata = json.dumps({k: v for k, v in metadata.items() if k != "model_state"})
            self.assertNotIn(str(data), serialized_metadata)
            self.assertNotIn("sample_0", serialized_metadata)
            metrics = root / "metrics.json"
            main(["evaluate", "--checkpoint", str(checkpoint), "--data-root", str(data),
                  "--output", str(metrics), "--device", "cpu"])
            self.assertEqual(json.loads(metrics.read_text())["images"], 2)
            predictions = root / "predictions"
            main(["predict", "--checkpoint", str(checkpoint), "--input-dir", str(data / "test" / "images"),
                  "--output-dir", str(predictions), "--device", "cpu"])
            masks = sorted(predictions.glob("*.png"))
            self.assertEqual(len(masks), 2)
            self.assertEqual(masks[0].name, "prediction_000001.png")
            self.assertTrue(set(np.unique(cv2.imread(str(masks[0]), 0))) <= {0, 255})
            with self.assertRaises(FileExistsError):
                main(["train", "--stage", "baseline", "--backbone", "unext_s", "--image-size", "64",
                      "--output-dir", str(base), *common])


if __name__ == "__main__":
    unittest.main()
