"""Training, checkpoint selection, evaluation, and prediction."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .backbones import SegmentationModel, create_backbone
from .data import ImageMaskDataset, image_files, image_tensor
from .losses import HybridBoundaryLoss
from .metrics import METRIC_NAMES, segmentation_metrics
from .refinement import BoundaryRepairConfig


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cv2.setNumThreads(1)


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


def data_loader(root, split, image_size, batch_size, workers, seed, *, augment=False):
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return DataLoader(
        ImageMaskDataset(root, split, image_size, augment=augment),
        batch_size=batch_size, shuffle=augment, num_workers=workers,
        generator=torch.Generator().manual_seed(seed), worker_init_fn=_worker_seed,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return device


def correction_strength(epoch: int, ramp_epochs: int = 6) -> float:
    """Half-cosine ramp; epochs are one-based."""
    if epoch < 0 or ramp_epochs < 0:
        raise ValueError("epoch and ramp_epochs must be non-negative")
    return 1.0 if ramp_epochs == 0 else 0.5 - 0.5 * math.cos(math.pi * min(1, epoch / ramp_epochs))


def select_checkpoint(stage, metrics, epoch, best, baseline_dice=None):
    """Return the best validation candidate, preserving the earliest exact tie."""
    candidate = {"epoch": epoch, "metrics": metrics}
    if stage == "baseline":
        score = (metrics["dice"], -metrics["hd95"])
        previous = None if best is None else (best["metrics"]["dice"], -best["metrics"]["hd95"])
    else:
        if baseline_dice is None:
            raise ValueError("BAAR selection requires the baseline validation Dice")
        if metrics["dice"] < baseline_dice - 0.005:
            return best
        score = (metrics["bf1_at_2"] + metrics["boundary_dice_w3"]) / 2
        previous = None if best is None else best["score"]
        candidate["score"] = score
    return candidate if previous is None or score > previous else best


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_checkpoint(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_config(path: str | None):
    config = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(config) - {"model", "loss"}:
        raise ValueError("Configuration accepts only 'model' and 'loss' sections")
    repair_config = BoundaryRepairConfig(**config.get("model", {}))
    repair_config.validate()
    loss_config = config.get("loss", {})
    criterion = HybridBoundaryLoss(**loss_config)
    return repair_config, criterion


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a checkpoint produced by this package")
    if checkpoint.get("stage") not in ("baseline", "baar"):
        raise ValueError("Checkpoint stage must be baseline or baar")
    return checkpoint


def model_from_checkpoint(path, device):
    checkpoint = _load_checkpoint(path)
    config = BoundaryRepairConfig(**checkpoint["model_config"])
    model = SegmentationModel(
        create_backbone(checkpoint["backbone"], checkpoint["image_size"]),
        config, refine=checkpoint["stage"] == "baar",
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None):
    """Average each metric over images; do not export filenames or case records."""
    model.eval()
    sums = {name: 0.0 for name in METRIC_NAMES}
    count = 0
    loss_sum = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        logits = model(image, strength=1.0)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model output")
        if criterion is not None:
            loss = criterion(logits, target)["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            loss_sum += float(loss) * len(image)
        for prediction, mask in zip(logits, target):
            values = segmentation_metrics(prediction[0] > 0, mask[0])
            for name in METRIC_NAMES:
                sums[name] += values[name]
        count += len(image)
    if not count:
        raise ValueError("The evaluation split is empty")
    result = {name: value / count for name, value in sums.items()}
    if criterion is not None:
        result["loss"] = loss_sum / count
    return result


def train(args) -> None:
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.ramp_epochs < 0:
        raise ValueError("ramp_epochs must be non-negative")
    if args.stage == "baar" and not args.baseline:
        raise ValueError("--baseline is required for BAAR training")
    if args.stage == "baseline" and args.baseline:
        raise ValueError("--baseline is only used for BAAR training")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    config, criterion = read_config(args.config)
    baseline_checkpoint = _load_checkpoint(args.baseline) if args.baseline else None
    if baseline_checkpoint is not None:
        if baseline_checkpoint["stage"] != "baseline":
            raise ValueError("--baseline must identify a baseline-stage checkpoint")
        name, image_size = baseline_checkpoint["backbone"], baseline_checkpoint["image_size"]
        if args.backbone is not None and args.backbone != name:
            raise ValueError("The backbone must match the baseline checkpoint")
        if args.image_size is not None and args.image_size != image_size:
            raise ValueError("The image size must match the baseline checkpoint")
    else:
        name, image_size = args.backbone or "unext", args.image_size or 256
    backbone = create_backbone(name, image_size)
    if baseline_checkpoint is not None:
        baseline = SegmentationModel(backbone, refine=False)
        baseline.load_state_dict(baseline_checkpoint["model_state"], strict=True)
    model = SegmentationModel(backbone, config, refine=args.stage == "baar").to(device)
    train_loader = data_loader(args.data_root, "train", image_size, args.batch_size,
                               args.workers, args.seed, augment=True)
    validation_loader = data_loader(args.data_root, "val", image_size, args.batch_size,
                                    args.workers, args.seed)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory for each training run")
    output.mkdir(parents=True, exist_ok=True)
    baseline_dice = None
    if args.stage == "baar":
        baseline_dice = evaluate_model(baseline.to(device), validation_loader, device)["dice"]
    epochs = args.epochs if args.epochs is not None else (200 if args.stage == "baseline" else 20)
    lr = 1e-4 if args.stage == "baseline" else 1e-3
    minimum_lr = 1e-6 if args.stage == "baseline" else 1e-5
    parameters = list(model.backbone.parameters() if model.refiner is None else model.refiner.parameters())
    optimizer = AdamW(parameters, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=minimum_lr)
    best, best_loss, stale = None, math.inf, 0
    history = []
    # Checkpoints store only model tensors, numerical settings and aggregate metrics.
    metadata = {
        "format_version": 1, "stage": args.stage, "backbone": name,
        "image_size": image_size, "seed": args.seed, "model_config": asdict(config),
        "loss_config": {**criterion.weights, "gt_band_radius": criterion.gt_band_radius,
                        "max_distance": criterion.max_distance},
        "training": {"epochs": epochs, "batch_size": args.batch_size, "lr": lr,
                     "min_lr": minimum_lr, "weight_decay": 1e-4, "gradient_clip": 1.0,
                     "ramp_epochs": args.ramp_epochs},
    }
    for epoch in range(1, epochs + 1):
        model.train()
        beta = 1.0 if args.stage == "baseline" else correction_strength(epoch, args.ramp_epochs)
        total, count = 0.0, 0
        for batch in train_loader:
            image, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, strength=beta)
            loss = criterion(logits, target)["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(image)
            count += len(image)
        validation = evaluate_model(model, validation_loader, device, criterion)
        metrics = {name: validation[name] for name in METRIC_NAMES}
        selected = select_checkpoint(args.stage, metrics, epoch, best, baseline_dice)
        if selected is not best:
            best = selected
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint(output / "best.pt", {**metadata, "model_state": state, "selection": best})
        improved = (not math.isfinite(best_loss)
                    or best_loss - validation["loss"] > max(1e-4, 0.002 * abs(best_loss)))
        if improved:
            best_loss, stale = validation["loss"], 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total / count, "validation_loss": validation["loss"],
                        "strength": beta, "learning_rate": optimizer.param_groups[0]["lr"], **metrics})
        _write_json(output / "history.json", history)
        print(f"Epoch {epoch}/{epochs} | loss {total/count:.5f} | "
              f"val Dice {metrics['dice']:.5f} | BF1@2 {metrics['bf1_at_2']:.5f}", flush=True)
        scheduler.step()
        if args.stage == "baseline" and epoch >= 40 and stale >= 12:
            break
    _write_json(output / "selection.json", {
        "status": "selected" if best is not None else "no_eligible_checkpoint",
        "selected": best, "baseline_validation_dice": baseline_dice,
    })
    if best is None:
        raise RuntimeError("No BAAR epoch met the validation Dice floor; no best.pt was created")
    print(f"Selected epoch {best['epoch']}; checkpoint: best.pt", flush=True)


def evaluate(args) -> None:
    seed_everything(0)
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    loader = data_loader(args.data_root, args.split, checkpoint["image_size"],
                         args.batch_size, args.workers, 0)
    result = evaluate_model(model, loader, device)
    _write_json(Path(args.output), {
        "split": args.split, "images": len(loader.dataset),
        "distance_unit": "resized-image pixels", "metrics": result,
    })
    print(json.dumps(result, indent=2), flush=True)


@torch.no_grad()
def predict(args) -> None:
    seed_everything(0)
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    inputs = image_files(args.input_dir)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty prediction output directory")
    if output.resolve() == Path(args.input_dir).resolve():
        raise ValueError("Prediction output must differ from the input directory")
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(inputs, 1):
        image = image_tensor(path, checkpoint["image_size"])[None].to(device)
        logits = model(image)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite prediction")
        mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("Unable to encode prediction")
        encoded.tofile(output / f"prediction_{index:06d}.png")
    print(f"Saved {len(inputs)} binary masks at {checkpoint['image_size']} x "
          f"{checkpoint['image_size']} pixels.", flush=True)
