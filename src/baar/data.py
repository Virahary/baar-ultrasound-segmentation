"""Image/mask loading with explicit, user-supplied dataset locations."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import MedicalImageAugmentation

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("The image directory does not exist")
    paths = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("The image directory contains no supported files")
    return paths


def read_gray(path: Path) -> np.ndarray:
    # imdecode handles Unicode paths on Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to decode an input image or mask")
    return image


def image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = cv2.resize(read_gray(path), (image_size, image_size),
                       interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


class ImageMaskDataset(Dataset):
    """Pair matching stems under root/{train,val,test}/{images,masks}."""

    def __init__(self, root: str | Path, split: str, image_size: int = 256,
                 *, augment: bool = False) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        directory = Path(root) / split
        self.images = image_files(directory / "images")
        mask_paths = image_files(directory / "masks")
        image_stems = [p.stem for p in self.images]
        mask_stems = [p.stem for p in mask_paths]
        if len(set(image_stems)) != len(image_stems) or len(set(mask_stems)) != len(mask_stems):
            raise ValueError("Each image and mask must have a unique filename stem")
        if set(image_stems) != set(mask_stems):
            raise ValueError("Every image must have one mask with the same filename stem")
        masks = {p.stem: p for p in mask_paths}
        self.masks = [masks[p.stem] for p in self.images]
        self.image_size = image_size
        self.augmentation = MedicalImageAugmentation() if augment else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = read_gray(self.images[index])
        mask = read_gray(self.masks[index])
        if image.shape != mask.shape:
            raise ValueError("Each image and its mask must have the same original dimensions")
        shape = (self.image_size, self.image_size)
        image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask = (cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0),
        }
