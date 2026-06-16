from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _logging_image_loader(path: str):
    try:
        return default_loader(path)
    except Exception as exc:
        print(
            f"[image-load-error] failed to load '{path}': {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise


def build_transforms(
    train: bool,
    vertical_flip: bool = False,
    image_size: int = 256,
    aug_crop_scale_min: float = 0.7,
    aug_rotation: float = 12.0,
    aug_brightness: float = 0.2,
    aug_contrast: float = 0.2,
    aug_saturation: float = 0.2,
    aug_hue: float = 0.05,
    aug_erasing_prob: float = 0.2,
    aug_erasing_scale_max: float = 0.2,
    aug_blur_prob: float = 0.0,
    aug_grayscale_prob: float = 0.0,
    aug_perspective_prob: float = 0.0,
) -> transforms.Compose:
    if train:
        transforms_list = [
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(aug_crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if vertical_flip:
            transforms_list.append(transforms.RandomVerticalFlip(p=0.1))
        transforms_list.append(transforms.RandomRotation(degrees=aug_rotation))
        if aug_perspective_prob > 0.0:
            transforms_list.append(transforms.RandomPerspective(distortion_scale=0.4, p=aug_perspective_prob))
        transforms_list.append(
            transforms.ColorJitter(
                brightness=aug_brightness,
                contrast=aug_contrast,
                saturation=aug_saturation,
                hue=aug_hue,
            )
        )
        if aug_grayscale_prob > 0.0:
            transforms_list.append(transforms.RandomGrayscale(p=aug_grayscale_prob))
        if aug_blur_prob > 0.0:
            transforms_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=aug_blur_prob))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        if aug_erasing_prob > 0.0:
            transforms_list.append(
                transforms.RandomErasing(p=aug_erasing_prob, scale=(0.02, aug_erasing_scale_max), ratio=(0.3, 3.3))
            )
        return transforms.Compose(transforms_list)
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _has_split_dirs(data_dir: Path) -> bool:
    return all((data_dir / name).is_dir() for name in ("train", "val", "test"))


def _has_train_val_dirs(data_dir: Path) -> bool:
    return all((data_dir / name).is_dir() for name in ("train", "val"))


class TransformDataset(Dataset):
    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


class RemapTargetsDataset(Dataset):
    def __init__(self, dataset: Dataset, label_map: dict[int, int]):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, self.label_map[label]


def _remap_split(
    train_ds: datasets.ImageFolder,
    other_ds: datasets.ImageFolder,
    split: str,
    check_classes: bool,
) -> Dataset:
    train_classes = train_ds.classes
    other_classes = other_ds.classes
    if train_classes != other_classes:
        missing = sorted(set(train_classes) - set(other_classes))
        extra = sorted(set(other_classes) - set(train_classes))
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if check_classes:
            raise ValueError(f"{split} classes do not match train classes: {', '.join(details)}")
        return other_ds

    if other_ds.class_to_idx == train_ds.class_to_idx:
        return other_ds

    if not check_classes:
        return other_ds

    label_map = {other_ds.class_to_idx[name]: train_ds.class_to_idx[name] for name in train_classes}
    return RemapTargetsDataset(other_ds, label_map)


def build_datasets(
    data_dir: str,
    split_ratios: Sequence[float],
    seed: int,
    vertical_flip: bool = False,
    check_split_classes: bool = True,
    image_size: int = 256,
    aug_crop_scale_min: float = 0.7,
    aug_rotation: float = 12.0,
    aug_brightness: float = 0.2,
    aug_contrast: float = 0.2,
    aug_saturation: float = 0.2,
    aug_hue: float = 0.05,
    aug_erasing_prob: float = 0.2,
    aug_erasing_scale_max: float = 0.2,
    aug_blur_prob: float = 0.0,
    aug_grayscale_prob: float = 0.0,
    aug_perspective_prob: float = 0.0,
) -> Tuple[Dataset, Dataset, Dataset]:
    aug_kwargs = dict(
        aug_crop_scale_min=aug_crop_scale_min,
        aug_rotation=aug_rotation,
        aug_brightness=aug_brightness,
        aug_contrast=aug_contrast,
        aug_saturation=aug_saturation,
        aug_hue=aug_hue,
        aug_erasing_prob=aug_erasing_prob,
        aug_erasing_scale_max=aug_erasing_scale_max,
        aug_blur_prob=aug_blur_prob,
        aug_grayscale_prob=aug_grayscale_prob,
        aug_perspective_prob=aug_perspective_prob,
    )
    root = Path(data_dir)
    if _has_split_dirs(root):
        train_ds = datasets.ImageFolder(
            root / "train",
            transform=build_transforms(True, vertical_flip, image_size, **aug_kwargs),
            loader=_logging_image_loader,
        )
        val_ds = datasets.ImageFolder(
            root / "val",
            transform=build_transforms(False, vertical_flip, image_size),
            loader=_logging_image_loader,
        )
        test_ds = datasets.ImageFolder(
            root / "test",
            transform=build_transforms(False, vertical_flip, image_size),
            loader=_logging_image_loader,
        )
        val_ds = _remap_split(train_ds, val_ds, "val", check_split_classes)
        test_ds = _remap_split(train_ds, test_ds, "test", check_split_classes)
        return train_ds, val_ds, test_ds
    if _has_train_val_dirs(root):
        train_ds = datasets.ImageFolder(
            root / "train",
            transform=build_transforms(True, vertical_flip, image_size, **aug_kwargs),
            loader=_logging_image_loader,
        )
        val_ds = datasets.ImageFolder(
            root / "val",
            transform=build_transforms(False, vertical_flip, image_size),
            loader=_logging_image_loader,
        )
        val_ds = _remap_split(train_ds, val_ds, "val", check_split_classes)
        return train_ds, val_ds, val_ds

    # Keep base dataset untransformed so split-specific transforms can be applied later.
    full_ds = datasets.ImageFolder(root, loader=_logging_image_loader)
    n_total = len(full_ds)
    if n_total == 0:
        raise ValueError(f"No images found in {data_dir}")

    ratios = list(split_ratios)
    if len(ratios) != 3:
        raise ValueError("split_ratios must have 3 values (train, val, test)")

    train_len = int(n_total * ratios[0])
    val_len = int(n_total * ratios[1])
    test_len = n_total - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(full_ds, [train_len, val_len, test_len], generator=generator)

    train_ds = TransformDataset(train_idx, build_transforms(True, vertical_flip, image_size, **aug_kwargs))
    val_ds = TransformDataset(val_idx, build_transforms(False, vertical_flip, image_size))
    test_ds = TransformDataset(test_idx, build_transforms(False, vertical_flip, image_size))

    return train_ds, val_ds, test_ds


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
