from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models


_SUPPORTED = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if arch not in _SUPPORTED:
        raise ValueError(f"Unsupported arch {arch}. Choose from {list(_SUPPORTED)}")

    model_fn = _SUPPORTED[arch]
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT if arch == "resnet18" else None
        if arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        if arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT

    model = model_fn(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, model: nn.Module, class_to_idx: Dict[str, int], meta: Dict[str, str]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, arch: str | None = None, device: torch.device | None = None):
    checkpoint = torch.load(path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    meta = checkpoint.get("meta", {})
    resolved_arch = arch or meta.get("arch", "resnet18")
    if meta.get("dann") == "True":
        from .dann import build_dann_model
        model = build_dann_model(num_classes=len(class_to_idx), arch=resolved_arch, pretrained=False)
    else:
        model = build_model(num_classes=len(class_to_idx), arch=resolved_arch, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, class_to_idx, meta
