"""DANN (Domain Adversarial Neural Network) components.

Implements:
  GradientReversalFunction  — autograd function that flips gradient sign
  GradientReversalLayer     — nn.Module wrapper
  DomainClassifier          — 2-layer MLP head for binary domain prediction
  DANNModel                 — ResNet backbone + species head + domain head
  build_dann_model          — factory function
  compute_lambda            — Ganin annealing schedule
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models

from .model import _SUPPORTED


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * -ctx.lambda_val, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)


class DomainClassifier(nn.Module):
    """Binary domain classifier head: source (0) vs target (1)."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        self.grl = GradientReversalLayer()

    def forward(self, features: torch.Tensor, lambda_val: float) -> torch.Tensor:
        self.grl.lambda_val = lambda_val
        return self.net(self.grl(features))


class DANNModel(nn.Module):
    """ResNet backbone shared between a species classifier and a domain classifier.

    forward(x) returns species logits only — compatible with eval.py and
    predict.py without modification.  Use forward_features / forward_species /
    forward_domain during DANN training.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, in_features: int) -> None:
        super().__init__()
        self.backbone = backbone          # ResNet with fc replaced by Identity
        self.species_head = nn.Linear(in_features, num_classes)
        self.domain_head = DomainClassifier(in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def forward_species(self, features: torch.Tensor) -> torch.Tensor:
        return self.species_head(features)

    def forward_domain(self, features: torch.Tensor, lambda_val: float) -> torch.Tensor:
        return self.domain_head(features, lambda_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_species(self.forward_features(x))


def build_dann_model(num_classes: int, arch: str = "resnet34", pretrained: bool = True) -> DANNModel:
    if arch not in _SUPPORTED:
        raise ValueError(f"Unsupported arch {arch}. Choose from {list(_SUPPORTED)}")

    model_fn = _SUPPORTED[arch]
    weights = None
    if pretrained:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT

    base = model_fn(weights=weights)
    in_features = base.fc.in_features
    base.fc = nn.Identity()
    return DANNModel(base, num_classes, in_features)


def compute_lambda(epoch: int, total_epochs: int, lambda_max: float = 0.5) -> float:
    """Ganin et al. annealing schedule: ramp lambda from 0 → lambda_max."""
    p = epoch / max(1, total_epochs)
    return lambda_max * (2.0 / (1.0 + pow(2.718281828, -10.0 * p)) - 1.0)
