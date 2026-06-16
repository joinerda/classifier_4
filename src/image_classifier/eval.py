from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch import nn
from .config import EvalConfig
from .data import build_datasets, build_loaders
from .model import get_device, load_checkpoint
from .utils import save_json


def _load_labels(labels_path: Optional[str] = None) -> Dict[str, str]:
    """Load integer-ID → species name mapping from a labels.txt file.
    Searches repo root (two levels up from this file) if no path given."""
    candidates = []
    if labels_path:
        candidates.append(Path(labels_path))
    candidates.append(Path(__file__).parent.parent / "labels.txt")
    for p in candidates:
        if p.exists():
            mapping = {}
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
            return mapping
    return {}


def _init_confusion(num_classes: int) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.int64)


def _update_confusion(confusion: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
    targets = targets.view(-1).to(torch.int64)
    preds = preds.view(-1).to(torch.int64)
    for t, p in zip(targets.tolist(), preds.tolist()):
        confusion[t, p] += 1


def evaluate_model(config: EvalConfig) -> Dict[str, Any]:
    device = get_device(config.device)
    model, _, meta = load_checkpoint(config.checkpoint_path, arch=None, device=device)
    image_size = config.image_size
    if image_size is None:
        image_size = int(meta.get("image_size", 256))

    train_ds, val_ds, test_ds = build_datasets(
        config.data_dir,
        (0.8, 0.1, 0.1),
        seed=config.seed,
        vertical_flip=False,
        check_split_classes=True,
        image_size=image_size,
    )
    _, _, test_loader = build_loaders(
        train_ds, val_ds, test_ds, config.batch_size, config.num_workers
    )

    model.to(device)
    model.eval()

    class_to_idx = getattr(test_ds, "class_to_idx", None)
    if class_to_idx is None:
        class_to_idx = test_ds.subset.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = _load_labels(config.labels_path)
    class_mapping = [
        {"idx": idx, "class": name, "name": labels.get(name, name)}
        for idx, name in sorted(idx_to_class.items())
    ]
    confusion = None
    if config.confusion_matrix or config.per_class:
        confusion = _init_confusion(len(class_mapping))

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            if confusion is not None:
                _update_confusion(confusion, targets.cpu(), preds.cpu())
            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(1, len(test_loader))
    acc = total_correct / max(1, total)

    report: Dict[str, Any] = {"test_loss": avg_loss, "test_acc": acc}

    if config.confusion_matrix:
        report["confusion_matrix"] = confusion.tolist() if confusion is not None else []
        report["class_mapping"] = class_mapping

    if config.per_class and confusion is not None:
        per_class: List[Dict[str, Any]] = []
        for item in class_mapping:
            idx = item["idx"]
            total_i = int(confusion[idx].sum().item())
            correct_i = int(confusion[idx, idx].item())
            acc_i = correct_i / total_i if total_i > 0 else 0.0
            per_class.append(
                {
                    "idx": idx,
                    "class": item["class"],
                    "name": item["name"],
                    "count": total_i,
                    "correct": correct_i,
                    "acc": acc_i,
                }
            )
        report["per_class"] = per_class
        report["class_mapping"] = class_mapping

    if config.report_path:
        save_json(config.report_path, report)

    return report
