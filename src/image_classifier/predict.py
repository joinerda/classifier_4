from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import torch
from PIL import Image
from torchvision import transforms

from .config import PredictConfig
from .data import IMAGENET_MEAN, IMAGENET_STD
from .model import get_device, load_checkpoint


def _prep_image(path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    img = Image.open(path).convert("RGB")
    return transform(img)


def predict_images(config: PredictConfig) -> Dict[str, Any]:
    device = get_device(config.device)
    model, class_to_idx, meta = load_checkpoint(config.checkpoint_path, device=device)
    model.to(device)
    model.eval()

    image_size = config.image_size
    if image_size is None:
        image_size = int(meta.get("image_size", 256))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_mapping = [{"idx": idx, "class": name} for idx, name in sorted(idx_to_class.items())]
    results: Dict[str, List[Dict[str, float]]] = {}

    with torch.no_grad():
        for path_str in config.image_paths:
            path = Path(path_str)
            image = _prep_image(path, image_size).unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_probs, top_idx = torch.topk(probs, k=min(config.top_k, probs.numel()))

            preds = [
                {"class": idx_to_class[idx.item()], "prob": float(prob.item())}
                for prob, idx in zip(top_probs, top_idx)
            ]
            results[str(path)] = preds

    return {
        "predictions": results,
        "class_mapping": class_mapping,
        "meta": meta,
    }
