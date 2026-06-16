from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

from .config import DANNConfig, EvalConfig, ManifestTrainConfig, TrainConfig
from .data import build_datasets, build_loaders
from .dann import build_dann_model, compute_lambda
from .manifest import build_datasets_from_manifest, build_dann_datasets_from_manifest
from .model import build_model, get_device, load_checkpoint, save_checkpoint
from .utils import dataclass_to_dict, ensure_dir, resolve_seed, run_id, save_json, set_seed, timestamp
from .eval import _load_labels, _init_confusion, _update_confusion, evaluate_model


def _build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError("optimizer must be adamw or sgd")


def _accuracy(logits, targets) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_model(config: TrainConfig) -> Dict[str, float]:
    seed = resolve_seed(config.seed)
    set_seed(seed, config.deterministic)

    device = get_device(config.device)

    if config.run_comment:
        print(f"[run-comment] {config.run_comment}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(
        config.data_dir,
        config.split_ratios,
        seed,
        vertical_flip=config.vertical_flip,
        check_split_classes=config.check_split_classes,
        image_size=config.image_size,
        aug_crop_scale_min=config.aug_crop_scale_min,
        aug_rotation=config.aug_rotation,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        aug_saturation=config.aug_saturation,
        aug_hue=config.aug_hue,
        aug_erasing_prob=config.aug_erasing_prob,
        aug_erasing_scale_max=config.aug_erasing_scale_max,
        aug_blur_prob=config.aug_blur_prob,
        aug_grayscale_prob=config.aug_grayscale_prob,
        aug_perspective_prob=config.aug_perspective_prob,
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, config.batch_size, config.num_workers
    )

    if isinstance(train_ds, datasets.ImageFolder):
        class_to_idx = train_ds.class_to_idx
    else:
        class_to_idx = train_ds.subset.dataset.class_to_idx

    model = build_model(len(class_to_idx), arch=config.arch, pretrained=config.pretrained)
    model.to(device)

    optimizer = _build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    rid = run_id()
    run_name = f"run-{timestamp()}" if rid is None else f"run-{timestamp()}-{rid}"
    run_dir = ensure_dir(Path(config.output_dir) / run_name)
    best_path = run_dir / "best.pt"

    history: List[Dict[str, float]] = []
    best_val = float("-inf")
    epochs_since_improve = 0

    train_start = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += _accuracy(logits, targets)

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_acc += _accuracy(logits, targets)

        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        scheduler.step()

        epoch_time = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_acc > best_val + config.early_stopping_min_delta:
            best_val = val_acc
            epochs_since_improve = 0
            save_checkpoint(
                str(best_path),
                model,
                class_to_idx,
                meta={
                    "arch": config.arch,
                    "seed": str(seed),
                    "data_dir": config.data_dir,
                    "image_size": str(config.image_size),
                },
            )
        else:
            epochs_since_improve += 1
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"epoch_time_sec={epoch_time:.2f}",
            flush=True,
        )

        if config.early_stopping_patience and epochs_since_improve >= config.early_stopping_patience:
            print(
                f"[early-stop] no val_acc improvement for {epochs_since_improve} epochs; stopping.",
                flush=True,
            )
            break

    eval_report = None
    if config.eval_report:
        eval_cfg = EvalConfig(
            data_dir=config.data_dir,
            checkpoint_path=str(best_path),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=seed,
            image_size=config.image_size,
            confusion_matrix=config.eval_confusion_matrix,
            per_class=config.eval_per_class,
            report_path=str(run_dir / "eval_report.json"),
            device=config.device,
        )
        eval_report = evaluate_model(eval_cfg)

    wall_time = time.time() - train_start
    history_payload = {
        "config": dataclass_to_dict(config),
        "history": history,
        "wall_time_sec": wall_time,
    }
    if config.run_comment:
        history_payload["run_comment"] = config.run_comment
    if eval_report is not None:
        history_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "history.json", history_payload)
    metrics_payload = {"best_val_acc": best_val, "wall_time_sec": wall_time}
    if eval_report is not None:
        metrics_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "metrics.json", metrics_payload)

    result = {"best_val_acc": best_val, "checkpoint": str(best_path), "seed": seed}
    if eval_report is not None:
        result["eval_report"] = str(run_dir / "eval_report.json")
    return result


def train_model_manifest(config: ManifestTrainConfig) -> Dict[str, float]:
    """Train using a manifest CSV to define train/val/test splits by domain."""
    seed = resolve_seed(config.seed)
    set_seed(seed, config.deterministic)

    device = get_device(config.device)

    if config.run_comment:
        print(f"[run-comment] {config.run_comment}", flush=True)

    train_domains = [d.strip() for d in config.train_domains.split(",") if d.strip()]
    test_domains = [d.strip() for d in config.test_domains.split(",") if d.strip()]

    aug_kwargs = dict(
        aug_crop_scale_min=config.aug_crop_scale_min,
        aug_rotation=config.aug_rotation,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        aug_saturation=config.aug_saturation,
        aug_hue=config.aug_hue,
        aug_erasing_prob=config.aug_erasing_prob,
        aug_erasing_scale_max=config.aug_erasing_scale_max,
        aug_blur_prob=config.aug_blur_prob,
        aug_grayscale_prob=config.aug_grayscale_prob,
        aug_perspective_prob=config.aug_perspective_prob,
    )

    train_ds, val_ds, test_ds = build_datasets_from_manifest(
        manifest_path=config.manifest_path,
        image_root=config.image_root,
        train_domains=train_domains,
        test_domains=test_domains,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        max_per_class=config.max_per_class,
        seed=seed,
        vertical_flip=config.vertical_flip,
        image_size=config.image_size,
        aug_kwargs=aug_kwargs,
    )

    print(
        f"[manifest] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={len(train_ds.class_to_idx)}",
        flush=True,
    )

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, config.batch_size, config.num_workers
    )

    class_to_idx = train_ds.class_to_idx
    model = build_model(len(class_to_idx), arch=config.arch, pretrained=config.pretrained)
    model.to(device)

    optimizer = _build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    rid = run_id()
    run_name = f"run-{timestamp()}" if rid is None else f"run-{timestamp()}-{rid}"
    run_dir = ensure_dir(Path(config.output_dir) / run_name)
    best_path = run_dir / "best.pt"

    history: List[Dict[str, float]] = []
    best_val = float("-inf")
    epochs_since_improve = 0

    train_start = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += _accuracy(logits, targets)

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_acc += _accuracy(logits, targets)

        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        scheduler.step()

        epoch_time = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_acc > best_val + config.early_stopping_min_delta:
            best_val = val_acc
            epochs_since_improve = 0
            save_checkpoint(
                str(best_path),
                model,
                class_to_idx,
                meta={
                    "arch": config.arch,
                    "seed": str(seed),
                    "manifest_path": config.manifest_path,
                    "train_domains": config.train_domains,
                    "test_domains": config.test_domains,
                    "image_size": str(config.image_size),
                },
            )
        else:
            epochs_since_improve += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"epoch_time_sec={epoch_time:.2f}",
            flush=True,
        )

        if config.early_stopping_patience and epochs_since_improve >= config.early_stopping_patience:
            print(
                f"[early-stop] no val_acc improvement for {epochs_since_improve} epochs; stopping.",
                flush=True,
            )
            break

    # --- Inline eval on test set using best checkpoint ---
    eval_report = None
    if config.eval_report and best_path.exists():
        print(f"[eval] running inline test-set evaluation from {best_path}", flush=True)
        model_eval, _, _ = load_checkpoint(str(best_path), arch=None, device=device)
        model_eval.to(device)
        model_eval.eval()

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        labels = _load_labels(config.labels_path)
        class_mapping = [
            {"idx": idx, "class": name, "name": labels.get(name, name)}
            for idx, name in sorted(idx_to_class.items())
        ]
        num_classes = len(class_mapping)
        confusion = _init_confusion(num_classes) if (config.eval_confusion_matrix or config.eval_per_class) else None

        total_loss_e = 0.0
        total_correct_e = 0
        total_e = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model_eval(images)
                loss = criterion(logits, targets)
                total_loss_e += loss.item()
                preds = torch.argmax(logits, dim=1)
                if confusion is not None:
                    _update_confusion(confusion, targets.cpu(), preds.cpu())
                total_correct_e += (preds == targets).sum().item()
                total_e += targets.size(0)

        avg_loss = total_loss_e / max(1, len(test_loader))
        test_acc = total_correct_e / max(1, total_e)
        print(f"[eval] test_loss={avg_loss:.4f} test_acc={test_acc:.4f}", flush=True)

        eval_report = {"test_loss": avg_loss, "test_acc": test_acc}

        if config.eval_confusion_matrix and confusion is not None:
            eval_report["confusion_matrix"] = confusion.tolist()
            eval_report["class_mapping"] = class_mapping

        if config.eval_per_class and confusion is not None:
            per_class: List[Dict[str, float]] = []
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
            eval_report["per_class"] = per_class
            eval_report["class_mapping"] = class_mapping

        report_path = run_dir / "eval_report.json"
        save_json(report_path, eval_report)
        print(f"[eval] report saved to {report_path}", flush=True)

    wall_time = time.time() - train_start
    history_payload = {
        "config": dataclass_to_dict(config),
        "history": history,
        "wall_time_sec": wall_time,
    }
    if config.run_comment:
        history_payload["run_comment"] = config.run_comment
    if eval_report is not None:
        history_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "history.json", history_payload)

    metrics_payload = {"best_val_acc": best_val, "wall_time_sec": wall_time}
    if eval_report is not None:
        metrics_payload["test_acc"] = eval_report.get("test_acc")
        metrics_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "metrics.json", metrics_payload)

    result: Dict[str, float] = {"best_val_acc": best_val, "checkpoint": str(best_path), "seed": seed}
    if eval_report is not None:
        result["test_acc"] = eval_report.get("test_acc")
        result["eval_report"] = str(run_dir / "eval_report.json")
    return result


def train_model_dann(config: DANNConfig) -> Dict[str, float]:
    """Train a DANN model for cross-domain adaptation."""
    seed = resolve_seed(config.seed)
    set_seed(seed, config.deterministic)
    device = get_device(config.device)

    if config.run_comment:
        print(f"[run-comment] {config.run_comment}", flush=True)

    aug_kwargs = dict(
        aug_crop_scale_min=config.aug_crop_scale_min,
        aug_rotation=config.aug_rotation,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        aug_saturation=config.aug_saturation,
        aug_hue=config.aug_hue,
        aug_erasing_prob=config.aug_erasing_prob,
        aug_erasing_scale_max=config.aug_erasing_scale_max,
        aug_blur_prob=config.aug_blur_prob,
        aug_grayscale_prob=config.aug_grayscale_prob,
        aug_perspective_prob=config.aug_perspective_prob,
    )

    src_train_ds, tgt_train_ds, val_ds, test_ds = build_dann_datasets_from_manifest(
        manifest_path=config.manifest_path,
        image_root=config.image_root,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        val_fraction=config.val_fraction,
        max_per_class=config.max_per_class,
        seed=seed,
        vertical_flip=config.vertical_flip,
        image_size=config.image_size,
        aug_kwargs=aug_kwargs,
    )

    print(
        f"[dann] src_train={len(src_train_ds)} tgt_train={len(tgt_train_ds)} "
        f"val={len(val_ds)} test={len(test_ds)} classes={len(src_train_ds.class_to_idx)}",
        flush=True,
    )

    source_loader = torch.utils.data.DataLoader(
        src_train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    target_loader = torch.utils.data.DataLoader(
        tgt_train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    class_to_idx = src_train_ds.class_to_idx
    model = build_dann_model(len(class_to_idx), arch=config.arch, pretrained=config.pretrained)
    model.to(device)

    optimizer = _build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    species_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    rid = run_id()
    run_name = f"run-{timestamp()}" if rid is None else f"run-{timestamp()}-{rid}"
    run_dir = ensure_dir(Path(config.output_dir) / run_name)
    best_path = run_dir / "best.pt"

    history: List[Dict[str, float]] = []
    best_val = float("-inf")
    epochs_since_improve = 0
    train_start = time.time()

    burnin_epochs = max(0, config.dann_burnin_epochs)
    min_epochs = max(0, config.min_epochs)

    for epoch in range(1, config.epochs + 1):
        if epoch <= burnin_epochs:
            lambda_val = 0.0
        else:
            dann_epoch = epoch - burnin_epochs
            dann_total_epochs = max(1, config.epochs - burnin_epochs)
            lambda_val = compute_lambda(dann_epoch, dann_total_epochs, config.dann_lambda_max)
        model.train()

        n_batches = max(len(source_loader), len(target_loader))
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        total_species_loss = 0.0
        total_domain_loss = 0.0
        total_species_acc = 0.0
        epoch_start = time.time()

        for _ in range(n_batches):
            try:
                src_images, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                src_images, src_labels = next(src_iter)
            try:
                tgt_images, tgt_labels = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                tgt_images, tgt_labels = next(tgt_iter)

            src_images = src_images.to(device)
            src_labels = src_labels.to(device)
            tgt_images = tgt_images.to(device)
            tgt_labels = tgt_labels.to(device)

            src_features = model.forward_features(src_images)
            tgt_features = model.forward_features(tgt_images)

            src_species_logits = model.forward_species(src_features)
            tgt_species_logits = model.forward_species(tgt_features)
            species_loss = (
                species_criterion(src_species_logits, src_labels)
                + species_criterion(tgt_species_logits, tgt_labels)
            )

            src_domain_logits = model.forward_domain(src_features, lambda_val)
            tgt_domain_logits = model.forward_domain(tgt_features, lambda_val)
            domain_labels_src = torch.zeros(src_images.size(0), 1, device=device)
            domain_labels_tgt = torch.ones(tgt_images.size(0), 1, device=device)
            domain_loss = (
                domain_criterion(src_domain_logits, domain_labels_src)
                + domain_criterion(tgt_domain_logits, domain_labels_tgt)
            )

            loss = species_loss + lambda_val * domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_species_loss += species_loss.item()
            total_domain_loss += domain_loss.item()
            total_species_acc += _accuracy(src_species_logits, src_labels)

        total_species_loss /= max(1, n_batches)
        total_domain_loss /= max(1, n_batches)
        total_species_acc /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                val_loss += species_criterion(logits, targets).item()
                val_acc += _accuracy(logits, targets)
        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        scheduler.step()
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_species_loss": total_species_loss,
            "train_domain_loss": total_domain_loss,
            "train_species_acc": total_species_acc,
            "lambda_val": lambda_val,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        })

        if val_acc > best_val + config.early_stopping_min_delta:
            best_val = val_acc
            epochs_since_improve = 0
            save_checkpoint(
                str(best_path),
                model,
                class_to_idx,
                meta={
                    "arch": config.arch,
                    "seed": str(seed),
                    "manifest_path": config.manifest_path,
                    "source_domain": config.source_domain,
                    "target_domain": config.target_domain,
                    "image_size": str(config.image_size),
                    "dann": "True",
                },
            )
        else:
            epochs_since_improve += 1

        print(
            f"epoch={epoch} "
            f"species_loss={total_species_loss:.4f} "
            f"domain_loss={total_domain_loss:.4f} "
            f"lambda={lambda_val:.4f} "
            f"train_acc={total_species_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"epoch_time_sec={epoch_time:.2f}",
            flush=True,
        )

        if (
            config.early_stopping_patience
            and epoch >= min_epochs
            and epochs_since_improve >= config.early_stopping_patience
        ):
            print(
                f"[early-stop] no val_acc improvement for {epochs_since_improve} epochs after min_epochs={min_epochs}; stopping.",
                flush=True,
            )
            break

    # --- Inline eval on target test set ---
    eval_report = None
    if config.eval_report and best_path.exists():
        print(f"[eval] running inline test-set evaluation from {best_path}", flush=True)
        model_eval, _, _ = load_checkpoint(str(best_path), arch=None, device=device)
        model_eval.to(device)
        model_eval.eval()

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        labels = _load_labels(config.labels_path)
        class_mapping = [
            {"idx": idx, "class": name, "name": labels.get(name, name)}
            for idx, name in sorted(idx_to_class.items())
        ]
        num_classes = len(class_mapping)
        confusion = _init_confusion(num_classes) if (config.eval_confusion_matrix or config.eval_per_class) else None

        total_loss_e = 0.0
        total_correct_e = 0
        total_e = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model_eval(images)
                total_loss_e += species_criterion(logits, targets).item()
                preds = torch.argmax(logits, dim=1)
                if confusion is not None:
                    _update_confusion(confusion, targets.cpu(), preds.cpu())
                total_correct_e += (preds == targets).sum().item()
                total_e += targets.size(0)

        avg_loss = total_loss_e / max(1, len(test_loader))
        test_acc = total_correct_e / max(1, total_e)
        print(f"[eval] test_loss={avg_loss:.4f} test_acc={test_acc:.4f}", flush=True)

        eval_report = {"test_loss": avg_loss, "test_acc": test_acc}

        if config.eval_confusion_matrix and confusion is not None:
            eval_report["confusion_matrix"] = confusion.tolist()
            eval_report["class_mapping"] = class_mapping

        if config.eval_per_class and confusion is not None:
            per_class: List[Dict[str, float]] = []
            for item in class_mapping:
                idx = item["idx"]
                total_i = int(confusion[idx].sum().item())
                correct_i = int(confusion[idx, idx].item())
                per_class.append({
                    "idx": idx,
                    "class": item["class"],
                    "name": item["name"],
                    "count": total_i,
                    "correct": correct_i,
                    "acc": correct_i / total_i if total_i > 0 else 0.0,
                })
            eval_report["per_class"] = per_class
            eval_report["class_mapping"] = class_mapping

        save_json(run_dir / "eval_report.json", eval_report)

    wall_time = time.time() - train_start
    history_payload = {
        "config": dataclass_to_dict(config),
        "history": history,
        "wall_time_sec": wall_time,
    }
    if config.run_comment:
        history_payload["run_comment"] = config.run_comment
    if eval_report is not None:
        history_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "history.json", history_payload)

    metrics_payload = {"best_val_acc": best_val, "wall_time_sec": wall_time}
    if eval_report is not None:
        metrics_payload["test_acc"] = eval_report.get("test_acc")
        metrics_payload["eval_report"] = str(run_dir / "eval_report.json")
    save_json(run_dir / "metrics.json", metrics_payload)

    result: Dict[str, float] = {"best_val_acc": best_val, "checkpoint": str(best_path), "seed": seed}
    if eval_report is not None:
        result["test_acc"] = eval_report.get("test_acc")
        result["eval_report"] = str(run_dir / "eval_report.json")
    return result
