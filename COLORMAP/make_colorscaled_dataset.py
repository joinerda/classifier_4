#!/usr/bin/env python3
"""Build a color-scaled copy of a lab/wild pollen dataset.

This script is designed for dataset roots that look like:

    <dataset_root>/
      lab_set/
        train/full
        train/cropped
        test/full
        test/cropped
      wild_set/
        train/full
        train/cropped
        test/full
        test/cropped
      manifest.csv

It creates a new dataset root where:
- `wild_set/` is copied unchanged
- `lab_set/full/` images are color-transferred toward the wild domain
- `lab_set/cropped/` is regenerated from the transformed full images + YOLO boxes
- `manifest.csv` is rebuilt from the new cropped folders

The script intentionally ignores the source `lab_set/cropped/` contents.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _import_cv2():
    import cv2

    return cv2


def _import_numpy():
    import numpy as np

    return np


def _import_rbf_interpolator():
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _iter_image_label_pairs(full_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for image_path in sorted(full_dir.iterdir()):
        if not image_path.is_file() or not _is_image(image_path):
            continue
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            yield image_path, label_path


def _load_color_transfer_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("pollen_color_transfer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load color transfer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _classify_reference_group(path: Path) -> str:
    stem = path.stem.lower()
    if "400x" in stem:
        return "400x"
    if "100x" in stem:
        return "100x"
    return "default"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _parse_yolo_boxes(label_path: Path, width: int, height: int) -> List[Tuple[int, int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int, int]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2.0) * width)
            y1 = int((cy - bh / 2.0) * height)
            x2 = int((cx + bw / 2.0) * width)
            y2 = int((cy + bh / 2.0) * height)
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((class_id, x1, y1, x2, y2))
    return boxes


def _regenerate_crops_from_full(full_dir: Path, cropped_root: Path) -> Dict[str, int]:
    cv2 = _import_cv2()
    counts = defaultdict(int)
    cropped_root.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in _iter_image_label_pairs(full_dir):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load transformed image {image_path}")
        height, width = image_bgr.shape[:2]
        boxes = _parse_yolo_boxes(label_path, width, height)
        for idx, (class_id, x1, y1, x2, y2) in enumerate(boxes):
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            class_dir = cropped_root / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{image_path.stem}_{idx:05d}{image_path.suffix.lower()}"
            out_path = class_dir / out_name
            if not cv2.imwrite(str(out_path), crop):
                raise RuntimeError(f"Failed writing crop {out_path}")
            counts[str(class_id)] += 1
    return dict(counts)


def _build_manifest_rows(data_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for domain, set_name in [("lab", "lab_set"), ("wild", "wild_set")]:
        for split in ("train", "test"):
            cropped_root = data_root / set_name / split / "cropped"
            if not cropped_root.exists():
                continue
            for class_dir in sorted(cropped_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                try:
                    class_id = int(class_dir.name)
                except ValueError:
                    continue
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and _is_image(image_path):
                        rows.append(
                            {
                                "path": image_path.relative_to(data_root).as_posix(),
                                "class_id": class_id,
                                "domain": domain,
                                "split": split,
                            }
                        )
    return rows


def _write_manifest(data_root: Path) -> Path:
    manifest_path = data_root / "manifest.csv"
    rows = _build_manifest_rows(data_root)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_id", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _build_reference_models(
    color_module,
    wild_full_dirs: Sequence[Path],
    *,
    group_by_magnification: bool,
    n_bg_clusters: int,
    n_fg_clusters: int,
    kernel: str,
    epsilon: float,
    l_strength: float,
) -> Dict[str, object]:
    cv2 = _import_cv2()
    grouped_pairs: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for full_dir in wild_full_dirs:
        for image_path, label_path in _iter_image_label_pairs(full_dir):
            key = _classify_reference_group(image_path) if group_by_magnification else "default"
            grouped_pairs[key].append((image_path, label_path))

    if not grouped_pairs:
        raise ValueError("No wild reference image/label pairs found.")

    models = {}
    for key, pairs in grouped_pairs.items():
        ref_bg, ref_fg = color_module.load_reference_pixels(
            [(str(img), str(lbl)) for img, lbl in pairs],
            verbose=False,
        )
        representative_ref = cv2.cvtColor(cv2.imread(str(pairs[0][0])), cv2.COLOR_BGR2RGB)
        models[key] = color_module.ColorTransferModel(
            ref_bg,
            ref_fg,
            representative_ref=representative_ref,
            n_bg_clusters=n_bg_clusters,
            n_fg_clusters=n_fg_clusters,
            kernel=kernel,
            epsilon=epsilon,
            l_strength=l_strength,
        )
    if "default" not in models:
        first_key = next(iter(models))
        models["default"] = models[first_key]
    return models


def _dedupe_anchor_pairs(src_pts, tgt_pts):
    np = _import_numpy()
    rounded = np.round(src_pts, decimals=6)
    buckets: Dict[Tuple[float, float, float], List[int]] = defaultdict(list)
    for idx, row in enumerate(rounded):
        buckets[tuple(row.tolist())].append(idx)

    src_rows = []
    tgt_rows = []
    for indices in buckets.values():
        src_rows.append(src_pts[indices[0]])
        tgt_rows.append(tgt_pts[indices].mean(axis=0))
    return np.asarray(src_rows, dtype=float), np.asarray(tgt_rows, dtype=float)


def _fit_rbf(src_pts, tgt_pts, kernel: str, epsilon: float):
    RBFInterpolator = _import_rbf_interpolator()

    needs_eps = kernel in (
        "multiquadric",
        "gaussian",
        "inverse_multiquadric",
        "inverse_quadratic",
    )
    fit_kwargs = {"kernel": kernel, "degree": 1}
    if needs_eps:
        fit_kwargs["epsilon"] = epsilon

    src_clean, tgt_clean = _dedupe_anchor_pairs(src_pts, tgt_pts)

    try:
        return RBFInterpolator(src_clean, tgt_clean, **fit_kwargs)
    except Exception:
        # Degenerate anchors can still appear after deduplication; a tiny amount
        # of smoothing makes the system solvable without materially changing the fit.
        return RBFInterpolator(src_clean, tgt_clean, smoothing=1e-6, **fit_kwargs)


def _transform_image_safe(model, color_module, image_path: Path, label_path: Path, output_full_dir: Path) -> None:
    cv2 = _import_cv2()
    np = _import_numpy()

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    src_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = src_img.shape[:2]
    mask = color_module.make_fg_mask(src_img.shape, color_module.parse_yolo(str(label_path), width, height))

    src_lab = color_module.rgb_to_lab(src_img)
    n_samples = 10_000
    src_bg_px = color_module.sample_pixels(src_lab, ~mask, n_samples)
    src_fg_px = color_module.sample_pixels(src_lab, mask, n_samples)

    src_bg_c = color_module.cluster_centers_sorted(src_bg_px, model.n_bg_clusters)
    src_fg_c = color_module.cluster_centers_sorted(src_fg_px, model.n_fg_clusters)

    src_pts = np.vstack([src_bg_c, src_fg_c, [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
    tgt_pts = np.vstack([model._ref_bg_c, model._ref_fg_c, [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])

    rbf = _fit_rbf(src_pts, tgt_pts, kernel=model.kernel, epsilon=model.epsilon)

    pixels = src_lab.reshape(-1, 3)
    rbf_out = rbf(pixels.astype(float))

    n_bg = max(len(src_bg_px), 1)
    n_fg = max(len(src_fg_px), 1)
    fg_weight = n_bg / n_fg
    all_src_l = np.concatenate([src_bg_px[:, 0], np.repeat(src_fg_px[:, 0], int(round(fg_weight)))])
    all_ref_l = np.concatenate([model._ref_bg_lab[:, 0], np.repeat(model._ref_fg_lab[:, 0], int(round(fg_weight)))])
    affine_l = color_module.affine_channel(
        pixels[:, 0],
        all_src_l.mean(),
        all_src_l.std(),
        all_ref_l.mean(),
        all_ref_l.std(),
    )

    out_lab = rbf_out.copy()
    out_lab[:, 0] = model.l_strength * rbf_out[:, 0] + (1.0 - model.l_strength) * affine_l
    out_lab = np.clip(out_lab, [0, -128, -128], [100, 127, 127])
    transformed = color_module.lab_to_rgb(out_lab.reshape(height, width, 3).astype(np.float32))

    output_full_dir.mkdir(parents=True, exist_ok=True)
    image_out = output_full_dir / image_path.name
    label_out = output_full_dir / label_path.name
    if not cv2.imwrite(str(image_out), cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed writing transformed image {image_out}")
    shutil.copy2(label_path, label_out)


def _transform_lab_fulls(
    color_module,
    models: Dict[str, object],
    source_full_dir: Path,
    output_full_dir: Path,
) -> int:
    output_full_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path, label_path in _iter_image_label_pairs(source_full_dir):
        key = _classify_reference_group(image_path)
        model = models.get(key, models["default"])
        _transform_image_safe(model, color_module, image_path, label_path, output_full_dir)
        count += 1
    return count


def _copy_other_full_files(src_full_dir: Path, dst_full_dir: Path) -> None:
    dst_full_dir.mkdir(parents=True, exist_ok=True)
    for path in src_full_dir.iterdir():
        if path.is_dir():
            continue
        if _is_image(path) or path.suffix.lower() == ".txt":
            continue
        shutil.copy2(path, dst_full_dir / path.name)


def _copy_non_cropped_set(src_set_root: Path, dst_set_root: Path) -> None:
    dst_set_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        src_split = src_set_root / split
        dst_split = dst_set_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        src_full = src_split / "full"
        dst_full = dst_split / "full"
        _copy_tree(src_full, dst_full)
        dst_cropped = dst_split / "cropped"
        _ensure_empty_dir(dst_cropped)


def build_colorscaled_dataset(
    *,
    source_root: Path,
    output_root: Path,
    color_transfer_module: Path,
    group_by_magnification: bool,
    n_bg_clusters: int,
    n_fg_clusters: int,
    kernel: str,
    epsilon: float,
    l_strength: float,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")

    color_module = _load_color_transfer_module(color_transfer_module)

    lab_src = source_root / "lab_set"
    wild_src = source_root / "wild_set"
    if not lab_src.exists() or not wild_src.exists():
        raise FileNotFoundError("Expected source_root to contain lab_set/ and wild_set/")

    output_root.mkdir(parents=True, exist_ok=False)

    # Copy wild data unchanged.
    _copy_tree(wild_src, output_root / "wild_set")

    # Create lab structure afresh: full is transformed, cropped is regenerated.
    for split in ("train", "test"):
        src_full = lab_src / split / "full"
        dst_full = output_root / "lab_set" / split / "full"
        dst_cropped = output_root / "lab_set" / split / "cropped"
        dst_full.mkdir(parents=True, exist_ok=True)
        dst_cropped.mkdir(parents=True, exist_ok=True)

        _copy_other_full_files(src_full, dst_full)

    models = _build_reference_models(
        color_module,
        wild_full_dirs=[wild_src / "train" / "full", wild_src / "test" / "full"],
        group_by_magnification=group_by_magnification,
        n_bg_clusters=n_bg_clusters,
        n_fg_clusters=n_fg_clusters,
        kernel=kernel,
        epsilon=epsilon,
        l_strength=l_strength,
    )

    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "test"):
        src_full = lab_src / split / "full"
        dst_full = output_root / "lab_set" / split / "full"
        dst_cropped = output_root / "lab_set" / split / "cropped"
        transformed = _transform_lab_fulls(color_module, models, src_full, dst_full)
        crop_counts = _regenerate_crops_from_full(dst_full, dst_cropped)
        summary[split] = {"transformed_full_images": transformed, "regenerated_crops": sum(crop_counts.values())}

    manifest_path = _write_manifest(output_root)

    print(f"[done] wrote colorscaled dataset: {output_root}")
    print(f"[done] manifest: {manifest_path}")
    for split, stats in summary.items():
        print(
            f"[done] lab/{split}: transformed_full_images={stats['transformed_full_images']} "
            f"regenerated_crops={stats['regenerated_crops']}"
        )
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a colorscaled copy of a lab/wild dataset and rebuild lab crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", required=True, help="Source dataset root containing lab_set/ and wild_set/.")
    parser.add_argument("--output-root", required=True, help="New dataset root to create.")
    parser.add_argument(
        "--color-transfer-module",
        default="DATASET_TEST/PollenImageIdea/color_transfer.py",
        help="Path to the reusable color_transfer.py module.",
    )
    parser.add_argument(
        "--group-by-magnification",
        action="store_true",
        default=True,
        help="Build separate wild reference pools for 100x vs 400x image names.",
    )
    parser.add_argument(
        "--no-group-by-magnification",
        dest="group_by_magnification",
        action="store_false",
        help="Use one global wild reference pool for all images.",
    )
    parser.add_argument("--n-bg-clusters", type=int, default=4)
    parser.add_argument("--n-fg-clusters", type=int, default=2)
    parser.add_argument(
        "--kernel",
        default="gaussian",
        choices=[
            "thin_plate_spline",
            "multiquadric",
            "gaussian",
            "inverse_multiquadric",
            "inverse_quadratic",
            "linear",
            "cubic",
            "quintic",
        ],
    )
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--l-strength", type=float, default=0.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    build_colorscaled_dataset(
        source_root=Path(args.source_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        color_transfer_module=Path(args.color_transfer_module).resolve(),
        group_by_magnification=args.group_by_magnification,
        n_bg_clusters=args.n_bg_clusters,
        n_fg_clusters=args.n_fg_clusters,
        kernel=args.kernel,
        epsilon=args.epsilon,
        l_strength=args.l_strength,
    )


if __name__ == "__main__":
    main()
