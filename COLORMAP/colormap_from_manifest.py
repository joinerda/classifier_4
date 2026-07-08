#!/usr/bin/env python3
"""Build a manifest-backed crop colormap dataset for pre-cropped images.

This workflow is designed for classifier datasets where the manifest already
indexes cropped images directly. It:

1. reads an existing manifest.csv
2. fits reference RGB channel statistics from selected reference rows
3. color-normalizes selected source rows toward that reference distribution
4. writes transformed images under a derived subdirectory beneath IMAGE_ROOT
5. writes a new manifest with updated `path` values for the transformed rows

The default experiment setup is:
- reference rows: domain=wild, split=train
- transform rows: domain=lab, split=train
- untouched rows: everything else, including wild test images
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for manifest-driven colormapping") from exc
    return np


def _import_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for manifest-driven colormapping") from exc
    return Image


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _read_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required = {"path", "class_id", "domain", "split"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def _iter_selected_rows(
    rows: Iterable[dict[str, str]],
    *,
    domain: str,
    splits: set[str],
) -> Iterable[dict[str, str]]:
    for row in rows:
        if row["domain"] == domain and row["split"] in splits:
            yield row


def _load_rgb_pixels(image_path: Path, *, max_pixels: int, rng) -> "object":
    np = _import_numpy()
    Image = _import_image()
    with Image.open(image_path) as img:
        pixels = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    if max_pixels > 0 and len(pixels) > max_pixels:
        indices = rng.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[indices]
    return pixels


def _fit_reference_stats(
    rows: list[dict[str, str]],
    *,
    image_root: Path,
    max_pixels_per_image: int,
    seed: int,
) -> tuple["object", "object", int]:
    np = _import_numpy()
    rng = np.random.default_rng(seed)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    image_count = 0

    for row in rows:
        image_path = image_root / row["path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest image not found: {image_path}")
        pixels = _load_rgb_pixels(image_path, max_pixels=max_pixels_per_image, rng=rng)
        if len(pixels) == 0:
            continue
        channel_sum += pixels.sum(axis=0)
        channel_sum_sq += np.square(pixels).sum(axis=0)
        pixel_count += len(pixels)
        image_count += 1

    if pixel_count == 0:
        raise ValueError("No reference pixels found for the selected manifest rows")

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sum_sq / pixel_count - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32), image_count


def _transfer_image_rgb(image_path: Path, out_path: Path, *, target_mean, target_std) -> None:
    np = _import_numpy()
    Image = _import_image()
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)

    flat = rgb.reshape(-1, 3)
    src_mean = flat.mean(axis=0)
    src_std = np.maximum(flat.std(axis=0), 1.0)
    out = (flat - src_mean) / src_std * target_std + target_mean
    out = np.clip(out, 0.0, 255.0).astype("uint8").reshape(rgb.shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(out_path)


def build_colormapped_manifest(
    *,
    image_root: Path,
    manifest_path: Path,
    output_manifest_path: Path,
    output_subdir: str,
    reference_domain: str,
    reference_splits: list[str],
    transform_domain: str,
    transform_splits: list[str],
    max_pixels_per_image: int,
    seed: int,
) -> Path:
    rows, fieldnames = _read_manifest(manifest_path)
    reference_rows = list(
        _iter_selected_rows(rows, domain=reference_domain, splits=set(reference_splits))
    )
    transform_rows = list(
        _iter_selected_rows(rows, domain=transform_domain, splits=set(transform_splits))
    )
    if not reference_rows:
        raise ValueError(
            f"No reference rows found for domain={reference_domain!r} splits={reference_splits!r}"
        )
    if not transform_rows:
        raise ValueError(
            f"No transform rows found for domain={transform_domain!r} splits={transform_splits!r}"
        )

    target_mean, target_std, reference_image_count = _fit_reference_stats(
        reference_rows,
        image_root=image_root,
        max_pixels_per_image=max_pixels_per_image,
        seed=seed,
    )

    output_prefix = Path(output_subdir)
    transformed = 0
    transform_keys = {
        (row["path"], row["domain"], row["split"])
        for row in transform_rows
    }
    output_rows: list[dict[str, str]] = []

    for row in rows:
        out_row = dict(row)
        row_key = (row["path"], row["domain"], row["split"])
        if row_key in transform_keys:
            source_rel = Path(row["path"])
            out_rel = output_prefix / row["domain"] / row["split"] / source_rel
            source_image = image_root / source_rel
            out_image = image_root / out_rel
            _transfer_image_rgb(
                source_image,
                out_image,
                target_mean=target_mean,
                target_std=target_std,
            )
            out_row["path"] = out_rel.as_posix()
            transformed += 1
        output_rows.append(out_row)

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[done] reference rows: {len(reference_rows)} from {reference_image_count} images")
    print(f"[done] transformed rows: {transformed}")
    print(f"[done] wrote derived manifest: {output_manifest_path}")
    print(f"[done] derived images under: {image_root / output_prefix}")
    return output_manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a derived manifest with colormapped cropped images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-root", required=True, help="Image root used by manifest paths.")
    parser.add_argument("--manifest", required=True, help="Base manifest to read.")
    parser.add_argument("--output-manifest", required=True, help="Derived manifest to write.")
    parser.add_argument(
        "--output-subdir",
        required=True,
        help="Subdirectory beneath IMAGE_ROOT where transformed images will be written.",
    )
    parser.add_argument("--reference-domain", default="wild")
    parser.add_argument("--reference-splits", default="train")
    parser.add_argument("--transform-domain", default="lab")
    parser.add_argument("--transform-splits", default="train")
    parser.add_argument("--max-pixels-per-image", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    build_colormapped_manifest(
        image_root=Path(args.image_root).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        output_manifest_path=Path(args.output_manifest).resolve(),
        output_subdir=args.output_subdir,
        reference_domain=args.reference_domain,
        reference_splits=_parse_csv_list(args.reference_splits),
        transform_domain=args.transform_domain,
        transform_splits=_parse_csv_list(args.transform_splits),
        max_pixels_per_image=args.max_pixels_per_image,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
