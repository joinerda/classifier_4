#!/usr/bin/env python3
"""Build a manifest CSV from metadata.json using split/domain settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_domains(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        default=env_or_default("IMAGE_ROOT", "pollen_labwild_test"),
        help="Root directory containing images and metadata.json.",
    )
    parser.add_argument(
        "--metadata",
        default=env_or_default("METADATA_PATH", ""),
        help="Optional explicit metadata.json path. Defaults to IMAGE_ROOT/metadata.json.",
    )
    parser.add_argument(
        "--manifest",
        default=env_or_default("MANIFEST", "pollen_labwild_test/manifest.csv"),
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--train-domains",
        default=env_or_default("TRAIN_DOMAINS", "wild"),
        help="Comma-separated domains assigned to training/validation.",
    )
    parser.add_argument(
        "--test-domains",
        default=env_or_default("TEST_DOMAINS", "wild"),
        help="Comma-separated domains assigned to test.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=float(env_or_default("TRAIN_TARGET_RATIO", "0.4")),
        help="If a domain is both train and test, train fraction for that domain.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(env_or_default("VAL_TARGET_RATIO", "0.1")),
        help="If a domain is both train and test, validation fraction for that domain.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=float(env_or_default("TEST_TARGET_RATIO", "0.5")),
        help="If a domain is both train and test, test fraction for that domain.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(env_or_default("MANIFEST_SEED", "42")),
        help="Shuffle seed used before assigning splits inside each domain.",
    )
    parser.add_argument(
        "--domain-field",
        default=env_or_default("DOMAIN_FIELD", "image_type"),
        help="Metadata field containing domain/source labels.",
    )
    parser.add_argument(
        "--class-field",
        default=env_or_default("CLASS_FIELD", "species"),
        help="Metadata field containing class labels.",
    )
    parser.add_argument(
        "--path-field",
        default=env_or_default("PATH_FIELD", "filename"),
        help="Metadata field containing image paths.",
    )
    return parser.parse_args()


def normalize_path(raw_path: str, image_root: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return path.relative_to(image_root).as_posix()
    return path.as_posix()


def assign_splits(
    n_total: int,
    in_train: bool,
    in_test: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[str]:
    if in_train and in_test:
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        return ["train"] * n_train + ["val"] * n_val + ["test"] * (n_total - n_train - n_val)
    if in_train:
        denom = train_ratio + val_ratio
        prop_train = train_ratio / denom if denom else 1.0
        n_train = int(n_total * prop_train)
        return ["train"] * n_train + ["val"] * (n_total - n_train)
    if in_test:
        return ["test"] * n_total
    return ["unknown"] * n_total


def main() -> int:
    args = parse_args()
    image_root = Path(args.image_root).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else image_root / "metadata.json"
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not image_root.exists():
        print(f"[error] image root not found: {image_root}", file=sys.stderr, flush=True)
        return 1
    if not metadata_path.exists():
        print(f"[error] metadata file not found: {metadata_path}", file=sys.stderr, flush=True)
        return 1

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        print(
            f"[error] train/val/test ratios must sum to 1.0; got {ratio_sum}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    train_domains = set(parse_domains(args.train_domains))
    test_domains = set(parse_domains(args.test_domains))

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, list):
        print("[error] metadata.json must contain a list of records", file=sys.stderr, flush=True)
        return 1

    required = {args.domain_field, args.class_field, args.path_field}
    missing = sorted(
        key for key in required if any(key not in record for record in metadata)
    )
    if missing:
        print(f"[error] metadata missing required fields: {missing}", file=sys.stderr, flush=True)
        return 1

    all_classes = sorted({record[args.class_field] for record in metadata})
    class_to_id = {label: idx for idx, label in enumerate(all_classes)}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in metadata:
        grouped[str(record[args.domain_field])].append(record)

    rng = random.Random(args.seed)
    rows: list[dict[str, str | int]] = []
    split_counts: Counter[tuple[str, str]] = Counter()

    for domain in sorted(grouped):
        items = list(grouped[domain])
        rng.shuffle(items)
        splits = assign_splits(
            len(items),
            domain in train_domains,
            domain in test_domains,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        for item, split in zip(items, splits):
            row = {
                "path": normalize_path(str(item[args.path_field]), image_root),
                "class_id": class_to_id[item[args.class_field]],
                "domain": domain,
                "split": split,
            }
            rows.append(row)
            split_counts[(domain, split)] += 1

    rows.sort(key=lambda row: (str(row["domain"]), str(row["split"]), str(row["path"])))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_id", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ready] manifest written: {manifest_path}", flush=True)
    print(f"[summary] rows={len(rows)} classes={len(class_to_id)}", flush=True)
    if rows:
        print("[summary] split counts by domain:", flush=True)
        for domain, split in sorted(split_counts):
            print(f"{domain:>12} {split:>7} {split_counts[(domain, split)]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
