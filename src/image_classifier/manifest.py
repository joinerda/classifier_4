"""Manifest-based image dataset utilities.

ManifestDataset
    A torch Dataset backed by a pandas DataFrame slice of manifest.csv.

build_datasets_from_manifest
    Reads manifest.csv, filters by domain/split, balances, splits val
    deterministically, and returns (train_ds, val_ds, test_ds).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from .data import build_transforms


class ManifestDataset(Dataset):
    """Image dataset driven by a pandas DataFrame row-slice of manifest.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from the manifest (columns: path, class_id, domain, split).
    transform : callable
        torchvision transform applied to each loaded PIL image.
    image_root : str | Path
        Absolute path to the directory that ``path`` values are relative to
        (i.e. the ``lab_wild_combined_feb_26/`` folder).
    class_to_idx : dict[str, int]
        Maps str(class_id) → consecutive integer index.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        image_root: str | Path,
        class_to_idx: Dict[str, int],
    ) -> None:
        self._df = df.reset_index(drop=True)
        self.transform = transform
        self.image_root = Path(image_root)
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        img_path = self.image_root / row["path"]
        image = default_loader(str(img_path))
        image = self.transform(image)
        label = self.class_to_idx[str(int(row["class_id"]))]
        return image, label


def _stratified_split(
    df: pd.DataFrame,
    class_ids: List,
    fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (keep, held_out) with stratification by class_id.

    Uses random sampling (not tail-slice) so held_out is representative
    regardless of path ordering.
    """
    keep_parts: List[pd.DataFrame] = []
    held_parts: List[pd.DataFrame] = []
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=seed)  # noqa – just seed init
    for cid in class_ids:
        cls_df = df[df["class_id"] == cid].sample(frac=1, random_state=seed)
        n = len(cls_df)
        n_held = max(1, math.floor(fraction * n)) if n > 1 else 0
        keep_parts.append(cls_df.iloc[n_held:])
        if n_held > 0:
            held_parts.append(cls_df.iloc[:n_held])
    keep = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=df.columns)
    held = pd.concat(held_parts, ignore_index=True) if held_parts else pd.DataFrame(columns=df.columns)
    return keep, held


def build_datasets_from_manifest(
    manifest_path: str | Path,
    image_root: str | Path,
    train_domains: Sequence[str],
    test_domains: Sequence[str],
    val_fraction: float = 0.15,
    test_fraction: Optional[float] = None,
    max_per_class: Optional[int] = None,
    seed: int = 1337,
    vertical_flip: bool = True,
    image_size: int = 224,
    aug_kwargs: Optional[dict] = None,
) -> Tuple[ManifestDataset, ManifestDataset, ManifestDataset]:
    """Build train / val / test ManifestDatasets from a manifest CSV.

    If ``test_fraction`` is None (default), the manifest's ``split`` column
    is used to separate train pool from test set.  ``train_domains`` filters
    the train pool and ``test_domains`` filters the test set independently.

    If ``test_fraction`` is set, the ``split`` column is ignored.  All rows
    matching *either* ``train_domains`` or ``test_domains`` are pooled, a
    stratified test set of size ``test_fraction`` is carved out first, then
    ``val_fraction`` is carved from the remainder for validation.  Use this
    when the manifest has no split metadata or you want a fresh random split.
    """
    if aug_kwargs is None:
        aug_kwargs = {}

    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)

    # --- class_to_idx from full manifest ---
    all_class_ids = sorted(manifest["class_id"].unique().tolist())
    class_to_idx: Dict[str, int] = {str(int(cid)): i for i, cid in enumerate(all_class_ids)}

    if test_fraction is not None:
        # --- Random-split mode: ignore manifest split column ---
        all_domains = sorted(set(list(train_domains)) | set(list(test_domains)))
        pool = manifest[manifest["domain"].isin(all_domains)].copy()
        pool = pool.sort_values("path").reset_index(drop=True)
        train_pool, test_df = _stratified_split(pool, all_class_ids, test_fraction, seed)
        print(
            f"[manifest] random-split mode: pool={len(pool)} "
            f"test_fraction={test_fraction} → test={len(test_df)} trainpool={len(train_pool)}",
            flush=True,
        )
    else:
        # --- Manifest-split mode: use split column ---
        train_pool = manifest[
            (manifest["split"] == "train") & (manifest["domain"].isin(list(train_domains)))
        ].copy()
        test_df = manifest[
            (manifest["split"] == "test") & (manifest["domain"].isin(list(test_domains)))
        ].copy()

    # Sort train pool by path for determinism before sampling / splitting
    train_pool = train_pool.sort_values("path").reset_index(drop=True)

    # --- 4. Balance per class ---
    if max_per_class is not None:
        balanced_parts: List[pd.DataFrame] = []
        for cid in all_class_ids:
            subset = train_pool[train_pool["class_id"] == cid]
            if len(subset) > max_per_class:
                subset = subset.sample(n=max_per_class, random_state=seed)
                subset = subset.sort_values("path")
            balanced_parts.append(subset)
        train_pool = pd.concat(balanced_parts, ignore_index=True)
        train_pool = train_pool.sort_values("path").reset_index(drop=True)

    # --- Stratified val split from train pool ---
    train_df, val_df = _stratified_split(train_pool, all_class_ids, val_fraction, seed)

    # --- 7. Build datasets ---
    train_transform = build_transforms(
        train=True,
        vertical_flip=vertical_flip,
        image_size=image_size,
        **aug_kwargs,
    )
    infer_transform = build_transforms(
        train=False,
        vertical_flip=False,
        image_size=image_size,
    )

    train_ds = ManifestDataset(train_df, train_transform, image_root, class_to_idx)
    val_ds = ManifestDataset(val_df, infer_transform, image_root, class_to_idx)
    test_ds = ManifestDataset(test_df, infer_transform, image_root, class_to_idx)

    return train_ds, val_ds, test_ds


def build_dann_datasets_from_manifest(
    manifest_path: str | Path,
    image_root: str | Path,
    source_domain: str,
    target_domain: str,
    val_fraction: float = 0.15,
    max_per_class: Optional[int] = None,
    seed: int = 1337,
    vertical_flip: bool = True,
    image_size: int = 224,
    aug_kwargs: Optional[dict] = None,
) -> Tuple[ManifestDataset, ManifestDataset, ManifestDataset, ManifestDataset]:
    """Build four ManifestDatasets for DANN training.

    Returns
    -------
    source_train_ds
        Source domain train images (species labels used for species + domain loss).
    target_train_ds
        Target domain train images (species labels used for species + domain loss).
    val_ds
        Source domain val images (used for early stopping on species accuracy).
    test_ds
        Target domain test images (held-out eval — the cross-domain number).
    """
    if aug_kwargs is None:
        aug_kwargs = {}

    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)

    all_class_ids = sorted(manifest["class_id"].unique().tolist())
    class_to_idx: Dict[str, int] = {str(int(cid)): i for i, cid in enumerate(all_class_ids)}

    def _build_train_pool(domain: str) -> pd.DataFrame:
        pool = manifest[
            (manifest["split"] == "train") & (manifest["domain"] == domain)
        ].copy().sort_values("path").reset_index(drop=True)
        if max_per_class is not None:
            parts: List[pd.DataFrame] = []
            for cid in all_class_ids:
                subset = pool[pool["class_id"] == cid]
                if len(subset) > max_per_class:
                    subset = subset.sample(n=max_per_class, random_state=seed).sort_values("path")
                parts.append(subset)
            pool = pd.concat(parts, ignore_index=True).sort_values("path").reset_index(drop=True)
        return pool

    src_pool = _build_train_pool(source_domain)
    tgt_pool = _build_train_pool(target_domain)

    src_train_df, src_val_df = _stratified_split(src_pool, all_class_ids, val_fraction, seed)

    test_df = manifest[
        (manifest["split"] == "test") & (manifest["domain"] == target_domain)
    ].copy()

    train_transform = build_transforms(
        train=True, vertical_flip=vertical_flip, image_size=image_size, **aug_kwargs
    )
    infer_transform = build_transforms(train=False, vertical_flip=False, image_size=image_size)

    source_train_ds = ManifestDataset(src_train_df, train_transform, image_root, class_to_idx)
    target_train_ds = ManifestDataset(tgt_pool, train_transform, image_root, class_to_idx)
    val_ds = ManifestDataset(src_val_df, infer_transform, image_root, class_to_idx)
    test_ds = ManifestDataset(test_df, infer_transform, image_root, class_to_idx)

    return source_train_ds, target_train_ds, val_ds, test_ds
