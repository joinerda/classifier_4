#!/usr/bin/env python3
"""
color_transfer.py — RBF-based color domain transfer for microscopy images.

Transforms lab microscope images to match the color distribution of wild
microscope images, preserving YOLO annotations unchanged.  Designed for use
both as a standalone CLI and as an importable module inside a larger pipeline.

── CLI usage ──────────────────────────────────────────────────────────────────

Single image:
    python color_transfer.py \\
        --source lab.jpg --source_label lab.txt \\
        --reference_dir ./WildTypeExamples/ \\
        --output_dir ./transferred/ --comparison

Batch (whole folder of lab images):
    python color_transfer.py \\
        --source_dir ./LabTypeExamples/ \\
        --reference_dir ./WildTypeExamples/ \\
        --output_dir ./transferred/

── Pipeline / module usage ────────────────────────────────────────────────────

    from color_transfer import ColorTransferModel

    model = ColorTransferModel.from_reference_dir("./WildTypeExamples/")

    model.transform_image(
        src_path   = "lab.jpg",
        label_path = "lab.txt",
        out_dir    = "./transferred/",
    )
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import RBFInterpolator


# ── YOLO utilities ────────────────────────────────────────────────────────────

def parse_yolo(label_path, img_w, img_h):
    boxes = []
    if not label_path or not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append((max(0, x1), max(0, y1),
                          min(img_w - 1, x2), min(img_h - 1, y2)))
    return boxes


def make_fg_mask(shape, boxes):
    mask = np.zeros(shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


# ── Color utilities ───────────────────────────────────────────────────────────

def sample_pixels(img, mask, n, seed=42):
    pixels = img[mask].astype(float)
    if len(pixels) == 0:
        return pixels
    rng = np.random.default_rng(seed)
    return pixels[rng.choice(len(pixels), size=min(n, len(pixels)), replace=False)]


def cluster_centers_sorted(pixels, k):
    if len(pixels) == 0:
        return np.empty((0, 3))
    k = min(k, len(pixels))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(pixels)
    centers = km.cluster_centers_
    lum = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    return centers[np.argsort(lum)]


def rgb_to_lab(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def rgb_pixels_to_lab(px_rgb):
    img_u8  = np.clip(px_rgb, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    img_lab = cv2.cvtColor(
        cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0,
        cv2.COLOR_BGR2Lab,
    )
    return img_lab.reshape(-1, 3).astype(float)


def lab_to_rgb(lab):
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]


def affine_channel(vals, src_mean, src_std, tgt_mean, tgt_std):
    if src_std < 1e-6:
        return np.full_like(vals, tgt_mean)
    return (vals - src_mean) / src_std * tgt_std + tgt_mean


# ── Reference pool loading ────────────────────────────────────────────────────

def ref_paths_from_dir(directory):
    pairs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(fname)[0]
        lbl  = os.path.join(directory, stem + '.txt')
        if not os.path.exists(lbl):
            continue
        pairs.append((os.path.join(directory, fname), lbl))
    return pairs


def load_reference_pixels(ref_paths, n_per_image=10_000, verbose=True):
    bg_pool, fg_pool = [], []
    for img_path, lbl_path in ref_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            if verbose:
                print(f"  [warn] cannot load {img_path}, skipping", file=sys.stderr)
            continue
        img  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = make_fg_mask(img.shape, parse_yolo(lbl_path, w, h))
        bg_pool.append(sample_pixels(img, ~mask, n_per_image))
        fg_pool.append(sample_pixels(img,  mask, n_per_image))
        if verbose:
            print(f"  {os.path.basename(img_path)}: "
                  f"{mask.sum():,} fg px, {(~mask).sum():,} bg px")
    return (np.vstack(bg_pool) if bg_pool else np.empty((0, 3)),
            np.vstack(fg_pool) if fg_pool else np.empty((0, 3)))


# ── ColorTransferModel ────────────────────────────────────────────────────────

class ColorTransferModel:
    """
    Encapsulates the wild-reference color statistics and transform parameters.
    Build once from a reference directory, then call transform_image() for
    each lab image — the reference pixel pool is never reloaded between images.

    Parameters
    ----------
    ref_bg_rgb, ref_fg_rgb : np.ndarray (N, 3)
        Pooled background and foreground pixel samples from wild reference
        images (float, RGB in [0, 255]).
    representative_ref : np.ndarray (H, W, 3) or None
        One reference image used for comparison strips.
    n_bg_clusters : int
        KMeans clusters for background anchors (default 4).
    n_fg_clusters : int
        KMeans clusters for foreground/pollen anchors (default 2).
    kernel : str
        RBF kernel (default 'gaussian').
    epsilon : float
        RBF epsilon, for kernels that require it (default 50.0).
    l_strength : float
        How much RBF controls the L channel [0–1].
        0 = pure affine (least contrast amplification), 1 = full RBF.
    """

    def __init__(self, ref_bg_rgb, ref_fg_rgb, representative_ref=None,
                 n_bg_clusters=4, n_fg_clusters=2,
                 kernel='gaussian', epsilon=50.0, l_strength=0.0):
        self.ref_bg_rgb       = ref_bg_rgb
        self.ref_fg_rgb       = ref_fg_rgb
        self.representative_ref = representative_ref
        self.n_bg_clusters    = n_bg_clusters
        self.n_fg_clusters    = n_fg_clusters
        self.kernel           = kernel
        self.epsilon          = epsilon
        self.l_strength       = l_strength

        # Pre-convert reference pools to LAB once
        self._ref_bg_lab = rgb_pixels_to_lab(ref_bg_rgb)
        self._ref_fg_lab = rgb_pixels_to_lab(ref_fg_rgb)

        # Pre-cluster reference side once (same for every source image)
        self._ref_bg_c = cluster_centers_sorted(self._ref_bg_lab, n_bg_clusters)
        self._ref_fg_c = cluster_centers_sorted(self._ref_fg_lab, n_fg_clusters)

    @classmethod
    def from_reference_dir(cls, directory, verbose=True, **kwargs):
        """Build a model by pooling all annotated images in a directory."""
        pairs = ref_paths_from_dir(directory)
        if not pairs:
            raise ValueError(f"No image+label pairs found in {directory}")
        if verbose:
            print(f"Loading {len(pairs)} reference image(s) from {directory}:")
        ref_bg, ref_fg = load_reference_pixels(pairs, verbose=verbose)
        rep = cv2.cvtColor(cv2.imread(pairs[0][0]), cv2.COLOR_BGR2RGB)
        if verbose:
            print(f"Reference pool — bg: {len(ref_bg):,} px  fg: {len(ref_fg):,} px\n")
        return cls(ref_bg, ref_fg, representative_ref=rep, **kwargs)

    @classmethod
    def from_reference_image(cls, img_path, label_path, verbose=True, **kwargs):
        """Build a model from a single reference image."""
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        img  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = make_fg_mask(img.shape, parse_yolo(label_path, w, h))
        ref_bg = sample_pixels(img, ~mask, 10_000)
        ref_fg = sample_pixels(img,  mask, 10_000)
        return cls(ref_bg, ref_fg, representative_ref=img, **kwargs)

    # ── core per-image transform ──────────────────────────────────────────────

    def transform(self, src_img, src_fg_mask):
        """
        Apply the color transfer to a single image array.

        Parameters
        ----------
        src_img : np.ndarray (H, W, 3) uint8 RGB
        src_fg_mask : np.ndarray (H, W) bool

        Returns
        -------
        np.ndarray (H, W, 3) uint8 RGB
        """
        src_lab = rgb_to_lab(src_img)
        n = 10_000

        src_bg_px = sample_pixels(src_lab, ~src_fg_mask, n)
        src_fg_px = sample_pixels(src_lab,  src_fg_mask, n)

        # Source-side clusters (change per image)
        src_bg_c = cluster_centers_sorted(src_bg_px, self.n_bg_clusters)
        src_fg_c = cluster_centers_sorted(src_fg_px, self.n_fg_clusters)

        src_pts = np.vstack([src_bg_c, src_fg_c,
                             [[0., 0., 0.], [100., 0., 0.]]])
        tgt_pts = np.vstack([self._ref_bg_c, self._ref_fg_c,
                             [[0., 0., 0.], [100., 0., 0.]]])

        # Fit RBF
        needs_eps = self.kernel in ('multiquadric', 'gaussian',
                                    'inverse_multiquadric', 'inverse_quadratic')
        rbf = (RBFInterpolator(src_pts, tgt_pts, kernel=self.kernel,
                               degree=1, epsilon=self.epsilon)
               if needs_eps else
               RBFInterpolator(src_pts, tgt_pts, kernel=self.kernel, degree=1))

        h, w   = src_lab.shape[:2]
        pixels = src_lab.reshape(-1, 3)
        rbf_out = rbf(pixels.astype(float))

        # Global affine L (bg and fg weighted equally regardless of pixel count)
        n_bg = max(len(src_bg_px), 1)
        n_fg = max(len(src_fg_px), 1)
        w_fg = n_bg / n_fg
        all_src_L = np.concatenate([src_bg_px[:, 0],
                                    np.repeat(src_fg_px[:, 0], int(round(w_fg)))])
        all_ref_L = np.concatenate([self._ref_bg_lab[:, 0],
                                    np.repeat(self._ref_fg_lab[:, 0], int(round(w_fg)))])
        affine_L = affine_channel(pixels[:, 0],
                                  all_src_L.mean(), all_src_L.std(),
                                  all_ref_L.mean(), all_ref_L.std())

        out_lab = rbf_out.copy()
        out_lab[:, 0] = (self.l_strength * rbf_out[:, 0]
                         + (1.0 - self.l_strength) * affine_L)
        out_lab = np.clip(out_lab, [0, -128, -128], [100, 127, 127])
        return lab_to_rgb(out_lab.reshape(h, w, 3).astype(np.float32))

    # ── file-level convenience ────────────────────────────────────────────────

    def transform_image(self, src_path, label_path, out_dir,
                        comparison=False, verbose=True):
        """
        Load src_path, apply transform, write result to out_dir.
        Copies the label file unchanged.  Returns path to output image.
        """
        bgr = cv2.imread(src_path)
        if bgr is None:
            raise FileNotFoundError(src_path)
        src_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w    = src_img.shape[:2]
        mask    = make_fg_mask(src_img.shape, parse_yolo(label_path, w, h))

        transformed = self.transform(src_img, mask)

        os.makedirs(out_dir, exist_ok=True)
        img_out = os.path.join(out_dir, os.path.basename(src_path))
        lbl_out = os.path.join(out_dir, os.path.basename(label_path))
        cv2.imwrite(img_out, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
        shutil.copy(label_path, lbl_out)

        if verbose:
            print(f"  {os.path.basename(src_path)} → {img_out}")

        if comparison and self.representative_ref is not None:
            stem     = os.path.splitext(os.path.basename(src_path))[0]
            cmp_path = os.path.join(out_dir, f"{stem}_comparison.jpg")
            _save_comparison(src_img, transformed, self.representative_ref, cmp_path)

        return img_out

    def transform_dir(self, src_dir, out_dir, comparison=False, verbose=True):
        """
        Batch-process all annotated images in src_dir.
        Returns (n_ok, n_skipped).
        """
        pairs = ref_paths_from_dir(src_dir)   # same helper — finds img+txt pairs
        if not pairs:
            raise ValueError(f"No image+label pairs found in {src_dir}")

        n_ok, n_skip = 0, 0
        if verbose:
            print(f"Processing {len(pairs)} image(s) from {src_dir}:")
        for img_path, lbl_path in pairs:
            try:
                self.transform_image(img_path, lbl_path, out_dir,
                                     comparison=comparison, verbose=verbose)
                n_ok += 1
            except Exception as exc:
                print(f"  [error] {os.path.basename(img_path)}: {exc}",
                      file=sys.stderr)
                n_skip += 1

        if verbose:
            print(f"\nDone — {n_ok} transformed, {n_skip} skipped.")
        return n_ok, n_skip


# ── QC strip ──────────────────────────────────────────────────────────────────

def _save_comparison(src_rgb, transformed_rgb, ref_rgb, out_path):
    target_h = 480

    def fit_h(img):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h),
                          interpolation=cv2.INTER_AREA)

    panels = []
    for img_rgb, label in [(src_rgb, 'Lab (source)'),
                            (transformed_rgb, 'Transformed'),
                            (ref_rgb, 'Wild (target)')]:
        panel = cv2.cvtColor(fit_h(img_rgb), cv2.COLOR_RGB2BGR)
        for color, thickness in [((255, 255, 255), 4), ((0, 0, 0), 1)]:
            cv2.putText(panel, label, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness,
                        cv2.LINE_AA)
        panels.append(panel)

    cv2.imwrite(out_path, np.hstack(panels))
    print(f"  comparison → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description='RBF color domain transfer: lab microscope → wild microscope style',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--source',      help='Single lab image to transform')
    src.add_argument('--source_dir',  help='Directory of lab images to batch-process')

    p.add_argument('--source_label',
                   help='YOLO .txt for --source (required when using --source)')

    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument('--reference_dir',   help='Directory of wild reference images + labels')
    ref.add_argument('--reference',       help='Single wild reference image')

    p.add_argument('--reference_label',
                   help='YOLO .txt for --reference (required when using --reference)')

    p.add_argument('--output_dir',    default='./transferred')
    p.add_argument('--n_bg_clusters', type=int,   default=4)
    p.add_argument('--n_fg_clusters', type=int,   default=2)
    p.add_argument('--kernel',        default='gaussian',
                   choices=['thin_plate_spline', 'multiquadric', 'gaussian',
                            'inverse_multiquadric', 'inverse_quadratic',
                            'linear', 'cubic', 'quintic'])
    p.add_argument('--epsilon',       type=float, default=50.0)
    p.add_argument('--l_strength',    type=float, default=0.0,
                   help='RBF control over L channel [0–1]. 0=affine only.')
    p.add_argument('--comparison',    action='store_true',
                   help='Save side-by-side QC image for each output')
    return p


def main():
    args = _build_parser().parse_args()

    # Validate argument combinations
    if args.source and not args.source_label:
        _build_parser().error("--source requires --source_label")
    if args.reference and not args.reference_label:
        _build_parser().error("--reference requires --reference_label")

    # Build model (loads + pools reference images once)
    if args.reference_dir:
        model = ColorTransferModel.from_reference_dir(
            args.reference_dir,
            n_bg_clusters=args.n_bg_clusters,
            n_fg_clusters=args.n_fg_clusters,
            kernel=args.kernel,
            epsilon=args.epsilon,
            l_strength=args.l_strength,
        )
    else:
        model = ColorTransferModel.from_reference_image(
            args.reference, args.reference_label,
            n_bg_clusters=args.n_bg_clusters,
            n_fg_clusters=args.n_fg_clusters,
            kernel=args.kernel,
            epsilon=args.epsilon,
            l_strength=args.l_strength,
        )

    # Run
    if args.source_dir:
        model.transform_dir(args.source_dir, args.output_dir,
                            comparison=args.comparison)
    else:
        model.transform_image(args.source, args.source_label, args.output_dir,
                              comparison=args.comparison)


if __name__ == '__main__':
    main()
