#!/usr/bin/env python3
"""Build the manifest for a train/test run and record the run's settings.

Resolves IMAGE_ROOT/MANIFEST/TRAIN_DOMAINS/TEST_DOMAINS/RUN_COMMENT, runs
build_manifest.py with them, then writes them as `export` lines to a file
that submit_train_test.sbatch sources automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD_MANIFEST = REPO_ROOT / "build_manifest.py"
COLORMAP_BUILDER = REPO_ROOT / "COLORMAP" / "make_colorscaled_dataset.py"
MANIFEST_COLORMAP_BUILDER = REPO_ROOT / "COLORMAP" / "colormap_from_manifest.py"
DEFAULT_COLOR_TRANSFER_MODULE = REPO_ROOT / "COLORMAP" / "color_transfer.py"


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", default=env_or_default("IMAGE_ROOT", "pollen_labwild_test"))
    parser.add_argument("--metadata", default=env_or_default("METADATA_PATH", ""))
    parser.add_argument("--manifest", default=env_or_default("MANIFEST", "pollen_labwild_test/manifest.csv"))
    parser.add_argument("--train-domains", default=env_or_default("TRAIN_DOMAINS", "wild"))
    parser.add_argument("--test-domains", default=env_or_default("TEST_DOMAINS", "wild"))
    parser.add_argument("--train-ratio", type=float, default=float(env_or_default("TRAIN_TARGET_RATIO", "0.4")))
    parser.add_argument("--val-ratio", type=float, default=float(env_or_default("VAL_TARGET_RATIO", "0.1")))
    parser.add_argument("--test-ratio", type=float, default=float(env_or_default("TEST_TARGET_RATIO", "0.5")))
    parser.add_argument("--seed", type=int, default=int(env_or_default("MANIFEST_SEED", "42")))
    parser.add_argument("--domain-field", default=env_or_default("DOMAIN_FIELD", "image_type"))
    parser.add_argument("--class-field", default=env_or_default("CLASS_FIELD", "species"))
    parser.add_argument("--path-field", default=env_or_default("PATH_FIELD", "filename"))
    parser.add_argument("--run-comment", default=env_or_default("RUN_COMMENT", ""))
    parser.add_argument(
        "--env-file",
        default=env_or_default("ENV_FILE", str(REPO_ROOT / "train_test_env.sh")),
        help="Path to the sourceable env file consumed by submit_train_test.sbatch.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip running build_manifest.py; only (re)write the env file.",
    )
    parser.add_argument(
        "--colormap-output-dir",
        default=env_or_default("COLORMAP_OUTPUT_DIR", ""),
        help="Optional derived image subdirectory for manifest-driven crop colormapping.",
    )
    parser.add_argument(
        "--colormap-manifest",
        default=env_or_default("COLORMAP_MANIFEST", ""),
        help="Optional derived manifest path for manifest-driven crop colormapping.",
    )
    parser.add_argument(
        "--colormap-reference-domain",
        default=env_or_default("COLORMAP_REFERENCE_DOMAIN", "wild"),
        help="Reference domain used to fit crop color statistics.",
    )
    parser.add_argument(
        "--colormap-reference-splits",
        default=env_or_default("COLORMAP_REFERENCE_SPLITS", "train"),
        help="Comma-separated reference splits used to fit crop color statistics.",
    )
    parser.add_argument(
        "--colormap-transform-domain",
        default=env_or_default("COLORMAP_TRANSFORM_DOMAIN", "lab"),
        help="Domain whose cropped images should be color-normalized.",
    )
    parser.add_argument(
        "--colormap-transform-splits",
        default=env_or_default("COLORMAP_TRANSFORM_SPLITS", "train"),
        help="Comma-separated splits whose cropped images should be color-normalized.",
    )
    parser.add_argument(
        "--colormap-max-pixels-per-image",
        type=int,
        default=int(env_or_default("COLORMAP_MAX_PIXELS_PER_IMAGE", "4096")),
        help="Maximum sampled pixels per image when fitting the wild reference distribution.",
    )
    parser.add_argument(
        "--colorscale-source-root",
        default=env_or_default("COLORSCALE_SOURCE_ROOT", ""),
        help="Optional source dataset root containing lab_set/ and wild_set/ to colorscale.",
    )
    parser.add_argument(
        "--colorscale-output-root",
        default=env_or_default("COLORSCALE_OUTPUT_ROOT", ""),
        help="Optional output dataset root to create for the colorscaled copy.",
    )
    parser.add_argument(
        "--color-transfer-module",
        default=env_or_default("COLOR_TRANSFER_MODULE", str(DEFAULT_COLOR_TRANSFER_MODULE)),
        help="Path to the reusable color_transfer.py module used by the optional colorscale workflow.",
    )
    parser.add_argument(
        "--group-by-magnification",
        action="store_true",
        default=True,
        help="Build separate wild reference pools for 100x vs 400x image names during colorscaling.",
    )
    parser.add_argument(
        "--no-group-by-magnification",
        dest="group_by_magnification",
        action="store_false",
        help="Use one global wild reference pool for all images during colorscaling.",
    )
    parser.add_argument("--n-bg-clusters", type=int, default=int(env_or_default("COLOR_N_BG_CLUSTERS", "4")))
    parser.add_argument("--n-fg-clusters", type=int, default=int(env_or_default("COLOR_N_FG_CLUSTERS", "2")))
    parser.add_argument(
        "--kernel",
        default=env_or_default("COLOR_KERNEL", "gaussian"),
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
        help="RBF kernel for optional colorscaling.",
    )
    parser.add_argument("--epsilon", type=float, default=float(env_or_default("COLOR_EPSILON", "50.0")))
    parser.add_argument("--l-strength", type=float, default=float(env_or_default("COLOR_L_STRENGTH", "0.0")))
    return parser.parse_args()


def _load_colorscale_builder():
    spec = importlib.util.spec_from_file_location("classifier4_colorscale_builder", COLORMAP_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load colorscale builder from {COLORMAP_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest_colormap_builder():
    spec = importlib.util.spec_from_file_location(
        "classifier4_manifest_colormap_builder",
        MANIFEST_COLORMAP_BUILDER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load manifest colormap builder from {MANIFEST_COLORMAP_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_manifest_build(
    *,
    args: argparse.Namespace,
    image_root: Path,
    manifest_path: Path,
    metadata_path: Path | None,
) -> int:
    build_cmd = [
        sys.executable,
        str(BUILD_MANIFEST),
        "--image-root", str(image_root),
        "--manifest", str(manifest_path),
        "--train-domains", args.train_domains,
        "--test-domains", args.test_domains,
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
        "--seed", str(args.seed),
        "--domain-field", args.domain_field,
        "--class-field", args.class_field,
        "--path-field", args.path_field,
    ]
    if metadata_path:
        build_cmd += ["--metadata", str(metadata_path)]

    print("[prepare] " + " ".join(build_cmd), flush=True)
    result = subprocess.run(build_cmd, check=False)
    if result.returncode != 0:
        print("[error] build_manifest.py failed; env file not written", file=sys.stderr, flush=True)
    return result.returncode


def _run_colorscale_build(
    *,
    args: argparse.Namespace,
    source_root: Path,
    output_root: Path,
    color_transfer_module: Path,
) -> Path:
    if not source_root.exists():
        raise FileNotFoundError(f"colorscale source root not found: {source_root}")
    if not COLORMAP_BUILDER.exists():
        raise FileNotFoundError(f"colorscale builder not found: {COLORMAP_BUILDER}")
    if not color_transfer_module.exists():
        raise FileNotFoundError(f"color transfer module not found: {color_transfer_module}")

    colorscale_builder = _load_colorscale_builder()
    print(
        "[prepare] "
        f"colorscale source={source_root} output={output_root} color_transfer_module={color_transfer_module}",
        flush=True,
    )
    manifest_path = colorscale_builder.build_colorscaled_dataset(
        source_root=source_root,
        output_root=output_root,
        color_transfer_module=color_transfer_module,
        group_by_magnification=args.group_by_magnification,
        n_bg_clusters=args.n_bg_clusters,
        n_fg_clusters=args.n_fg_clusters,
        kernel=args.kernel,
        epsilon=args.epsilon,
        l_strength=args.l_strength,
    )
    return Path(manifest_path).resolve()


def _default_colormap_manifest_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}__colormapped{manifest_path.suffix}")


def _default_colormap_output_dir(manifest_path: Path) -> str:
    return f"colormapped_{manifest_path.stem}"


def _run_manifest_colormap_build(
    *,
    args: argparse.Namespace,
    image_root: Path,
    manifest_path: Path,
    output_manifest_path: Path,
    output_dir: str,
) -> Path:
    if not MANIFEST_COLORMAP_BUILDER.exists():
        raise FileNotFoundError(f"manifest colormap builder not found: {MANIFEST_COLORMAP_BUILDER}")

    builder = _load_manifest_colormap_builder()
    print(
        "[prepare] "
        f"manifest colormap image_root={image_root} manifest={manifest_path} "
        f"output_manifest={output_manifest_path} output_dir={output_dir}",
        flush=True,
    )
    derived_manifest = builder.build_colormapped_manifest(
        image_root=image_root,
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        output_subdir=output_dir,
        reference_domain=args.colormap_reference_domain,
        reference_splits=[part.strip() for part in args.colormap_reference_splits.split(",") if part.strip()],
        transform_domain=args.colormap_transform_domain,
        transform_splits=[part.strip() for part in args.colormap_transform_splits.split(",") if part.strip()],
        max_pixels_per_image=args.colormap_max_pixels_per_image,
        seed=args.seed,
    )
    return Path(derived_manifest).resolve()


def main() -> int:
    args = parse_args()
    colorscale_enabled = bool(args.colorscale_source_root or args.colorscale_output_root)
    manifest_colormap_enabled = bool(args.colormap_output_dir or args.colormap_manifest)

    if colorscale_enabled and not (args.colorscale_source_root and args.colorscale_output_root):
        print(
            "[error] colorscale workflow requires both --colorscale-source-root and --colorscale-output-root",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if colorscale_enabled and manifest_colormap_enabled:
        print(
            "[error] choose either the structured colorscale workflow or the manifest-driven crop colormap workflow",
            file=sys.stderr,
            flush=True,
        )
        return 1

    if colorscale_enabled:
        source_root = Path(args.colorscale_source_root).expanduser().resolve()
        output_root = Path(args.colorscale_output_root).expanduser().resolve()
        color_transfer_module = Path(args.color_transfer_module).expanduser().resolve()
        image_root = output_root
        manifest_path = output_root / "manifest.csv"
        if not args.skip_build:
            try:
                manifest_path = _run_colorscale_build(
                    args=args,
                    source_root=source_root,
                    output_root=output_root,
                    color_transfer_module=color_transfer_module,
                )
            except Exception as exc:
                print(f"[error] colorscale build failed: {exc}", file=sys.stderr, flush=True)
                return 1
        elif not manifest_path.exists():
            print(
                f"[error] expected existing colorscaled manifest when using --skip-build: {manifest_path}",
                file=sys.stderr,
                flush=True,
            )
            return 1
    else:
        image_root = Path(args.image_root).expanduser().resolve()
        base_manifest_path = Path(args.manifest).expanduser().resolve()
        manifest_path = base_manifest_path
        metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else None
        if not args.skip_build:
            result = _run_manifest_build(
                args=args,
                image_root=image_root,
                manifest_path=base_manifest_path,
                metadata_path=metadata_path,
            )
            if result != 0:
                return result
        if manifest_colormap_enabled:
            output_manifest_path = (
                Path(args.colormap_manifest).expanduser().resolve()
                if args.colormap_manifest
                else _default_colormap_manifest_path(base_manifest_path)
            )
            output_dir = args.colormap_output_dir or _default_colormap_output_dir(base_manifest_path)
            if not args.skip_build:
                try:
                    manifest_path = _run_manifest_colormap_build(
                        args=args,
                        image_root=image_root,
                        manifest_path=base_manifest_path,
                        output_manifest_path=output_manifest_path,
                        output_dir=output_dir,
                    )
                except Exception as exc:
                    print(f"[error] manifest colormap build failed: {exc}", file=sys.stderr, flush=True)
                    return 1
            else:
                manifest_path = output_manifest_path
                if not manifest_path.exists():
                    print(
                        f"[error] expected existing colormapped manifest when using --skip-build: {manifest_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1

    default_run_comment = f"{args.train_domains}__{args.test_domains}"
    if colorscale_enabled or manifest_colormap_enabled:
        default_run_comment += "__colorscaled"
    run_comment = args.run_comment or default_run_comment

    env_vars = {
        "IMAGE_ROOT": str(image_root),
        "MANIFEST": str(manifest_path),
        "TRAIN_DOMAINS": args.train_domains,
        "TEST_DOMAINS": args.test_domains,
        "RUN_COMMENT": run_comment,
    }
    export_lines = [f"export {name}={shlex.quote(value)}" for name, value in env_vars.items()]
    # ":=" only fills unset/empty vars, so this file won't clobber values a caller
    # (e.g. train_test.py) already exported into the job's inherited environment.
    default_lines = [f': "${{{name}:={shlex.quote(value)}}}"' for name, value in env_vars.items()]

    print("\n[env] settings for submit_train_test.sbatch:", flush=True)
    for line in export_lines:
        print(line, flush=True)

    env_file = Path(args.env_file).expanduser().resolve()
    env_file.write_text(
        "# Generated by prepare_train_test.py -- sourced by submit_train_test.sbatch\n"
        + "\n".join(default_lines)
        + "\n"
    )
    print(f"\n[ready] wrote env file: {env_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
