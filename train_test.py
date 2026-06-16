#!/usr/bin/env python3
"""Submit a manifest-driven training/eval job through sbatch."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sbatch-script",
        default=env_or_default("SBATCH_SCRIPT", "submit_train_test.sbatch"),
        help="SBATCH script to submit.",
    )
    parser.add_argument("--code-dir", default=env_or_default("CODE_DIR", "src"))
    parser.add_argument("--manifest", default=env_or_default("MANIFEST", "pollen_labwild_test/manifest.csv"))
    parser.add_argument("--image-root", default=env_or_default("IMAGE_ROOT", "pollen_labwild_test"))
    parser.add_argument("--train-domains", default=env_or_default("TRAIN_DOMAINS", "wild"))
    parser.add_argument("--test-domains", default=env_or_default("TEST_DOMAINS", "wild"))
    parser.add_argument("--run-comment", default=env_or_default("RUN_COMMENT", "wild__wild"))
    parser.add_argument("--output-dir", default=env_or_default("OUTPUT_DIR", "runs"))
    parser.add_argument("--arch", default=env_or_default("ARCH", env_or_default("MODEL_NAME", "resnet18")))
    parser.add_argument("--epochs", default=env_or_default("EPOCHS", "100"))
    parser.add_argument("--batch-size", default=env_or_default("BATCH_SIZE", "16"))
    parser.add_argument("--num-workers", default=env_or_default("NUM_WORKERS", "2"))
    parser.add_argument("--lr", default=env_or_default("LR", env_or_default("LEARNING_RATE", "0.0001")))
    parser.add_argument("--image-size", default=env_or_default("IMAGE_SIZE", "224"))
    parser.add_argument("--val-fraction", default=env_or_default("VAL_FRACTION", "0.15"))
    parser.add_argument("--max-per-class", default=env_or_default("MAX_PER_CLASS", "0"))
    parser.add_argument("--partition", default=env_or_default("SBATCH_PARTITION", "gpu_long"))
    parser.add_argument("--gres", default=env_or_default("SBATCH_GRES", "gpu:1"))
    parser.add_argument("--cpus-per-task", default=env_or_default("SBATCH_CPUS_PER_TASK", "8"))
    parser.add_argument("--mem", default=env_or_default("SBATCH_MEM", "100G"))
    parser.add_argument("--time", default=env_or_default("SBATCH_TIME", "10:00:00"))
    parser.add_argument("--print-only", action="store_true", help="Print the sbatch command instead of running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sbatch_script = Path(args.sbatch_script).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    code_dir = Path(args.code_dir).expanduser().resolve()
    submit_dir = sbatch_script.parent
    logs_dir = submit_dir / "logs"

    missing_paths = [str(path) for path in (sbatch_script, manifest_path, image_root, code_dir) if not path.exists()]
    if missing_paths and not args.print_only:
        print(f"[error] missing required paths: {missing_paths}", file=sys.stderr, flush=True)
        return 1
    if missing_paths:
        print(f"[warn] missing paths (print-only mode): {missing_paths}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "CODE_DIR": str(code_dir),
            "MANIFEST": str(manifest_path),
            "IMAGE_ROOT": str(image_root),
            "TRAIN_DOMAINS": args.train_domains,
            "TEST_DOMAINS": args.test_domains,
            "RUN_COMMENT": args.run_comment,
            "OUTPUT_DIR": args.output_dir,
            "ARCH": args.arch,
            "EPOCHS": args.epochs,
            "BATCH_SIZE": args.batch_size,
            "NUM_WORKERS": args.num_workers,
            "LR": args.lr,
            "IMAGE_SIZE": args.image_size,
            "VAL_FRACTION": args.val_fraction,
            "MAX_PER_CLASS": args.max_per_class,
        }
    )

    cmd = [
        "sbatch",
        "--partition",
        args.partition,
        "--gres",
        args.gres,
        "--cpus-per-task",
        args.cpus_per_task,
        "--mem",
        args.mem,
        "--time",
        args.time,
        str(sbatch_script),
    ]
    print("[submit] " + " ".join(cmd), flush=True)
    print(
        f"[config] train={args.train_domains} test={args.test_domains} "
        f"manifest={manifest_path} image_root={image_root} arch={args.arch}",
        flush=True,
    )

    if args.print_only:
        return 0

    logs_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
