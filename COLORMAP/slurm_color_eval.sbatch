#!/bin/bash
#SBATCH --job-name=imgclf-color-eval
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load APPTAINER

cd "$SLURM_SUBMIT_DIR"

set -euo pipefail

mkdir -p logs

IMAGE="docker://pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime"
WORKDIR="$PWD"
APPTAINER_USER_TMP="/tmp/apptainer-${USER:-${LOGNAME:-unknown}}"
APPTAINER_USER_CACHE="$APPTAINER_USER_TMP/cache"

mkdir -p "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE"
chmod 700 "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_USER_TMP"
export APPTAINER_CACHEDIR="$APPTAINER_USER_CACHE"

DATASET_ROOT="${DATASET_ROOT:-DATASET_TEST/lab_wild_combined_feb_26_labcolorscaled}"
MANIFEST="${MANIFEST:-$DATASET_ROOT/manifest.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATASET_ROOT}"
RUNS_DIR="${RUNS_DIR:-runs_color}"
REPORT_DIR="${REPORT_DIR:-eval_reports_color}"
SEED="${SEED:-1337}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PER_CLASS="${MAX_PER_CLASS:-400}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
LABELS_PATH="${LABELS_PATH:-labels.txt}"
COLOR_COMBINED_CHECKPOINT="${COLOR_COMBINED_CHECKPOINT:-}"
COLOR_LAB_CHECKPOINT="${COLOR_LAB_CHECKPOINT:-}"
COLOR_WILD_CHECKPOINT="${COLOR_WILD_CHECKPOINT:-}"

export DATASET_ROOT MANIFEST IMAGE_ROOT RUNS_DIR REPORT_DIR SEED BATCH_SIZE NUM_WORKERS \
  MAX_PER_CLASS VAL_FRACTION LABELS_PATH COLOR_COMBINED_CHECKPOINT \
  COLOR_LAB_CHECKPOINT COLOR_WILD_CHECKPOINT

CMD=$(cat <<'EOF'
python -m pip -q install --user certifi pandas

cd /workspace
export PYTHONUNBUFFERED=1

CERT=""
CERT="$(python - <<'PY'
import sys
try:
    import certifi
    print(certifi.where())
except Exception as exc:
    print("", end="")
    print(f"[warn] certifi not available inside container: {exc}", file=sys.stderr)
PY
)"

if [[ -n "$CERT" ]]; then
  export SSL_CERT_FILE="$CERT"
  export REQUESTS_CA_BUNDLE="$CERT"
fi

python -u - <<'PY'
import json
import os
from pathlib import Path

import torch
from torch import nn

from image_classifier.eval import _init_confusion, _load_labels, _update_confusion
from image_classifier.manifest import build_datasets_from_manifest
from image_classifier.model import get_device, load_checkpoint
from image_classifier.utils import save_json


RUN_COMMENTS = [
    "color_labscaled_combined_train",
    "color_labscaled_lab_train",
    "color_labscaled_wild_train",
]
EVAL_DOMAINS = ["lab", "wild"]
OVERRIDE_ENV = {
    "color_labscaled_combined_train": "COLOR_COMBINED_CHECKPOINT",
    "color_labscaled_lab_train": "COLOR_LAB_CHECKPOINT",
    "color_labscaled_wild_train": "COLOR_WILD_CHECKPOINT",
}


def find_latest_checkpoint(run_comment: str, runs_dir: Path) -> Path:
    override = os.environ.get(OVERRIDE_ENV[run_comment], "").strip()
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path
    latest = None
    for run_dir in sorted(runs_dir.glob("run-*")):
        history_path = run_dir / "history.json"
        best_path = run_dir / "best.pt"
        if not history_path.exists() or not best_path.exists():
            continue
        history = json.loads(history_path.read_text())
        if history.get("run_comment") == run_comment:
            latest = best_path
    return latest


def evaluate_manifest_checkpoint(checkpoint_path: Path, eval_domain: str, report_path: Path) -> dict:
    device = get_device(None)
    model, class_to_idx, meta = load_checkpoint(str(checkpoint_path), arch=None, device=device)
    image_size = int(meta.get("image_size", 224))

    train_ds, val_ds, test_ds = build_datasets_from_manifest(
        manifest_path=os.environ["MANIFEST"],
        image_root=os.environ["IMAGE_ROOT"],
        train_domains=["lab"],
        test_domains=[eval_domain],
        val_fraction=float(os.environ["VAL_FRACTION"]),
        test_fraction=None,
        max_per_class=int(os.environ["MAX_PER_CLASS"]),
        seed=int(os.environ["SEED"]),
        vertical_flip=False,
        image_size=image_size,
        aug_kwargs={},
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(os.environ["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(os.environ["NUM_WORKERS"]),
    )

    model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = _load_labels(os.environ.get("LABELS_PATH"))
    class_mapping = [
        {"idx": idx, "class": name, "name": labels.get(name, name)}
        for idx, name in sorted(idx_to_class.items())
    ]
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
            _update_confusion(confusion, targets.cpu(), preds.cpu())
            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(1, len(test_loader))
    test_acc = total_correct / max(1, total)

    per_class = []
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

    report = {
        "checkpoint": str(checkpoint_path),
        "eval_domain": eval_domain,
        "test_loss": avg_loss,
        "test_acc": test_acc,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "class_mapping": class_mapping,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report)
    return report


runs_dir = Path(os.environ["RUNS_DIR"])
report_dir = Path(os.environ["REPORT_DIR"])
report_dir.mkdir(parents=True, exist_ok=True)

for run_comment in RUN_COMMENTS:
    checkpoint = find_latest_checkpoint(run_comment, runs_dir)
    if checkpoint is None:
        print(f"[skip] no checkpoint yet for run_comment={run_comment}")
        continue
    print(f"[eval] run_comment={run_comment} checkpoint={checkpoint}")
    for eval_domain in EVAL_DOMAINS:
        report_path = report_dir / f"{run_comment}__{eval_domain}_test.json"
        report = evaluate_manifest_checkpoint(checkpoint, eval_domain, report_path)
        print(
            f"[eval] {run_comment} on {eval_domain}: "
            f"test_acc={report['test_acc']:.4f} report={report_path}"
        )
PY
EOF
)

apptainer exec --nv \
  --bind "$WORKDIR:/workspace" \
  "$IMAGE" \
  bash -lc "$CMD"
