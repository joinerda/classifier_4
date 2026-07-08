#!/bin/bash
#SBATCH --job-name=imgclf-color-train
#SBATCH --partition=gpu_long
##SBATCH --nodelist=gn01
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
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

# --- dataset / experiment defaults ---
DATASET_ROOT="${DATASET_ROOT:-DATASET_TEST/lab_wild_combined_feb_26_labcolorscaled}"
MANIFEST="${MANIFEST:-$DATASET_ROOT/manifest.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATASET_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_color}"
MAX_PER_CLASS="${MAX_PER_CLASS:-400}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"

# --- training defaults ---
SEED="${SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-0}"
ARCH="${ARCH:-resnet34}"
PRETRAINED="${PRETRAINED:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VERTICAL_FLIP="${VERTICAL_FLIP:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EVAL_REPORT="${EVAL_REPORT:-0}"   # train-only by default; use slurm_color_eval.sbatch for held-out evals
EVAL_CONFUSION="${EVAL_CONFUSION:-1}"
EVAL_PER_CLASS="${EVAL_PER_CLASS:-1}"

# DANN support remains available but off by default.
USE_DANN="${USE_DANN:-0}"
DANN_LAMBDA_MAX="${DANN_LAMBDA_MAX:-0.01}"
DANN_BURNIN_EPOCHS="${DANN_BURNIN_EPOCHS:-0}"
MIN_EPOCHS="${MIN_EPOCHS:-$DANN_BURNIN_EPOCHS}"

# augmentation defaults
AUG_CROP_SCALE_MIN="${AUG_CROP_SCALE_MIN:-0.7}"
AUG_ROTATION="${AUG_ROTATION:-12.0}"
AUG_BRIGHTNESS="${AUG_BRIGHTNESS:-0.5}"
AUG_CONTRAST="${AUG_CONTRAST:-0.5}"
AUG_SATURATION="${AUG_SATURATION:-0.5}"
AUG_HUE="${AUG_HUE:-0.15}"
AUG_ERASING_PROB="${AUG_ERASING_PROB:-0.2}"
AUG_ERASING_SCALE_MAX="${AUG_ERASING_SCALE_MAX:-0.2}"
AUG_BLUR_PROB="${AUG_BLUR_PROB:-0.3}"
AUG_GRAYSCALE_PROB="${AUG_GRAYSCALE_PROB:-0.1}"
AUG_PERSPECTIVE_PROB="${AUG_PERSPECTIVE_PROB:-0.2}"

export DATASET_ROOT MANIFEST IMAGE_ROOT OUTPUT_DIR MAX_PER_CLASS VAL_FRACTION \
  SEED DETERMINISTIC ARCH PRETRAINED EPOCHS BATCH_SIZE LR WEIGHT_DECAY OPTIMIZER \
  NUM_WORKERS VERTICAL_FLIP EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA IMAGE_SIZE \
  EVAL_REPORT EVAL_CONFUSION EVAL_PER_CLASS USE_DANN DANN_LAMBDA_MAX \
  DANN_BURNIN_EPOCHS MIN_EPOCHS AUG_CROP_SCALE_MIN AUG_ROTATION AUG_BRIGHTNESS \
  AUG_CONTRAST AUG_SATURATION AUG_HUE AUG_ERASING_PROB AUG_ERASING_SCALE_MAX \
  AUG_BLUR_PROB AUG_GRAYSCALE_PROB AUG_PERSPECTIVE_PROB

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
  echo "[ssl] using cert bundle: $CERT"
fi

python -u - <<'PY'
import os
import subprocess
import sys

scenarios = [
    ("color_labscaled_combined_train", "lab,wild"),
    ("color_labscaled_lab_train", "lab"),
    ("color_labscaled_wild_train", "wild"),
]

base_cmd = [
    sys.executable, "-u", "-m", "image_classifier.cli", "train-manifest",
    "--manifest", os.environ["MANIFEST"],
    "--image-root", os.environ["IMAGE_ROOT"],
    "--output-dir", os.environ["OUTPUT_DIR"],
    "--arch", os.environ["ARCH"],
    "--epochs", os.environ["EPOCHS"],
    "--batch-size", os.environ["BATCH_SIZE"],
    "--lr", os.environ["LR"],
    "--weight-decay", os.environ["WEIGHT_DECAY"],
    "--optimizer", os.environ["OPTIMIZER"],
    "--num-workers", os.environ["NUM_WORKERS"],
    "--val-fraction", os.environ["VAL_FRACTION"],
    "--max-per-class", os.environ["MAX_PER_CLASS"],
    "--seed", os.environ["SEED"],
    "--image-size", os.environ["IMAGE_SIZE"],
]

if os.environ["PRETRAINED"] == "0":
    base_cmd.append("--no-pretrained")
if os.environ["VERTICAL_FLIP"] == "1":
    base_cmd.append("--vertical-flip")
else:
    base_cmd.append("--no-vertical-flip")
if os.environ["DETERMINISTIC"] == "1":
    base_cmd.append("--deterministic")
if int(os.environ["EARLY_STOP_PATIENCE"]) > 0:
    base_cmd.extend([
        "--early-stopping-patience", os.environ["EARLY_STOP_PATIENCE"],
        "--early-stopping-min-delta", os.environ["EARLY_STOP_MIN_DELTA"],
    ])
if os.environ["USE_DANN"] == "1":
    base_cmd.extend([
        "--use-dann",
        "--dann-lambda-max", os.environ["DANN_LAMBDA_MAX"],
    ])
    if int(os.environ["MIN_EPOCHS"]) > 0:
        base_cmd.extend(["--min-epochs", os.environ["MIN_EPOCHS"]])
    if int(os.environ["DANN_BURNIN_EPOCHS"]) > 0:
        base_cmd.extend(["--dann-burnin-epochs", os.environ["DANN_BURNIN_EPOCHS"]])
if os.environ["EVAL_REPORT"] == "1":
    base_cmd.append("--eval-report")
    if os.environ["EVAL_CONFUSION"] == "1":
        base_cmd.append("--eval-confusion-matrix")
    if os.environ["EVAL_PER_CLASS"] == "1":
        base_cmd.append("--eval-per-class")
else:
    base_cmd.append("--no-eval-report")
base_cmd.extend([
    "--aug-crop-scale-min", os.environ["AUG_CROP_SCALE_MIN"],
    "--aug-rotation", os.environ["AUG_ROTATION"],
    "--aug-brightness", os.environ["AUG_BRIGHTNESS"],
    "--aug-contrast", os.environ["AUG_CONTRAST"],
    "--aug-saturation", os.environ["AUG_SATURATION"],
    "--aug-hue", os.environ["AUG_HUE"],
    "--aug-erasing-prob", os.environ["AUG_ERASING_PROB"],
    "--aug-erasing-scale-max", os.environ["AUG_ERASING_SCALE_MAX"],
    "--aug-blur-prob", os.environ["AUG_BLUR_PROB"],
    "--aug-grayscale-prob", os.environ["AUG_GRAYSCALE_PROB"],
    "--aug-perspective-prob", os.environ["AUG_PERSPECTIVE_PROB"],
])

for run_comment, train_domains in scenarios:
    cmd = list(base_cmd)
    cmd.extend([
        "--run-comment", run_comment,
        "--train-domains", train_domains,
        "--test-domains", "wild",
    ])
    print(f"[train] scenario={run_comment} train_domains={train_domains}")
    print("[train] command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
PY
EOF
)

apptainer exec --nv \
  --bind "$WORKDIR:/workspace" \
  "$IMAGE" \
  bash -lc "$CMD"
