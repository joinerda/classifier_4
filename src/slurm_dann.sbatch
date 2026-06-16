#!/bin/bash
#SBATCH --job-name=imgclf-dann
#SBATCH --partition=gpu_long          # ← set to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
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

# ── configure these per run ────────────────────────────────────────────────
# DANN requires exactly one source domain and one target domain (no comma lists)
SOURCE_DOMAIN="${SOURCE_DOMAIN:-domain_a}"       # domain to train classifier on
TARGET_DOMAIN="${TARGET_DOMAIN:-domain_b}"       # domain to adapt toward and evaluate on
MANIFEST="${MANIFEST:-data/manifest.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data}"
MAX_PER_CLASS="${MAX_PER_CLASS:-400}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
DANN_LAMBDA_MAX="${DANN_LAMBDA_MAX:-0.01}"       # domain loss weight ceiling
DANN_BURNIN_EPOCHS="${DANN_BURNIN_EPOCHS:-0}"    # epochs at lambda=0 before annealing starts
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
LABELS_PATH="${LABELS_PATH:-}"
# ──────────────────────────────────────────────────────────────────────────

mkdir -p "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE"
chmod 700 "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_USER_TMP"
export APPTAINER_CACHEDIR="$APPTAINER_USER_CACHE"

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
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MIN_EPOCHS="${MIN_EPOCHS:-$DANN_BURNIN_EPOCHS}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RUN_COMMENT="${RUN_COMMENT:-${SOURCE_DOMAIN}__${TARGET_DOMAIN}}"

# aggressive augmentation defaults for domain adaptation
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

EXTRA_FLAGS=()
[[ "$PRETRAINED" -eq 0 ]] && EXTRA_FLAGS+=(--no-pretrained)
if [[ "$VERTICAL_FLIP" -eq 0 ]]; then
  EXTRA_FLAGS+=(--no-vertical-flip)
else
  EXTRA_FLAGS+=(--vertical-flip)
fi
if [[ "$EARLY_STOP_PATIENCE" -gt 0 ]]; then
  EXTRA_FLAGS+=(--early-stopping-patience "$EARLY_STOP_PATIENCE")
  EXTRA_FLAGS+=(--early-stopping-min-delta "$EARLY_STOP_MIN_DELTA")
fi
EXTRA_FLAGS+=(--use-dann --dann-lambda-max "$DANN_LAMBDA_MAX")
[[ "$MIN_EPOCHS" -gt 0 ]] && EXTRA_FLAGS+=(--min-epochs "$MIN_EPOCHS")
[[ "$DANN_BURNIN_EPOCHS" -gt 0 ]] && EXTRA_FLAGS+=(--dann-burnin-epochs "$DANN_BURNIN_EPOCHS")
EXTRA_FLAGS+=(--eval-report --eval-confusion-matrix --eval-per-class)
[[ -n "$LABELS_PATH" ]] && EXTRA_FLAGS+=(--labels-path "$LABELS_PATH")
EXTRA_FLAGS+=(
  --aug-crop-scale-min "$AUG_CROP_SCALE_MIN"
  --aug-rotation "$AUG_ROTATION"
  --aug-brightness "$AUG_BRIGHTNESS"
  --aug-contrast "$AUG_CONTRAST"
  --aug-saturation "$AUG_SATURATION"
  --aug-hue "$AUG_HUE"
  --aug-erasing-prob "$AUG_ERASING_PROB"
  --aug-erasing-scale-max "$AUG_ERASING_SCALE_MAX"
  --aug-blur-prob "$AUG_BLUR_PROB"
  --aug-grayscale-prob "$AUG_GRAYSCALE_PROB"
  --aug-perspective-prob "$AUG_PERSPECTIVE_PROB"
)
EXTRA_FLAGS_STR="${EXTRA_FLAGS[*]}"

export SOURCE_DOMAIN TARGET_DOMAIN MANIFEST IMAGE_ROOT MAX_PER_CLASS VAL_FRACTION \
  DANN_LAMBDA_MAX DANN_BURNIN_EPOCHS MIN_EPOCHS OUTPUT_DIR ARCH EPOCHS BATCH_SIZE \
  LR WEIGHT_DECAY OPTIMIZER NUM_WORKERS SEED DETERMINISTIC IMAGE_SIZE RUN_COMMENT \
  EXTRA_FLAGS_STR

CMD=$(cat <<'EOF'

python -m pip -q install --user certifi pandas

cd /workspace
export PYTHONUNBUFFERED=1

CERT="$(python - <<'PY'
import sys
try:
    import certifi; print(certifi.where())
except Exception as e:
    print("", end="")
    print(f"[warn] certifi unavailable: {e}", file=sys.stderr)
PY
)"
[[ -n "$CERT" ]] && export SSL_CERT_FILE="$CERT" REQUESTS_CA_BUNDLE="$CERT" && echo "[ssl] $CERT"

python -u - <<'PY'
import torch
print(f"[gpu-check] torch={torch.__version__}")
print(f"[gpu-check] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[gpu-check] device={torch.cuda.get_device_name(0)}")
PY

RUN_COMMENT_FLAG=()
[[ -n "$RUN_COMMENT" ]] && RUN_COMMENT_FLAG=(--run-comment "$RUN_COMMENT")

python -u -m image_classifier.cli train-manifest \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --train-domains "$SOURCE_DOMAIN" \
  --test-domains "$TARGET_DOMAIN" \
  --output-dir "$OUTPUT_DIR" \
  --arch "$ARCH" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --weight-decay $WEIGHT_DECAY \
  --optimizer "$OPTIMIZER" \
  --num-workers $NUM_WORKERS \
  --val-fraction $VAL_FRACTION \
  --max-per-class $MAX_PER_CLASS \
  --seed $SEED \
  --image-size $IMAGE_SIZE \
  "${RUN_COMMENT_FLAG[@]}" \
  $EXTRA_FLAGS_STR \
  $([[ "$DETERMINISTIC" -eq 1 ]] && echo --deterministic)

EOF
)

apptainer exec --nv \
  --bind "$WORKDIR:/workspace" \
  "$IMAGE" \
  bash -lc "$CMD"
