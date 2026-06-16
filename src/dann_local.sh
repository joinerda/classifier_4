#!/bin/bash
# Local DANN training script — no Apptainer required.
# Run directly: bash dann_local.sh
# Or submit to SLURM without a container by adding #SBATCH headers and calling via sbatch.

set -euo pipefail

# ── optional: activate your Python environment ────────────────────────────
# conda activate myenv
# source /path/to/venv/bin/activate
# ─────────────────────────────────────────────────────────────────────────

# ── configure these per run ───────────────────────────────────────────────
SOURCE_DOMAIN="${SOURCE_DOMAIN:-domain_a}"
TARGET_DOMAIN="${TARGET_DOMAIN:-domain_b}"
MANIFEST="${MANIFEST:-data/manifest.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data}"
MAX_PER_CLASS="${MAX_PER_CLASS:-400}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
DANN_LAMBDA_MAX="${DANN_LAMBDA_MAX:-0.01}"
DANN_BURNIN_EPOCHS="${DANN_BURNIN_EPOCHS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
LABELS_PATH="${LABELS_PATH:-}"
# ─────────────────────────────────────────────────────────────────────────

SEED="${SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-0}"
ARCH="${ARCH:-resnet34}"
PRETRAINED="${PRETRAINED:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VERTICAL_FLIP="${VERTICAL_FLIP:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MIN_EPOCHS="${MIN_EPOCHS:-$DANN_BURNIN_EPOCHS}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RUN_COMMENT="${RUN_COMMENT:-${SOURCE_DOMAIN}__${TARGET_DOMAIN}}"
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

RUN_COMMENT_FLAG=()
[[ -n "$RUN_COMMENT" ]] && RUN_COMMENT_FLAG=(--run-comment "$RUN_COMMENT")

export PYTHONUNBUFFERED=1

python -u - <<'PY'
import torch
print(f"[gpu-check] torch={torch.__version__}")
print(f"[gpu-check] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[gpu-check] device={torch.cuda.get_device_name(0)}")
PY

python -u -m image_classifier.cli train-manifest \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --train-domains "$SOURCE_DOMAIN" \
  --test-domains "$TARGET_DOMAIN" \
  --output-dir "$OUTPUT_DIR" \
  --arch "$ARCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --num-workers "$NUM_WORKERS" \
  --val-fraction "$VAL_FRACTION" \
  --max-per-class "$MAX_PER_CLASS" \
  --seed "$SEED" \
  --image-size "$IMAGE_SIZE" \
  "${RUN_COMMENT_FLAG[@]}" \
  "${EXTRA_FLAGS[@]}" \
  $([[ "$DETERMINISTIC" -eq 1 ]] && echo --deterministic)
