#!/bin/bash
#SBATCH --job-name=imgclf
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PARTITION="${SBATCH_PARTITION:-gpu_long}"
GRES="${SBATCH_GRES:-gpu:1}"
CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-8}"
MEM="${SBATCH_MEM:-100G}"
TIME_LIMIT="${SBATCH_TIME:-10:00:00}"

if [[ "${SLURM_JOB_PARTITION:-}" != "$PARTITION" ]]; then
  echo "[note] requested partition via env: $PARTITION"
fi
if [[ "${SLURM_JOB_GRES:-}" != "$GRES" ]]; then
  echo "[note] requested gres via env: $GRES"
fi
echo "[note] sbatch header defaults remain in effect unless you override them with sbatch flags."
echo "[note] for strict resource overrides use: sbatch --partition=<p> --gres=<g> --cpus-per-task=<n> --mem=<m> --time=<t> submit_train_test.sbatch"

module purge
module load APPTAINER

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ENV_FILE_ARG="${1:-}"
if [[ -n "$ENV_FILE_ARG" ]]; then
  if [[ "$ENV_FILE_ARG" = /* ]]; then
    ENV_FILE="$ENV_FILE_ARG"
  else
    ENV_FILE="$SLURM_SUBMIT_DIR/$ENV_FILE_ARG"
  fi
else
  ENV_FILE="${ENV_FILE:-$SLURM_SUBMIT_DIR/train_test_env.sh}"
fi
if [[ -f "$ENV_FILE" ]]; then
  echo "[note] sourcing prepared env file: $ENV_FILE"
  source "$ENV_FILE"
fi

IMAGE="${APPTAINER_IMAGE:-docker://pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime}"
WORKDIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${CODE_DIR:-$WORKDIR/src}"
APPTAINER_USER_TMP="${APPTAINER_TMPDIR:-/tmp/apptainer-${USER:-${LOGNAME:-unknown}}}"
APPTAINER_USER_CACHE="${APPTAINER_CACHEDIR:-$APPTAINER_USER_TMP/cache}"

TRAIN_DOMAINS="${TRAIN_DOMAINS:-wild}"
TEST_DOMAINS="${TEST_DOMAINS:-wild}"
MANIFEST="${MANIFEST:-$WORKDIR/pollen_labwild_test/manifest.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$WORKDIR/pollen_labwild_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/runs}"
RUN_COMMENT="${RUN_COMMENT:-${TRAIN_DOMAINS}__${TEST_DOMAINS}}"

ARCH="${ARCH:-resnet18}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-0.0001}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
MAX_PER_CLASS="${MAX_PER_CLASS:-0}"
TEST_FRACTION="${TEST_FRACTION:-}"
PRETRAINED="${PRETRAINED:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-0}"
VERTICAL_FLIP="${VERTICAL_FLIP:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
LABELS_PATH="${LABELS_PATH:-}"

mkdir -p "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE" "$OUTPUT_DIR"
chmod 700 "$APPTAINER_USER_TMP" "$APPTAINER_USER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_USER_TMP"
export APPTAINER_CACHEDIR="$APPTAINER_USER_CACHE"
export CODE_DIR MANIFEST IMAGE_ROOT OUTPUT_DIR TRAIN_DOMAINS TEST_DOMAINS RUN_COMMENT
export ARCH EPOCHS BATCH_SIZE NUM_WORKERS LR IMAGE_SIZE VAL_FRACTION MAX_PER_CLASS TEST_FRACTION
export PRETRAINED WEIGHT_DECAY OPTIMIZER SEED DETERMINISTIC VERTICAL_FLIP
export EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA LABELS_PATH

CMD=$(cat <<'EOF'
python -m pip -q install --user certifi pandas
cd "$CODE_DIR"
export PYTHONUNBUFFERED=1

python -u - <<'PY'
import torch
print(f"[gpu-check] torch={torch.__version__}")
print(f"[gpu-check] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[gpu-check] device={torch.cuda.get_device_name(0)}")
PY

RUN_COMMENT_FLAG=()
if [[ -n "$RUN_COMMENT" ]]; then
  RUN_COMMENT_FLAG=(--run-comment "$RUN_COMMENT")
fi

TEST_FRACTION_FLAG=()
if [[ -n "$TEST_FRACTION" ]]; then
  TEST_FRACTION_FLAG=(--test-fraction "$TEST_FRACTION")
fi

PRETRAINED_FLAG=()
if [[ "$PRETRAINED" -eq 0 ]]; then
  PRETRAINED_FLAG=(--no-pretrained)
fi

VERTICAL_FLIP_FLAG=(--vertical-flip)
if [[ "$VERTICAL_FLIP" -eq 0 ]]; then
  VERTICAL_FLIP_FLAG=(--no-vertical-flip)
fi

EARLY_STOP_FLAGS=()
if [[ "$EARLY_STOP_PATIENCE" -gt 0 ]]; then
  EARLY_STOP_FLAGS=(
    --early-stopping-patience "$EARLY_STOP_PATIENCE"
    --early-stopping-min-delta "$EARLY_STOP_MIN_DELTA"
  )
fi

LABELS_FLAG=()
if [[ -n "$LABELS_PATH" ]]; then
  LABELS_FLAG=(--labels-path "$LABELS_PATH")
fi

DETERMINISTIC_FLAG=()
if [[ "$DETERMINISTIC" -eq 1 ]]; then
  DETERMINISTIC_FLAG=(--deterministic)
fi

MAX_PER_CLASS_FLAG=()
if [[ "$MAX_PER_CLASS" != "0" ]]; then
  MAX_PER_CLASS_FLAG=(--max-per-class "$MAX_PER_CLASS")
fi

python -u -m image_classifier.cli train-manifest \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --train-domains "$TRAIN_DOMAINS" \
  --test-domains "$TEST_DOMAINS" \
  --output-dir "$OUTPUT_DIR" \
  --arch "$ARCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --num-workers "$NUM_WORKERS" \
  --val-fraction "$VAL_FRACTION" \
  --image-size "$IMAGE_SIZE" \
  --seed "$SEED" \
  --eval-report \
  --eval-confusion-matrix \
  --eval-per-class \
  "${RUN_COMMENT_FLAG[@]}" \
  "${TEST_FRACTION_FLAG[@]}" \
  "${PRETRAINED_FLAG[@]}" \
  "${VERTICAL_FLIP_FLAG[@]}" \
  "${EARLY_STOP_FLAGS[@]}" \
  "${LABELS_FLAG[@]}" \
  "${DETERMINISTIC_FLAG[@]}" \
  "${MAX_PER_CLASS_FLAG[@]}"
EOF
)

echo "[config] code_dir=$CODE_DIR"
echo "[config] manifest=$MANIFEST"
echo "[config] image_root=$IMAGE_ROOT"
echo "[config] train_domains=$TRAIN_DOMAINS"
echo "[config] test_domains=$TEST_DOMAINS"
echo "[config] output_dir=$OUTPUT_DIR"

apptainer exec --nv \
  --bind "$WORKDIR:$WORKDIR" \
  "$IMAGE" \
  bash -lc "$CMD"
