# Image Classifier

ResNet-based image classifier with support for multi-domain datasets, manifest-driven
data pipelines, and optional domain adversarial training (DANN).

Designed to run on SLURM clusters via Apptainer (Singularity). No framework installation
required beyond the PyTorch container.

---

## Contents

```
image_classifier/        Python package — all training, eval, and prediction logic
slurm_train_test.sbatch  Supervised training job (Apptainer/SLURM)
slurm_dann.sbatch        DANN training job (Apptainer/SLURM)
train_local.sh           Supervised training script (local / no container)
dann_local.sh            DANN training script (local / no container)
requirements.txt         Python dependencies
labels.txt               Optional: class_id → human-readable name mapping
README.md                This file
```

---

## Environment

Two options depending on your setup.

### Option A — Local (no container)

Install Python dependencies, then run `train_local.sh` or `dann_local.sh` directly.

```bash
# Install torch first (see notes in requirements.txt for CUDA versions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install numpy pandas Pillow

# Run training
bash train_local.sh
```

The scripts have a commented-out section at the top for activating a conda or venv
environment — uncomment the appropriate line if needed.

### Option B — SLURM + Apptainer (no local install required)

- Requires Apptainer (formerly Singularity) and SLURM
- Container `docker://pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` is pulled
  automatically on first run and cached in `/tmp/apptainer-<user>/`
- `pandas` is installed inside the container at job start
- Adjust `#SBATCH --partition`, `--time`, `--mem`, `--cpus-per-task` for your cluster

```bash
mkdir -p logs
sbatch slurm_train_test.sbatch
```

---

## Data and Manifest Format

Training is driven entirely by a **manifest CSV file**. Images can live anywhere on
disk in any directory layout — the manifest is the only index the pipeline reads.

### manifest.csv columns

| column     | type    | description |
|------------|---------|-------------|
| `path`     | string  | Path to the image, **relative to `--image-root`**. Forward slashes. |
| `class_id` | integer | Class label. Any consistent integer scheme works. |
| `domain`   | string  | Arbitrary string identifying the image source (e.g. `lab`, `wild`, `site_a`). Used to select train/test subsets. |
| `split`    | string  | `"train"` or `"test"`. Controls which rows are eligible for training vs. held-out evaluation. |

Additional columns are ignored.

### Example rows

```
path,class_id,domain,split
images/site_a/class_3/img001.jpg,3,site_a,train
images/site_b/class_3/img002.jpg,3,site_b,train
images/site_a/class_3/img099.jpg,3,site_a,test
```

### Rules

- Paths must be relative (not absolute). They are joined with `--image-root` at runtime,
  so the manifest is portable across machines — only `--image-root` needs to change.
- All class IDs that will ever appear in any experiment must be present in the manifest.
  The pipeline builds a stable `class_id → index` mapping from
  `sorted(unique class_ids in the full manifest)`. If a class is missing from the
  manifest the index mapping shifts and checkpoints become incompatible.
- The `split` column is the canonical train/test boundary. The pipeline never uses
  `test` rows for training or validation.
- If your dataset has no pre-assigned train/test split, set `TEST_FRACTION` (e.g. `0.15`)
  in the SLURM script. The pipeline will then ignore the `split` column and carve a
  stratified test set randomly from all matching rows before training.

### Generating the manifest

Any tool that produces a CSV with the four columns above is sufficient. There is no
required directory structure. Write the file to a convenient path and point
`--manifest` at it.

---

## labels.txt (optional)

A plain-text file that maps integer class IDs to human-readable names. Only used for
display in eval reports — training works without it.

Format: one entry per line, `<class_id> <name>`.

```
1 Acer (Maple)
2 Alnus (Alder)
3 Betula (Birch)
```

Pass `--labels-path labels.txt` (or set `LABELS_PATH` in the SLURM script) to enable.
If omitted, eval reports show numeric class IDs.

---

## Quick Start

**1. Prepare your manifest**

Generate `manifest.csv` with the four required columns using whatever tool or query
is appropriate for your dataset. Place it somewhere accessible, e.g. `data/manifest.csv`.

**2. Configure and run**

Edit the "configure these per run" block near the top of either script.
Minimal example — single domain, in-domain test:

```bash
TRAIN_DOMAINS="site_a"
TEST_DOMAINS="site_a"
MANIFEST="data/manifest.csv"
IMAGE_ROOT="data"
MAX_PER_CLASS="200"       # reduce if your dataset is small
EPOCHS="50"               # reduce for a quick smoke test
```

**Local:**
```bash
bash train_local.sh
# or override inline:
TRAIN_DOMAINS=site_a EPOCHS=50 bash train_local.sh
```

**SLURM + Apptainer:**
```bash
mkdir -p logs
sbatch slurm_train_test.sbatch
tail -f logs/imgclf-<jobid>.out
```

**3. What to expect in the log**

The job stdout will show:

```
[ssl] using cert bundle: /root/.local/lib/python3.10/...
[gpu-check] torch=2.1.2
[gpu-check] cuda_available=True
[gpu-check] device=NVIDIA A100 80GB PCIe
[manifest] train=340 val=60 test=120 classes=8
[run-comment] site_a__site_a
[epoch  1/50] train_loss=2.041 train_acc=0.124 | val_loss=1.890 val_acc=0.201 | 14.2s
[epoch  2/50] train_loss=1.723 train_acc=0.312 | val_loss=1.512 val_acc=0.389 | 13.8s
...
[best] epoch=12 val_acc=0.847
[eval] test_acc=0.831 test_loss=0.521
{"test_acc": 0.831, "test_loss": 0.521, "eval_report": "runs/run-.../eval_report.json"}
```

If `cuda_available=False` appears, the job landed on a CPU node — check your
`#SBATCH --gres` and `--partition` settings.

**4. Check results**

Each run writes to `runs/run-<date>-<jobid>/`:

```
best.pt          PyTorch checkpoint (best validation accuracy)
history.json     Per-epoch train/val loss and accuracy, config, timing
metrics.json     Final test metrics summary
eval_report.json Full eval: test accuracy, per-class stats, confusion matrix
```

---

## Training Parameters

All parameters can be overridden by setting environment variables before `sbatch`,
e.g. `ARCH=resnet50 LR=1e-4 sbatch slurm_train_test.sbatch`.

### Data

| variable | default | description |
|---|---|---|
| `TRAIN_DOMAINS` | `domain_a` | Comma-separated domain(s) for training |
| `TEST_DOMAINS` | `domain_a` | Comma-separated domain(s) for test eval |
| `MANIFEST` | `data/manifest.csv` | Path to manifest CSV |
| `IMAGE_ROOT` | `data` | Root directory for resolving manifest paths |
| `MAX_PER_CLASS` | `400` | Max training images per class; 0 or unset = no cap |
| `VAL_FRACTION` | `0.15` | Fraction of train pool held out for validation |
| `TEST_FRACTION` | _(empty)_ | If set, ignores manifest `split` column and carves a random stratified test set of this fraction from all rows matching either domain. Useful when the manifest has no pre-assigned splits. |

### Model

| variable | default | description |
|---|---|---|
| `ARCH` | `resnet34` | Network: `resnet18`, `resnet34`, `resnet50` |
| `PRETRAINED` | `1` | `1` = initialize from ImageNet weights; `0` = random init |
| `EPOCHS` | `100` | Maximum training epochs |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `LR` | `3e-5` | Learning rate (AdamW; pretrained models typically need low values) |
| `WEIGHT_DECAY` | `1e-4` | L2 regularization |
| `OPTIMIZER` | `adamw` | `adamw` or `sgd` |
| `IMAGE_SIZE` | `224` | Input crop size in pixels |
| `EARLY_STOP_PATIENCE` | `20` | Stop after N epochs without val_acc improvement (monitors val accuracy, not loss) |
| `EARLY_STOP_MIN_DELTA` | `0.0001` | Minimum val_acc improvement to reset early stopping counter |
| `MIN_EPOCHS` | `0` | Do not allow early stopping before this epoch count |

### Run control

| variable | default | description |
|---|---|---|
| `SEED` | `1337` | Random seed for data shuffling and augmentation |
| `DETERMINISTIC` | `0` | `1` = force cuDNN deterministic mode (slower, fully reproducible) |
| `OUTPUT_DIR` | `runs` | Directory for run outputs |
| `NUM_WORKERS` | `8` | DataLoader worker processes for image loading |
| `RUN_COMMENT` | `<train>__<test>` | Freeform label stored in `history.json` for experiment tracking |
| `LABELS_PATH` | _(empty)_ | Optional path to `labels.txt` for human-readable names in eval reports |
| `EVAL_REPORT` | `1` | `1` = write `eval_report.json` after training |
| `EVAL_CONFUSION` | `1` | `1` = include confusion matrix in eval report |
| `EVAL_PER_CLASS` | `1` | `1` = include per-class accuracy breakdown in eval report |

### DANN (slurm_dann.sbatch only)

| variable | default | description |
|---|---|---|
| `USE_DANN` | `0` | `1` = enable domain adversarial training (slurm_train_test.sbatch only; slurm_dann.sbatch always uses DANN) |
| `DANN_LAMBDA_MAX` | `0.01` | Domain loss weight ceiling. Start low (0.01–0.1); too high destabilizes species classification. |
| `DANN_BURNIN_EPOCHS` | `0` | Epochs to train with lambda=0 before DANN annealing begins |

### Augmentation

The augmentation pipeline during training always includes:
- `RandomResizedCrop(IMAGE_SIZE, scale=(AUG_CROP_SCALE_MIN, 1.0))`
- `RandomHorizontalFlip` (always on, not configurable)
- `RandomVerticalFlip(p=0.1)` — enabled when `VERTICAL_FLIP=1`
- `ColorJitter` with the brightness/contrast/saturation/hue params below
- `Normalize` using ImageNet mean and std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)

The ImageNet normalization is fixed regardless of dataset. If your images have very
different statistics, consider preprocessing them to match or fine-tuning with
`PRETRAINED=0`.

| variable | default | description |
|---|---|---|
| `VERTICAL_FLIP` | `1` | `1` = include `RandomVerticalFlip(p=0.1)` in training transforms |
| `AUG_CROP_SCALE_MIN` | `0.7` | Minimum scale for random resized crop |
| `AUG_ROTATION` | `12.0` | Max rotation degrees |
| `AUG_BRIGHTNESS` | `0.2` | Color jitter brightness range |
| `AUG_CONTRAST` | `0.2` | Color jitter contrast range |
| `AUG_SATURATION` | `0.2` | Color jitter saturation range |
| `AUG_HUE` | `0.05` | Color jitter hue range |
| `AUG_ERASING_PROB` | `0.2` | Random erasing probability |
| `AUG_ERASING_SCALE_MAX` | `0.2` | Random erasing max patch fraction |
| `AUG_BLUR_PROB` | `0.0` | Gaussian blur probability (0 = off) |
| `AUG_GRAYSCALE_PROB` | `0.0` | Random grayscale probability (0 = off) |
| `AUG_PERSPECTIVE_PROB` | `0.0` | Random perspective warp probability (0 = off) |

For aggressive augmentation (domain adaptation scenarios), try:
`AUG_BRIGHTNESS=0.5 AUG_CONTRAST=0.5 AUG_SATURATION=0.5 AUG_HUE=0.15 AUG_BLUR_PROB=0.3 AUG_GRAYSCALE_PROB=0.1 AUG_PERSPECTIVE_PROB=0.2`

Inference (val and test) uses only center crop + ImageNet normalization — no augmentation.

---

## Output Structure

Each run creates a timestamped directory under `OUTPUT_DIR`:

```
runs/
└── run-20260604-143021-98765/
    ├── best.pt            ← checkpoint with best validation accuracy
    ├── history.json       ← full training log (see below)
    ├── metrics.json       ← final test metrics summary
    └── eval_report.json   ← detailed evaluation output
```

### history.json

```json
{
  "run_id": "98765",
  "config": { ... all training parameters ... },
  "epochs": [
    { "epoch": 1, "train_loss": 1.23, "train_acc": 0.45, "val_loss": 1.10, "val_acc": 0.51 },
    ...
  ],
  "best_epoch": 47,
  "best_val_acc": 0.823,
  "eval_report": "runs/run-.../eval_report.json"
}
```

### metrics.json

```json
{
  "test_acc": 0.784,
  "test_loss": 0.612,
  "eval_report": "runs/run-.../eval_report.json"
}
```

### eval_report.json

```json
{
  "test_acc": 0.784,
  "test_loss": 0.612,
  "per_class": [
    { "idx": 0, "class": "3", "name": "Betula (Birch)", "n": 120, "correct": 98, "acc": 0.817 },
    ...
  ],
  "confusion_matrix": [ [...], ... ],
  "class_mapping": { "3": 0, "7": 1, ... }
}
```

`class_mapping` shows how manifest `class_id` values map to the consecutive
integer indices used internally (sorted ascending order of all class IDs in the manifest).

---

## Multi-Domain Experiments

Set `TRAIN_DOMAINS` and `TEST_DOMAINS` to any domain strings present in your manifest.
Multiple domains can be comma-separated for training.

Examples (assuming domains `lab` and `wild`):

| scenario | `TRAIN_DOMAINS` | `TEST_DOMAINS` |
|---|---|---|
| in-domain lab | `lab` | `lab` |
| cross-domain: lab → wild | `lab` | `wild` |
| combined → wild | `lab,wild` | `wild` |

---

## Domain Adversarial Training (DANN)

`slurm_dann.sbatch` trains with a gradient-reversal domain classifier to reduce the
feature distribution gap between two domains.

Key differences from standard training:
- Requires exactly one source domain and one target domain (no comma lists)
- `DANN_LAMBDA_MAX` controls the domain loss weight ceiling; start low (e.g. `0.01`) —
  high values cause the species classifier to destabilize
- `DANN_BURNIN_EPOCHS` runs standard supervised training before DANN annealing begins;
  `MIN_EPOCHS` prevents early stopping from firing during burn-in
- Early stopping monitors source-domain val accuracy (not target)
- The checkpoint saved is `best.pt` on source val accuracy; the eval in `eval_report.json`
  is on the target domain test set

```bash
SOURCE_DOMAIN=site_a TARGET_DOMAIN=site_b DANN_LAMBDA_MAX=0.01 sbatch slurm_dann.sbatch
```

---

## Standalone Evaluation and Prediction

Run against an existing checkpoint without submitting a new training job:

```bash
# evaluate on a directory of images (ImageFolder layout)
python -m image_classifier.cli test \
  --data-dir path/to/test_images \
  --checkpoint runs/run-<id>/best.pt \
  --per-class --confusion-matrix \
  --report-path eval_report.json

# predict top-3 classes for individual images
python -m image_classifier.cli predict \
  --checkpoint runs/run-<id>/best.pt \
  --images img1.jpg img2.jpg \
  --top-k 3
```

These commands require PyTorch and torchvision installed locally (or run inside the
Apptainer container).
