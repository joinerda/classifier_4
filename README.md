
RESNET based classifier for classifying pollen images

## Full run walkthrough

This walks through a complete train/test cycle on the Slurm + Apptainer cluster
workflow: build the manifest, submit the job, and check the results.

### 0. Prerequisites

- A dataset directory containing `metadata.json` and the referenced images
  (e.g. `pollen_query_20260616_170602/`, produced by the data export step).
- Access to `sbatch`/`squeue` on the cluster, with Apptainer available.
- A local Python (3.9+) for running the prep/submit scripts — these only use
  the standard library, so a plain `venv` is enough; the training itself runs
  inside the Apptainer container, which installs its own dependencies
  (`pip install --user certifi pandas`) at job start.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1. Build the manifest

`prepare_train_test.py` builds `manifest.csv` from `metadata.json` and writes
the resolved settings to `train_test_env.sh`, which `submit_train_test.sbatch`
sources automatically:

```bash
python prepare_train_test.py \
  --train-domains "wild" \
  --test-domains "wild" \
  --image-root pollen_query_20260616_170602 \
  --manifest pollen_query_20260616_170602/manifest.csv
```

This:

1. Runs `build_manifest.py` against `metadata.json` in the given
   `--image-root`, writing `manifest.csv` and printing a per-domain/split
   summary.
2. Prints the resolved `IMAGE_ROOT`, `MANIFEST`, `TRAIN_DOMAINS`,
   `TEST_DOMAINS`, and `RUN_COMMENT` as `export` lines, for reference or for
   pasting into a shell.
3. Writes those same settings to `train_test_env.sh` (as conditional
   defaults, so they won't override anything already set in the
   environment) for `submit_train_test.sbatch` to pick up.

Re-run this step whenever the dataset, domain split, or manifest path
changes. Use `--skip-build` to only regenerate `train_test_env.sh` without
rebuilding the manifest.

### 2. Submit the training/eval job

Two equivalent ways to submit, depending on whether you want the script to
manage `sbatch` resource flags for you:

**Option A — submit directly, using the prepared env file:**

```bash
sbatch submit_train_test.sbatch
```

`train_test_env.sh` (written in step 1) is sourced automatically and fills
in `IMAGE_ROOT`, `MANIFEST`, `TRAIN_DOMAINS`, `TEST_DOMAINS`, and
`RUN_COMMENT`. Override Slurm resources with flags, e.g.:

```bash
sbatch --partition=gpu_long --gres=gpu:1 --time=04:00:00 submit_train_test.sbatch
```

**Option B — submit via `train_test.py`:**

```bash
python train_test.py \
  --code-dir src \
  --manifest pollen_query_20260616_170602/manifest.csv \
  --image-root pollen_query_20260616_170602 \
  --train-domains wild \
  --test-domains wild
```

Add `--print-only` first to see the exact `sbatch` command without
submitting. Values passed here take priority over `train_test_env.sh`.

Other training knobs (`ARCH`, `EPOCHS`, `BATCH_SIZE`, `LR`, etc.) can be set
as environment variables before either submission method; see the top of
`submit_train_test.sbatch` for the full list and defaults.

### 3. Monitor the job

```bash
squeue -u "$USER"
tail -f logs/imgclf-<jobid>.out
```

### 4. Check results

Each run writes to its own subdirectory under `OUTPUT_DIR` (default
`runs/`), containing:

- `best.pt` — best checkpoint
- `metrics.json` / `history.json` — training metrics
- `eval_report.json` — test-set evaluation (confusion matrix and per-class
  metrics are enabled by default in `submit_train_test.sbatch`)
