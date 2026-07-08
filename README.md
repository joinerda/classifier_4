
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

### 1. Prepare the experiment

`prepare_train_test.py` builds the manifest from `metadata.json`, prints the
resolved settings, and writes a sourceable env file for
`submit_train_test.sbatch`.

#### Baseline: no colormapping (`wild -> wild`)

Use this to train and evaluate only on the wild domain:

```bash
python prepare_train_test.py \
  --image-root pollen_query_20260616_170602 \
  --manifest pollen_query_20260616_170602/manifest_wild__wild.csv \
  --train-domains wild \
  --test-domains wild \
  --run-comment wild__wild \
  --env-file train_test_env_wild__wild.sh
```

This produces:

- `pollen_query_20260616_170602/manifest_wild__wild.csv`
- `train_test_env_wild__wild.sh`

#### Comparison: colormapped lab training crops (`lab,wild -> wild`)

Use this to test whether color-normalized lab crops help training for wild
test images.

First install the extra prep dependencies:

```bash
python -m pip install -r COLORMAP/requirements_colormap.txt
```

Then prepare the run:

```bash
python prepare_train_test.py \
  --image-root pollen_query_20260616_170602 \
  --manifest pollen_query_20260616_170602/manifest_combined__wild.csv \
  --train-domains "lab,wild" \
  --test-domains wild \
  --run-comment combined__wild__colormapped \
  --env-file train_test_env_combined__wild__colormapped.sh \
  --colormap-output-dir colormapped_manifest_combined__wild
```

This produces:

- `pollen_query_20260616_170602/manifest_combined__wild.csv`
- `pollen_query_20260616_170602/manifest_combined__wild__colormapped.csv`
- transformed lab training crops under
  `pollen_query_20260616_170602/colormapped_manifest_combined__wild/`
- `train_test_env_combined__wild__colormapped.sh`

Default colormapping behavior:

1. Reference color statistics are fit from `wild/train`.
2. Only `lab/train` rows are rewritten to point at transformed images.
3. `wild/test` is left untouched.
4. `IMAGE_ROOT` stays the original export root; only `MANIFEST` changes to the
   derived colormapped manifest.
5. `--skip-build` works here too, but only if the derived manifest already
   exists.

### Older structured colorscale workflow

The repo also still carries the older `lab_set/` + `wild_set/` workflow in
`COLORMAP/make_colorscaled_dataset.py`. That path expects full images and
label files and is mainly useful for the earlier detection-style dataset
layout, not the current pre-cropped export workflow.

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

To submit a specific prepared run config without exporting `ENV_FILE`
manually:

```bash
sbatch submit_train_test.sbatch train_test_env_wild__wild.sh
sbatch submit_train_test.sbatch train_test_env_combined__wild__colormapped.sh
```

**Option B — submit via `train_test.py`:**

```bash
python train_test.py \
  --code-dir src \
  --manifest pollen_query_20260616_170602/manifest_wild__wild.csv \
  --image-root pollen_query_20260616_170602 \
  --train-domains wild \
  --test-domains wild
```

Colormapped comparison:

```bash
python train_test.py \
  --code-dir src \
  --manifest pollen_query_20260616_170602/manifest_combined__wild__colormapped.csv \
  --image-root pollen_query_20260616_170602 \
  --train-domains lab,wild \
  --test-domains wild \
  --run-comment combined__wild__colormapped
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
