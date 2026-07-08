# Colormap Plan

## Goal

Create a reusable workflow to color-normalize the **lab** images so they look
more like the **wild** microscope domain, then rerun the manifest-based
classification experiments on that color-scaled dataset.

The working hypothesis is that lab-to-wild transfer is being hurt by a strong
color-domain shift, and that replacing the lab color distribution with a
wild-like color distribution may improve generalization while preserving the
ground-truth labels available on the lab images.

## Dataset setup

Work is being sandboxed inside `DATASET_TEST/`.

Relevant dataset roots:

- Original copy:
  `DATASET_TEST/lab_wild_combined_feb_26`
- Color-scaled copy:
  `DATASET_TEST/lab_wild_combined_feb_26_labcolorscaled`

The manifest format fits `DATASET_TEST` as-is.  The manifest paths are relative
like:

- `lab_set/train/cropped/...`
- `lab_set/test/cropped/...`
- `wild_set/train/cropped/...`
- `wild_set/test/cropped/...`

and those paths exist under the copied dataset root.

## Key design decisions

### 1. Do not overwrite the original dataset

The color-scaled dataset is built as a new copy so the original manifest-based
dataset remains intact for comparison and for future preprocessing variants.

### 2. Ignore the existing lab `cropped/` folders

The color transfer method described in
`DATASET_TEST/PollenImageIdea/README.md` works on the **full** images with YOLO
annotations, not on the already-cropped classification tiles.

So the correct flow is:

1. transform lab `full/` images
2. keep YOLO labels unchanged
3. regenerate lab `cropped/` tiles from the transformed full images
4. rebuild the manifest from the regenerated crops

### 3. Keep wild data unchanged

Only the lab images are color-transferred.  The wild `full/` and `cropped/`
data are copied unchanged into the new dataset root.

### 4. Use pooled wild reference colors, not one template image

The current preprocessing uses a pooled wild reference model built from many
wild full images plus their YOLO labels.  It does not choose one matched wild
template per lab image.

The current script groups reference pools by magnification when possible:

- `100x`
- `400x`
- fallback `default`

## Implemented preprocessing

Added reusable preprocessing script:

- `make_colorscaled_dataset.py`

This script:

- takes a dataset root containing `lab_set/` and `wild_set/`
- creates a new output root
- copies `wild_set/` unchanged
- transforms lab `full/` images using
  `DATASET_TEST/PollenImageIdea/color_transfer.py`
- regenerates lab crops from YOLO boxes
- writes a fresh `manifest.csv`

Suggested command:

```bash
python make_colorscaled_dataset.py \
  --source-root DATASET_TEST/lab_wild_combined_feb_26 \
  --output-root DATASET_TEST/lab_wild_combined_feb_26_labcolorscaled \
  --color-transfer-module DATASET_TEST/PollenImageIdea/color_transfer.py \
  --group-by-magnification
```

### Stability fix added

The first run hit:

- `numpy.linalg.LinAlgError: Singular matrix`

This came from the RBF solve in the color-transfer fitting step.

The wrapper script was updated to make the transform more robust by:

- deduplicating near-identical anchor colors
- averaging target anchors for duplicate source anchors
- retrying the RBF fit with tiny smoothing if the exact solve is singular

## Environment support

Added:

- `requirements_colormap.txt`

with:

- `opencv-python-headless`
- `scikit-learn`
- `scipy`
- `numpy`

## Training / evaluation workflow changes

The repo already uses manifest-based training as the primary workflow.

For the color-scaled dataset, two new SLURM scripts were added:

- `slurm_color_experiment.sbatch`
- `slurm_color_eval.sbatch`

### `slurm_color_experiment.sbatch`

Trains three scenarios on the color-scaled dataset root:

- `color_labscaled_combined_train` with `train_domains=lab,wild`
- `color_labscaled_lab_train` with `train_domains=lab`
- `color_labscaled_wild_train` with `train_domains=wild`

Defaults:

- dataset root:
  `DATASET_TEST/lab_wild_combined_feb_26_labcolorscaled`
- output runs directory:
  `runs_color`
- DANN support remains available but off by default

### `slurm_color_eval.sbatch`

Finds the latest checkpoint for each of the three run comments above and
evaluates each one on:

- color-scaled `lab` test
- `wild` test

This produces six reports under:

- `eval_reports_color/`

Expected report names:

- `color_labscaled_combined_train__lab_test.json`
- `color_labscaled_combined_train__wild_test.json`
- `color_labscaled_lab_train__lab_test.json`
- `color_labscaled_lab_train__wild_test.json`
- `color_labscaled_wild_train__lab_test.json`
- `color_labscaled_wild_train__wild_test.json`

## Current job status

The color-map training and evaluation batches were completed.

Training:

- `15469` completed all three training scenarios

Evaluation:

- `15471` completed the full six-report evaluation set

## Rationale vs DANN

DANN was explored previously and kept as an option, but it is not the default
training path.  The current color-map experiments are being run in the standard
manifest-based training setup because:

- DANN improved some cross-domain cases but did not beat the supervised
  combined-domain baseline overall
- color normalization is a simpler, more targeted intervention for the observed
  lab-to-wild shift

## Final results

Held-out accuracies from `eval_reports_color/`:

- `combined -> colormapped_lab`: `77.22%`
- `combined -> wild`: `73.95%`
- `colormapped_lab -> colormapped_lab`: `76.86%`
- `colormapped_lab -> wild`: `8.48%`
- `wild -> colormapped_lab`: `16.79%`
- `wild -> wild`: `72.06%`

## Conclusion

Color-mapping the lab images did not improve the cross-domain problem.

The key failure mode is still `lab -> wild`: after color transfer,
`colormapped_lab -> wild` is only `8.48%`, which is worse than the earlier
non-colormapped lab-to-wild baseline. The reverse direction is also weak:
`wild -> colormapped_lab` is `16.79%`.

The only configuration that remained strong was supervised mixed-domain
training:

- `combined -> wild`: `73.95%`
- `wild -> wild`: `72.06%`

So the practical takeaway is:

- color normalization did not close the domain gap
- combined supervised training is still the best-performing practical approach
- the domain shift is not explained by color alone

## Next steps

1. Keep the color-map pipeline available as a reusable preprocessing tool, but
   do not treat it as a current accuracy-improving path.
2. Keep standard manifest training, especially `combined_train`, as the main
   benchmark.
3. If domain adaptation work continues, focus on other sources of shift beyond
   color.
