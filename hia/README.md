# HIA (Histopathology Image Analysis) - CNN Reproducibility Guide

This directory contains a **CNN-only subset** of HIA used in the EC subtyping benchmark paper workflow.

The code is based on and adapted from the original HIA/deep histology pipeline and related prior work. Please credit the original authors and references when reusing this code:

- Kather et al., Nature Medicine 2019: <https://www.nature.com/articles/s41591-019-0462-y>
- Echle et al., Gastroenterology 2020: <https://www.sciencedirect.com/science/article/pii/S0016508520348186>
- DeepHistology (earlier Matlab framework): <https://github.com/jnkather/DeepHistology>

## What Is Included In This Public Subset

- Classical tile-based CNN training and validation
- Cross-validation workflow
- External deployment workflow
- AUC/statistics aggregation utilities
- Heatmap generation scripts

Supported `modelName` values in this subset:

- `resnet18`
- `resnet50`
- `densenet`
- `efficient`

## Environment Setup (Conda)

Create the HIA environment from the repository root:

```bash
conda env create -f envs/hia.yml
conda activate ec-hia
```

For strict reproducibility, prefer exporting and storing the exact original env used for training:

```bash
conda list --explicit > envs/hia-explicit.txt
```

An exact snapshot is already included in this repository at `envs/hia-explicit.txt`.

## Reproducing The Paper CNN Workflow

The workflow used in the paper is:

1. Run **cross-validation training** per architecture (`densenet`, `resnet18`, `resnet50`, `efficient`)
2. Collect fold-level and total statistics (generated automatically in training reports/csvs)
3. Run **external deployment once per fold** using each fold checkpoint (`bestModelFold1..k`)
4. Compute/report deployment statistics
5. Run final aggregated model comparison in `stamp/eval_stats` (CNN + transformer summary tables/plots)
6. Generate subtype heatmaps and GradCAM visualizations

Only reusable config templates are kept under `hia/example_runs/templates/` in this public repo.

## Step 1: Cross-Validation Training

Use `hia/example_runs/templates/cnn_crossval_template.txt` as the starting config.

Run:

```bash
python hia/Main.py --adressExp "/path/to/cnn_crossval_experiment.txt"
```

Main outputs (under the generated experiment folder):

- `RESULTS/bestModelFold1..k`
- `RESULTS/finalModelFold1..k`
- `RESULTS/TRAIN_HISTORY_FOLD_*.csv`
- `RESULTS/TEST_RESULT_TILE_BASED_FOLD_*.csv`
- `RESULTS/TEST_RESULT_PATIENT_BASED_FOLD_*.csv`
- `RESULTS/TEST_RESULT_TILE_BASED_TOTAL.csv`
- `RESULTS/TEST_RESULT_PATIENT_BASED_TOTAL.csv`
- `Report.txt` (fold AUCs + total AUC + confidence intervals)

## Step 2: External Deployment Per Fold

Use `hia/example_runs/templates/cnn_deploy_template.txt` as the deploy config.

Run deployment once per fold with the fold checkpoint:

```bash
python hia/Classic_Deployment.py \
  --adressExp "/path/to/cnn_deploy_fold1.txt" \
  --modelAdr "/path/to/crossval_run/RESULTS/bestModelFold1"
```

Repeat for folds 2..k (`bestModelFold2`, ..., `bestModelFold5`).

Deployment outputs:

- `RESULTS/TEST_RESULT_TILE_BASED_FULL.csv`
- `RESULTS/TEST_RESULT_PATIENT_BASED_FULL.csv`
- `SPLITS/TestSplit.csv`
- `Report.txt`

## Step 3: Final CNN Statistics In STAMP (`stamp/eval_stats`)

For paper-level comparison across folds/models, use the STAMP evaluation utilities:

- Copy/adapt `stamp/eval_stats/model_paths_example.py` into your own model-path mapping file (for example, `model_paths.py`).
- Populate your CNN fold/deploy output paths in that mapping.
- Run `stamp/eval_stats/evaluate_models.py` to generate final aggregated metrics and plots.

This is the step used to summarize CNN baselines together with transformer models in one evaluation framework.

## Step 4: Heatmaps

Subtype heatmap:

```bash
python hia/heatmap.py \
  --wsi_dir "/path/to/WSI_TILE_DIR" \
  --model_path "/path/to/bestModelFoldX" \
  --model_name densenet \
  --num_classes 4 \
  --class_names "CNV-H" "CNV-L" "MSI-H" "POLE" \
  --output_dir "/path/to/output/heatmaps"
```

GradCAM heatmap:

```bash
python hia/heatmap_tilegradcam.py \
  --wsi_dir "/path/to/WSI_TILE_DIR" \
  --model_path "/path/to/bestModelFoldX" \
  --model_name densenet \
  --num_classes 4 \
  --output_dir "/path/to/output/gradcam"
```

## Data Layout Expectations

Each cohort root in `dataDir_train` / `dataDir_test` should contain:

- One tile folder: `BLOCKS_NORM_MACENKO` or `BLOCKS_NORM_VAHADANE` or `BLOCKS_NORM_REINHARD` or `BLOCKS`
- `*_CLINI.xlsx`
- `*_SLIDE.xlsx` or `*_SLIDE.csv`
- `FEATURES` (for downstream steps that use extracted features)

## Notes

- `Classic_Deployment.py` takes `--modelAdr` from the command line, not from the experiment file.
- Keep config values consistent (e.g. `projectDetails` text should match `modelName`).
- Use JSON booleans/numbers where possible (`true`, `false`, `0`) for cleaner config files.

## Attribution Statement For Reuse

If you reuse this workflow in publications or derivative repositories, please:

- Cite the original HIA-related publications listed above
- Cite the EC subtyping benchmark paper accompanying this release
- Keep a clear note that this package is an adapted CNN-focused subset of the broader HIA workflow
