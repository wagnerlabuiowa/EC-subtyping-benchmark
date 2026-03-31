# EC-Subtyping Benchmark

Code for reproducing and extending the results in:

> **Real-World Benchmarking and Validation of Foundation Model Transformers for Endometrial Cancer Subtyping from Histopathology**
>
> Wagner VM, Cosgrove CM, Chen SJ, Griffin DT, Suster MI, Goodfellow MJ, Baiocchi JG.
>
> Preprint: [https://doi.org/10.21203/rs.3.rs-7689962/v1](https://doi.org/10.21203/rs.3.rs-7689962/v1)

This repository benchmarks six open-source histopathology foundation encoders
(ViT-B/16, CTransPath, Prov-GigaPath, H-Optimus-0, UNI-2-h, Virchow2) with
two MIL aggregation strategies (TransMIL, CLAM) against four ImageNet-pretrained
CNN baselines (ResNet-18, ResNet-50, DenseNet, EfficientNet) for four-class
molecular subtyping of endometrial cancer (POLE, dMMR, NSMP, p53abn) from
H&E whole-slide images.

---

## Repository Structure

```
EC-subtyping-benchmark/
├── stamp/                  # Modified STAMP pipeline (preprocessing, training, deploy, eval)
│   ├── config.yaml         # Template configuration (placeholder paths)
│   ├── cli.py              # CLI entry point
│   ├── modeling/            # TransMIL, CLAM, training loop, data loading
│   ├── eval_stats/          # Cross-model comparison scripts
│   └── tiatoolbox/          # HoVer-Net nucleus segmentation helpers
├── hia/                    # CNN baseline training/deployment (HIA subset)
│   ├── Main.py              # CNN cross-validation entry point
│   ├── Classic_Deployment.py
│   ├── example_runs/templates/
│   └── README.md            # Detailed CNN workflow guide
├── envs/                   # Conda environment definitions and lockfiles
│   ├── stamp.yml / hia.yml / tiahover.yml
│   ├── *-explicit.txt       # Exact package snapshots
│   └── *-history.yml        # From-history exports
└── README.md               # This file
```

---

## Prerequisites

- **OS**: Ubuntu Linux (tested on 20.04/22.04). Other platforms may work but are not tested.
- **GPU**: CUDA-capable GPU(s) required for preprocessing and training.
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
- **OpenSlide**:

```bash
apt update && apt install -y openslide-tools libgl1-mesa-glx
```

- **Hugging Face token**: Required to download foundation encoder weights. Obtain from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and request access for gated models (UNI-2-h, Virchow2, Prov-GigaPath, etc.) where required.

---

## Environment Setup

Three separate conda environments are used:

| Environment | Purpose | YAML |
|---|---|---|
| STAMP | Feature extraction, transformer training, statistics, heatmaps | `envs/stamp.yml` |
| HIA | CNN baseline training and deployment | `envs/hia.yml` |
| TIA/HoVer | Nucleus segmentation and morphometrics | `envs/tiahover.yml` |

**Create from YAML (recommended starting point):**

```bash
conda env create -f envs/stamp.yml
conda env create -f envs/hia.yml
conda env create -f envs/tiahover.yml
```

**Or recreate exact environments from lockfiles:**

```bash
conda create -n STAMP --file envs/stamp-explicit.txt
conda create -n HIA --file envs/hia-explicit.txt
conda create -n TIAHOVER --file envs/tiahover-explicit.txt
```

> [!TIP]
> The `*-explicit.txt` files capture the exact packages used to produce the results in the paper.
> To regenerate from your own environments:
> ```bash
> conda activate <env_name>
> conda list --explicit > envs/<env_name>-explicit.txt
> conda env export --from-history > envs/<env_name>-history.yml
> ```

---

## Required Data Structure

### Clinical and slide tables

Prepare two tables per cohort:

- **Clinical table** (`.xlsx` or `.csv`) with at least:
  - `PATIENT` — unique patient identifier
  - Target column (e.g. `Subtype` with values POLE / dMMR / NSMP / p53abn)
- **Slide table** (`.xlsx` or `.csv`) with at least:
  - `PATIENT`
  - `FILENAME` — slide identifier that maps to the feature-file stem

### Whole-slide images

A directory of WSIs (`.svs`, `.ndpi`, `.tif/.tiff`, `.mrxs`, ...) pointed to by `preprocessing.wsi_dir` in `config.yaml`.

### For external deployment

Provide separate clinical/slide tables for the external cohort via `modeling.d_clini_table` and `modeling.d_slide_table` in `config.yaml`.

---

## Reproducing the Benchmark

### 1 &mdash; Foundation Model Transformer Pipeline (STAMP)

```bash
conda activate STAMP    # or the name from envs/stamp.yml
```

#### 1.1 Initialize and download encoder weights

```bash
stamp init                              # generates config.yaml in cwd
# Edit config.yaml: set paths, target label, encoder, etc.
# (see stamp/config.yaml template in this repo)

export HF_TOKEN=hf_...                  # required for HuggingFace-hosted encoders
stamp setup                             # downloads encoder weights
```

Supported encoders: `ctp` (CTransPath), `uni`, `uni2`, `virchow2`, `h-optimus-0`, `prov-gigapath`, `ViT-B16`.
If you change `preprocessing.feat_extractor`, re-run `stamp setup`.

#### 1.2 Feature extraction (multi-GPU)

```bash
stamp preprocess_ddp
```

If deploying to an external cohort, update `preprocessing.wsi_dir` / `preprocessing.output_dir` and preprocess that cohort as well.

#### 1.3 Cross-validation training (5-fold)

Set `modeling.n_splits: 5` in `config.yaml`, then:

```bash
stamp crossval
```

#### 1.4 Statistics on CV folds

Fill `modeling.statistics.pred_csvs` with fold prediction files (`.../fold-0/patient-preds.csv` through `.../fold-4/patient-preds.csv`):

```bash
stamp statistics
```

#### 1.5 Deploy each fold model to external cohort

For each fold `k` in {0..4}:
- set `modeling.model_path` to `.../fold-k/export.pkl`
- set `modeling.d_output_dir` (e.g. `.../deploy_fk/`)

```bash
stamp deploy
```

#### 1.6 Statistics on deploy folds

Fill `modeling.statistics.pred_csvs` with deploy prediction files:

```bash
stamp statistics
```

#### 1.7 Heatmaps

Set `heatmaps.model_path`, `heatmaps.feature_dir`, and `heatmaps.wsi_dir`:

```bash
stamp heatmaps
```

#### 1.8 Nucleus segmentation and morphometrics (TIAToolbox / HoVer-Net)

```bash
conda activate ec-tiahover   # see name: in envs/tiahover.yml
```

Helper scripts live under `stamp/tiatoolbox/` (`hover_complete.py`, `hover_img.py`, `hov_stats.py`).
Update placeholder paths in those scripts to point at your tile output and experiment directories.

Official TIAToolbox guidance for HoVer-Net nucleus instance segmentation:

- Documentation: [08 &mdash; Nucleus instance segmentation](https://tia-toolbox.readthedocs.io/en/v1.0.0/_notebooks/08-nucleus-instance-segmentation.html)
- Notebook: [08-nucleus-instance-segmentation.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/08-nucleus-instance-segmentation.ipynb)

---

### 2 &mdash; CNN Baselines (HIA)

```bash
conda activate ec-hia   # see name: in envs/hia.yml
```

CNN baselines (`resnet18`, `resnet50`, `densenet`, `efficient`) are trained and deployed via the `hia/` directory.
See [`hia/README.md`](hia/README.md) for the full workflow; a summary follows.

#### 2.1 Cross-validation training

Use `hia/example_runs/templates/cnn_crossval_template.txt` as a starting config:

```bash
python hia/Main.py --adressExp /path/to/cnn_crossval_experiment.txt
```

#### 2.2 External deployment per fold

Use `hia/example_runs/templates/cnn_deploy_template.txt`:

```bash
python hia/Classic_Deployment.py \
  --adressExp /path/to/cnn_deploy_fold.txt \
  --modelAdr /path/to/crossval_run/RESULTS/bestModelFold1
```

Repeat for each fold checkpoint (`bestModelFold1` through `bestModelFold5`).

---

### 3 &mdash; Final Cross-Model Comparison (`stamp/eval_stats`)

After all transformer and CNN runs are complete:

1. Copy `stamp/eval_stats/model_paths_example.py` to your own path config.
2. Populate CNN and transformer fold/deploy output directories.
3. Run:

```bash
python stamp/eval_stats/evaluate_models.py \
  --path_config /path/to/model_paths.py \
  --out_dir /path/to/results \
  --level patient
```

This generates the aggregated metrics, ROC curves, and comparison tables reported in the paper.

---

## Available STAMP Commands

```bash
stamp init            # create config.yaml in current directory
stamp setup           # download encoder weights and resources
stamp config          # print resolved configuration
stamp preprocess      # single-GPU feature extraction
stamp preprocess_ddp  # multi-GPU feature extraction
stamp crossval        # k-fold cross-validation training
stamp train           # train a single model
stamp deploy          # deploy a trained model on an external set
stamp statistics      # ROC curves, AUCs, confidence intervals
stamp heatmaps        # attention heatmap generation
```

> [!NOTE]
> By default STAMP reads `config.yaml` from the current working directory.
> Use `--config /path/to/other.yaml` to override.
> Run `stamp init` to generate a fresh config with default values.

---

## Data and Code Availability

Public datasets used in this benchmark:

- TCGA via the NCI Genomic Data Commons: <https://portal.gdc.cancer.gov>
- CPTAC via The Cancer Imaging Archive: <https://www.cancerimagingarchive.net>

Digital histopathology slides from the institutional clinical cohort contain protected health information subject to HIPAA/IRB restrictions and are not publicly posted.
De-identified derivatives (tile-level features, slide-level predictions, heatmaps) may be shared under a data use agreement upon reasonable request, pending institutional approvals.

### Open-source frameworks

| Framework | Repository | Role in this benchmark |
|---|---|---|
| STAMP | [KatherLab/STAMP](https://github.com/KatherLab/STAMP) | WSI preprocessing, foundation model feature extraction, MIL training/deploy/statistics |
| HIA | [KatherLab/HIA](https://github.com/KatherLab/HIA) | CNN tile-level training and deployment |
| CLAM | [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) | Attention-based MIL aggregation (CLAM-SB/MB architecture) |
| TIA Toolbox | [TissueImageAnalytics/tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) &middot; [docs v1.0.0](https://tiatoolbox.readthedocs.io/en/v1.0.0/index.html) | Nucleus instance segmentation |
| HoVer-Net | [vqdang/hover_net](https://github.com/vqdang/hover_net) | Nucleus segmentation and classification |

### Foundation models

| Model | Weights |
|---|---|
| ViT-Base | [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) |
| CTransPath | [jamesdolezal/CTransPath](https://huggingface.co/jamesdolezal/CTransPath) |
| Prov-GigaPath | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| H-Optimus-0 | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| UNI-2-h | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| Virchow2 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{wagner2025ecsubtyping,
  title   = {Real-World Benchmarking and Validation of Foundation Model Transformers
             for Endometrial Cancer Subtyping from Histopathology},
  author  = {Wagner, Vincent M. and Cosgrove, Casey M. and Chen, Stephanie J.
             and Griffin, Daniel T. and Suster, Martina I.
             and Goodfellow, Michael J. and Baiocchi, John G.},
  year    = {2025},
  doi     = {10.21203/rs.3.rs-7689962/v1},
  note    = {Under review at npj Precision Oncology}
}
```

This work builds on the STAMP protocol; if you use the STAMP pipeline please also cite:

```bibtex
@article{elnahhas2025stamp,
  title   = {From Whole-Slide Image to Biomarker Prediction: End-to-End Weakly
             Supervised Deep Learning in Computational Pathology},
  author  = {El Nahhas, Omar S. M. and van Treeck, Marko and W{\"o}lflein, Georg
             and Unger, Michaela and Ligero, Marta and Lenz, Tim and Wagner, Sophia J.
             and Hewitt, Katherine J. and Khader, Firas and Foersch, Sebastian
             and Truhn, Daniel and Kather, Jakob Nikolas},
  journal = {Nature Protocols},
  volume  = {20},
  pages   = {293--316},
  year    = {2025}
}
```

If you use the CLAM aggregation module, please cite:

```bibtex
@article{lu2021clam,
  title     = {Data-Efficient and Weakly Supervised Computational Pathology on
               Whole-Slide Images},
  author    = {Lu, Ming Y. and Williamson, Drew F. K. and Chen, Tiffany Y.
               and Chen, Richard J. and Barbieri, Matteo and Mahmood, Faisal},
  journal   = {Nature Biomedical Engineering},
  volume    = {5},
  number    = {6},
  pages     = {555--570},
  year      = {2021}
}
```
