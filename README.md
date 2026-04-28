# guided-brightspot

White Matter Hyperintensities (WMH) are regions of abnormal signal in brain MRI associated with neurological disease, and accurately segmenting them from 3D FLAIR and T1 scans is a challenging task due to their variability in size, shape, and location across patients and scanner sites. This project investigates whether a classical image processing pipeline can be used not as a standalone segmentor, but as a source of spatial guidance for a deep learning model. My idea is to run a fast thresholding and morphology pipeline on a FLAIR/T1 difference map to produce a per-patient soft probability map of WMH candidates, then inject that map into a 3D U-Net at every skip connection via element-wise scaling (`skip_guided = skip × (1 + soft_map)`). This biases the decoder toward classically-flagged regions without adding any learnable parameters → the two models being compared have identical capacity, just different inductive biases. The classical soft maps are computed using SimpleITK (intensity thresholding on the FLAIR–T1 difference, morphological cleaning, and Gaussian smoothing), and both models are trained end-to-end using MONAI's 3D U-Net with patch-based training on the [WMH 2017 Challenge](https://www.kaggle.com/datasets/farahmo/wmh-dataset) dataset (60 train / 110 test cases across 3/5 scanner sites).

## Results

| Method | Train DICE | Test DICE |
|--------|-----------|----------|
| Classical pipeline | 0.146 ± 0.198 | 0.177 ± 0.228 |
| Baseline U-Net | 0.698 ± 0.138 | 0.639 ± 0.143 |
| **Guided U-Net** | **0.734 ± 0.097** | **0.694 ± 0.124** |

Guided improves over baseline by +3.6% train DICE and +5.5% test DICE.

---

## Quick Start

For the TA of 16-725A, please follow these instructions to run the trained Guided U-Net on the included test set and see the evaluation metrics. 

1. Install dependencies ([Setup](#setup))
2. Run from the repo root:
   ```bash
   python evaluation/run_inference.py
   ```

Results are saved to `data/project_submission/results/`:
- `metrics.csv` — per-case DICE and HD95 for all 110 test cases
- `sample_results.png` — FLAIR / ground truth / prediction for a spread of cases

> To reproduce everything from scratch (download data, preprocess, train, full evaluation), follow the [Pipeline](#pipeline) section below.
> You can also find additional images, loss curves, graphs of evaluation metrics, etc. from my training runs in `outputs/evaluation`, `outputs/figures` and `visualizations`. All of these outputs are explained later in this README.


---

## Repo Structure

```
guided-brightspot/
├── preprocessing/
│   ├── preprocess_rawData.py       # Resample, crop, normalize, binarize raw data
│   ├── compute_soft_maps.py        # Generate classical segmentations + soft maps
│   └── grid_search_soft_maps.py    # Hyperparameter search for classical pipeline
├── training/
│   ├── models.py                   # BaselineUNet and GuidedUNet definitions
│   ├── dataset.py                  # MONAI dataloaders (patch-based train, whole-vol val)
│   └── train.py                    # Training script (see CLI flags below)
├── evaluation/
│   ├── metrics.py                  # DICE, HD95, lesion-level F1
│   └── grid_search_results/        # Classical pipeline grid search CSVs
├── notebooks/                      # Analysis notebooks (run in order listed below)
├── data/                           # Raw + processed data (gitignored)
│   ├── wmh_data/                   # Original WMH 2017 Challenge dataset
│   ├── processed/                  # Preprocessed: 200×200×48 vox, 1.0×1.0×3.0 mm
│   └── download_instructions.md
├── outputs/                        # Training outputs (gitignored)
│   ├── checkpoints/                # Model weights
│   ├── logs/                       # Training history (JSON)
│   ├── figures/                    # Training curve plots
│   └── evaluation/                 # Metric CSVs + evaluation figures
└── visualizations/                 # Figures saved by exploration/verification notebooks
```

---

## Setup

```bash
conda create -n brightspot python=3.10
conda activate brightspot
pip install "monai[all]" SimpleITK nibabel scikit-image scikit-learn matplotlib jupyterlab kaggle
```

> `monai[all]` is large. If you hit `No space left on device`, point `PIP_CACHE_DIR` and `TMPDIR` to a directory with sufficient storage before installing.

---

## Data

Download the WMH 2017 Challenge dataset from Kaggle → see [`data/download_instructions.md`](data/download_instructions.md) for exact steps.

---

## Pipeline

Run all commands from the **repo root**.

### 1. Preprocess raw data
```bash
python preprocessing/preprocess_rawData.py --split training --split test --verify
```
Resamples to 1.0×1.0×3.0 mm, crops/pads to 200×200×48, z-score normalizes on brain voxels, binarizes masks. Output goes to `data/processed/`.

### 2. Compute classical soft maps
```bash
python preprocessing/compute_soft_maps.py --split training --split test
```
Runs the classical thresholding + morphology pipeline. Writes `soft_map.nii` and `classical_seg.nii` into each processed case directory.

### 3. Train
```bash
python training/train.py --model baseline
python training/train.py --model guided
```
Checkpoints saved to `outputs/checkpoints/`, training logs to `outputs/logs/`.

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | `baseline` or `guided` |
| `--epochs` | 300 | Training epochs |
| `--lr` | 5e-5 | Initial learning rate |
| `--batch_size` | 2 | Samples per step |
| `--gpu` | 0 | GPU index |
| `--resume` | — | Path to checkpoint to resume from |
| `--val_interval` | 5 | Validate every N epochs |
| `--cache_rate` | 1.0 | MONAI CacheDataset cache fraction |
| `--num_workers` | 4 | DataLoader worker processes |
| `--seed` | 42 | Random seed |

### 4. Evaluate
Open the notebooks in order (see below).

### Other scripts

**Grid search for classical pipeline hyperparameters** (`preprocessing/grid_search_soft_maps.py`):
Sweeps thresholding/morphology parameters to find the best classical segmentation settings on the training set. Results are saved to `evaluation/grid_search_results/`. Configure the search space and paths at the top of the file, then run:
```bash
python preprocessing/grid_search_soft_maps.py
```

---

## Notebooks

*Note: In the notebooks, you may have to change the paths (`DATA_ROOT`, `TEST_ROOT`, etc.) that are usually declared in the first few cells to match your absolute path.*

| Notebook | Purpose |
|----------|---------|
| `data_exploration.ipynb` | Inspect raw data → shape, spacing, intensity variation across sites |
| `verify_processed.ipynb` | Confirm preprocessing → uniform shape/spacing, z-score norm, binary masks |
| `evaluate_classical_and_softmap.ipynb` | Classical baseline metrics + soft map health checks |
| `plot_training_curves.ipynb` | Training loss, validation DICE, learning rate schedule |
| `model_evaluation.ipynb` | Full comparison: classical vs baseline vs guided (DICE, HD95, per-site) |

---

## Outputs

```
outputs/
├── checkpoints/
│   ├── baseline_best.pt    # Best baseline checkpoint
│   ├── baseline_last.pt    # Checkpoint at final epoch
│   ├── guided_best.pt      # Best guided checkpoint
│   └── guided_last.pt      # Checkpoint at final epoch
├── logs/
│   ├── baseline_history.json
│   └── guided_history.json
├── figures/
│   ├── train_loss.png              # Training loss curve (both models)
│   ├── val_dice.png                # Validation DICE curve (both models)
│   ├── lr_schedule.png             # Learning rate over epochs
│   └── training_summary.png        # Combined 3-panel summary figure
└── evaluation/
    ├── train_eval.csv                     # Per-case metrics, 60 training cases × 3 methods
    ├── test_eval.csv                      # Per-case metrics, 110 test cases × 3 methods
    ├── dice_distributions.png             # DICE histogram across all 3 methods
    ├── dice_by_site_training.png          # Per-case DICE by site, training set
    ├── dice_by_site_test.png              # Per-case DICE by site, test set
    ├── bmw_classical_training.png         # Best/Median/Worst cases for classical pipeline
    ├── bmw_baseline_training.png          # Best/Median/Worst cases for Baseline U-Net
    ├── bmw_guided_training.png            # Best/Median/Worst cases for Guided U-Net
    ├── greatest_improvement_training.png  # Cases where Guided most outperforms Baseline (train)
    ├── greatest_improvement_test.png      # Cases where Guided most outperforms Baseline (test)
    └── presentation_grid_training.png     # Large composite grid across all 3 methods
```

`visualizations/` holds figures from the exploration and verification notebooks:
- `exploration_slices.png` / `exploration_intensity_dists.png` — raw data inspection (from `data_exploration.ipynb`)
- `verification_training_set.png` / `verification_test_set.png` / `verification_lesion_load.png` — preprocessing QC (from `verify_processed.ipynb`)
- `classical_and_softmap_qualitative.png` — classical pipeline steps + soft map overlay (from `evaluate_classical_and_softmap.ipynb`)

---

## Dataset

**WMH 2017 Challenge** → 3D FLAIR + T1 brain MRI with manual WMH lesion masks.

- **Training:** 60 cases across 3 sites (Utrecht, Singapore, Amsterdam/GE3T), stratified 80/20 train/val split
- **Test:** 110 cases across 5 sites (includes 2 out-of-distribution scanners: GE1T5, Philips)

Each preprocessed case contains:

| File | Description |
|------|-------------|
| `FLAIR.nii` | z-score normalized FLAIR |
| `T1.nii` | z-score normalized T1 |
| `wmh.nii` | Binary ground truth mask |
| `soft_map.nii` | Classical soft probability map [0, 1] |
| `classical_seg.nii` | Classical segmentation output |
