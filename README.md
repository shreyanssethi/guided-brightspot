# guided-brightspot

Research Question: Does injecting a classical soft probability map into a 3D U-Net's skip connections improve White Matter Hyperintensity (WMH) segmentation?

Two models are compared on the [WMH 2017 Challenge](https://www.kaggle.com/datasets/farahmo/wmh-dataset) dataset (60 train / 110 test cases, 3/5 scanner sites):

- **BaselineUNet** — standard MONAI 3D U-Net with FLAIR + T1 as input
- **GuidedUNet** — same architecture, but each skip connection is scaled by a classical soft probability map: `skip_guided = skip × (1 + soft_map)`

The guidance adds zero learnable parameters — same model capacity, different inductive bias.

## Results

| Method | Train DICE | Test DICE |
|--------|-----------|----------|
| Classical pipeline | 0.146 ± 0.198 | 0.177 ± 0.228 |
| Baseline U-Net | 0.698 ± 0.138 | 0.639 ± 0.143 |
| **Guided U-Net** | **0.734 ± 0.097** | **0.694 ± 0.124** |

Guided improves over baseline by +3.6% train DICE and +5.5% test DICE.

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

Download the WMH 2017 Challenge dataset from Kaggle — see [`data/download_instructions.md`](data/download_instructions.md) for exact steps.

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

| Notebook | Purpose |
|----------|---------|
| `data_exploration.ipynb` | Inspect raw data — shape, spacing, intensity variation across sites |
| `verify_processed.ipynb` | Confirm preprocessing — uniform shape/spacing, z-score norm, binary masks |
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
├── figures/                # Training curve PNGs
└── evaluation/
    ├── train_eval.csv      # Per-case metrics, all 60 training cases × 3 methods
    ├── test_eval.csv       # Per-case metrics, all 110 test cases × 3 methods
    └── *.png               # DICE distributions, per-site plots, qualitative grids
```

`visualizations/` holds figures from the exploration and verification notebooks.

---

## Dataset

**WMH 2017 Challenge** — 3D FLAIR + T1 brain MRI with manual WMH lesion masks.

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
