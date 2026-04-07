"""
training/dataset.py

MONAI dataset and dataloader construction for WMH segmentation.

Produces two dataloaders:
    - train_loader: patch-based, augmented, 48 cases (16 per site)
    - val_loader:   whole-volume, no augmentation, 12 cases (4 per site)

Split strategy: site-stratified 80/20.
    Utrecht:       patients 0-15 → train, 16-19 → val
    Singapore:     patients 0-15 → train, 16-19 → val
    Amsterdam/GE3T: patients 0-15 → train, 16-19 → val
Sorting is alphabetical by patient ID within each site, which is
deterministic and keeps the same 4 patients as val every run.

Data keys per sample dict:
    flair    → (1, H, W, D) float32 tensor
    t1       → (1, H, W, D) float32 tensor
    wmh      → (1, H, W, D) uint8  tensor  (binary 0/1)
    soft_map → (1, H, W, D) float32 tensor

Training input to U-Net:
    Baseline:  torch.cat([flair, t1], dim=1)           → (B, 2, H, W, D)
    Guided:    torch.cat([flair, t1, soft_map], dim=1) → (B, 3, H, W, D)

References for augmentation choices:
    - Li et al. 2018 (WMH challenge winner): mirroring, rotation, scaling
    - Kuijf et al. 2019 (WMH challenge paper): augmentation key to cross-scanner
      generalization
    - MONAI HW5 notebook (from class): CacheDataset + RandCropByPosNegLabeld pattern
"""

import torch
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ConcatItemsd,
)
from monai.data import CacheDataset, DataLoader


# ── Config ─────────────────────────────────────────────────────────────────────

PROCESSED_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/processed/training')

SITES = ['Utrecht', 'Singapore', 'Amsterdam/GE3T']

# 80/20 split per site: first 16 alphabetically → train, last 4 → val
N_TRAIN_PER_SITE = 16
N_VAL_PER_SITE   = 4

# Patch size for RandCropByPosNegLabeld (x, y, z)
# Full z-depth (48), crop in-plane to 96×96
# Fits comfortably in 20GB VRAM at batch_size=2
PATCH_SIZE = (96, 96, 48)

# Number of patches sampled per volume per training step
# 4 patches × 2 batch = 8 effective sub-volumes per iteration
NUM_SAMPLES = 4

# Ratio of foreground (WMH) to background patches
# 2:1 biases toward lesion-containing patches given WMH sparsity
POS_NEG_RATIO = 2  # pos=2, neg=1


# ── Data dict construction ─────────────────────────────────────────────────────

def build_data_dicts(split='training'):
    """
    Build list of data dicts for a given split.

    Each dict contains file paths:
        {'flair': Path, 't1': Path, 'wmh': Path, 'soft_map': Path}

    Returns:
        train_dicts (list), val_dicts (list)
    """
    train_dicts = []
    val_dicts   = []

    for site in SITES:
        site_dir = PROCESSED_ROOT / site
        if not site_dir.exists():
            print(f'WARNING: {site_dir} not found, skipping')
            continue

        # Sort alphabetically for deterministic split
        patient_dirs = sorted(
            [d for d in site_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name
        )

        if len(patient_dirs) != 20:
            print(f'WARNING: {site} has {len(patient_dirs)} patients, expected 20')

        for i, pdir in enumerate(patient_dirs):
            flair_path    = pdir / 'FLAIR.nii'
            t1_path       = pdir / 'T1.nii'
            wmh_path      = pdir / 'wmh.nii'
            soft_map_path = pdir / 'soft_map.nii'

            missing = [p.name for p in [flair_path, t1_path, wmh_path, soft_map_path]
                       if not p.exists()]
            if missing:
                print(f'WARNING: {site}/{pdir.name} missing {missing}, skipping')
                continue

            d = {
                'flair':    str(flair_path),
                't1':       str(t1_path),
                'wmh':      str(wmh_path),
                'soft_map': str(soft_map_path),
                # Metadata — not used by transforms, useful for debugging
                'site':     site,
                'pid':      pdir.name,
            }

            if i < N_TRAIN_PER_SITE:
                train_dicts.append(d)
            else:
                val_dicts.append(d)

    return train_dicts, val_dicts


# ── Transforms ─────────────────────────────────────────────────────────────────

# Keys that hold image data (not metadata strings)
IMAGE_KEYS = ['flair', 't1', 'wmh', 'soft_map']

# Keys for intensity augmentation (exclude mask and soft_map)
INTENSITY_KEYS = ['flair', 't1']


def get_train_transforms():
    """
    Training transforms: load → channel → patch crop → augmentation → tensor.

    Augmentation follows WMH challenge literature:
        - Random flips along all three axes (mirroring)
        - Random 90° rotations in-plane
        - Random intensity scale + shift on FLAIR and T1
          (simulates scanner variability across sites)

    RandCropByPosNegLabeld:
        - Samples NUM_SAMPLES patches per volume per call
        - pos=2, neg=1 → 2/3 of patches centered on WMH voxels
        - label_key='wmh' drives the foreground/background decision
        - soft_map follows the same crop (same spatial_size, same random state)
    """
    return Compose([
        # Load NIfTI files from disk
        LoadImaged(keys=IMAGE_KEYS),

        # Add channel dimension: (H,W,D) → (1,H,W,D)
        EnsureChannelFirstd(keys=IMAGE_KEYS),

        # Patch sampling — biased toward WMH voxels
        RandCropByPosNegLabeld(
            keys=IMAGE_KEYS,
            label_key='wmh',
            spatial_size=PATCH_SIZE,
            pos=POS_NEG_RATIO,
            neg=1,
            num_samples=NUM_SAMPLES,
            image_key='flair',
            image_threshold=0,  # any non-zero FLAIR voxel is valid center
        ),

        # Spatial augmentation — applied identically to all keys
        RandFlipd(keys=IMAGE_KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=IMAGE_KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=IMAGE_KEYS, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=IMAGE_KEYS, prob=0.5, max_k=3, spatial_axes=(0, 1)),

        # Intensity augmentation — FLAIR and T1 only, not mask or soft_map
        # Simulates scanner-to-scanner intensity variability
        # Reference: Li et al. 2018 (WMH challenge winner)
        RandScaleIntensityd(keys=INTENSITY_KEYS, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=INTENSITY_KEYS, offsets=0.1, prob=0.5),

        # Convert to float32 PyTorch tensors
        EnsureTyped(keys=IMAGE_KEYS, dtype=torch.float32),
    ])


def get_val_transforms():
    """
    Validation transforms: load → channel → tensor. No augmentation, no patching.
    Full volumes are passed to sliding_window_inference during validation.
    """
    return Compose([
        LoadImaged(keys=IMAGE_KEYS),
        EnsureChannelFirstd(keys=IMAGE_KEYS),
        EnsureTyped(keys=IMAGE_KEYS, dtype=torch.float32),
    ])


# ── Dataset and DataLoader construction ────────────────────────────────────────

def get_dataloaders(
    batch_size: int = 2,
    num_workers: int = 4,
    cache_rate: float = 1.0,
    seed: int = 42,
):
    """
    Build and return train and validation DataLoaders.

    Args:
        batch_size  (int):   Training batch size. Default 2 — fits in 20GB GPU
                             with PATCH_SIZE=(96,96,48) and NUM_SAMPLES=4.
        num_workers (int):   DataLoader worker processes.
        cache_rate  (float): Fraction of data to cache in RAM. 1.0 caches all
                             48 training cases — recommended since IO otherwise
                             dominates with NIfTI files.
        seed        (int):   Random seed for reproducibility.

    Returns:
        train_loader (DataLoader): Patch-based, shuffled, augmented.
        val_loader   (DataLoader): Whole-volume, ordered, no augmentation.
        train_dicts  (list):       Raw data dicts for reference.
        val_dicts    (list):       Raw data dicts for reference.
    """
    from monai.utils import set_determinism
    set_determinism(seed=seed)

    train_dicts, val_dicts = build_data_dicts()

    print(f'Split: {len(train_dicts)} train / {len(val_dicts)} val')
    for site in SITES:
        n_tr = sum(1 for d in train_dicts if d['site'] == site)
        n_va = sum(1 for d in val_dicts   if d['site'] == site)
        print(f'  {site}: {n_tr} train, {n_va} val')

    # CacheDataset loads and transforms all cases once, then caches the result
    # Subsequent epochs read from RAM — much faster than re-loading NIfTI each time
    # Reference: MONAI HW5 notebook (course material)
    train_ds = CacheDataset(
        data=train_dicts,
        transform=get_train_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    val_ds = CacheDataset(
        data=val_dicts,
        transform=get_val_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,       # always 1 for whole-volume sliding window inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dicts, val_dicts


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Quick sanity check — run directly to verify the dataloader works.

    python training/dataset.py

    Expected output:
        Split: 48 train / 12 val
        Utrecht:        16 train, 4 val
        Singapore:      16 train, 4 val
        Amsterdam/GE3T: 16 train, 4 val
        First train batch:
          flair:    torch.Size([2, 4, 1, 96, 96, 48])  ← [batch, n_patches, C, H, W, D]
          t1:       torch.Size([2, 4, 1, 96, 96, 48])
          wmh:      torch.Size([2, 4, 1, 96, 96, 48])
          soft_map: torch.Size([2, 4, 1, 96, 96, 48])
        First val batch:
          flair:    torch.Size([1, 1, 200, 200, 48])   ← full volume
    """
    print('Building dataloaders...')
    train_loader, val_loader, train_dicts, val_dicts = get_dataloaders(
        batch_size=2,
        num_workers=0,   # 0 for easier debugging
        cache_rate=0.0,  # skip cache for speed during check
    )

    print('\nChecking first training batch...')
    batch = next(iter(train_loader))
    for key in ['flair', 't1', 'wmh', 'soft_map']:
        print(f'  {key}: {batch[key].shape}  dtype={batch[key].dtype}')

    print('\nChecking first validation batch...')
    val_batch = next(iter(val_loader))
    for key in ['flair', 't1', 'wmh', 'soft_map']:
        print(f'  {key}: {val_batch[key].shape}  dtype={val_batch[key].dtype}')

    # Verify soft_map is never all-zero
    sm = batch['soft_map']
    n_zero = (sm.view(sm.shape[0], sm.shape[1], -1).sum(-1) == 0).sum()
    print(f'\nZero soft_map patches in batch: {n_zero.item()} '
          f'(expected 0 with new pipeline)')

    print('\nDataloader check complete.')