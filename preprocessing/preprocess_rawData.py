"""
Preprocesses WMH dataset (training or test) into a common space.

Steps per case:
  1. Resample FLAIR, T1, and WMH mask to target spacing (1.0, 1.0, 3.0)
  2. Crop/pad to target shape (200, 200, 48)
  3. Binarize mask: 1 = WMH, 0 = everything else (treat label 2, which is other pathologies, as background)
  4. Z-score normalize FLAIR and T1 on brain voxels only (arr > 0)
  5. Save to data/processed/{training|test}/{site}/{patient_id}/

Usage:
  python preprocess_rawData.py                                              # process training set
  python preprocess_rawData.py --split test                                 # process test set
  python preprocess_rawData.py --split training --split test                # process both
  python preprocess_rawData.py --split training --split test  --verify      # verify flag prints shapes/spacing/mask for 5 samples as sanity check
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# Target preprocessing parameters
TARGET_SPACING = (1.0, 1.0, 3.0)   # x, y, z in mm
TARGET_SIZE    = (200, 200, 48)     # x, y, z in voxels

# Data layout (TODO: Change to match your setup)
DATA_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/wmh_data')
OUT_ROOT  = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/processed')

# Sites per split — Amsterdam has subdirs so we have to list full relative paths
SITES = {
    'training': ['Utrecht', 'Singapore', 'Amsterdam/GE3T'],
    'test':  ['Utrecht', 'Singapore', 'Amsterdam/GE3T',
              'Amsterdam/GE1T5', 'Amsterdam/Philips_VU .PETMR_01.'],
}


# Core preprocessing functions 

def resample_image(image, new_spacing, interpolator=sitk.sitkLinear):
    """Resample a SimpleITK image to a new voxel spacing."""
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def crop_or_pad(image, target_size):
    """
    Crop or zero-pad a SimpleITK image to target_size (x, y, z).
    Padding is symmetric; cropping takes the center region.
    """
    current_size = list(image.GetSize())  # (x, y, z)
    lower_pad, upper_pad = [], []
    lower_crop, upper_crop = [], []

    for i in range(3):
        diff = target_size[i] - current_size[i]
        if diff >= 0:
            lower_pad.append(diff // 2)
            upper_pad.append(diff - diff // 2)
            lower_crop.append(0)
            upper_crop.append(0)
        else:
            lower_pad.append(0)
            upper_pad.append(0)
            lower_crop.append((-diff) // 2)
            upper_crop.append((-diff) - (-diff) // 2)

    # Crop first then pad
    if any(v > 0 for v in lower_crop + upper_crop):
        image = sitk.Crop(image, lower_crop, upper_crop)

    if any(v > 0 for v in lower_pad + upper_pad):
        image = sitk.ConstantPad(image, lower_pad, upper_pad, 0)

    return image


def zscore_normalize(image):
    """
    Z-score normalize a SimpleITK image using brain voxels only (value > 0).
    Returns a float32 SimpleITK image.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    brain_mask = arr > 0
    if brain_mask.sum() == 0:
        # Fallback: normalize over all voxels
        mean, std = arr.mean(), arr.std()
    else:
        mean = arr[brain_mask].mean()
        std  = arr[brain_mask].std()

    std = std if std > 1e-8 else 1.0  # avoid division by zero
    arr = (arr - mean) / std
    arr[~brain_mask] = 0.0  # keep background at zero

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def binarize_mask(mask_image):
    """
    Binarize WMH mask: label 1 = WMH (foreground), everything else = 0.
    Label 2 = other pathology, treated as background per project decision.
    """
    arr = sitk.GetArrayFromImage(mask_image)
    binary = (arr == 1).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(mask_image)
    return out


def preprocess_case(flair_path, t1_path, wmh_path, out_dir):
    """
    Full preprocessing pipeline for one patient case.
    Saves FLAIR.nii, T1.nii, wmh.nii to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the data from wmh_dataset
    flair = sitk.ReadImage(str(flair_path), sitk.sitkFloat32)
    t1    = sitk.ReadImage(str(t1_path),    sitk.sitkFloat32)
    wmh   = sitk.ReadImage(str(wmh_path),   sitk.sitkFloat32)

    # 1. Resample — use nearest neighbour for mask to avoid interpolation artifacts
    flair = resample_image(flair, TARGET_SPACING, sitk.sitkLinear)
    t1    = resample_image(t1,    TARGET_SPACING, sitk.sitkLinear)
    wmh   = resample_image(wmh,   TARGET_SPACING, sitk.sitkNearestNeighbor)

    # 2. Crop / pad to fixed size
    flair = crop_or_pad(flair, TARGET_SIZE)
    t1    = crop_or_pad(t1,    TARGET_SIZE)
    wmh   = crop_or_pad(wmh,   TARGET_SIZE)

    # 3. Binarize mask
    wmh = binarize_mask(wmh)

    # 4. Z-score normalize FLAIR and T1
    flair = zscore_normalize(flair)
    t1    = zscore_normalize(t1)

    # 5. Save the standardized versions
    sitk.WriteImage(flair, str(out_dir / 'FLAIR.nii'))
    sitk.WriteImage(t1,    str(out_dir / 'T1.nii'))
    sitk.WriteImage(wmh,   str(out_dir / 'wmh.nii'))


def discover_cases(split):
    """Return list of (site, patient_id, flair_path, t1_path, wmh_path) for a split."""
    split_root = DATA_ROOT / split
    cases = []
    for site in SITES[split]:
        site_dir = split_root / site
        if not site_dir.exists():
            print(f'  WARNING: {site_dir} not found, skipping')
            continue
        patient_dirs = sorted(
            [d for d in site_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name
        )
        for pdir in patient_dirs:
            flair = pdir / 'pre' / 'FLAIR.nii'
            t1    = pdir / 'pre' / 'T1.nii'
            wmh   = pdir / 'wmh.nii'
            if not all(p.exists() for p in [flair, t1, wmh]):
                print(f'  WARNING: missing files for {site}/{pdir.name}, skipping')
                continue
            cases.append({
                'site':       site,
                'patient_id': pdir.name,
                'flair':      flair,
                't1':         t1,
                'wmh':        wmh,
            })
    return cases


def run_split(split):
    """Preprocess all cases for a given split."""
    print(f'\n{"="*60}')
    print(f'Processing {split} set')
    print(f'{"="*60}')

    cases = discover_cases(split)
    print(f'Found {len(cases)} cases\n')

    for i, case in enumerate(cases):
        out_dir = OUT_ROOT / split / case['site'] / case['patient_id']

        # Skip if already processed
        if all((out_dir / f).exists() for f in ['FLAIR.nii', 'T1.nii', 'wmh.nii']):
            print(f'[{i+1:3d}/{len(cases)}] SKIP {case["site"]}/{case["patient_id"]} (already processed)')
            continue

        print(f'[{i+1:3d}/{len(cases)}] Processing {case["site"]}/{case["patient_id"]}')
        try:
            preprocess_case(case['flair'], case['t1'], case['wmh'], out_dir)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

    print(f'\nDone. Output saved to {OUT_ROOT / split}')


def verify_output(split):
    """Quick sanity check on processed output — print shapes and mask stats."""
    print(f'\n--- Verification ({split}) ---')
    split_out = OUT_ROOT / split
    checked = 0
    for case_dir in sorted(split_out.rglob('FLAIR.nii')):
        patient_dir = case_dir.parent
        flair = sitk.ReadImage(str(patient_dir / 'FLAIR.nii'))
        wmh   = sitk.ReadImage(str(patient_dir / 'wmh.nii'))
        wmh_arr = sitk.GetArrayFromImage(wmh)
        unique_vals = np.unique(wmh_arr)
        print(f'  {patient_dir.parent.name}/{patient_dir.name}: '
              f'shape={flair.GetSize()}, spacing={tuple(round(s,2) for s in flair.GetSpacing())}, '
              f'mask_vals={unique_vals}, lesion_vox={int(wmh_arr.sum())}')
        checked += 1
        if checked >= 5:
            print(f'  ... (showing first 5 only)')
            break


# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess WMH dataset')
    parser.add_argument(
        '--split', action='append', choices=['training', 'test'],
        help='Which split(s) to process. Defaults to training only. '
             'Use --split test or --split training --split test'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='After preprocessing, print a quick verification summary'
    )
    args = parser.parse_args()

    splits = args.split if args.split else ['training']

    for split in splits:
        run_split(split)
        if args.verify:
            verify_output(split)