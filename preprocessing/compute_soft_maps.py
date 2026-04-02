"""
preprocessing/compute_soft_maps2.py

Runs a classical SimpleITK pipeline on each preprocessed FLAIR volume to produce:

  1. soft_map.nii      -- Soft probability map (values 0→1) used as guidance
                          signal injected into U-Net skip connections during training.

  2. classical_seg.nii -- Hard binary segmentation from the classical pipeline alone
                          (values 0 or 1). Used in ablation study to compare:
                          - Classical only (this file)
                          - U-Net only (baseline model output)
                          - Guided U-Net (guided model output)

Pipeline (per patient):
  Step 1 — Build eroded brain mask from T1
            T1 non-zero voxels define the brain region. We erode this mask
            inward by BRAIN_EROSION_RADIUS voxels to eliminate the skull
            boundary, which is the primary source of false positives when
            thresholding FLAIR.
            Reference: Ghafoorian et al. 2017 (use of brain mask to restrict
            WMH candidate region)

  Step 2 — Percentile threshold on per-patient FLAIR-T1 difference map
            Both FLAIR and T1 are z-scored (mean=0, std=1 on brain voxels).
            WMH appear bright on FLAIR but only moderately bright on T1, so
            (FLAIR - T1) is strongly positive for WMH and near zero or negative
            for normal WM (bright on both) and CSF (dark on both).
            Because WMH occupies only a tiny fraction of brain voxels, we use
            a per-patient percentile threshold on the diff map within the brain
            mask rather than Otsu. This selects only the top tail of diff
            values (e.g. top 3% if using the 97th percentile), which is more
            appropriate for highly imbalanced lesion distributions.
            A secondary absolute diff floor further prevents marking normal
            tissue when WMH burden is very low. A secondary FLAIR gate
            (FLAIR >= FLAIR_MIN_THRESHOLD) further excludes CSF partial-volume
            voxels with suppressed (negative) FLAIR.
            References: Schmidt et al. 2012 (LST lesion growth algorithm uses
            T1 tissue segmentation + FLAIR); Griffanti et al. 2016 (BIANCA:
            T1+FLAIR multimodal input outperforms FLAIR-only).

  Step 3 — Morphological opening
            Removes small isolated detections while preserving genuine WMH.
            Reference: Gonzalez-Villà et al. 2016

  Step 4 — Connected component filtering
            Removes components smaller than MIN_LESION_VOXELS.
            Reference: Caligiuri et al. 2015

  Step 5 — Save classical_seg.nii (binary mask at this point)

  Step 6 — Gaussian smoothing → soft_map.nii
            Our novel design choice: converts hard binary mask to soft
            probabilities for injection into U-Net skip connections.

Usage:
  python preprocessing/compute_soft_maps2.py                  # training set
  python preprocessing/compute_soft_maps2.py --split test     # test set
  python preprocessing/compute_soft_maps2.py --split training --split test
"""

import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ── Parameters ────────────────────────────────────────────────────────────────

# Secondary gate applied after diff threshold: FLAIR must also be above this
# z-score to exclude CSF partial-volume voxels where FLAIR is suppressed.
FLAIR_MIN_THRESHOLD = 0.5

# Per-patient percentile threshold on the brain-masked diff map.
# Example: 97.0 keeps only the top 3% of diff values as WMH candidates.
DIFF_PERCENTILE = 96.0

# Absolute floor on the diff map. Prevents the percentile rule from selecting
# normal tissue when WMH burden is extremely low.
DIFF_MIN_THRESHOLD = 0.3

# Erosion radius (voxels) applied to the T1 brain mask before thresholding.
# Removes the skull boundary, which is the main source of false positives.
# 5 voxels at 1mm in-plane spacing removes ~5mm of outer brain boundary.
BRAIN_EROSION_RADIUS = 7

# Morphological opening kernel radius (voxels).
MORPH_RADIUS = 1

# Minimum connected component size (voxels) to keep.
MIN_LESION_VOXELS = 15

# Gaussian smoothing sigma (voxels) for soft map generation.
GAUSSIAN_SIGMA = 1.0

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/processed')

SITES = {
    'training': ['Utrecht', 'Singapore', 'Amsterdam/GE3T'],
    'test':     ['Utrecht', 'Singapore', 'Amsterdam/GE3T',
                 'Amsterdam/GE1T5', 'Amsterdam/Philips_VU .PETMR_01.'],
}


# ── Pipeline steps ────────────────────────────────────────────────────────────

def build_eroded_brain_mask(t1_image, erosion_radius=BRAIN_EROSION_RADIUS):
    """
    Build a brain mask from the T1 image and erode it inward.

    T1 non-zero voxels define the brain region. Erosion removes the outermost
    voxels of the brain boundary, eliminating the skull rim which is
    consistently the main source of false positives when thresholding FLAIR.

    Args:
        t1_image       (sitk.Image): Preprocessed T1 volume (z-scored)
        erosion_radius (int):        Erosion radius in voxels

    Returns:
        sitk.Image: Binary brain mask (1=brain interior, 0=outside/boundary)
    """
    t1_arr    = sitk.GetArrayFromImage(t1_image)
    brain_arr = (t1_arr != 0).astype(np.uint8)
    brain_img = sitk.GetImageFromArray(brain_arr)
    brain_img.CopyInformation(t1_image)

    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(erosion_radius)
    erode_filter.SetForegroundValue(1)
    return erode_filter.Execute(brain_img)


def percentile_diff_threshold(flair_image, t1_image, brain_mask_image,
                              diff_percentile=DIFF_PERCENTILE,
                              diff_min=DIFF_MIN_THRESHOLD,
                              flair_min=FLAIR_MIN_THRESHOLD):
    """
    Per-patient percentile threshold on the FLAIR-T1 difference map.

    Computes (FLAIR - T1) within the eroded brain mask, then selects candidate
    voxels whose diff value is above a patient-specific percentile threshold
    (e.g. 97th percentile = top 3% of brain voxels by diff value).

    Because WMH occupies only a tiny fraction of brain voxels, this is more
    appropriate than Otsu for the highly imbalanced diff-value distribution.
    An absolute floor on the diff map is also enforced to avoid selecting
    normal tissue in cases with very low WMH burden.

    A secondary FLAIR gate (flair_arr >= flair_min) removes CSF partial-volume
    voxels whose FLAIR signal is suppressed regardless of the diff value.

    Args:
        flair_image      (sitk.Image): Z-scored FLAIR volume
        t1_image         (sitk.Image): Z-scored T1 volume
        brain_mask_image (sitk.Image): Eroded binary brain mask
        diff_percentile  (float):      Percentile on brain-masked diff values
        diff_min         (float):      Absolute minimum diff threshold
        flair_min        (float):      Minimum FLAIR z-score gate

    Returns:
        sitk.Image: Binary candidate WMH mask (1=candidate, 0=background)
        float:      Effective per-patient diff threshold used for logging

    References: Schmidt et al. 2012 (LST); Griffanti et al. 2016 (BIANCA).
    """
    flair_arr = sitk.GetArrayFromImage(flair_image)
    t1_arr    = sitk.GetArrayFromImage(t1_image)
    mask_arr  = sitk.GetArrayFromImage(brain_mask_image).astype(bool)

    diff_arr = flair_arr - t1_arr

    # Compute percentile threshold on brain-masked diff values only
    brain_diff_vals = diff_arr[mask_arr]
    percentile_threshold = np.percentile(brain_diff_vals, diff_percentile)
    threshold = max(float(percentile_threshold), float(diff_min))

    # Apply threshold + secondary FLAIR gate
    candidate_arr = np.zeros_like(flair_arr, dtype=np.uint8)
    candidate_arr[(diff_arr >= threshold) & (flair_arr >= flair_min) & mask_arr] = 1

    out = sitk.GetImageFromArray(candidate_arr)
    out.CopyInformation(flair_image)
    return out, threshold


def morphological_opening(binary_image, radius=MORPH_RADIUS):
    """
    Apply morphological opening (erosion then dilation) to a binary image.

    Removes small isolated detections smaller than the kernel radius while
    preserving larger connected WMH structures.

    Reference: Gonzalez-Villà et al. 2016
    """
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelRadius(radius)
    opening_filter.SetForegroundValue(1)
    return opening_filter.Execute(binary_image)


def remove_small_components(binary_image, min_voxels=MIN_LESION_VOXELS):
    """
    Remove connected components smaller than min_voxels.

    Reference: Caligiuri et al. 2015, Neuroinformatics.

    Returns:
        cleaned binary image, n_removed, n_total_components
    """
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    labeled      = cc_filter.Execute(binary_image)
    n_components = cc_filter.GetObjectCount()

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    labeled_arr = sitk.GetArrayFromImage(labeled)
    keep_mask   = np.zeros_like(labeled_arr, dtype=np.uint8)
    removed = 0

    for label_id in range(1, n_components + 1):
        if stats.GetNumberOfPixels(label_id) >= min_voxels:
            keep_mask[labeled_arr == label_id] = 1
        else:
            removed += 1

    out = sitk.GetImageFromArray(keep_mask)
    out.CopyInformation(binary_image)
    return out, removed, n_components


def gaussian_smooth_to_soft_map(binary_image, sigma=GAUSSIAN_SIGMA):
    """
    Smooth a binary mask with a Gaussian filter to produce a soft probability map.

    Converts hard 0/1 classical segmentation into continuous probability-like
    values for injection into U-Net skip connections.

    Note: This step is our novel design choice, not from the WMH literature.
    """
    float_img = sitk.Cast(binary_image, sitk.sitkFloat32)
    smoothed  = sitk.SmoothingRecursiveGaussian(float_img, sigma=sigma)

    clamp_filter = sitk.ClampImageFilter()
    clamp_filter.SetLowerBound(0.0)
    clamp_filter.SetUpperBound(1.0)
    return clamp_filter.Execute(smoothed)


# ── Per-case pipeline ─────────────────────────────────────────────────────────

def compute_maps_for_case(case_dir):
    """
    Run the full classical pipeline for one patient case.

    Reads:   {case_dir}/FLAIR.nii
             {case_dir}/T1.nii
    Writes:  {case_dir}/classical_seg.nii
             {case_dir}/soft_map.nii

    Returns a dict of statistics for logging.
    """
    flair = sitk.ReadImage(str(case_dir / 'FLAIR.nii'), sitk.sitkFloat32)
    t1    = sitk.ReadImage(str(case_dir / 'T1.nii'),    sitk.sitkFloat32)

    # Step 1 — Eroded brain mask from T1
    brain_mask = build_eroded_brain_mask(t1, erosion_radius=BRAIN_EROSION_RADIUS)

    # Step 2 — Per-patient percentile threshold on FLAIR-T1 difference map
    binary, threshold = percentile_diff_threshold(
        flair,
        t1,
        brain_mask,
        diff_percentile=DIFF_PERCENTILE,
        diff_min=DIFF_MIN_THRESHOLD,
        flair_min=FLAIR_MIN_THRESHOLD,
    )

    # Step 3 — Morphological opening
    binary = morphological_opening(binary, radius=MORPH_RADIUS)

    # Step 4 — Remove small components
    binary, n_removed, n_total = remove_small_components(binary, min_voxels=MIN_LESION_VOXELS)

    # Step 5 — Save hard binary segmentation
    sitk.WriteImage(sitk.Cast(binary, sitk.sitkUInt8),
                    str(case_dir / 'classical_seg.nii'))

    # Step 6 — Gaussian smooth → soft map
    soft_map = gaussian_smooth_to_soft_map(binary, sigma=GAUSSIAN_SIGMA)
    sitk.WriteImage(soft_map, str(case_dir / 'soft_map.nii'))

    binary_arr = sitk.GetArrayFromImage(binary)
    soft_arr   = sitk.GetArrayFromImage(soft_map)

    return {
        'threshold':           round(float(threshold), 3),
        'n_components_before': n_total,
        'n_removed':           n_removed,
        'lesion_voxels':       int(binary_arr.sum()),
        'soft_map_max':        round(float(soft_arr.max()), 4),
    }


# ── Discovery and runner ──────────────────────────────────────────────────────

def discover_cases(split):
    split_dir = PROCESSED_ROOT / split
    cases = []
    for site in SITES[split]:
        site_dir = split_dir / site
        if not site_dir.exists():
            print(f'  WARNING: {site_dir} not found, skipping')
            continue
        for patient_dir in sorted(site_dir.iterdir(), key=lambda x: x.name):
            if (patient_dir / 'FLAIR.nii').exists() and \
               (patient_dir / 'T1.nii').exists():
                cases.append(patient_dir)
    return cases


def run_split(split):
    print(f'\n{"="*60}')
    print(f'Computing soft maps — {split} set')
    print(f'Parameters: percentile_on_diff_map (per-patient), '
          f'diff_percentile={DIFF_PERCENTILE}, '
          f'diff_min={DIFF_MIN_THRESHOLD}, '
          f'flair_min={FLAIR_MIN_THRESHOLD}, '
          f'brain_erosion={BRAIN_EROSION_RADIUS}px, '
          f'morph_radius={MORPH_RADIUS}, '
          f'min_lesion_voxels={MIN_LESION_VOXELS}, '
          f'gaussian_sigma={GAUSSIAN_SIGMA}')
    print(f'{"="*60}\n')

    cases = discover_cases(split)
    print(f'Found {len(cases)} cases\n')

    print(f"{'#':<5} {'Case':<35} {'Threshold':>10} {'Components':>12} "
          f"{'Removed':>9} {'Lesion vox':>12} {'Soft max':>10}")
    print('-' * 100)

    for i, case_dir in enumerate(cases):
        label = f'{case_dir.parent.name}/{case_dir.name}'
        try:
            stats = compute_maps_for_case(case_dir)
            print(f"{i+1:<5} {label:<35} "
                  f"{stats['threshold']:>10} "
                  f"{stats['n_components_before']:>12} "
                  f"{stats['n_removed']:>9} "
                  f"{stats['lesion_voxels']:>12} "
                  f"{stats['soft_map_max']:>10}")
        except Exception as e:
            print(f'{i+1:<5} {label:<35} ERROR: {e}')

    print(f'\nDone. Outputs written to {PROCESSED_ROOT / split}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute classical soft maps and segmentations for WMH dataset'
    )
    parser.add_argument(
        '--split', action='append', choices=['training', 'test'],
        help='Split(s) to process. Defaults to training only.'
    )
    args = parser.parse_args()

    splits = args.split if args.split else ['training']
    for split in splits:
        run_split(split)