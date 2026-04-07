"""
preprocessing/compute_soft_maps.py

Runs a classical SimpleITK pipeline on each preprocessed FLAIR volume to produce:

  1. soft_map.nii      -- Soft probability map (values 0→1) used as guidance
                          signal injected into U-Net skip connections during training.
                          Derived from the CONTINUOUS FLAIR-T1 difference map
                          (before any thresholding), so it is never all-zero.

  2. classical_seg.nii -- Hard binary segmentation from the classical pipeline alone
                          (values 0 or 1). Used in ablation study to compare:
                          - Classical only (this file)
                          - U-Net only (baseline model output)
                          - Guided U-Net (guided model output)

Key design decision — two separate derivations from the same diff map:

    diff_map = FLAIR - T1 (within eroded brain mask)
        │
        ├─► soft_map path:
        │       clip negative values to 0 (WMH are positive in diff)
        │       clip upper tail at SOFT_MAP_CLIP_PERCENTILE (removes outliers)
        │       normalize to [0, 1] within brain mask
        │       Gaussian smooth
        │       → soft_map.nii  ← ALWAYS non-zero, smooth spatial prior
        │
        └─► classical_seg path:
                percentile threshold (DIFF_PERCENTILE) + absolute floor (DIFF_MIN_THRESHOLD)
                secondary FLAIR gate (FLAIR >= FLAIR_MIN_THRESHOLD)
                morphological opening
                connected component filtering
                → classical_seg.nii  ← hard binary, used for ablation only

    Previously, soft_map.nii was derived by Gaussian-smoothing classical_seg.nii.
    This meant empty classical segs (common for low-burden patients when no voxels
    survive the threshold + DIFF_MIN_THRESHOLD floor) produced all-zero soft maps,
    which when injected into U-Net skip connections would zero out feature maps
    entirely — actively harmful rather than neutral.

    Deriving soft_map from the continuous diff map fixes this: every patient has
    a non-zero FLAIR-T1 signal within their brain, so the soft map always provides
    a meaningful spatial prior regardless of lesion burden.

Pipeline per patient:
  Step 1 — Build eroded brain mask from T1
            Reference: Ghafoorian et al. 2017

  Step 2 — Compute FLAIR-T1 difference map within brain mask
            WMH are bright on FLAIR but moderately bright on T1, so diff is
            strongly positive for WMH and near zero for normal WM and CSF.
            References: Schmidt et al. 2012 (LST); Griffanti et al. 2016 (BIANCA)

  Step 3 — Derive soft_map from continuous diff map (NEW)
            Clip negatives to 0, clip upper tail, normalize to [0,1], Gaussian smooth.
            Saved as soft_map.nii.

  Step 4 — Derive classical_seg from thresholded diff map (unchanged)
            Percentile threshold + DIFF_MIN_THRESHOLD floor + FLAIR gate
            + morphological opening + connected component filtering.
            Saved as classical_seg.nii.

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

# --- Soft map parameters ---

# Upper percentile at which the diff map is clipped before normalisation.
# Prevents a small number of extreme outlier voxels (e.g. blood vessels)
# from compressing the effective range for WMH voxels.
SOFT_MAP_CLIP_PERCENTILE = 99.0

# --- Classical seg parameters (unchanged from grid search best) ---

# Secondary gate: FLAIR must be above this z-score to pass.
# Excludes CSF partial-volume voxels whose FLAIR is suppressed.
FLAIR_MIN_THRESHOLD = 0.5

# Per-patient percentile threshold on the brain-masked diff map.
# 96.0 → top 4% of diff values are candidate WMH.
DIFF_PERCENTILE = 96.0

# Absolute floor on the diff map. Prevents the percentile rule from selecting
# normal tissue when WMH burden is extremely low.
# NOTE: this is what caused all-zero maps previously — DIFF_MIN_THRESHOLD was
# raised above the per-patient percentile for low-burden patients. This only
# affects classical_seg now, NOT soft_map.
DIFF_MIN_THRESHOLD = 0.3

# --- Shared parameters ---

# Erosion radius (voxels) for the T1 brain mask.
BRAIN_EROSION_RADIUS = 7

# Morphological opening kernel radius (voxels) — classical seg only.
MORPH_RADIUS = 1

# Minimum connected component size (voxels) — classical seg only.
MIN_LESION_VOXELS = 15

# Gaussian smoothing sigma (voxels) — applied to both soft map and classical seg.
GAUSSIAN_SIGMA = 1.0

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/processed')

SITES = {
    'training': ['Utrecht', 'Singapore', 'Amsterdam/GE3T'],
    'test':     ['Utrecht', 'Singapore', 'Amsterdam/GE3T',
                 'Amsterdam/GE1T5', 'Amsterdam/Philips_VU .PETMR_01.'],
}


# ── Shared step ───────────────────────────────────────────────────────────────

def build_eroded_brain_mask(t1_image, erosion_radius=BRAIN_EROSION_RADIUS):
    """
    Build a brain mask from T1 non-zero voxels, eroded inward.

    Erosion removes the skull boundary, which is the primary source of false
    positives when thresholding FLAIR.

    Args:
        t1_image       (sitk.Image): Preprocessed z-scored T1 volume
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


def compute_diff_map(flair_image, t1_image, brain_mask_image):
    """
    Compute the FLAIR-T1 difference map within the eroded brain mask.

    Both images are z-scored so the diff is on a comparable scale.
    WMH: FLAIR >> T1  → large positive diff
    Normal WM: FLAIR ≈ T1 → diff near zero
    CSF: FLAIR suppressed, T1 dark → diff near zero or slightly negative

    Returns:
        np.ndarray: diff array (same shape as FLAIR), zeros outside brain mask
        np.ndarray: boolean brain mask array
    """
    flair_arr = sitk.GetArrayFromImage(flair_image).astype(np.float32)
    t1_arr    = sitk.GetArrayFromImage(t1_image).astype(np.float32)
    mask_arr  = sitk.GetArrayFromImage(brain_mask_image).astype(bool)

    diff_arr = np.zeros_like(flair_arr)
    diff_arr[mask_arr] = flair_arr[mask_arr] - t1_arr[mask_arr]

    return diff_arr, mask_arr


# ── Soft map path ─────────────────────────────────────────────────────────────

def diff_map_to_soft_map(diff_arr, mask_arr, flair_image,
                          clip_percentile=SOFT_MAP_CLIP_PERCENTILE,
                          gaussian_sigma=GAUSSIAN_SIGMA):
    """
    Convert a continuous FLAIR-T1 diff map into a soft probability map.

    Steps:
      1. Clip negative values to 0 — WMH are positive in the diff map;
         negative regions (e.g. CSF artifacts) carry no useful signal.
      2. Clip the upper tail at clip_percentile of brain diff values —
         prevents outlier voxels (vessels) from compressing the WMH range.
      3. Min-max normalize within brain mask to [0, 1].
      4. Gaussian smooth to produce spatially continuous probabilities.

    This map is NEVER all-zero — every brain has positive diff values.

    Args:
        diff_arr       (np.ndarray):  FLAIR-T1 diff array
        mask_arr       (np.ndarray):  Boolean brain mask array
        flair_image    (sitk.Image):  Source image for CopyInformation
        clip_percentile (float):      Upper clip percentile on brain diff values
        gaussian_sigma  (float):      Gaussian smoothing sigma in voxels

    Returns:
        sitk.Image: Soft map (float32, values in [0, 1])
    """
    # Step 1 — clip negatives
    positive_diff = np.clip(diff_arr, 0, None)

    # Step 2 — clip upper tail within brain mask
    brain_pos_vals = positive_diff[mask_arr]
    clip_val = float(np.percentile(brain_pos_vals, clip_percentile))
    clipped  = np.clip(positive_diff, 0, clip_val)

    # Step 3 — min-max normalize within brain mask
    brain_vals = clipped[mask_arr]
    v_min = float(brain_vals.min())
    v_max = float(brain_vals.max())
    denom = v_max - v_min if v_max > v_min else 1.0

    normalized = np.zeros_like(clipped, dtype=np.float32)
    normalized[mask_arr] = (clipped[mask_arr] - v_min) / denom

    # Step 4 — Gaussian smooth
    norm_img = sitk.GetImageFromArray(normalized)
    norm_img.CopyInformation(flair_image)
    smoothed = sitk.SmoothingRecursiveGaussian(norm_img, sigma=gaussian_sigma)

    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(0.0)
    clamp.SetUpperBound(1.0)
    return clamp.Execute(smoothed)


# ── Classical seg path ────────────────────────────────────────────────────────

def threshold_diff_map(diff_arr, mask_arr, flair_image,
                        diff_percentile=DIFF_PERCENTILE,
                        diff_min=DIFF_MIN_THRESHOLD,
                        flair_min=FLAIR_MIN_THRESHOLD):
    """
    Apply percentile threshold to the diff map to get a binary WMH candidate mask.

    Computes a per-patient percentile threshold on brain diff values, with an
    absolute floor (diff_min) to avoid selecting noise in very healthy patients,
    and a secondary FLAIR gate to exclude CSF partial-volume voxels.

    Note: if diff_min > per-patient percentile, the mask may be all-zero for
    low-burden patients. This is acceptable here because this output is only
    used for classical_seg.nii (ablation), NOT for the soft map.

    Returns:
        sitk.Image: Binary candidate mask
        float:      Effective threshold used
    """
    flair_arr = sitk.GetArrayFromImage(flair_image).astype(np.float32)

    brain_diff_vals      = diff_arr[mask_arr]
    percentile_threshold = np.percentile(brain_diff_vals, diff_percentile)
    threshold            = max(float(percentile_threshold), float(diff_min))

    candidate_arr = np.zeros_like(diff_arr, dtype=np.uint8)
    candidate_arr[(diff_arr >= threshold) & (flair_arr >= flair_min) & mask_arr] = 1

    out = sitk.GetImageFromArray(candidate_arr)
    out.CopyInformation(flair_image)
    return out, threshold


def morphological_opening(binary_image, radius=MORPH_RADIUS):
    """
    Apply morphological opening to remove small isolated detections.
    Reference: Gonzalez-Villà et al. 2016
    """
    f = sitk.BinaryMorphologicalOpeningImageFilter()
    f.SetKernelRadius(radius)
    f.SetForegroundValue(1)
    return f.Execute(binary_image)


def remove_small_components(binary_image, min_voxels=MIN_LESION_VOXELS):
    """
    Remove connected components smaller than min_voxels.
    Reference: Caligiuri et al. 2015, Neuroinformatics.

    Returns: cleaned binary image, n_removed, n_total
    """
    cc = sitk.ConnectedComponentImageFilter()
    cc.SetFullyConnected(False)
    labeled = cc.Execute(binary_image)
    n_total = cc.GetObjectCount()

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    labeled_arr = sitk.GetArrayFromImage(labeled)
    keep_mask   = np.zeros_like(labeled_arr, dtype=np.uint8)
    removed = 0

    for lid in range(1, n_total + 1):
        if stats.GetNumberOfPixels(lid) >= min_voxels:
            keep_mask[labeled_arr == lid] = 1
        else:
            removed += 1

    out = sitk.GetImageFromArray(keep_mask)
    out.CopyInformation(binary_image)
    return out, removed, n_total


# ── Per-case pipeline ─────────────────────────────────────────────────────────

def compute_maps_for_case(case_dir):
    """
    Run the full pipeline for one patient case.

    Reads:   {case_dir}/FLAIR.nii, {case_dir}/T1.nii
    Writes:  {case_dir}/soft_map.nii       (continuous, never zero)
             {case_dir}/classical_seg.nii  (hard binary, may be zero for low-burden)

    Returns a dict of statistics for logging.
    """
    flair = sitk.ReadImage(str(case_dir / 'FLAIR.nii'), sitk.sitkFloat32)
    t1    = sitk.ReadImage(str(case_dir / 'T1.nii'),    sitk.sitkFloat32)

    # Shared — eroded brain mask and diff map
    brain_mask           = build_eroded_brain_mask(t1)
    diff_arr, mask_arr   = compute_diff_map(flair, t1, brain_mask)

    # Soft map path — continuous diff → normalize → smooth (NEVER zero)
    soft_map = diff_map_to_soft_map(diff_arr, mask_arr, flair)
    sitk.WriteImage(soft_map, str(case_dir / 'soft_map.nii'))

    # Classical seg path — threshold → morph → components → binary
    binary, threshold = threshold_diff_map(diff_arr, mask_arr, flair)
    binary = morphological_opening(binary)
    binary, n_removed, n_total = remove_small_components(binary)
    sitk.WriteImage(sitk.Cast(binary, sitk.sitkUInt8),
                    str(case_dir / 'classical_seg.nii'))

    # Stats for logging
    soft_arr   = sitk.GetArrayFromImage(soft_map)
    binary_arr = sitk.GetArrayFromImage(binary)

    return {
        'threshold':     round(float(threshold), 3),
        'lesion_voxels': int(binary_arr.sum()),
        'n_removed':     n_removed,
        'n_total_comp':  n_total,
        'soft_map_max':  round(float(soft_arr.max()), 4),
        'soft_map_mean': round(float(soft_arr[soft_arr > 0].mean()), 4)
                         if soft_arr.max() > 0 else 0.0,
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
        for pdir in sorted(site_dir.iterdir(), key=lambda x: x.name):
            if (pdir / 'FLAIR.nii').exists() and (pdir / 'T1.nii').exists():
                cases.append(pdir)
    return cases


def run_split(split):
    print(f'\n{"="*60}')
    print(f'Computing soft maps — {split} set')
    print(f'Soft map:     continuous diff → clip@{SOFT_MAP_CLIP_PERCENTILE}pct → '
          f'normalize → Gaussian(σ={GAUSSIAN_SIGMA})')
    print(f'Classical seg: diff@{DIFF_PERCENTILE}pct + floor={DIFF_MIN_THRESHOLD} + '
          f'flair_min={FLAIR_MIN_THRESHOLD} + morph={MORPH_RADIUS} + '
          f'min_vox={MIN_LESION_VOXELS}')
    print(f'Brain erosion: {BRAIN_EROSION_RADIUS}px')
    print(f'{"="*60}\n')

    cases = discover_cases(split)
    print(f'Found {len(cases)} cases\n')

    print(f"{'#':<5} {'Case':<35} {'Threshold':>10} {'Lesion vox':>12} "
          f"{'Removed':>9} {'Soft max':>10} {'Soft mean':>10}")
    print('-' * 100)

    for i, case_dir in enumerate(cases):
        label = f'{case_dir.parent.name}/{case_dir.name}'
        try:
            s = compute_maps_for_case(case_dir)
            # Flag cases where classical seg is empty
            flag = ' ← empty seg' if s['lesion_voxels'] == 0 else ''
            print(f"{i+1:<5} {label:<35} "
                  f"{s['threshold']:>10} "
                  f"{s['lesion_voxels']:>12}{flag}"
                  f"{s['n_removed']:>9} "
                  f"{s['soft_map_max']:>10} "
                  f"{s['soft_map_mean']:>10}")
        except Exception as e:
            print(f'{i+1:<5} {label:<35} ERROR: {e}')

    print(f'\nDone. Outputs written to {PROCESSED_ROOT / split}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute soft maps and classical segmentations for WMH dataset'
    )
    parser.add_argument(
        '--split', action='append', choices=['training', 'test'],
        help='Split(s) to process. Defaults to training only.'
    )
    args = parser.parse_args()

    splits = args.split if args.split else ['training']
    for split in splits:
        run_split(split)