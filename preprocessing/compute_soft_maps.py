"""
Runs a classical SimpleITK pipeline on each preprocessed FLAIR volume to produce:

  1. soft_map.nii   --> Soft probability map (values 0→1) used as guidance signal (injected into U-Net skip connections during training). Produced by Gaussian-smoothing the cleaned binary mask

  2. classical_seg.nii --> Hard binary segmentation from the classical pipeline alone (will have values 0 or 1). Used in ablation study to compare:
                          - Classical only (this file)
                          - U-Net only (baseline model output)
                          - Hybrid U-Net (guided model output)

Pipeline (This will run PER patient):
  Step 1 — Otsu threshold on brain voxels of FLAIR
            Automatically finds the optimal intensity cutoff separating bright/hyperintense tissue from normal brain.

  Step 2 — Morphological opening (erosion --> dilation)
            Removes small isolated false positive detections ("speckle" noise) while preserving larger genuine WMH clusters.

  Step 3 — Connected component filtering
            Removes any remaining connected component smaller than MIN_LESION_VOXELS. Filters CSF-adjacent false positives that make it through morphological cleanup.

  Step 4 — Save classical_seg.nii (binary mask at this point, for ablation studies)

  Step 5 — Gaussian smoothing of binary mask to get soft_map.nii
            Converts the hard binary mask into soft probabilities.
            Voxels near lesion centers get values close to 1,
            boundary voxels get intermediate values, background near 0.
            This is part of the novel design choice for the hybrid approach I'm trying

Some of the papers I looked at to motivate this pipeline:
Caligiuri et al. 2015, Neuroinformatics --> https://link.springer.com/article/10.1007/s12021-015-9260-y
Gonzalez-Villà et al. 2016 --> https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2016.00033/full

Usage:
  python preprocessing/compute_soft_maps.py                  # training set
  python preprocessing/compute_soft_maps.py --split test     # test set
  python preprocessing/compute_soft_maps.py --split training --split test
"""

import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# {── Parameters ────────────────────────────────────────────────────────────}

# Morphological opening kernel radius (in voxels)
# Radius of 1 removes isolated single-voxel detections
MORPH_RADIUS = 1

# Minimum connected component size (voxels) to keep
# Components smaller than this are treated as false positives
# 3mm slice thickness × ~1mm in-plane --> 3 voxels ≈ ~3mm³ minimum lesion
MIN_LESION_VOXELS = 3

# Gaussian smoothing sigma (in voxels) for soft map generation
# Higher sigma = smoother/wider soft probabilities around each lesion
# 1.0 is a conservative starting choice that keeps the map tightly around lesions
GAUSSIAN_SIGMA = 1.0


# {── Paths (Replace with yours) ────────────────────────────────────────────────────────────}
PROCESSED_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot/data/processed')

SITES = {
    'training': ['Utrecht', 'Singapore', 'Amsterdam/GE3T'],
    'test':     ['Utrecht', 'Singapore', 'Amsterdam/GE3T',
                 'Amsterdam/GE1T5', 'Amsterdam/Philips_VU .PETMR_01.'],
}


#{ ── Pipeline steps ────────────────────────────────────────────────────────────}

def otsu_threshold(flair_image):
    """
    Apply Otsu thresholding to brain voxels of the FLAIR image (Should find the intensity threshold to minimize intra-class variance)
    Applied only to brain voxels (intensity > 0) so the background does not skew the threshold estimate.
    Returns a binary SimpleITK image (1 = candidate WMH, 0 = background).
    """
    arr = sitk.GetArrayFromImage(flair_image)

    # Mask (non-zero voxels) to restrict Otsu to brain tissue
    brain_mask = (arr > 0).astype(np.uint8)
    brain_mask_img = sitk.GetImageFromArray(brain_mask)
    brain_mask_img.CopyInformation(flair_image)

    # Run Otsu: insideValue=1 means voxels ABOVE threshold are foreground
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    otsu_filter.SetMaskValue(1)
    binary = otsu_filter.Execute(flair_image, brain_mask_img)

    # Ensure background (arr == 0) stays 0 regardless of threshold
    binary_arr = sitk.GetArrayFromImage(binary)
    binary_arr[arr <= 0] = 0
    out = sitk.GetImageFromArray(binary_arr.astype(np.uint8))
    out.CopyInformation(flair_image)

    return out, otsu_filter.GetThreshold()


def morphological_opening(binary_image, radius=MORPH_RADIUS):
    """
    Apply morphological opening (erosion followed by dilation) to a binary image.

    Opening removes small isolated detections (radius < kernel) while
    preserving larger connected structures. This cleans up single-voxel
    false positives common near CSF boundaries after thresholding.
    """
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelRadius(radius)
    opening_filter.SetForegroundValue(1)
    return opening_filter.Execute(binary_image)


def remove_small_components(binary_image, min_voxels=MIN_LESION_VOXELS):
    """
    Remove connected components smaller than min_voxels.

    After thresholding and morphological opening, small isolated clusters
    remain — often near CSF or vessel boundaries. Connected component
    analysis labels each cluster; we discard those below the size threshold.

    Returns cleaned binary image and count of removed components.
    """
    # Label connected components
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)  # 6-connectivity (face neighbours only)
    labeled = cc_filter.Execute(binary_image)
    n_components = cc_filter.GetObjectCount()

    # Get sizes of each component
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    # Build mask of components to keep
    labeled_arr = sitk.GetArrayFromImage(labeled)
    keep_mask   = np.zeros_like(labeled_arr, dtype=np.uint8)
    removed = 0

    for label in range(1, n_components + 1):
        size = stats.GetNumberOfPixels(label)
        if size >= min_voxels:
            keep_mask[labeled_arr == label] = 1
        else:
            removed += 1

    out = sitk.GetImageFromArray(keep_mask)
    out.CopyInformation(binary_image)
    return out, removed, n_components


def gaussian_smooth_to_soft_map(binary_image, sigma=GAUSSIAN_SIGMA):
    """
    Smooth a binary mask with a Gaussian filter to produce a soft probability map.

    This converts the hard 0/1 classical segmentation into a continuous
    probability-like map where:
      - Voxels at lesion centers → values close to 1
      - Voxels at lesion boundaries → intermediate values (0.3–0.7)
      - Background far from lesions → values close to 0

    This soft map is injected into U-Net skip connections as a spatial
    attention signal. Using soft values rather than hard binary allows the
    network to learn to partially trust uncertain boundary regions.

    Note: This step is our novel design choice and not directly taken
    from the WMH literature.
    """
    # Cast to float first for Gaussian
    float_img = sitk.Cast(binary_image, sitk.sitkFloat32)
    smoothed  = sitk.SmoothingRecursiveGaussian(float_img, sigma=sigma)

    # Clip to [0, 1] — Gaussian can produce tiny negative values at boundaries
    clamp_filter = sitk.ClampImageFilter()
    clamp_filter.SetLowerBound(0.0)
    clamp_filter.SetUpperBound(1.0)
    soft_map = clamp_filter.Execute(smoothed)

    return soft_map


# ── Per-case pipeline ─────────────────────────────────────────────────────────

def compute_maps_for_case(case_dir):
    """
    Run the full classical pipeline for one patient case.

    Reads:   {case_dir}/FLAIR.nii
    Writes:  {case_dir}/classical_seg.nii  (hard binary segmentation)
             {case_dir}/soft_map.nii       (soft probability map for U-Net)

    Returns a dict of statistics for logging.
    """
    flair_path        = case_dir / 'FLAIR.nii'
    classical_seg_out = case_dir / 'classical_seg.nii'
    soft_map_out      = case_dir / 'soft_map.nii'

    flair = sitk.ReadImage(str(flair_path), sitk.sitkFloat32)

    # Step 1 —-> Otsu threshold
    binary, threshold = otsu_threshold(flair)

    # Step 2 —-> Morphological opening
    binary = morphological_opening(binary, radius=MORPH_RADIUS)

    # Step 3 —-> Remove small connected components
    binary, n_removed, n_total = remove_small_components(binary, min_voxels=MIN_LESION_VOXELS)

    # Step 4 —-> Save hard binary segmentation (classical_seg)
    sitk.WriteImage(sitk.Cast(binary, sitk.sitkUInt8), str(classical_seg_out))

    # Step 5 —-> Gaussian smooth for soft map
    soft_map = gaussian_smooth_to_soft_map(binary, sigma=GAUSSIAN_SIGMA)
    sitk.WriteImage(soft_map, str(soft_map_out))

    # Collect stats for logging
    binary_arr   = sitk.GetArrayFromImage(binary)
    soft_arr     = sitk.GetArrayFromImage(soft_map)
    lesion_vox   = int(binary_arr.sum())
    soft_max     = float(soft_arr.max())

    return {
        'threshold':   round(threshold, 3),
        'n_components_before': n_total,
        'n_removed':   n_removed,
        'lesion_voxels': lesion_vox,
        'soft_map_max':  round(soft_max, 4),
    }


# ── Runner ──────────────────────────────────────────────────────

def discover_cases(split):
    """Return sorted list of processed case directories for a split."""
    split_dir = PROCESSED_ROOT / split
    cases = []
    for site in SITES[split]:
        site_dir = split_dir / site
        if not site_dir.exists():
            print(f'  WARNING: {site_dir} not found, skipping')
            continue
        for patient_dir in sorted(site_dir.iterdir(), key=lambda x: x.name):
            if (patient_dir / 'FLAIR.nii').exists():
                cases.append(patient_dir)
    return cases


def run_split(split):
    print(f'\n{"="*60}')
    print(f'Computing soft maps — {split} set')
    print(f'Parameters: morph_radius={MORPH_RADIUS}, '
          f'min_lesion_voxels={MIN_LESION_VOXELS}, '
          f'gaussian_sigma={GAUSSIAN_SIGMA}')
    print(f'{"="*60}\n')

    cases = discover_cases(split)
    print(f'Found {len(cases)} cases\n')

    print(f"{'#':<5} {'Case':<35} {'Threshold':>10} {'Components':>12} "
          f"{'Removed':>9} {'Lesion vox':>12} {'Soft max':>10}")
    print('-' * 100)

    for i, case_dir in enumerate(cases):
        # Skip if already computed
        if (case_dir / 'soft_map.nii').exists() and \
           (case_dir / 'classical_seg.nii').exists():
            print(f'{i+1:<5} {str(case_dir.parent.name+"/"+case_dir.name):<35} SKIP (already computed)')
            continue

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


# ── Main ───────────────────────────────────────────────────────────────

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