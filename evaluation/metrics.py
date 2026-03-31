"""
evaluation/metrics.py

Reusable evaluation metrics for WMH segmentation.
Used consistently across all three ablation conditions:
  - Classical only (classical_seg.nii vs wmh.nii)
  - U-Net baseline (model output vs wmh.nii)
  - Guided U-Net (model output vs wmh.nii)

Metrics implemented:
  - DICE coefficient
  - Hausdorff distance (95th percentile)
  - Lesion-level F1 (precision/recall over individual connected lesions)

All functions accept numpy arrays (not SimpleITK images) so they work
identically whether called from a notebook or from a PyTorch training loop.
"""

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.ndimage import binary_erosion, generate_binary_structure


# ── Voxel-level metrics ───────────────────────────────────────────────────────

def dice_coefficient(pred, gt):
    """
    Compute the DICE similarity coefficient between two binary arrays.

    DICE = 2 * |pred ∩ gt| / (|pred| + |gt|)

    Returns 1.0 if both pred and gt are empty (no lesions, correct prediction).
    Returns 0.0 if only one is empty (complete miss or false positive volume).

    Args:
        pred (np.ndarray): Binary predicted segmentation (0/1), any shape
        gt   (np.ndarray): Binary ground truth mask (0/1), same shape

    Returns:
        float: DICE score in [0, 1]
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0  # both empty — perfect agreement
    return float(2.0 * intersection / denom)


def hausdorff_distance_95(pred, gt, spacing=(1.0, 1.0, 3.0)):
    """
    Compute the 95th percentile Hausdorff distance between two binary masks.

    Standard Hausdorff measures the worst-case boundary error.
    The 95th percentile variant ignores the top 5% of outliers, making it
    more robust to small segmentation artifacts.

    Distance is computed in mm using the provided voxel spacing.

    Args:
        pred    (np.ndarray): Binary predicted segmentation (z, y, x)
        gt      (np.ndarray): Binary ground truth mask (z, y, x)
        spacing (tuple):      Voxel spacing in mm (x, y, z). Default=(1,1,3)

    Returns:
        float: 95th percentile Hausdorff distance in mm,
               or np.inf if pred or gt is empty
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() or not gt.any():
        return np.inf  # undefined if either mask is empty

    # Extract surface voxels via erosion
    struct = generate_binary_structure(pred.ndim, 1)  # 6-connectivity
    pred_surface = pred ^ binary_erosion(pred, struct)
    gt_surface   = gt   ^ binary_erosion(gt,   struct)

    # Get coordinates of surface voxels, scaled by spacing
    # Note: numpy array is (z, y, x), spacing tuple is (x, y, z)
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])

    pred_coords = np.argwhere(pred_surface) * spacing_zyx
    gt_coords   = np.argwhere(gt_surface)   * spacing_zyx

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.inf

    # Compute pairwise distances efficiently using broadcasting
    # For large surfaces this could be slow — chunked if needed
    def directed_hausdorff_95(a, b):
        # For each point in a, find min distance to b
        # Done in chunks to avoid memory issues on large surfaces
        chunk_size = 500
        min_dists = []
        for i in range(0, len(a), chunk_size):
            chunk = a[i:i+chunk_size]
            dists = np.sqrt(((chunk[:, None] - b[None, :]) ** 2).sum(axis=2))
            min_dists.append(dists.min(axis=1))
        return np.concatenate(min_dists)

    pred_to_gt = directed_hausdorff_95(pred_coords, gt_coords)
    gt_to_pred = directed_hausdorff_95(gt_coords, pred_coords)

    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(all_distances, 95))


# ── Lesion-level metrics ──────────────────────────────────────────────────────

def lesion_level_f1(pred, gt, iou_threshold=0.1):
    """
    Compute lesion-level precision, recall, and F1.

    Unlike voxel-level DICE, this treats each connected component (lesion)
    as a separate detection. A predicted lesion is a true positive if it
    overlaps with a ground truth lesion by at least iou_threshold (IoU).

    This metric penalises fragmented predictions and rewards finding
    individual lesions, which matters clinically.

    Args:
        pred          (np.ndarray): Binary predicted segmentation
        gt            (np.ndarray): Binary ground truth
        iou_threshold (float):      Minimum IoU to count as a true positive

    Returns:
        dict with keys: precision, recall, f1, n_pred, n_gt, n_tp
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    # Label connected components
    pred_labeled, n_pred = nd_label(pred)
    gt_labeled,   n_gt   = nd_label(gt)

    if n_gt == 0 and n_pred == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'n_pred': 0, 'n_gt': 0, 'n_tp': 0}

    if n_gt == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
                'n_pred': n_pred, 'n_gt': 0, 'n_tp': 0}

    if n_pred == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
                'n_pred': 0, 'n_gt': n_gt, 'n_tp': 0}

    # For each predicted lesion, check if it matches any GT lesion
    tp = 0
    matched_gt = set()

    for pred_id in range(1, n_pred + 1):
        pred_mask = (pred_labeled == pred_id)
        # Find GT lesions that overlap with this prediction
        overlapping_gt = np.unique(gt_labeled[pred_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        for gt_id in overlapping_gt:
            if gt_id in matched_gt:
                continue
            gt_mask = (gt_labeled == gt_id)
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(gt_id)
                break  # one GT lesion per prediction

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / n_gt   if n_gt   > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'n_pred':    n_pred,
        'n_gt':      n_gt,
        'n_tp':      tp,
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def evaluate_case(pred_arr, gt_arr, spacing=(1.0, 1.0, 3.0)):
    """
    Run all metrics for a single case and return a results dict.

    Args:
        pred_arr (np.ndarray): Binary predicted segmentation (z, y, x)
        gt_arr   (np.ndarray): Binary ground truth mask (z, y, x)
        spacing  (tuple):      Voxel spacing in mm (x, y, z)

    Returns:
        dict with keys: dice, hausdorff95, precision, recall, f1,
                        n_pred_lesions, n_gt_lesions, n_tp_lesions
    """
    pred_bin = (pred_arr > 0.5).astype(bool)
    gt_bin   = (gt_arr   > 0.5).astype(bool)

    dice  = dice_coefficient(pred_bin, gt_bin)
    hd95  = hausdorff_distance_95(pred_bin, gt_bin, spacing=spacing)
    lesion = lesion_level_f1(pred_bin, gt_bin)

    return {
        'dice':           round(dice, 4),
        'hausdorff95':    round(hd95, 2) if not np.isinf(hd95) else np.inf,
        'precision':      lesion['precision'],
        'recall':         lesion['recall'],
        'f1':             lesion['f1'],
        'n_pred_lesions': lesion['n_pred'],
        'n_gt_lesions':   lesion['n_gt'],
        'n_tp_lesions':   lesion['n_tp'],
    }