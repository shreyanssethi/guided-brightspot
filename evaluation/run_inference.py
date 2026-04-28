"""
evaluation/run_inference.py

Runs the pretrained Guided U-Net on the test set in data/project_submission/test/,
reports DICE and HD95 metrics, and saves sample input/output visualizations.

Usage:
    python evaluation/run_inference.py           # GPU 0
    python evaluation/run_inference.py --gpu 1   # different GPU

Outputs (written to data/project_submission/results/):
    metrics.csv        -- per-case DICE, HD95, precision, recall, F1 for all 110 cases
    sample_results.png -- FLAIR | Ground Truth | Prediction for a spread of cases
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
from monai.inferers import sliding_window_inference

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.models import build_guided
from evaluation.metrics import evaluate_case

# ── Paths ──────────────────────────────────────────────────────────────────────
SUBMISSION_DIR = REPO_ROOT / 'data' / 'project_submission'
TEST_DIR       = SUBMISSION_DIR / 'test'
CKPT_PATH      = REPO_ROOT / 'outputs' / 'checkpoints' / 'guided_best.pt'
RESULTS_DIR    = SUBMISSION_DIR / 'results'

# ── Inference config (matches training) ────────────────────────────────────────
PATCH_SIZE = (96, 96, 48)
SW_BATCH   = 2
OVERLAP    = 0.25
SPACING    = (1.0, 1.0, 3.0)


def load_nii(path):
    """Load a NIfTI file to a float32 numpy array shaped (x, y, z).

    SimpleITK's GetArrayFromImage returns (z, y, x). We reverse all axes to
    match MONAI's NibabelReader convention (x, y, z), which is what the model
    was trained and evaluated with.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)   # (z, y, x)
    return arr.transpose(2, 1, 0).astype(np.float32)  # → (x, y, z)


def discover_cases(test_dir):
    cases = []
    for flair_path in sorted(test_dir.rglob('FLAIR.nii')):
        case_dir = flair_path.parent
        parts    = case_dir.relative_to(test_dir).parts
        pid      = parts[-1]
        site     = str(Path(*parts[:-1]))
        cases.append({
            'site':     site,
            'pid':      pid,
            'flair':    case_dir / 'FLAIR.nii',
            't1':       case_dir / 'T1.nii',
            'soft_map': case_dir / 'soft_map.nii',
            'wmh':      case_dir / 'wmh.nii',
        })
    return cases


def predict_volume(model, flair, t1, soft, device):
    """Sliding-window inference on a full volume. Returns (H, W, D) uint8."""
    x  = torch.cat([
        torch.tensor(flair[None, None]).float().to(device),
        torch.tensor(t1[None, None]).float().to(device),
    ], dim=1)                                                    # (1, 2, H, W, D)
    sm = torch.tensor(soft[None, None]).float().to(device)       # (1, 1, H, W, D)

    with torch.no_grad():
        def _fwd(patch):
            sm_patch = F.interpolate(sm, size=patch.shape[2:],
                                     mode='trilinear', align_corners=False)
            return model(patch, sm_patch)

        logits = sliding_window_inference(x, PATCH_SIZE, SW_BATCH, _fwd, overlap=OVERLAP)

    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def best_slice(arr):
    """Return axial index of the slice with the most foreground voxels."""
    counts = arr.sum(axis=(0, 1))
    idx = int(counts.argmax())
    return max(1, min(idx, arr.shape[2] - 2))


def save_samples(cases, df, model, device, n_samples=9):
    """Save a grid: n_samples rows × 3 cols (FLAIR | GT | Prediction)."""
    df_sorted = df.sort_values('dice').reset_index(drop=True)
    indices   = np.linspace(0, len(df_sorted) - 1, n_samples, dtype=int)
    selected  = df_sorted.iloc[indices]

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 3))
    col_titles = ['FLAIR', 'Ground Truth', 'Prediction']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11)

    case_lookup = {(c['site'], c['pid']): c for c in cases}

    for row_i, (_, row) in enumerate(selected.iterrows()):
        case  = case_lookup[(row['site'], row['pid'])]
        flair = load_nii(case['flair'])
        t1    = load_nii(case['t1'])
        soft  = load_nii(case['soft_map'])
        gt    = load_nii(case['wmh']).astype(np.uint8)
        pred  = predict_volume(model, flair, t1, soft, device)

        sl   = best_slice(gt) if gt.max() > 0 else flair.shape[2] // 2
        f2d  = flair[:, :, sl]
        f2d  = (f2d - f2d.min()) / (f2d.max() - f2d.min() + 1e-8)
        gt2d = gt[:, :, sl]
        pr2d = pred[:, :, sl]

        row_label = f"{row['site']}/{row['pid']}\nDICE={row['dice']:.3f}"

        for col_i, (mask, color) in enumerate([(None, None), (gt2d, 'lime'), (pr2d, 'orangered')]):
            ax = axes[row_i, col_i]
            ax.imshow(f2d, cmap='gray')
            if mask is not None and mask.max() > 0:
                rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                r, g, b = mcolors.to_rgb(color)
                rgba[mask > 0] = [r, g, b, 0.6]
                ax.imshow(rgba)
            ax.axis('off')

        axes[row_i, 0].set_ylabel(row_label, fontsize=7, rotation=0,
                                   ha='right', va='center', labelpad=60)

    plt.tight_layout()
    out_path = RESULTS_DIR / 'sample_results.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample visualizations saved --> {out_path}")


def print_summary(df):
    print("\n=== Guided U-Net -- Test Set Results ===")
    print(f"{'Metric':<16} {'Mean':>7} +/- {'Std':<7}")
    print("-" * 36)
    for col, label in [('dice', 'DICE'), ('hausdorff95', 'HD95 (mm)')]:
        finite = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"{label:<16} {finite.mean():>7.3f} +/- {finite.std():<7.3f}")

    print("\nPer-site DICE (mean +/- std):")
    for site, grp in df.groupby('site'):
        print(f"  {site:<38} {grp['dice'].mean():.3f} +/- {grp['dice'].std():.3f}  (n={len(grp)})")


def parse_args():
    p = argparse.ArgumentParser(
        description='Run Guided U-Net inference on the project_submission test set'
    )
    p.add_argument('--gpu', type=int, default=0, help='GPU index (default: 0)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = build_guided().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Discover cases
    cases = discover_cases(TEST_DIR)
    if not cases:
        raise RuntimeError(f"No test cases found under {TEST_DIR}")
    print(f"Found {len(cases)} test cases\n")

    # Inference + metrics
    records = []
    for i, case in enumerate(cases, 1):
        flair = load_nii(case['flair'])
        t1    = load_nii(case['t1'])
        soft  = load_nii(case['soft_map'])
        gt    = load_nii(case['wmh']).astype(np.uint8)

        pred    = predict_volume(model, flair, t1, soft, device)
        metrics = evaluate_case(pred, gt, spacing=SPACING)
        records.append({'site': case['site'], 'pid': case['pid'], **metrics})
        print(f"  [{i:3d}/{len(cases)}] {case['site']}/{case['pid']:<6}  "
              f"DICE={metrics['dice']:.3f}  HD95={metrics['hausdorff95']:.1f} mm")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nPer-case metrics saved --> {csv_path}")

    print_summary(df)

    print("\nGenerating sample visualizations...")
    save_samples(cases, df, model, device)
    print("\nDone.")


if __name__ == '__main__':
    main()
