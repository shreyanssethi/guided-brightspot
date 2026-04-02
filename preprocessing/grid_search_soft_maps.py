import sys
import time
import itertools
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ── Paths / repo setup ────────────────────────────────────────────────────────

REPO_ROOT = Path('/data/users/ssethi2/mmml_repos/guided-brightspot')
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import evaluate_case


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compute_soft_maps = load_module_from_path(
    'compute_soft_maps_module',
    REPO_ROOT / 'preprocessing' / 'compute_soft_maps.py',
)


# ── Config ────────────────────────────────────────────────────────────────────

SPLIT = 'training'
PROCESSED_ROOT = REPO_ROOT / 'data' / 'processed' / SPLIT
SITES = ['Utrecht', 'Singapore', 'Amsterdam/GE3T']
TARGET_SPACING = (1.0, 1.0, 3.0)  # x, y, z in mm

OUTPUT_DIR = REPO_ROOT / 'evaluation' / 'grid_search_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use fixed filenames so resume works naturally across reruns
SUMMARY_CSV = OUTPUT_DIR / f'grid_search_summary_{SPLIT}.csv'
PER_CASE_CSV = OUTPUT_DIR / f'grid_search_per_case_{SPLIT}.csv'


# ── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    'FLAIR_MIN_THRESHOLD':   [0.1, 0.3, 0.5],
    'DIFF_PERCENTILE':       [95.0, 96.0, 97.0, 98.0, 99.0],
    'DIFF_MIN_THRESHOLD':    [0.3, 0.5, 0.7],
    'BRAIN_EROSION_RADIUS':  [3, 5, 7],
    'MIN_LESION_VOXELS':     [3, 7, 15],
}

FIXED_PARAMS = {
    'MORPH_RADIUS': 1,
    'GAUSSIAN_SIGMA': 1.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

PARAM_COLUMNS = [
    'FLAIR_MIN_THRESHOLD',
    'DIFF_PERCENTILE',
    'DIFF_MIN_THRESHOLD',
    'BRAIN_EROSION_RADIUS',
    'MIN_LESION_VOXELS',
    'MORPH_RADIUS',
    'GAUSSIAN_SIGMA',
]


def iter_param_combinations(search_space: dict):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def set_compute_soft_maps_params(module, params: dict):
    for key, value in params.items():
        setattr(module, key, value)


def params_to_key(params: dict):
    """
    Stable tuple key for dedup / resume.
    """
    return tuple(params[col] for col in PARAM_COLUMNS)


def load_existing_results():
    """
    Load existing summary/per-case CSVs if present so we can resume.
    """
    summary_rows = []
    per_case_rows = []

    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        if 'rank' in summary_df.columns:
            summary_df = summary_df.drop(columns=['rank'])
        summary_rows = summary_df.to_dict(orient='records')
    else:
        summary_df = pd.DataFrame()

    if PER_CASE_CSV.exists():
        per_case_df = pd.read_csv(PER_CASE_CSV)
        per_case_rows = per_case_df.to_dict(orient='records')
    else:
        per_case_df = pd.DataFrame()

    return summary_rows, per_case_rows, summary_df, per_case_df


def get_completed_param_keys(summary_rows):
    completed = set()
    for row in summary_rows:
        try:
            key = tuple(row[col] for col in PARAM_COLUMNS)
            completed.add(key)
        except KeyError:
            continue
    return completed


def evaluate_current_outputs(combo_pbar=None):
    """
    Evaluate current classical_seg.nii outputs already written into the
    normal processed/training folders by compute_soft_maps.run_split(...).
    """
    per_case_rows = []

    for site in tqdm(SITES, desc='Evaluating sites', leave=False):
        site_dir = PROCESSED_ROOT / site
        if not site_dir.exists():
            tqdm.write(f'WARNING: {site_dir} not found, skipping')
            continue

        patient_dirs = sorted(site_dir.iterdir(), key=lambda x: x.name)
        for patient_dir in patient_dirs:
            classical_path = patient_dir / 'classical_seg.nii'
            gt_path = patient_dir / 'wmh.nii'

            if not classical_path.exists() or not gt_path.exists():
                tqdm.write(f'WARNING: missing files for {site}/{patient_dir.name}, skipping')
                continue

            pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(classical_path)))
            gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

            metrics = evaluate_case(pred_arr, gt_arr, spacing=TARGET_SPACING)
            metrics['site'] = site
            metrics['patient_id'] = patient_dir.name
            per_case_rows.append(metrics)

    if not per_case_rows:
        raise RuntimeError('No evaluation rows found. Check processed outputs and GT paths.')

    df = pd.DataFrame(per_case_rows)

    finite_hd = df.loc[df['hausdorff95'] != np.inf, 'hausdorff95']
    n_inf_hd = int((df['hausdorff95'] == np.inf).sum())

    summary_row = {
        'n_cases': len(df),
        'mean_dice': round(float(df['dice'].mean()), 4),
        'std_dice': round(float(df['dice'].std()), 4),
        'min_dice': round(float(df['dice'].min()), 4),
        'max_dice': round(float(df['dice'].max()), 4),

        'mean_hd95': round(float(finite_hd.mean()), 4) if len(finite_hd) > 0 else np.inf,
        'std_hd95': round(float(finite_hd.std()), 4) if len(finite_hd) > 0 else np.inf,
        'n_inf_hd95': n_inf_hd,

        'mean_precision': round(float(df['precision'].mean()), 4),
        'mean_recall': round(float(df['recall'].mean()), 4),
        'mean_f1': round(float(df['f1'].mean()), 4),

        'mean_n_pred_lesions': round(float(df['n_pred_lesions'].mean()), 4),
        'mean_n_gt_lesions': round(float(df['n_gt_lesions'].mean()), 4),
        'mean_n_tp_lesions': round(float(df['n_tp_lesions'].mean()), 4),
    }

    return per_case_rows, summary_row


def rank_summary_df(df: pd.DataFrame):
    if len(df) == 0:
        return df

    ranked = df.copy()

    # If resuming from an existing CSV, rank may already be present
    if 'rank' in ranked.columns:
        ranked = ranked.drop(columns=['rank'])

    ranked = ranked.sort_values(
        by=['mean_dice', 'mean_f1', 'mean_hd95', 'n_inf_hd95'],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    return ranked


def save_results(summary_rows, all_per_case_rows):
    summary_df = pd.DataFrame(summary_rows)
    summary_df = rank_summary_df(summary_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    per_case_df = pd.DataFrame(all_per_case_rows)
    per_case_df.to_csv(PER_CASE_CSV, index=False)

    return summary_df, per_case_df


# ── Main grid search ──────────────────────────────────────────────────────────

def main():
    combos = list(iter_param_combinations(SEARCH_SPACE))
    total = len(combos)

    summary_rows, all_per_case_rows, _, _ = load_existing_results()
    completed_keys = get_completed_param_keys(summary_rows)

    combos_to_run = []
    for combo in combos:
        current_params = {}
        current_params.update(FIXED_PARAMS)
        current_params.update(combo)
        if params_to_key(current_params) not in completed_keys:
            combos_to_run.append(current_params)

    print('=' * 80)
    print('Grid search for classical WMH soft-map pipeline')
    print(f'Split: {SPLIT}')
    print(f'Processed root: {PROCESSED_ROOT}')
    print(f'Total parameter combinations: {total}')
    print(f'Already completed: {len(completed_keys)}')
    print(f'Remaining to run: {len(combos_to_run)}')
    print(f'Summary CSV: {SUMMARY_CSV}')
    print(f'Per-case CSV: {PER_CASE_CSV}')
    print('Resume behavior: existing completed parameter sets are skipped.')
    print('=' * 80)
    print()

    if len(combos_to_run) == 0:
        print('Nothing left to run.')
        return

    start_all = time.time()

    combo_pbar = tqdm(combos_to_run, desc='Grid search combos', unit='combo')

    for current_params in combo_pbar:
        run_start = time.time()

        combo_pbar.set_postfix({
            'pct': current_params['DIFF_PERCENTILE'],
            'diff_min': current_params['DIFF_MIN_THRESHOLD'],
            'flair_min': current_params['FLAIR_MIN_THRESHOLD'],
            'erode': current_params['BRAIN_EROSION_RADIUS'],
            'minvox': current_params['MIN_LESION_VOXELS'],
        })

        tqdm.write('-' * 80)
        tqdm.write('Running combo:')
        for k, v in current_params.items():
            tqdm.write(f'  {k} = {v}')

        set_compute_soft_maps_params(compute_soft_maps, current_params)

        # Generate outputs using your existing pipeline
        compute_soft_maps.run_split(SPLIT)

        # Evaluate current outputs
        per_case_rows, summary_row = evaluate_current_outputs()

        elapsed = time.time() - run_start

        summary_row['elapsed_sec'] = round(elapsed, 2)
        for key, value in current_params.items():
            summary_row[key] = value
        summary_rows.append(summary_row)

        for row in per_case_rows:
            row = dict(row)
            row['elapsed_sec'] = round(elapsed, 2)
            for key, value in current_params.items():
                row[key] = value
            all_per_case_rows.append(row)

        # Save after every combo so interruption is safe
        summary_df, _ = save_results(summary_rows, all_per_case_rows)

        tqdm.write(
            f"Done | mean Dice={summary_row['mean_dice']:.4f} | "
            f"mean F1={summary_row['mean_f1']:.4f} | "
            f"mean HD95={summary_row['mean_hd95']} | "
            f"inf HD95={summary_row['n_inf_hd95']} | "
            f"elapsed={elapsed:.1f}s"
        )

        if len(summary_df) > 0:
            best = summary_df.iloc[0]
            tqdm.write(
                f"Current best | Dice={best['mean_dice']:.4f} | "
                f"F1={best['mean_f1']:.4f} | "
                f"pct={best['DIFF_PERCENTILE']} | "
                f"diff_min={best['DIFF_MIN_THRESHOLD']} | "
                f"flair_min={best['FLAIR_MIN_THRESHOLD']} | "
                f"erode={best['BRAIN_EROSION_RADIUS']} | "
                f"minvox={best['MIN_LESION_VOXELS']}"
            )

    total_elapsed = time.time() - start_all

    summary_df, per_case_df = save_results(summary_rows, all_per_case_rows)

    print('=' * 80)
    print('Grid search complete')
    print(f'Total elapsed: {total_elapsed / 60:.1f} min')
    print(f'Summary CSV:   {SUMMARY_CSV}')
    print(f'Per-case CSV:  {PER_CASE_CSV}')
    print('=' * 80)
    print()

    print('Top 10 parameter settings:')
    cols_to_show = [
        'rank',
        'mean_dice',
        'mean_f1',
        'mean_hd95',
        'n_inf_hd95',
        'FLAIR_MIN_THRESHOLD',
        'DIFF_PERCENTILE',
        'DIFF_MIN_THRESHOLD',
        'BRAIN_EROSION_RADIUS',
        'MIN_LESION_VOXELS',
        'MORPH_RADIUS',
        'GAUSSIAN_SIGMA',
    ]
    print(summary_df[cols_to_show].head(10).to_string(index=False))


if __name__ == '__main__':
    main()