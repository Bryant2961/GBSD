"""Diagnose NLL/std scale for uncertainty variants, separating
guard-validation points from blind-test points.

v3.37 changes:
  * Each (case, variant) row is now reported TWICE: once on the guard
    subset (where the std was fit) and once on the blind-test subset
    (where the std must generalize).
  * If a saved prediction npz does not contain a guard_mask (legacy
    runs), the script falls back to the std_calibration_mask, and if
    that is also absent it reports on the full grid with `eval_grid`
    marked as 'full_grid_no_split'.
  * Adds a 'nll_formula' column and a 'std_floor' column so the JCP
    reviewer can audit the calibration setup at a glance.

The fix for the 5.4e5 -> -11.0 NLL span the reviewer flagged: report
the raw uncalibrated std on the blind subset, and the calibrated std
on the same subset. If the gap is due to calibration generalizing
correctly, both subsets will show consistent behavior. If the gap is
due to fitting noise on the calibration subset, the blind subset will
show degraded NLL.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import (
    CASES,
    TABLE_DIR,
    collect_runs,
    ensure_output_dirs,
    error_std_corr,
    load_npz,
    nll_gaussian,
    coverage95,
    avg_interval_width,
    write_csv,
)


STD_KEYS = [
    ('final_calibrated_std', 'bayesian_mean', 'bayesian_std'),
    ('raw_mc_std', 'bayesian_dense_mean', 'bayesian_raw_std'),
    ('dense_student_std', 'bayesian_dense_mean', 'bayesian_dense_std'),
    ('structured_candidate_std', 'bayesian_structured_mean',
     'bayesian_structured_std'),
]


def _extract_masks(data) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return (guard_mask, blind_mask, source_label) at flat-index level.

    Source labels:
      * 'guard_blind_split'           — pre-declared masks from blind_split
      * 'calibration_mask_complement' — legacy checkerboard mask present
      * 'no_split'                    — no mask info; return None, None
    """
    if 'guard_mask' in data and 'blind_test_mask' in data:
        gm = np.asarray(data['guard_mask']).astype(bool).reshape(-1)
        bm = np.asarray(data['blind_test_mask']).astype(bool).reshape(-1)
        return gm, bm, 'guard_blind_split'
    if 'std_calibration_mask' in data:
        mask = np.asarray(data['std_calibration_mask']).astype(bool).reshape(-1)
        return mask, ~mask, 'calibration_mask_complement'
    return None, None, 'no_split'


def _row(record_name, seed, case, variant, mean_key, std_key,
         mean, std, exact, mask, eval_grid_label, std_floor):
    if mask is not None:
        flat_mean = np.asarray(mean).reshape(-1)[mask]
        flat_std = np.asarray(std).reshape(-1)[mask]
        flat_exact = np.asarray(exact).reshape(-1)[mask]
        n_eval = int(mask.sum())
    else:
        flat_mean = np.asarray(mean).reshape(-1)
        flat_std = np.asarray(std).reshape(-1)
        flat_exact = np.asarray(exact).reshape(-1)
        n_eval = int(flat_mean.size)
    flat_std = np.maximum(flat_std, std_floor)
    return {
        'run_id': record_name, 'seed': seed, 'case': case,
        'variant': variant,
        'mean_key': mean_key, 'std_key': std_key,
        'eval_grid': eval_grid_label,
        'n_eval': n_eval,
        'std_floor': std_floor,
        'std_min': float(np.min(flat_std)),
        'std_median': float(np.median(flat_std)),
        'std_mean': float(np.mean(flat_std)),
        'std_max': float(np.max(flat_std)),
        'floor_fraction': float(np.mean(flat_std == std_floor)),
        'coverage95': coverage95(flat_mean, flat_std, flat_exact),
        'corr': error_std_corr(flat_mean, flat_std, flat_exact),
        'nll': nll_gaussian(flat_mean, flat_std, flat_exact),
        'avg_interval_width': avg_interval_width(flat_std),
        'nll_formula': '0.5*log(2*pi*std^2) + 0.5*(error^2)/std^2; constant retained',
    }


def build_rows(seed0_root: str | None, std_floor: float):
    rows = []
    for record in collect_runs(seed0_root):
        for case in CASES:
            data = load_npz(record, case)
            if data is None or 'exact' not in data:
                continue
            exact = data['exact']
            guard_mask, blind_mask, split_source = _extract_masks(data)

            for variant, mean_key, std_key in STD_KEYS:
                if mean_key not in data or std_key not in data:
                    continue
                mean = data[mean_key]
                std = data[std_key]

                if guard_mask is not None and blind_mask is not None:
                    rows.append(_row(record.name, record.seed, case, variant,
                                     mean_key, std_key, mean, std, exact,
                                     guard_mask, f'guard_val:{split_source}',
                                     std_floor))
                    rows.append(_row(record.name, record.seed, case, variant,
                                     mean_key, std_key, mean, std, exact,
                                     blind_mask, f'blind_test:{split_source}',
                                     std_floor))
                else:
                    rows.append(_row(record.name, record.seed, case, variant,
                                     mean_key, std_key, mean, std, exact,
                                     None, 'full_grid_no_split', std_floor))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed0-root', default=None)
    parser.add_argument('--std-floor', type=float, default=1e-12)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root, args.std_floor)
    if not rows:
        raise SystemExit('No uncertainty arrays found.')
    df = pd.DataFrame(rows).sort_values(['case', 'variant', 'eval_grid', 'seed'])
    write_csv(df, TABLE_DIR / 'nll_sanity_check_raw.csv')

    summary = df.groupby(['case', 'variant', 'eval_grid'], dropna=False).agg(
        n=('nll', 'size'),
        nll_mean=('nll', 'mean'),
        nll_std=('nll', 'std'),
        coverage95_mean=('coverage95', 'mean'),
        corr_mean=('corr', 'mean'),
        avg_interval_width_mean=('avg_interval_width', 'mean'),
        std_min_min=('std_min', 'min'),
        std_median_mean=('std_median', 'mean'),
        std_max_max=('std_max', 'max'),
        floor_fraction_mean=('floor_fraction', 'mean'),
    ).reset_index()
    write_csv(summary, TABLE_DIR / 'nll_sanity_check_mean_std.csv')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
