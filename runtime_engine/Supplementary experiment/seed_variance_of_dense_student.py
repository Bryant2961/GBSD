"""Seed-variance of the distilled dense Bayesian student.

NOTE ON NAMING (v3.37 JCP review correction)
============================================
This script previously lived as `deep_ensemble_uq_baseline.py`. That name
was misleading: a "Deep Ensemble PINN baseline" in the literature means
N independently-trained PINNs from scratch on the PDE loss, with no
shared teacher.

What this script actually does is stack the means of the SAME
GBSD-distilled MC-Dropout student across seeds. Every member here is the
output of:
    Teacher PINN  ->  MC-Dropout student distilled from that teacher
so the teacher channel is fully baked into every "member". The right
interpretation is therefore "seed-variance of the proposed method's dense
student", not an independent UQ baseline.

For the real Deep Ensemble PINN baseline see
`direct_mc_dropout_pinn_baseline.py`.

This script is retained because the per-seed variance is still a useful
diagnostic: it bounds how much of the calibrated uncertainty is reducible
by an ensemble vs. truly epistemic about the underlying PINN.
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
    evaluate_prediction,
    load_npz,
    write_csv,
)


def build_rows(seed0_root: str | None):
    rows = []
    records = collect_runs(seed0_root)
    for case in CASES:
        members = []
        exact = None
        used = []
        for record in records:
            data = load_npz(record, case)
            if (data is None or 'bayesian_dense_mean' not in data
                    or 'exact' not in data):
                continue
            members.append(np.asarray(data['bayesian_dense_mean'],
                                      dtype=float).reshape(-1))
            exact = np.asarray(data['exact'], dtype=float).reshape(-1)
            used.append(record.name)
        if len(members) < 2 or exact is None:
            continue

        # If any record carries a blind_test_mask, restrict to its
        # intersection so the report is on held-out points.
        blind_intersection = None
        for record in records:
            data = load_npz(record, case)
            if data is not None and 'blind_test_mask' in data:
                bm = np.asarray(data['blind_test_mask']).astype(bool).reshape(-1)
                blind_intersection = (bm if blind_intersection is None
                                      else blind_intersection & bm)
        if blind_intersection is not None and blind_intersection.sum() >= 10:
            members = [m[blind_intersection] for m in members]
            exact = exact[blind_intersection]
            eval_grid = 'blind_test_intersection'
        else:
            eval_grid = 'full_grid'

        stack = np.vstack(members)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=1)
        std = np.maximum(std, 1e-12)
        metrics = evaluate_prediction(mean, std, exact)
        row = {
            'case': case,
            'variant': 'seed_variance_of_dense_bayesian_student',
            'is_independent_baseline': False,
            'is_deep_ensemble_pinn': False,
            'shares_teacher_distillation': True,
            'eval_grid': eval_grid,
            'n_members': len(members),
            'members': ';'.join(used),
        }
        row.update(metrics)
        rows.append(row)

        # Per-seed MC-Dropout average (one member at a time).
        individual_rows = []
        for record in records:
            data = load_npz(record, case)
            if (data is None or 'bayesian_dense_mean' not in data
                    or 'exact' not in data
                    or 'bayesian_dense_std' not in data):
                continue
            m_arr = np.asarray(data['bayesian_dense_mean']).reshape(-1)
            s_arr = np.asarray(data['bayesian_dense_std']).reshape(-1)
            e_arr = np.asarray(data['exact']).reshape(-1)
            if (blind_intersection is not None
                    and blind_intersection.size == e_arr.size
                    and blind_intersection.sum() >= 10):
                m_arr = m_arr[blind_intersection]
                s_arr = s_arr[blind_intersection]
                e_arr = e_arr[blind_intersection]
            individual_rows.append(evaluate_prediction(m_arr, s_arr, e_arr))
        if individual_rows:
            avg = pd.DataFrame(individual_rows).mean(numeric_only=True).to_dict()
            avg_row = {
                'case': case,
                'variant': 'mean_individual_mc_dropout_dense_student',
                'is_independent_baseline': False,
                'is_deep_ensemble_pinn': False,
                'shares_teacher_distillation': True,
                'eval_grid': eval_grid,
                'n_members': len(individual_rows),
                'members': 'per-seed average',
            }
            avg_row.update({k: float(v) if not math.isnan(float(v))
                            else math.nan
                            for k, v in avg.items()})
            rows.append(avg_row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed0-root', default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root)
    if not rows:
        raise SystemExit('No multi-seed runs found.')
    df = pd.DataFrame(rows).sort_values(['case', 'variant'])
    # Write under the new name AND keep the old filename for back-compat,
    # so the existing make_appendix_figures.py and command files continue
    # to find a CSV until the user updates them.
    write_csv(df, TABLE_DIR / 'seed_variance_of_dense_student.csv')
    write_csv(df, TABLE_DIR / 'deep_ensemble_uq_baseline.csv')  # legacy name
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
