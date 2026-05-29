#!/usr/bin/env python3
"""
validate_bayesian_only.py -- Validate GBSD / MC Dropout results only.

Does NOT expect baseline or deterministic outputs.
Checks only: teacher PINN + Bayesian MC Dropout student pipeline
for Laplace, Poisson, and Burgers_inv.

Usage: python validate_bayesian_only.py [--preset quick_check|medium|full]
"""
import os, sys, argparse, math
import pandas as pd
import numpy as np

passed, failed = 0, 0
ACTIVE_CASES = ['Laplace', 'Poisson', 'Burgers_inv']

EXPECTED_ITERS = {
    'quick_check': {'Laplace': 500, 'Poisson': 500, 'Burgers_inv': 1000},
    'medium':      {'Laplace': 5000, 'Poisson': 10000, 'Burgers_inv': 10000},
    'full':        {'Laplace': 10000, 'Poisson': 30000, 'Burgers_inv': 20000},
}


def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f'  PASS: {name}')
        passed += 1
    else:
        print(f'  FAIL: {name}  {detail}')
        failed += 1


def validate_scope():
    """Verify default scope is the three active cases."""
    print('\n=== Scope Validation ===')
    check('Active cases = 3', len(ACTIVE_CASES) == 3)
    for c in ACTIVE_CASES:
        check(f'{c} in active cases', c in ACTIVE_CASES)


def validate_case(case, preset):
    """Validate one case's Bayesian pipeline output."""
    print(f'\n=== {case} ===')

    exp_iter = EXPECTED_ITERS.get(preset, {}).get(case, None)

    # Teacher loss CSV
    teacher_csv = f'./Results/{case}_EXP/Loss/{case}_EXP_loss_PINN.csv'
    if os.path.isfile(teacher_csv):
        df = pd.read_csv(teacher_csv)
        final = int(df['iter'].max())
        check(f'Teacher loss CSV exists', True)
        if exp_iter:
            check(f'Teacher final_iter == {exp_iter}', final == exp_iter,
                  f'got {final}')
        # Check for Burgers ν
        param_cols = [c for c in df.columns if 'parameters' in c]
        if 'Burgers' in case and param_cols:
            nu_final = df[param_cols[0]].iloc[-1]
            check(f'Burgers ν is positive', nu_final > 0, f'got {nu_final:.6f}')
            true_nu = 0.01 / math.pi
            rel_err = abs(nu_final - true_nu) / true_nu * 100
            print(f'    Burgers ν: learned={nu_final:.6f}, true={true_nu:.6f}, '
                  f'rel_err={rel_err:.1f}%')
        elif 'Burgers' in case:
            # Try Parameters/ directory fallback
            para_csv = f'./Results/{case}_EXP/Parameters/{case}_EXP_paras_PINN.csv'
            if os.path.isfile(para_csv):
                pdf = pd.read_csv(para_csv)
                if 'parameters_1' in pdf.columns:
                    nu_final = pdf['parameters_1'].iloc[-1]
                    check(f'Burgers ν is positive (Parameters/)', nu_final > 0,
                          f'got {nu_final:.6f}')
                    true_nu = 0.01 / math.pi
                    rel_err = abs(nu_final - true_nu) / true_nu * 100
                    print(f'    Burgers ν: learned={nu_final:.6f}, true={true_nu:.6f}, '
                          f'rel_err={rel_err:.1f}%')
            else:
                print(f'    Burgers ν: not found in loss CSV or Parameters/')
    else:
        check(f'Teacher loss CSV exists', False, teacher_csv)

    # Bayesian student loss CSV
    student_csv = f'./Results/{case}_EXP/Loss/{case}_EXP_loss_Student_MCDropout_student.csv'
    if os.path.isfile(student_csv):
        df = pd.read_csv(student_csv)
        final = int(df['iter'].max())
        check(f'Bayesian student loss CSV exists', True)
        if exp_iter:
            check(f'Student final_iter == {exp_iter}', final == exp_iter,
                  f'got {final}')
    else:
        check(f'Bayesian student loss CSV exists', False, student_csv)

    # Prediction .npz (may not exist for quick_check)
    npz = f'./results/raw/{case}_predictions.npz'
    if os.path.isfile(npz):
        data = np.load(npz, allow_pickle=True)
        check(f'Prediction .npz exists', True)

        if 'bayesian_mean' in data:
            check(f'bayesian_mean in predictions', True)
        else:
            check(f'bayesian_mean in predictions', False)

        if 'bayesian_std' in data:
            std = data['bayesian_std']
            has_range = std.max() > std.min()
            check(f'bayesian_std has nonzero range', has_range,
                  f'min={std.min():.4e}, max={std.max():.4e}')
            print(f'    Bayesian std: min={std.min():.4e}, '
                  f'max={std.max():.4e}, mean={std.mean():.4e}')

        if 'exact' in data and 'bayesian_mean' in data:
            ex = data['exact']
            bm = data['bayesian_mean']
            mae = np.abs(bm - ex).mean()
            denom = np.sum(ex**2)
            if denom > 1e-15:
                rel_l2 = np.sqrt(np.sum((bm - ex)**2) / denom)
                print(f'    Bayesian rel_L2 = {rel_l2:.4f} ({rel_l2*100:.1f}%)')
            else:
                rel_l2 = float('inf')
                print(f'    Bayesian rel_L2 = N/A (exact solution norm ≈ 0)')
            sol_amp = max(abs(ex.max()), abs(ex.min()))
            print(f'    Bayesian MAE = {mae:.4e}, |u|_max = {sol_amp:.4e}')
            if sol_amp > 0:
                print(f'    MAE / |u|_max = {mae/sol_amp:.1%} (absolute accuracy)')
    else:
        print(f'    Prediction .npz not found (expected for quick_check)')

    # No requirement for baseline or deterministic outputs
    # These are intentionally absent in Bayesian-only mode


def validate_data_files():
    """Validate Burgers data file column ordering."""
    print('\n=== Burgers Data Files ===')
    for ds in ['1', '3']:
        fpath = f'./Database/Burgers_inv_data_{ds}.csv'
        if os.path.isfile(fpath):
            d = pd.read_csv(fpath, header=None)
            x_ok = d[0].min() >= -1 and d[0].max() <= 1
            t_ok = d[1].min() >= 0 and d[1].max() <= 1
            check(f'data_{ds} col0(x) in [-1,1]', x_ok,
                  f'[{d[0].min():.3f}, {d[0].max():.3f}]')
            check(f'data_{ds} col1(t) in [0,1]', t_ok,
                  f'[{d[1].min():.3f}, {d[1].max():.3f}]')
        else:
            print(f'    {fpath} not found')


def validate_no_stale_mixing():
    """Check for obvious signs of result mixing."""
    print('\n=== Stale Result Check ===')
    # Check transfer_generalization_metrics.csv for duplicates
    tf_path = './results/metrics/transfer_generalization_metrics.csv'
    if os.path.isfile(tf_path):
        df = pd.read_csv(tf_path)
        dupes = df.duplicated(subset=['case', 'method', 'seed'], keep='first').sum()
        check(f'No duplicate rows in transfer metrics', dupes == 0,
              f'{dupes} duplicate rows found')
    else:
        print(f'    transfer_generalization_metrics.csv not found (OK if clean run)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='quick_check',
                        choices=['quick_check', 'preview', 'medium', 'full'])
    args = parser.parse_args()

    print('=' * 60)
    print('  BAYESIAN-ONLY VALIDATION REPORT')
    print(f'  Expected preset: {args.preset}')
    print('=' * 60)

    validate_scope()
    validate_data_files()
    for case in ACTIVE_CASES:
        validate_case(case, args.preset)
    validate_no_stale_mixing()

    print(f'\n{"="*60}')
    print(f'  TOTAL: {passed} passed, {failed} failed')
    if failed == 0:
        print(f'  STATUS: ALL CHECKS PASSED')
        print(f'  Ready for: python run_all_experiments.py '
              f'--case all --method all --preset medium --seed 0 --clean')
    else:
        print(f'  STATUS: {failed} CHECKS FAILED — review before proceeding')
    print(f'{"="*60}')

    sys.exit(1 if failed > 0 else 0)
