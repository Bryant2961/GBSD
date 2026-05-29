# run_all_experiments.py -- Master orchestrator for GBSD experiments
"""
Default mode: GBSD / MC Dropout only on Laplace, Poisson, Burgers_inv.
Comparison methods (baseline, deterministic) are disabled by default.

Usage:
    # Default Bayesian-only validation
    python run_all_experiments.py --case all --method all --preset quick_check --seed 0 --clean

    # Explicit single-case
    python run_all_experiments.py --case Laplace --method bayesian --preset medium --seed 0 --clean

    # Re-enable comparisons
    python run_all_experiments.py --case all --method all --preset full --seed 0 --clean --include_comparisons
"""
import os, sys, time, argparse, shutil, random
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default active scope (Bayesian validation phase)
DEFAULT_CASES = ['Laplace', 'Poisson', 'Burgers_inv']
ALL_CASES = ['Laplace', 'Poisson', 'Burgers_inv']
DEFAULT_METHODS = ['bayesian']
ALL_METHODS = ['baseline', 'deterministic', 'bayesian']

BUDGETS = {
    'smoke': {
        'Laplace':     {'teacher':   20, 'student':   20, 'recon':   20},
        'Burgers_inv': {'teacher':   20, 'student':   20, 'recon':   20},
        'Poisson':     {'teacher':   20, 'student':   20, 'recon':   20},
    },
    'quick_check': {
        'Laplace':     {'teacher':  500, 'student':  500, 'recon':  500},
        'Burgers_inv': {'teacher': 1000, 'student': 1000, 'recon': 1000},
        'Poisson':     {'teacher':  500, 'student':  500, 'recon':  500},
    },
    'preview': {
        'Laplace':     {'teacher': 1000, 'student': 1000, 'recon': 1000},
        'Burgers_inv': {'teacher': 1000, 'student': 1000, 'recon': 1000},
        'Poisson':     {'teacher': 1000, 'student': 1000, 'recon': 1000},
    },
    'medium': {
        'Laplace':     {'teacher':  5000, 'student':  5000, 'recon':  5000},
        'Burgers_inv': {'teacher': 10000, 'student': 10000, 'recon': 10000},
        'Poisson':     {'teacher': 10000, 'student': 10000, 'recon': 10000},
    },
    'full': {
        'Laplace':     {'teacher': 10000, 'student': 10000, 'recon': 15000},
        'Burgers_inv': {'teacher': 20000, 'student': 20000, 'recon': 20000},
        'Poisson':     {'teacher': 30000, 'student': 30000, 'recon': 15000},
    },
}
COMMON = dict(weight_rgl=1e-5, cluster_mode='relative', cluster_distance=0.1)


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_run_id(preset, seed):
    return f'{preset}_s{seed}_{time.strftime("%Y%m%d_%H%M%S")}'


def _load_config_quick(case):
    path = f'./Config/{case}_EXP.csv'
    if not os.path.isfile(path):
        return {}
    raw = pd.read_csv(path, header=None)
    if str(raw.iloc[0, 0]).strip().lower() in ('key', 'names'):
        raw = raw.iloc[1:].reset_index(drop=True)
    cfg = {}
    for _, row in raw.iterrows():
        key = str(row.iloc[0]).strip()
        val = str(row.iloc[1]).strip()
        if key and key != 'nan' and val and val != 'nan':
            cfg[key] = val
    return cfg


def _method_label(method):
    labels = {
        'bayesian': 'Guarded Final Source / MC Dropout',
        'deterministic': 'Structured Candidate',
        'baseline': 'Teacher PINN',
    }
    return labels.get(method, method)


def _stage_header(case, method, stage_num, stage_total, stage_name,
                  preset='', run_id=''):
    label = _method_label(method)
    print(f'\n{"="*60}')
    print(f'  [{case} | {label} | Stage {stage_num}/{stage_total}] {stage_name}')
    if preset: print(f'  Preset: {preset}  Run ID: {run_id}')
    print(f'{"="*60}')


def _print_experiment_plan(cases, methods, preset, seed, run_id, device,
                           include_comparisons):
    print(f'\n{"="*60}')
    print(f'  EXPERIMENT PLAN')
    print(f'{"="*60}')
    print(f'  Run ID:  {run_id}')
    print(f'  Preset:  {preset}')
    print(f'  Seed:    {seed}')
    print(f'  Device:  {device}')
    print(f'\n  Active cases:')
    for i, c in enumerate(cases, 1):
        print(f'    {i}. {c}')
    print(f'\n  Active methods:')
    for i, m in enumerate(methods, 1):
        print(f'    {i}. {_method_label(m)}')
    print(f'\n  Comparison methods: {"enabled" if include_comparisons else "disabled by default"}')
    if not include_comparisons:
        print(f'  → To enable comparisons: use --include_comparisons')

    # Budget table
    print(f'\n  Training Budget Table ({preset}):')
    print(f'  {"Case":<14} {"Teacher":<10} {"Student":<10} {"Recon":<10} {"Equal?":<6}')
    print(f'  {"-"*50}')
    for c in cases:
        b = BUDGETS[preset][c]
        equal = 'YES' if b['teacher'] == b['student'] else 'NO'
        print(f'  {c:<14} {b["teacher"]:<10} {b["student"]:<10} {b["recon"]:<10} {equal:<6}')
    print(f'{"="*60}\n')


def _print_stage0(case, method, preset, run_id, device, b):
    """Stage 0: Pre-run validation and configuration."""
    cfg = _load_config_quick(case)
    grid = cfg.get('grid_node_num', '?')
    dr = cfg.get('dropout_rate', '0.25')

    _stage_header(case, method, 0, 5, 'Pre-run validation and configuration',
                  preset, run_id)
    print(f'  teacher_train_steps:   {b["teacher"]}')
    print(f'  student_train_steps:   {b["student"]}')
    print(f'  recon_steps:           {b["recon"]}')
    print(f'  grid_node_num:         {grid}')
    print(f'  dropout_rate:          {dr}')
    print(f'  train_dropout_rate:    {cfg.get("train_dropout_rate", "0.02")}')
    print(f'  distill_noise:         {cfg.get("distill_noise", "0.0")}')
    print(f'  cluster_distance:      {cfg.get("cluster_distance", "0.1")}')
    print(f'  student_pde_weight:    {cfg.get("student_pde_weight", "0.0")}')
    print(f'  device:                {device}')

    if 'Burgers' in case:
        true_nu = 0.01 / np.pi
        print(f'  True ν (0.01/π):       {true_nu:.6f}')
        print(f'  data_weight:           {cfg.get("data_weight", "1.0")}')
        print(f'  nu_prior_weight:       {cfg.get("nu_prior_weight", "0.0")}')
        data_serial = cfg.get('data_serial', '1').split(',')
        for ds in data_serial:
            fpath = f'./Database/Burgers_inv_data_{ds.strip()}.csv'
            if os.path.isfile(fpath):
                d = pd.read_csv(fpath, header=None).values
                print(f'  Data {ds.strip()}: x=[{d[:,0].min():.3f},{d[:,0].max():.3f}] '
                      f't=[{d[:,1].min():.3f},{d[:,1].max():.3f}]')
    print()


def _post_run_summary(case, method, preset, run_id, b):
    """Post-run stage summary."""
    print(f'\n{"="*60}')
    print(f'  POST-RUN STAGE SUMMARY')
    print(f'  EXPERIMENT: {case}')
    print(f'  METHOD: {_method_label(method)}')
    print(f'  RUN ID: {run_id}')
    print(f'{"="*60}')

    # Check loss CSVs
    loss_dir = f'./Results/{case}_EXP/Loss'
    if os.path.isdir(loss_dir):
        for csv_file in sorted(os.listdir(loss_dir)):
            if csv_file.endswith('.csv'):
                try:
                    df = pd.read_csv(f'{loss_dir}/{csv_file}')
                    final_iter = int(df['iter'].max()) if 'iter' in df.columns else '?'
                    final_loss = f'{df["loss"].iloc[-1]:.4e}' if 'loss' in df.columns else '?'
                    param_cols = [c for c in df.columns if 'parameters' in c]
                    nu_str = ''
                    if param_cols:
                        nu_str = f', ν={df[param_cols[0]].iloc[-1]:.6f}'
                    print(f'  Loss: {csv_file}: iter={final_iter}, loss={final_loss}{nu_str}')
                except Exception:
                    pass

    # Check predictions
    npz_path = f'./results/raw/{case}_predictions.npz'
    if os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        if 'bayesian_std' in data:
            std = data['bayesian_std']
            print(f'  Bayesian std: min={std.min():.4e}, max={std.max():.4e}, mean={std.mean():.4e}')
        if 'exact' in data and 'bayesian_mean' in data:
            ex, bm = data['exact'], data['bayesian_mean']
            rel_l2 = np.sqrt(np.sum((bm - ex)**2) / max(1e-15, np.sum(ex**2)))
            print(f'  Bayesian rel_L2={rel_l2:.4f} ({rel_l2*100:.1f}%), MAE={np.abs(bm-ex).mean():.4e}')
    else:
        print(f'  Predictions: not yet generated (run --predict_only or wait for full pipeline)')

    fig_dir = f'./Results/{case}_EXP/Figures'
    if os.path.isdir(fig_dir):
        figs = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
        print(f'  Figures: {len(figs)} in {fig_dir}')

    print(f'{"="*60}\n')


def run_training(case, method, preset, seed=0, run_id='', device='cpu',
                 student_pde_weight_override=None):
    import Module.Training as Training
    _set_global_seed(seed)

    b = BUDGETS[preset][case]
    teacher_gap = max(1, b['teacher'] // 5)
    student_gap = max(1, b['student'] // 5)
    recon_gap = max(1, b['recon'] // 5)

    # Stage 0
    _print_stage0(case, method, preset, run_id, device, b)

    def configure(task):
        task.train_steps = b['teacher']
        task.student_train_steps = b['student']
        task.train_ratio = 1.0
        task.epochs_recon = b['recon']
        task.teacher_print_every = teacher_gap
        task.student_print_every = student_gap
        task.recon_print_every = recon_gap
        task.print_every = recon_gap
        task.pace_record_state = 0
        task.pace_record_gap = [teacher_gap]; task.pace_record_skip = [0]
        task.milestone = [b['teacher']//2, int(b['teacher']*0.8)]
        task.lambda_distill = 1.0
        # Pass stage context for logging
        task._run_context = {
            'case': case, 'method': method, 'preset': preset,
            'run_id': run_id, 'method_label': _method_label(method),
        }
        if preset == 'smoke':
            task.poisson_use_lbfgs = False
            task.poisson_lbfgs_steps = 0
            task.poisson_mean_refine_steps = min(
                getattr(task, 'poisson_mean_refine_steps', 0), 20)
            task.poisson_mean_refine_lbfgs_steps = 0
            task.mean_refine_steps = min(
                getattr(task, 'mean_refine_steps', 0), 20)
            task.mean_refine_lbfgs_steps = 0
            task.anchor_pretrain_steps = min(
                getattr(task, 'anchor_pretrain_steps', 0), 20)
            task.residual_pretrain_steps = min(
                getattr(task, 'residual_pretrain_steps', 0), 20)
            print('  [Smoke] Internal refinement/pretraining capped at 20 steps; L-BFGS disabled.')
        return task

    n_mc = 10 if preset in ('quick_check', 'preview') else 100

    if method == 'baseline':
        _stage_header(case, method, 1, 5, 'Direct PINN / PINN-post training', preset, run_id)
        t = Training.model(case, 'EXP', **COMMON)
        configure(t); t.train()

    elif method == 'deterministic':
        _stage_header(case, method, 1, 5, 'Teacher PINN + Deterministic Student', preset, run_id)
        t = Training.model(case, 'EXP', distill=True, **COMMON)
        configure(t); t.train()

    elif method == 'bayesian':
        _stage_header(case, method, 1, 5, 'Teacher PINN + Bayesian MC Dropout Student',
                      preset, run_id)
        t = Training.model(case, 'EXP', student_type='mc_dropout',
                           bayesian_recon=True, include_uq=True,
                           n_mc_samples=n_mc, **COMMON)
        configure(t)
        # CLI override for student_pde_weight
        if student_pde_weight_override is not None:
            t.student_pde_weight = student_pde_weight_override
            print(f'  [Override] student_pde_weight = {student_pde_weight_override}')
        t.train()

    # Post-run summary
    _post_run_summary(case, method, preset, run_id, b)


def run_predictions(case, n_samples=200, device='cpu', seed=0):
    from utils.posterior_predict import predict_and_save
    predict_and_save(case, n_samples=n_samples, device_str=device, seed=seed)


def run_metrics(case):
    from utils.metrics import compute_all_metrics, save_metrics
    npz = f'./results/raw/{case}_predictions.npz'
    if not os.path.isfile(npz):
        print(f'  [{case}] No prediction .npz — skipping metrics')
        return
    data = np.load(npz, allow_pickle=True)
    exact = data.get('exact')
    rows = []
    if 'pinn_pred' in data:
        rows.append(compute_all_metrics(data['pinn_pred'], None, exact,
                                         case=case, method='PINN'))
    if 'det_student_pred' in data:
        rows.append(compute_all_metrics(data['det_student_pred'], None, exact,
                                         case=case, method='Det_Student'))
    if 'bayesian_mean' in data:
        source = 'selected'
        if 'bayesian_source' in data:
            try:
                source = str(data['bayesian_source'].reshape(-1)[0])
            except Exception:
                source = 'selected'
        row = compute_all_metrics(data['bayesian_mean'],
                                  data.get('bayesian_std'), exact,
                                  case=case, method='Bayesian_MCDropout',
                                  source=source)
        if 'bayesian_std_source' in data:
            try:
                row['std_source'] = str(data['bayesian_std_source'].reshape(-1)[0])
            except Exception:
                row['std_source'] = 'selected'
        if 'bayesian_structured_compression' in data:
            row['structured_compression'] = float(
                data['bayesian_structured_compression'].reshape(-1)[0])
        rows.append(row)
        if ('std_calibration_mask' in data
                and data.get('bayesian_std') is not None
                and exact is not None):
            mask = data['std_calibration_mask'].astype(bool).reshape(-1)
            eval_mask = ~mask
            if int(eval_mask.sum()) >= 10:
                split_row = compute_all_metrics(
                    data['bayesian_mean'].reshape(-1)[eval_mask],
                    data['bayesian_std'].reshape(-1)[eval_mask],
                    exact.reshape(-1)[eval_mask],
                    case=case,
                    method='Bayesian_MCDropout_UQEvalSplit',
                    source=f'{source}_calibration_holdout')
                split_row['calibration_eval_split'] = (
                    'grid_excluding_temperature_fit')
                split_row['calibration_eval_points'] = int(eval_mask.sum())
                if 'bayesian_std_source' in data:
                    try:
                        split_row['std_source'] = str(
                            data['bayesian_std_source'].reshape(-1)[0])
                    except Exception:
                        split_row['std_source'] = 'selected'
                if 'bayesian_structured_compression' in data:
                    split_row['structured_compression'] = float(
                        data['bayesian_structured_compression'].reshape(-1)[0])
                rows.append(split_row)
    if 'bayesian_dense_mean' in data:
        row = compute_all_metrics(data['bayesian_dense_mean'],
                                  data.get('bayesian_dense_std'), exact,
                                  case=case, method='Bayesian_DenseStudent',
                                  source='dense_student')
        row['std_source'] = 'raw_mc_dropout'
        rows.append(row)
    if 'bayesian_structured_mean' in data:
        row = compute_all_metrics(data['bayesian_structured_mean'],
                                  data.get('bayesian_structured_std'), exact,
                                  case=case, method='Bayesian_StructuredCandidate',
                                  source='structured')
        row['std_source'] = 'raw_mc_dropout'
        if 'bayesian_structured_compression' in data:
            row['structured_compression'] = float(
                data['bayesian_structured_compression'].reshape(-1)[0])
        rows.append(row)
    if rows:
        save_metrics(rows, f'./results/metrics/{case}_metrics.csv')


def generate_figures(cases, bayesian_only=True):
    """Generate figures. In bayesian_only mode, skip comparison figures gracefully."""
    print(f'\n{"="*60}')
    print(f'  GENERATING FIGURES (bayesian_only={bayesian_only})')
    print(f'{"="*60}')

    # Framework diagram
    try:
        from plot_framework_diagram import draw_framework
        draw_framework()
    except Exception as e:
        print(f'  Framework diagram skipped: {e}')

    # Per-case figures
    from plot_results import plot_all_for_case
    for c in cases:
        try:
            plot_all_for_case(c)
        except Exception as e:
            print(f'  [{c}] Figure generation error (non-fatal): {e}')

    # Calibration
    from utils.calibration import run_from_npz
    for c in cases:
        try:
            run_from_npz(c)
        except Exception as e:
            print(f'  [{c}] Calibration skipped: {e}')

    # Parameter inversion
    try:
        from plot_parameter_inversion import plot_trajectory
        for c in cases:
            if 'inv' in c.lower():
                plot_trajectory(c)
    except Exception as e:
        print(f'  Parameter inversion plot skipped: {e}')

    # Skip comparison-only figures in bayesian_only mode
    if bayesian_only:
        print(f'  Skipping multi-method comparison figures (bayesian_only mode)')
        print(f'  Skipping posterior weight-std figures (MC Dropout ≠ weight posterior)')
    else:
        try:
            from plot_transfer_generalization import plot
            plot()
        except Exception as e:
            print(f'  Transfer plot skipped: {e}')
        try:
            from plot_posterior_structure import plot_posterior_structure
            for c in cases:
                plot_posterior_structure(c)
        except Exception as e:
            print(f'  Posterior structure skipped: {e}')


def _filter_csv_by_case(path, cases_to_remove):
    if not os.path.isfile(path):
        return
    try:
        df = pd.read_csv(path)
        if 'case' not in df.columns:
            return
        before = len(df)
        df = df[~df['case'].isin(cases_to_remove)]
        if len(df) < before:
            df.to_csv(path, index=False)
            print(f'  Filtered {before - len(df)} rows from {path}')
        if len(df) == 0:
            os.remove(path)
    except Exception:
        pass


def clean_results(cases):
    import shutil
    for case in cases:
        for p in [f'./Results/{case}_EXP', f'./results/raw/{case}_predictions.npz']:
            if os.path.isdir(p):
                shutil.rmtree(p); print(f'  Cleaned: {p}')
            elif os.path.isfile(p):
                os.remove(p); print(f'  Cleaned: {p}')

        if os.path.isdir('./results/figures'):
            patterns = [case.lower(), case, f'posterior_structure_{case.lower()}']
            for f in os.listdir('./results/figures'):
                if any(p in f for p in patterns):
                    os.remove(f'./results/figures/{f}')
                    print(f'  Cleaned: ./results/figures/{f}')

        if os.path.isdir('./results/raw'):
            raw_patterns = [
                f'posterior_structure_{case.lower()}.npz',
                f'postior_structure_{case.lower()}.npz',
            ]
            for f in os.listdir('./results/raw'):
                if f in raw_patterns:
                    os.remove(f'./results/raw/{f}')
                    print(f'  Cleaned: ./results/raw/{f}')

        for p in [f'./results/metrics/{case}_metrics.csv',
                  f'./results/predictions/{case}', f'./results/logs/{case}']:
            if os.path.isfile(p):
                os.remove(p); print(f'  Cleaned: {p}')
            elif os.path.isdir(p):
                shutil.rmtree(p); print(f'  Cleaned: {p}')

    for g in ['./results/metrics/calibration_metrics.csv',
              './results/metrics/parameter_inversion_metrics.csv',
              './results/metrics/transfer_generalization_metrics.csv']:
        _filter_csv_by_case(g, cases)
    print(f'  Clean complete for: {cases}\n')


def archive_run_outputs(cases, run_id):
    """Archive outputs so multi-seed sweeps keep every run."""
    archive_root = os.path.join('.', 'Results', 'supplementary', 'runs', run_id)
    os.makedirs(archive_root, exist_ok=True)
    print(f'\n{"="*60}')
    print('  ARCHIVING RUN OUTPUTS')
    print(f'  Destination: {archive_root}')
    print(f'{"="*60}')

    for case in cases:
        case_src = os.path.join('.', 'Results', f'{case}_EXP')
        case_dst = os.path.join(archive_root, f'{case}_EXP')
        if os.path.isdir(case_src):
            if os.path.exists(case_dst):
                shutil.rmtree(case_dst)
            shutil.copytree(case_src, case_dst,
                            ignore=shutil.ignore_patterns('__pycache__'))
            print(f'  Archived {case}_EXP')

        pred_src = os.path.join('.', 'results', 'raw',
                                f'{case}_predictions.npz')
        if os.path.isfile(pred_src):
            raw_dst = os.path.join(archive_root, 'raw')
            os.makedirs(raw_dst, exist_ok=True)
            shutil.copy2(pred_src,
                         os.path.join(raw_dst, f'{case}_predictions.npz'))

        metrics_src = os.path.join('.', 'results', 'metrics',
                                   f'{case}_metrics.csv')
        if os.path.isfile(metrics_src):
            metrics_dst = os.path.join(archive_root, 'metrics')
            os.makedirs(metrics_dst, exist_ok=True)
            shutil.copy2(metrics_src,
                         os.path.join(metrics_dst, f'{case}_metrics.csv'))

    figures_src = os.path.join('.', 'results', 'figures')
    if os.path.isdir(figures_src):
        figures_dst = os.path.join(archive_root, 'figures')
        if os.path.exists(figures_dst):
            shutil.rmtree(figures_dst)
        shutil.copytree(figures_src, figures_dst,
                        ignore=shutil.ignore_patterns('__pycache__'))
        print('  Archived figures')

    for name in ['calibration_metrics.csv',
                 'parameter_inversion_metrics.csv',
                 'transfer_generalization_metrics.csv',
                 'guard_decision.csv',
                 'blind_test_metrics.csv',
                 'inference_time.csv']:
        src = os.path.join('.', 'results', 'metrics', name)
        if os.path.isfile(src):
            metrics_dst = os.path.join(archive_root, 'metrics')
            os.makedirs(metrics_dst, exist_ok=True)
            shutil.copy2(src, os.path.join(metrics_dst, name))

    splits_src = os.path.join('.', 'results', 'splits')
    if os.path.isdir(splits_src):
        splits_dst = os.path.join(archive_root, 'splits')
        if os.path.exists(splits_dst):
            shutil.rmtree(splits_dst)
        shutil.copytree(splits_src, splits_dst,
                        ignore=shutil.ignore_patterns('__pycache__'))
        print('  Archived guard/blind splits')


def write_environment_manifest(archive_root, args, run_id, cases, methods,
                               elapsed_s):
    """Write reproducibility metadata next to archived outputs."""
    import ctypes
    import json
    import platform
    import torch as _torch

    ram_total_bytes = None
    try:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        memory = MemoryStatusEx()
        memory.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            ram_total_bytes = int(memory.ullTotalPhys)
    except Exception:
        ram_total_bytes = None

    info = {
        'run_id': run_id,
        'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'elapsed_seconds': float(elapsed_s),
        'python_version': sys.version.split(' ')[0],
        'python_full': sys.version,
        'torch_version': _torch.__version__,
        'cuda_available': bool(_torch.cuda.is_available()),
        'cuda_version': _torch.version.cuda,
        'cudnn_version': (_torch.backends.cudnn.version()
                          if _torch.cuda.is_available() else None),
        'gpu_name': (_torch.cuda.get_device_name(0)
                     if _torch.cuda.is_available() else None),
        'gpu_memory_total_bytes': (
            int(_torch.cuda.get_device_properties(0).total_memory)
            if _torch.cuda.is_available() else None),
        'ram_total_bytes': ram_total_bytes,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'seed': int(args.seed),
        'cases': list(cases),
        'methods': list(methods),
        'preset': args.preset,
        'n_mc': int(args.n_mc),
        'include_comparisons': bool(args.include_comparisons),
        'argv': sys.argv,
    }
    os.makedirs(archive_root, exist_ok=True)
    with open(os.path.join(archive_root, 'environment.json'), 'w',
              encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    cfg_src = os.path.join('.', 'Config')
    cfg_dst = os.path.join(archive_root, 'config_snapshot')
    if os.path.isdir(cfg_src):
        if os.path.exists(cfg_dst):
            shutil.rmtree(cfg_dst)
        shutil.copytree(cfg_src, cfg_dst,
                        ignore=shutil.ignore_patterns('__pycache__'))
    print(f'  Environment manifest written: {archive_root}/environment.json')


def main():
    PRESETS = list(BUDGETS.keys())
    parser = argparse.ArgumentParser(description='GBSD Experiment Runner')
    parser.add_argument('--case', default='all',
                        choices=ALL_CASES + ['all'])
    parser.add_argument('--method', default='all',
                        choices=ALL_METHODS + ['all'])
    parser.add_argument('--preset', default='quick_check', choices=PRESETS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--figures_only', action='store_true')
    parser.add_argument('--predict_only', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--n_mc', type=int, default=200)
    parser.add_argument('--include_comparisons', action='store_true',
                        help='Enable baseline and deterministic methods')
    parser.add_argument('--student_pde_weight', type=float, default=None,
                        help='Override student PDE weight (physics-informed distillation)')
    parser.add_argument('--teacher_steps', type=int, default=None,
                        help='Override teacher train_steps for selected cases')
    parser.add_argument('--student_steps', type=int, default=None,
                        help='Override student_train_steps for selected cases')
    parser.add_argument('--recon_steps', type=int, default=None,
                        help='Override reconstruction steps for selected cases')
    args = parser.parse_args()

    # Resolve cases
    if args.case == 'all':
        cases = list(ALL_CASES)
    else:
        cases = [args.case]

    # Resolve methods
    if args.method == 'all':
        if args.include_comparisons:
            methods = list(ALL_METHODS)
        else:
            methods = list(DEFAULT_METHODS)
    elif args.method in ('baseline', 'deterministic') and not args.include_comparisons:
        print(f'  Comparison methods are disabled by default. '
              f'Use --include_comparisons to run {args.method}.')
        sys.exit(0)
    else:
        methods = [args.method]

    if any(v is not None for v in
           (args.teacher_steps, args.student_steps, args.recon_steps)):
        for c in cases:
            b = dict(BUDGETS[args.preset][c])
            if args.teacher_steps is not None:
                b['teacher'] = args.teacher_steps
            if args.student_steps is not None:
                b['student'] = args.student_steps
            if args.recon_steps is not None:
                b['recon'] = args.recon_steps
            BUDGETS[args.preset][c] = b

    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    run_id = _make_run_id(args.preset, args.seed)
    bayesian_only = (methods == ['bayesian'])
    start = time.time()

    _print_experiment_plan(cases, methods, args.preset, args.seed, run_id,
                           device, args.include_comparisons)

    if args.clean:
        clean_results(cases)

    if args.figures_only:
        for c in cases:
            try: run_predictions(c, n_samples=args.n_mc, device=device, seed=args.seed)
            except Exception as e: print(f'  [{c}] Prediction failed: {e}')
            try: run_metrics(c)
            except Exception as e: print(f'  [{c}] Metrics failed: {e}')
        generate_figures(cases, bayesian_only=bayesian_only)

    elif args.predict_only:
        for c in cases:
            try: run_predictions(c, n_samples=args.n_mc, device=device, seed=args.seed)
            except Exception as e: print(f'  [{c}] Prediction failed: {e}')
            try: run_metrics(c)
            except Exception as e: print(f'  [{c}] Metrics failed: {e}')

    else:
        # Training pipeline
        for c in cases:
            for m in methods:
                run_training(c, m, args.preset, args.seed,
                             run_id=run_id, device=device,
                             student_pde_weight_override=args.student_pde_weight)
        for c in cases:
            try:
                run_predictions(c, n_samples=args.n_mc, device=device, seed=args.seed)
            except Exception as e:
                print(f'  [{c}] Prediction failed (non-fatal): {e}')
            try:
                run_metrics(c)
            except Exception as e:
                print(f'  [{c}] Metrics failed (non-fatal): {e}')
        generate_figures(cases, bayesian_only=bayesian_only)

    elapsed = time.time() - start
    archive_run_outputs(cases, run_id)
    archive_root = os.path.join('.', 'Results', 'supplementary', 'runs', run_id)
    write_environment_manifest(archive_root, args, run_id, cases, methods, elapsed)

    print(f'\n{"="*60}')
    print(f'  COMPLETE. Run ID: {run_id}')
    print(f'  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  Cases: {cases}')
    print(f'  Methods: {[_method_label(m) for m in methods]}')
    print(f'  Figures:  ./results/figures/')
    print(f'  Metrics:  ./results/metrics/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
