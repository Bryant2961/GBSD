# utils/calibration.py — Figure 5: Error-uncertainty consistency + coverage calibration
"""
Computes: scatter, binned means, correlation, coverage curve, PICP, ECE.
Reads from .npz prediction files (real data only — no fabrication).

Usage: python utils/calibration.py --case Laplace
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paper_style import setup_paper_style
from utils.metrics import picp, mpiw_metric, nll_gaussian, calibration_error

FIG_DIR = './results/figures'
MET_DIR = './results/metrics'

def full_calibration(mean, std, exact, case, method='Bayesian $\\Psi$-NN', n_bins=15):
    """Run full calibration analysis and generate figures + metrics."""
    setup_paper_style()
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(MET_DIR, exist_ok=True)
    abs_err = np.abs(mean - exact).flatten()
    stds = std.flatten()

    # ── Scatter: error vs uncertainty ──
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(stds, abs_err, s=2, alpha=0.25, c='steelblue', rasterized=True, label='Test points')
    bins = np.percentile(stds, np.linspace(0, 100, n_bins+1))
    bx, by = [], []
    for i in range(n_bins):
        mask = (stds >= bins[i]) & (stds < bins[i+1] + 1e-10)
        if mask.sum() > 0: bx.append(np.mean(stds[mask])); by.append(np.mean(abs_err[mask]))
    ax.plot(bx, by, 'o-', c='orangered', lw=2.5, ms=6, label='Bin-wise mean', zorder=5)
    mx = max(stds.max(), abs_err.max()) * 1.05
    ax.plot([0, mx], [0, mx], '--', c='gray', alpha=0.6, label='y = x (ideal)')
    corr = float(np.corrcoef(stds, abs_err)[0,1]) if len(stds)>1 else 0
    ax.set_xlabel('Predictive std', fontsize=12); ax.set_ylabel('Absolute error', fontsize=12)
    ax.set_title(f'{case} — Error vs Uncertainty (corr={corr:.3f})', fontsize=12)
    ax.legend(fontsize=9); fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_error_uncertainty_consistency_{case}.{ext}')
    plt.close(fig)

    # ── Coverage calibration curve ──
    confidences = np.arange(0.50, 0.995, 0.025)
    coverages = [picp(mean.flatten(), std.flatten(), exact.flatten(), c) for c in confidences]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(confidences, coverages, 'o-', c='#d62728', lw=2.5, ms=5, label='Empirical coverage')
    ax.plot([0.5, 1.0], [0.5, 1.0], '--', c='gray', lw=1.5, label='Ideal')
    ax.fill_between(confidences, coverages, confidences, alpha=0.15, color='#d62728')
    ax.set_xlabel('Nominal confidence', fontsize=12); ax.set_ylabel('Empirical coverage (PICP)', fontsize=12)
    ax.set_title(f'{case} — Coverage Calibration', fontsize=12)
    ax.legend(fontsize=10); ax.set_xlim(0.48, 1.02); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3); fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_coverage_calibration_{case}.{ext}')
    plt.close(fig)

    # ── Metrics CSV ──
    m_flat, s_flat, e_flat = mean.flatten(), std.flatten(), exact.flatten()
    metrics = {
        'case': case, 'method': method, 'seed': 0,
        'std_error_corr': corr,
        'coverage_50': picp(m_flat, s_flat, e_flat, 0.50),
        'coverage_80': picp(m_flat, s_flat, e_flat, 0.80),
        'coverage_90': picp(m_flat, s_flat, e_flat, 0.90),
        'coverage_95': picp(m_flat, s_flat, e_flat, 0.95),
        'avg_interval_width_95': mpiw_metric(s_flat, 0.95),
        'nll': nll_gaussian(m_flat, s_flat, e_flat),
        'ece': calibration_error(mean, std, exact),
    }
    # Optional fields are injected by run_from_npz when available.
    if hasattr(full_calibration, '_extra_metrics'):
        metrics.update(full_calibration._extra_metrics)

    path = f'{MET_DIR}/calibration_metrics.csv'
    df_new = pd.DataFrame([metrics])
    if os.path.isfile(path):
        df_existing = pd.read_csv(path)
        # Remove old rows for this case, then append new
        df_existing = df_existing[df_existing['case'] != case]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)
    print(f'  Calibration for {case} → {path}')
    return metrics

def run_from_npz(case, npz_dir='./results/raw'):
    npz = f'{npz_dir}/{case}_predictions.npz'
    if not os.path.isfile(npz):
        print(f'  {npz} not found — run: python utils/posterior_predict.py --case {case}'); return None
    data = np.load(npz, allow_pickle=True)
    if 'bayesian_mean' not in data or 'exact' not in data:
        print(f'  Skipping {case}: missing bayesian_mean or exact'); return None
    extra = {}
    if 'std_temperature_factor' in data:
        extra['std_temperature_factor'] = float(data['std_temperature_factor'].reshape(-1)[0])
    if 'bayesian_std_source' in data:
        extra['std_source'] = str(data['bayesian_std_source'].reshape(-1)[0])
    if 'std_calibration_source' in data:
        extra['std_calibration_source'] = str(data['std_calibration_source'].reshape(-1)[0])
    if 'std_calibration_n' in data:
        extra['std_calibration_points'] = int(data['std_calibration_n'].reshape(-1)[0])
    if 'std_feature_weights' in data:
        extra['std_feature_weights'] = ';'.join(
            str(x) for x in data['std_feature_weights'].reshape(-1))

    mean = data['bayesian_mean']
    std = data['bayesian_std']
    exact = data['exact']
    if 'std_calibration_mask' in data:
        mask = data['std_calibration_mask'].astype(bool).reshape(-1)
        eval_mask = ~mask
        if int(eval_mask.sum()) >= 10:
            mean = mean.reshape(-1)[eval_mask]
            std = std.reshape(-1)[eval_mask]
            exact = exact.reshape(-1)[eval_mask]
            extra['calibration_eval_split'] = 'grid_excluding_temperature_fit'
            extra['calibration_eval_points'] = int(eval_mask.sum())
        else:
            extra['calibration_eval_split'] = 'full_grid_mask_too_small'
            extra['calibration_eval_points'] = int(mean.size)
    elif extra.get('std_calibration_source') == 'heldout':
        extra['calibration_eval_split'] = 'full_grid_holdout_temperature'
        extra['calibration_eval_points'] = int(mean.size)

    full_calibration._extra_metrics = extra
    try:
        return full_calibration(mean, std, exact, case)
    finally:
        full_calibration._extra_metrics = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all')
    args = parser.parse_args()
    cases = ['Laplace','Burgers_inv','Poisson'] if args.case=='all' else [args.case]
    results = []
    for c in cases:
        m = run_from_npz(c)
        if m: results.append(m)
    if results:
        print('\n  Calibration summary:')
        print(pd.DataFrame(results).to_string(index=False))
