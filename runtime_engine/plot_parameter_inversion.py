# plot_parameter_inversion.py — Figure 8: Parameter trajectory + errorbar comparison
"""
Burgers parameter inversion: the true viscosity is nu = 0.01/pi ≈ 0.003183.
Config `para_ctrl = 0.003183` already represents nu = 0.01/pi.
It must NOT be divided by pi again.

Usage: python plot_parameter_inversion.py --case Burgers_inv
"""
import os, sys, argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paper_style import get_paper_name, get_style, setup_paper_style

FIG_DIR = './results/figures'
MET_DIR = './results/metrics'

# TRUE PARAMETER VALUES (in the same scale as para_undetermin in the PDE)
# Burgers: net_f uses u_t + u*u_x - para_undetermin[0]*u_xx = 0
# So para_undetermin converges to nu = 0.01/pi ≈ 0.003183
# Config para_ctrl = 0.003183 IS the true nu value; do NOT divide by pi again.
TRUE_PARAMS = {
    'Burgers_inv': {
        'parameters_1': 0.01 / math.pi,  # ≈ 0.003183
    },
}
PARAM_LABELS = {
    'Burgers_inv': r'$\nu$ (diffusion coefficient in $u_t + uu_x = \nu u_{xx}$)',
}

PARAM_TRAJECTORY_LABELS = {
    'PINN': 'Teacher PINN',
    'PINN_post_minus': 'Symmetry-corrected PINN',
}

PARAM_TRAJECTORY_STYLE = {
    'Teacher PINN': {'color': '#1f77b4', 'ls': '-', 'lw': 2.0},
    'Symmetry-corrected PINN': {'color': '#ff7f0e', 'ls': '-', 'lw': 2.0},
}


def _trajectory_label(raw_name):
    return PARAM_TRAJECTORY_LABELS.get(raw_name, get_paper_name(raw_name))

def plot_trajectory(case):
    setup_paper_style()
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(MET_DIR, exist_ok=True)
    para_dir = f'./Results/{case}_EXP/Parameters'
    if not os.path.isdir(para_dir):
        print(f'  No parameter data for {case}'); return

    csv_files = [f for f in os.listdir(para_dir) if f.endswith('.csv')]
    if not csv_files: return

    true_params = TRUE_PARAMS.get(case, {})
    param_label = PARAM_LABELS.get(case, 'Parameter value')

    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []

    for f in sorted(csv_files):
        df = pd.read_csv(f'{para_dir}/{f}')
        raw_name = f.replace(f'{case}_EXP_paras_', '').replace('.csv', '')
        paper_name = _trajectory_label(raw_name)
        style = PARAM_TRAJECTORY_STYLE.get(paper_name, get_style(paper_name))

        for pc in [c for c in df.columns if c.startswith('parameters_')]:
            ax.plot(df['iter'], df[pc], style['ls'], color=style['color'],
                    lw=style['lw'], label=paper_name, alpha=0.9)

            final_val = df[pc].iloc[-1]
            true_val = true_params.get(pc)
            if true_val is not None:
                rows.append({
                    'case': case, 'method': paper_name, 'parameter': pc,
                    'true_value': true_val, 'estimated_mean': final_val,
                    'estimated_std': np.nan,  # deterministic — no posterior std
                    'absolute_error': abs(final_val - true_val),
                    'relative_error': abs(final_val - true_val) / (abs(true_val) + 1e-15),
                    'seed': 0,
                })

    # Draw true value
    for pname, tv in true_params.items():
        ax.axhline(y=tv, color='black', ls='--', lw=2,
                    label=f'True value ({tv:.6f})')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(param_label, fontsize=11)
    ax.set_title(f'{case} - Parameter Inversion Trajectory', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Add note about scale
    ax.annotate(f'Note: true $\\nu = 0.01/\\pi \\approx$ {list(true_params.values())[0]:.6f}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=8, va='top', color='gray',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_parameter_posterior_trajectory_{case.lower()}.{ext}')
    plt.close(fig)

    if rows:
        path = f'{MET_DIR}/parameter_inversion_metrics.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'  Parameter metrics → {path}')
    print(f'  Fig 8a: {case}')


def plot_errorbar_comparison(case):
    """Bar chart: final parameter estimates with error bars across methods."""
    setup_paper_style()
    path = f'{MET_DIR}/parameter_inversion_metrics.csv'
    if not os.path.isfile(path): return

    df = pd.read_csv(path)
    df = df[df['case'] == case]
    if df.empty: return

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = df['method'].unique()
    x = np.arange(len(methods))
    true_val = df['true_value'].iloc[0]

    ax.axhline(y=true_val, color='black', ls='--', lw=2, label='True value')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, method in enumerate(methods):
        sub = df[df['method'] == method]
        mean = sub['estimated_mean'].values[0]
        std = sub['estimated_std'].values[0] if not np.isnan(sub['estimated_std'].values[0]) else 0
        ax.bar(i, mean, yerr=2*std, capsize=5, color=colors[i % len(colors)],
               alpha=0.8, label=method, width=0.6)

    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(PARAM_LABELS.get(case, 'Parameter'))
    ax.set_title(f'{case} — Parameter Inversion Comparison')
    ax.legend(fontsize=8); fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_parameter_inversion_errorbar.{ext}')
    plt.close(fig)
    print(f'  Fig 8b: {case}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='Burgers_inv')
    args = parser.parse_args()
    plot_trajectory(args.case)
    plot_errorbar_comparison(args.case)
