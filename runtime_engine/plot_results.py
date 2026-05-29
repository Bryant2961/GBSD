# plot_results.py — Figures 2-4: Loss comparison, PDE results, uncertainty fields
"""
All legends use paper-quality method names. Colors/styles are consistent.
Reads from Results/ loss CSVs and results/raw/ prediction .npz files.

Usage:
    python plot_results.py --case Laplace
    python plot_results.py --case all
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paper_style import get_paper_name, get_style, setup_paper_style, STYLE

CASES = ['Laplace', 'Burgers_inv', 'Poisson']
RESULTS_DIR = './Results'
FIG_DIR = './results/figures'
RAW_DIR = './results/raw'

def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def _common_loss_iteration_cap(loss_dir):
    caps = []
    for csv_file in sorted(os.listdir(loss_dir)):
        if not csv_file.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(f'{loss_dir}/{csv_file}', usecols=['iter'])
            if not df.empty:
                caps.append(float(df['iter'].max()))
        except Exception:
            pass
    return int(min(caps)) if caps else None


def _rel_l2(pred, exact):
    denom = np.sum(exact ** 2)
    return np.sqrt(np.sum((pred - exact) ** 2) / max(denom, 1e-15))


def _smooth_for_plot(arr, sigma=1.0):
    if gaussian_filter is None:
        return arr
    return gaussian_filter(arr, sigma=sigma)


def _loss_plot_name(raw_label):
    """Use objective labels because these losses are not the same metric."""
    if raw_label == 'PINN' or raw_label.startswith('PINN_post'):
        return 'Teacher PINN objective'
    if raw_label == 'PINN_student' or raw_label.startswith('PsiNN'):
        return 'Structured Candidate objective'
    if 'Student_MCDropout' in raw_label:
        return 'Dense Bayesian Student objective'
    return get_paper_name(raw_label)


# ─── Figure 2a: Training loss comparison ───
def fig2a_loss_comparison(case):
    setup_paper_style()
    loss_dir = f'{RESULTS_DIR}/{case}_EXP/Loss'
    if not os.path.isdir(loss_dir): return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = []
    fair_iter = _common_loss_iteration_cap(loss_dir)

    for csv_file in sorted(os.listdir(loss_dir)):
        if not csv_file.endswith('.csv'): continue
        raw_label = csv_file.replace(f'{case}_EXP_loss_', '').replace('.csv', '')
        paper_name = _loss_plot_name(raw_label)
        style = get_style(paper_name)

        # Skip duplicates
        if paper_name in plotted: continue

        try:
            df = pd.read_csv(f'{loss_dir}/{csv_file}')
            if fair_iter is not None and 'iter' in df.columns:
                df = df[df['iter'] <= fair_iter]
                if df.empty:
                    continue
            col = 'loss' if 'loss' in df.columns else df.columns[1]
            ax.semilogy(df['iter'], df[col], style['ls'], color=style['color'],
                        label=paper_name,
                        lw=style['lw'], alpha=0.9)
            plotted.append(paper_name)
        except: pass

    ax.set_xlabel('Iteration'); ax.set_ylabel('Training objective (log scale)')
    if fair_iter is not None:
        ax.set_xlim(0, fair_iter)
    ax.set_title(f'{case} - Optimization Diagnostics')
    ax.legend(fontsize=8, ncol=1, loc='upper right'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_training_conflict_loss_{case}.{ext}')
    plt.close(fig)
    print(f'  Fig 2a: {case}')


# ─── Figure 2b: Loss components for Bayesian student ───
def fig2b_loss_components(case):
    setup_paper_style()
    # Try Bayesian student first, then deterministic student
    for suffix in ['Student_MCDropout_student', 'PINN_student']:
        csv_path = f'{RESULTS_DIR}/{case}_EXP/Loss/{case}_EXP_loss_{suffix}.csv'
        if os.path.isfile(csv_path):
            method_label = get_paper_name(suffix); break
    else:
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    component_map = [
        ('loss', 'Total loss', '#1f77b4', '-'),
        ('loss_teach', 'Distillation loss', '#ff7f0e', '--'),
        ('loss_rgl', 'KL / Regularization', '#2ca02c', '-.'),
        ('loss_student_d', 'Data loss', '#d62728', ':'),
        ('loss_f', 'PDE residual', '#9467bd', ':'),
        ('loss_b', 'Boundary loss', '#8c564b', ':'),
    ]
    for col, label, color, ls in component_map:
        if col in df.columns and df[col].abs().max() > 0:
            ax.semilogy(df['iter'], df[col], ls, lw=1.8, label=label, color=color)

    ax.set_xlabel('Iteration'); ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'{case} — {method_label} Loss Components')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_loss_components_{case}.{ext}')
    plt.close(fig)

    # Save loss log CSV copy
    log_dir = f'./results/logs/{case}'
    os.makedirs(log_dir, exist_ok=True)
    df.to_csv(f'{log_dir}/loss_history_bayesian_student.csv', index=False)
    print(f'  Fig 2b: {case}')


# ─── Figure 3: Core PDE results (2 rows: predictions + errors) ───
def fig3_core_results(case):
    setup_paper_style()
    npz = f'{RAW_DIR}/{case}_predictions.npz'
    if not os.path.isfile(npz):
        print(f'  Fig 3: {case} — no .npz (run: python utils/posterior_predict.py --case {case})'); return
    data = np.load(npz, allow_pickle=True)
    if 'X1' not in data: print(f'  Fig 3: {case} — no grid data'); return
    X1, X2 = data['X1'], data['X2']
    n = X1.shape[0]
    R = lambda a: (a[:,0] if a.ndim==2 and a.shape[1]==1 else a.flatten()).reshape(n,n)

    # Build panel lists
    row1, row1_t, row1_c = [], [], []  # predictions
    row2, row2_t, row2_c = [], [], []  # errors + uncertainty

    prediction_panels = [
        ('exact', 'Reference Solution'),
        ('pinn_pred', 'Teacher PINN'),
        ('bayesian_dense_mean', 'Dense Bayesian Student'),
        ('bayesian_structured_mean', 'Structured Candidate'),
        ('bayesian_mean', 'Guarded Final Source'),
    ]
    for key, title in prediction_panels:
        if key in data:
            row1.append(R(data[key])); row1_t.append(title); row1_c.append('RdBu_r')
    if 'bayesian_structured_mean' not in data and 'det_student_pred' in data:
        row1.append(R(data['det_student_pred']))
        row1_t.append('Structured Candidate')
        row1_c.append('RdBu_r')

    # Row 2: errors + uncertainty
    metric_rows = []
    error_panels = [
        ('pinn_pred', 'Teacher PINN'),
        ('bayesian_dense_mean', 'Dense Bayesian Student'),
        ('bayesian_structured_mean', 'Structured Candidate'),
        ('bayesian_mean', 'Guarded Final Source'),
    ]
    for key, label in error_panels:
        if 'exact' in data and key in data:
            err = np.abs(data[key] - data['exact'])
            rel = _rel_l2(data[key], data['exact'])
            row2.append(R(err))
            row2_t.append(f'|Error| {label}')
            row2_c.append('hot')
            metric_rows.append((label, err.mean(), err.max(), rel))
    if ('exact' in data and 'bayesian_structured_mean' not in data
            and 'det_student_pred' in data):
        err = np.abs(data['det_student_pred'] - data['exact'])
        rel = _rel_l2(data['det_student_pred'], data['exact'])
        row2.append(R(err))
        row2_t.append('|Error| Structured Candidate')
        row2_c.append('hot')
        metric_rows.append(('Structured Candidate', err.mean(), err.max(), rel))
    if 'bayesian_std' in data:
        std_raw = R(data['bayesian_std'])
        std_plot = _smooth_for_plot(std_raw, sigma=1.0)
        std_source = ''
        if 'bayesian_std_source' in data:
            try:
                std_source = str(data['bayesian_std_source'].reshape(-1)[0])
            except Exception:
                std_source = ''
        std_title = 'Calibrated Std\n(smoothed for display)'
        if 'structure_disagreement' in std_source:
            std_title = 'Calibrated Std\n(disagreement feature)'
        if 'temperature_calibrated' in std_source:
            std_title = 'Calibrated Std\n(temperature scaled)'
        row2.append(std_plot); row2_t.append(std_title); row2_c.append('hot')

    # Draw. Use a 3+2 layout for five-panel blocks so each field remains
    # readable after being scaled to manuscript column width.
    max_cols = 3
    row1_rows = int(np.ceil(len(row1) / max_cols)) if row1 else 0
    row2_rows = int(np.ceil(len(row2) / max_cols)) if row2 else 0
    nrows = row1_rows + row2_rows
    if nrows == 0:
        return
    fig = plt.figure(figsize=(12.6, 2.55*nrows), constrained_layout=False)
    grid = fig.add_gridspec(nrows, 6, wspace=0.92, hspace=0.40)
    fig.subplots_adjust(left=0.055, right=0.93, top=0.985, bottom=0.035)

    def compact_tick_label(value, _pos):
        if value == 0:
            return '0'
        if abs(value) < 1e-3 or abs(value) >= 1e3:
            return f'{value:.1e}'.replace('e-0', 'e-').replace('e+0', 'e+')
        return f'{value:.3g}'

    def polish_colorbar(cbar):
        cbar.formatter = FuncFormatter(compact_tick_label)
        cbar.ax.tick_params(labelsize=7, pad=1)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

    def add_centered_axis(row, index_in_row, count_in_row):
        spans = {
            3: [(0, 2), (2, 4), (4, 6)],
            2: [(1, 3), (3, 5)],
            1: [(2, 4)],
        }[count_in_row]
        start, end = spans[index_in_row]
        return fig.add_subplot(grid[row, start:end])

    def add_axis_for_panel(panel_index, total_panels, row_offset=0):
        row_in_block, index_in_row = divmod(panel_index, max_cols)
        remaining = total_panels - row_in_block * max_cols
        count_in_row = min(max_cols, remaining)
        return add_centered_axis(row_offset + row_in_block, index_in_row, count_in_row)

    # Shared colorbar range for Row 1 (solution panels), ensuring fair visual comparison.
    if row1:
        all_vals = np.concatenate([p.flatten() for p in row1])
        vmin1, vmax1 = all_vals.min(), all_vals.max()
        # Symmetrize for diverging colormaps
        vabs = max(abs(vmin1), abs(vmax1))
        vmin1, vmax1 = -vabs, vabs

    solution_axes, solution_im = [], None
    for i,(panel,title,cmap) in enumerate(zip(row1,row1_t,row1_c)):
        ax = add_axis_for_panel(i, len(row1))
        levels = np.linspace(vmin1, vmax1, 50)
        solution_im = ax.contourf(
            X1, X2, panel, levels=levels, cmap=cmap,
            vmin=vmin1, vmax=vmax1, extend='both')
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        solution_axes.append(ax)
    if solution_axes and solution_im is not None:
        polish_colorbar(fig.colorbar(
            solution_im, ax=solution_axes, fraction=0.022, pad=0.02))

    row2_offset = row1_rows
    for i,(panel,title,cmap) in enumerate(zip(row2,row2_t,row2_c)):
        ax = add_axis_for_panel(i, len(row2), row2_offset)
        im = ax.contourf(X1, X2, panel, levels=50, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        polish_colorbar(fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04))

    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_{case.lower()}_mean_error_uncertainty.{ext}',
                    bbox_inches='tight', dpi=300 if ext == 'png' else None)
    plt.close(fig)

    # Save raw prediction arrays
    pred_dir = f'./results/predictions/{case}'
    os.makedirs(pred_dir, exist_ok=True)
    if 'exact' in data: np.savez(f'{pred_dir}/ground_truth.npz', u=data['exact'], X1=X1, X2=X2)
    if 'pinn_pred' in data: np.savez(f'{pred_dir}/teacher_pinn_prediction.npz', u=data['pinn_pred'])
    if 'bayesian_dense_mean' in data: np.savez(f'{pred_dir}/dense_bayesian_student_prediction.npz', u=data['bayesian_dense_mean'])
    if 'bayesian_structured_mean' in data: np.savez(f'{pred_dir}/structured_candidate_prediction.npz', u=data['bayesian_structured_mean'])
    if 'det_student_pred' in data: np.savez(f'{pred_dir}/structured_candidate_legacy_prediction.npz', u=data['det_student_pred'])
    if 'bayesian_mean' in data:
        np.savez(f'{pred_dir}/bayesian_mean_std_error.npz',
                 mean=data['bayesian_mean'], std=data['bayesian_std'],
                 error=np.abs(data['bayesian_mean']-data['exact']) if 'exact' in data else np.array([]))
    for name, mean_err, max_err, rel in metric_rows:
        print(f'    {case} {name}: mean_error={mean_err:.4e}, '
              f'max_error={max_err:.4e}, rel_L2={rel:.4e}')
    print(f'  Fig 3: {case}')


# ─── Figure 4: Uncertainty field (3-panel) ───
def fig4_uncertainty_field(case):
    setup_paper_style()
    npz = f'{RAW_DIR}/{case}_predictions.npz'
    if not os.path.isfile(npz): return
    data = np.load(npz, allow_pickle=True)
    if 'bayesian_mean' not in data or 'X1' not in data: return
    X1, X2 = data['X1'], data['X2']
    n = X1.shape[0]
    R = lambda a: (a[:,0] if a.ndim==2 and a.shape[1]==1 else a.flatten()).reshape(n,n)

    mean_2d = R(data['bayesian_mean'])
    std_raw = R(data['bayesian_std'])
    std_2d = _smooth_for_plot(std_raw, sigma=1.0)
    std_source = ''
    if 'bayesian_std_source' in data:
        try:
            std_source = str(data['bayesian_std_source'].reshape(-1)[0])
        except Exception:
            std_source = ''

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].contourf(X1, X2, mean_2d, levels=50, cmap='RdBu_r')
    axes[0].set_title('Guarded Final Source', fontsize=12)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].contourf(X1, X2, std_2d, levels=50, cmap='hot')
    if 'temperature_calibrated' in std_source:
        std_title = 'Calibrated Std (temperature scaled)'
    elif 'structure_disagreement' in std_source:
        std_title = 'Calibrated Std (disagreement feature)'
    else:
        std_title = 'Calibrated Std (display-smoothed)'
    axes[1].set_title(std_title, fontsize=12)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if 'exact' in data:
        err = R(np.abs(data['bayesian_mean'] - data['exact']))
        im2 = axes[2].contourf(X1, X2, err, levels=50, cmap='hot')
        axes[2].set_title('Absolute Error', fontsize=12)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, 'No exact solution', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Absolute Error (N/A)')

    for ax in axes: ax.set_aspect('equal')
    fig.suptitle(f'{case} - Final Prediction and Uncertainty', fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_uncertainty_field_{case.lower()}.{ext}',
                    bbox_inches='tight', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'  Fig 4: {case}')


def plot_all_for_case(case):
    print(f'\n{"="*60}\n  Generating figures for: {case}\n{"="*60}')
    ensure_dirs()
    fig2a_loss_comparison(case)
    fig2b_loss_components(case)
    fig3_core_results(case)
    fig4_uncertainty_field(case)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all', choices=CASES+['all'])
    args = parser.parse_args()
    cases = CASES if args.case == 'all' else [args.case]
    for c in cases: plot_all_for_case(c)
