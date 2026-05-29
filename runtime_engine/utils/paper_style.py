# utils/paper_style.py — Consistent method naming, colors, and styles (v3.37)
"""
Centralized mapping from internal file names to paper-quality labels.
Import this in every plotting script for consistency.

v3.37 changes (JCP review terminology pass):
  * Added "Deterministic PINN" and "Direct MC-Dropout PINN" entries.
  * Renamed "Reference Solution" canonical key to match the manuscript.
  * Each entry's display string is exactly the term the manuscript uses,
    so plot legends and table headers never disagree with the prose.
"""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Method name mapping (file name fragment → paper label) ──
# Order matters when the lookup falls back to fragment matching: the
# entries are queried longest-first, so prefer specific keys.
METHOD_NAMES = {
    # Teacher PINNs and their post-processed variants.
    'PINN': 'Teacher PINN',
    'PINN_post_plus': 'Teacher PINN',
    'PINN_post_minus': 'Teacher PINN',
    'PINN_post_poisson': 'Teacher PINN',
    'PINN_post_flow': 'Teacher PINN',
    # Distilled deterministic student (deterministic baseline).
    'Student_Deterministic_student': 'Deterministic PINN student',
    # Structured candidates (post-clustering).
    'PINN_student': 'Structured Candidate',
    'PsiNN_laplace': 'Structured Candidate',
    'PsiNN_burgers': 'Structured Candidate',
    'PsiNN_poisson': 'Structured Candidate',
    'PsiNN_flow': 'Structured Candidate',
    # Dense Bayesian student (the standard GBSD student).
    'Student_MCDropout_student': 'Dense Bayesian Student',
    # Direct MC-Dropout PINN baseline (no teacher, no distillation).
    'Direct_MCDropout_PINN':       'Direct MC-Dropout PINN',
    'Student_MCDropout_direct':    'Direct MC-Dropout PINN',
    'direct_mc_dropout_pinn':      'Direct MC-Dropout PINN',
}

# Simplified mapping for cross-case comparison
METHOD_SHORT = {
    'PINN_post':                 'Teacher PINN',
    'Direct_MCDropout':          'Direct MC-Dropout PINN',
    'direct_mc_dropout':         'Direct MC-Dropout PINN',
    'Student_Deterministic':     'Deterministic PINN student',
    'Student_MCDropout':         'Dense Bayesian Student',
    'PINN':                      'Teacher PINN',
    'PsiNN':                     'Structured Candidate',
    'PINN_student':              'Structured Candidate',
}

# ── Colors and line styles per method category ──
STYLE = {
    'Reference Solution':         {'color': '#111111', 'ls': '--', 'lw': 2.0},
    'Teacher PINN':               {'color': '#1f77b4', 'ls': '-',  'lw': 2.0},
    'Deterministic PINN student': {'color': '#7f7f7f', 'ls': ':',  'lw': 2.0},
    'Direct MC-Dropout PINN':     {'color': '#e377c2', 'ls': '-',  'lw': 2.0},
    'Dense Bayesian Student':     {'color': '#d62728', 'ls': '-',  'lw': 2.5},
    'Structured Candidate':       {'color': '#2ca02c', 'ls': '-.', 'lw': 2.0},
    'Guarded Final Source':       {'color': '#9467bd', 'ls': '-',  'lw': 2.5},
}

# ── True parameter values per problem ──
# For Burgers_inv: PDE is u_t + u*u_x - nu*u_xx = 0
# Config para_ctrl = 0.003183 ≈ 0.01/pi IS the true nu.
# Do NOT divide by pi again.
TRUE_PARAMS = {
    'Burgers_inv': {
        'parameters_1': {
            'true_value': 0.01 / math.pi,
            'label': r'$\nu$ (diffusion coefficient)',
            'display_name': r'$\nu$',
        }
    },
}

# ── Eval grid configs per case ──
GRID_CONFIGS = {
    'Laplace':     {'xmin': -1, 'xmax': 1, 'ymin': -1, 'ymax': 1, 'xlabel': 'x', 'ylabel': 'y'},
    'Burgers_inv': {'xmin': -1, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'xlabel': 'x', 'ylabel': 't'},
    'Poisson':     {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'xlabel': 'x', 'ylabel': 'y'},
}


def get_paper_name(raw_name):
    """Convert raw file-name fragment to paper-quality label.

    Longest-key match wins to avoid 'PINN' matching 'Direct_MCDropout_PINN'.
    """
    if raw_name in METHOD_NAMES:
        return METHOD_NAMES[raw_name]
    # Sort by length descending so 'Direct_MCDropout' beats 'PINN'.
    for key in sorted(METHOD_SHORT.keys(), key=len, reverse=True):
        if key in raw_name:
            return METHOD_SHORT[key]
    return raw_name


def get_style(paper_name):
    """Get color/linestyle for a paper method name. Longest match wins."""
    for key in sorted(STYLE.keys(), key=len, reverse=True):
        if key in paper_name:
            return STYLE[key]
    return {'color': '#888888', 'ls': '-', 'lw': 1.5}


def setup_paper_style():
    """Set matplotlib defaults for paper quality."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
