import Module.Training as Training

import torch
torch.random.manual_seed(1234)

# =============================================================================
# Configuration
#
#   MODE   : 'baseline', 'bayesian', or 'all'
#   PRESET : 'preview' (fast debugging) or 'full' (paper-quality runs)
# =============================================================================

MODE   = 'all'
PRESET = 'preview'

# =============================================================================
# Per-problem budgets
# =============================================================================

BUDGETS = {
    'preview': {
        'Laplace':     {'teacher':  500, 'student':  300, 'recon':  500, 'print': 100},
        'Burgers_inv': {'teacher':  500, 'student':  500, 'recon': 1000, 'print': 100},
        'Poisson':     {'teacher':  500, 'student':  300, 'recon':  500, 'print': 100},
        'Flow':        {'teacher':  500, 'student':  500, 'recon': 1000, 'print': 100},
    },
    'full': {
        'Laplace':     {'teacher': 10000, 'student':  5000, 'recon': 15000, 'print': 1000},
        'Burgers_inv': {'teacher': 15000, 'student': 10000, 'recon': 20000, 'print': 1000},
        'Poisson':     {'teacher': 15000, 'student':  7500, 'recon': 15000, 'print': 1000},
        'Flow':        {'teacher': 50000, 'student': 50000, 'recon': 30000, 'print': 5000},
    },
}

# Safe defaults used for both presets
COMMON_KWARGS = dict(
    weight_rgl=1e-5,
    cluster_mode='relative',
    cluster_distance=0.1,
)

BAYESIAN_KWARGS = dict(
    student_type='mc_dropout',
    bayesian_recon=True,
    include_uq=True,
    n_mc_samples=10 if PRESET == 'preview' else 100,
)

PROBLEMS_BASELINE = ['Laplace', 'Burgers_inv', 'Poisson', 'Flow']
PROBLEMS_BAYESIAN = ['Laplace', 'Burgers_inv', 'Poisson']


def apply_budget(task, problem):
    """Apply per-problem budget from the active preset."""
    b = BUDGETS[PRESET][problem]
    task.train_steps = b['teacher']
    task.student_train_steps = b['student']   # explicit, not ratio-derived
    task.train_ratio = 1.0                     # deprecated
    task.epochs_recon = b['recon']
    task.print_every  = b['print']
    task.pace_record_state = 0

    # Sane milestone / print cadence
    gap = b['print']
    task.pace_record_gap  = [gap]
    task.pace_record_skip = [0]
    task.milestone = [b['teacher'] // 2, int(b['teacher'] * 0.8)]

    # Reconstruction: start distillation high, anneal down
    task.lambda_distill = 1.0
    return task


# =============================================================================
# BASELINE
# =============================================================================

if MODE in ('baseline', 'all'):
    print('\n' + '=' * 70)
    print(f'  BASELINE  [{PRESET.upper()}]')
    print('=' * 70)

    for problem in PROBLEMS_BASELINE:
        print(f'\n{"─" * 60}')
        print(f'  {problem}')
        print(f'{"─" * 60}')

        # -- Direct training: PINN + PINN-post --
        task = Training.model(problem, 'EXP', **COMMON_KWARGS)
        apply_budget(task, problem)
        task.train()

        # -- Structured-distillation pipeline: deterministic distillation --
        task = Training.model(problem, 'EXP', distill=True, **COMMON_KWARGS)
        apply_budget(task, problem)
        task.train()

    # Burgers_inv_distill: full-mode only
    if PRESET == 'full':
        print(f'\n{"─" * 60}')
        print(f'  Burgers_inv_distill')
        print(f'{"─" * 60}')
        Training.model('Burgers_inv_distill', 'EXP', **COMMON_KWARGS).train()

# =============================================================================
# BAYESIAN
# =============================================================================

if MODE in ('bayesian', 'all'):
    print('\n' + '=' * 70)
    print(f'  BAYESIAN  [{PRESET.upper()}]')
    print('=' * 70)

    for problem in PROBLEMS_BAYESIAN:
        print(f'\n{"─" * 60}')
        print(f'  {problem} (Bayesian MC Dropout)')
        print(f'{"─" * 60}')

        task = Training.model(problem, 'EXP', **COMMON_KWARGS, **BAYESIAN_KWARGS)
        apply_budget(task, problem)
        task.train()
