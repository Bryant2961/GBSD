# coding = utf-8
"""
Poisson Equation Runner

Problem: u_xx + u_yy = f(x,y)  on  [0,1]^2
BC:      u = 0 on all boundaries (Dirichlet)
"""
import numpy as np
import torch

from Module.Training import (
    BayesianPsiNNTrainer, poisson_residual
)
from Module.StructureDiscovery import StructureDiscovery
from Module.StructuredPINN import build_structured_pinn
from Module.ReconstructionTrainer import ReconstructionTrainer, get_boundary_fn
from Module.Evaluation import evaluate_reconstruction


PROBLEM_NAME = 'Poisson'


def exact_solution(x_eval, n_terms=4):
    """Fourier series exact solution for the Poisson problem."""
    x = x_eval[:, 0:1]
    y = x_eval[:, 1:2]
    u = np.zeros_like(x) if isinstance(x, np.ndarray) else torch.zeros_like(x)
    pi = np.pi
    for k in range(1, n_terms + 1):
        if isinstance(x, np.ndarray):
            u = u + 0.5 / (2 * pi**2) * ((-1)**(k+1)) * \
                np.sin(k * pi * x) * np.sin(k * pi * y)
        else:
            u = u + 0.5 / (2 * pi**2) * ((-1)**(k+1)) * \
                torch.sin(k * pi * x) * torch.sin(k * pi * y)
    return u


def run_problem(config):
    """
    Run the GBSD runtime pipeline for the Poisson equation.

    Args:
        config: argparse.Namespace with all CLI parameters

    Returns:
        dict with pipeline results
    """
    from Problems._runner import run_pipeline
    return run_pipeline(
        config=config,
        problem_name=PROBLEM_NAME,
        pde_residual_fn=poisson_residual,
        exact_solution_fn=exact_solution,
    )
