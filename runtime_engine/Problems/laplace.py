# coding = utf-8
"""
Laplace Equation Runner

Problem: u_xx + u_yy = 0  on  [-1,1]^2
Exact:   u(x,y) = x^3 - 3xy^2
"""
import numpy as np
import torch

from Module.Training import (
    BayesianPsiNNTrainer, laplace_residual
)
from Module.StructureDiscovery import StructureDiscovery
from Module.StructuredPINN import build_structured_pinn
from Module.ReconstructionTrainer import ReconstructionTrainer, get_boundary_fn
from Module.Evaluation import evaluate_reconstruction


PROBLEM_NAME = 'Laplace'


def exact_solution(x_eval):
    """Exact: u = x^3 - 3xy^2"""
    x = x_eval[:, 0:1]
    y = x_eval[:, 1:2]
    return x ** 3 - 3 * x * y ** 2


def run_problem(config):
    """
    Run the GBSD runtime pipeline for the Laplace equation.

    Stages:
      1. Teacher PINN training
      2. Student distillation (or skip if vanilla)
      3. Structure discovery
      4. Structured reconstruction
      5. Evaluation

    Args:
        config: argparse.Namespace with all CLI parameters

    Returns:
        dict with pipeline results
    """
    from Problems._runner import run_pipeline
    return run_pipeline(
        config=config,
        problem_name=PROBLEM_NAME,
        pde_residual_fn=laplace_residual,
        exact_solution_fn=exact_solution,
    )
