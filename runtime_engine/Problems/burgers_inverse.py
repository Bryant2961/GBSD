# coding = utf-8
"""
Burgers Inverse Problem Runner

Problem: u_t + u * u_x = nu * u_xx  (inverse: infer nu)
Domain:  x in [-1,1], t in [0,1]
IC:      u(x,0) = -sin(pi*x)
BC:      u(-1,t) = u(1,t) = 0
"""
import numpy as np
import torch

from Module.Training import (
    BayesianPsiNNTrainer, burgers_residual
)
from Module.StructureDiscovery import StructureDiscovery
from Module.StructuredPINN import build_structured_pinn
from Module.ReconstructionTrainer import ReconstructionTrainer, get_boundary_fn
from Module.Evaluation import evaluate_reconstruction


PROBLEM_NAME = 'Burgers_inv'


def exact_solution(x_eval):
    """Interpolate the generated high-resolution Burgers reference solution."""
    import os
    ref_path = './Database/Burgers_inv_reference.npz'
    if not os.path.isfile(ref_path):
        return None
    from scipy.interpolate import RegularGridInterpolator
    data = np.load(ref_path, allow_pickle=True)
    interp = RegularGridInterpolator(
        (data['t'], data['x']), data['u'],
        bounds_error=False, fill_value=None)
    x_np = x_eval.detach().cpu().numpy() if torch.is_tensor(x_eval) else x_eval
    u_np = interp(np.column_stack([x_np[:, 1], x_np[:, 0]])).reshape(-1, 1)
    if torch.is_tensor(x_eval):
        return torch.tensor(u_np, dtype=torch.float32, device=x_eval.device)
    return u_np


def run_problem(config):
    """
    Run the GBSD runtime pipeline for the Burgers equation.

    Args:
        config: argparse.Namespace with all CLI parameters

    Returns:
        dict with pipeline results
    """
    from Problems._runner import run_pipeline
    return run_pipeline(
        config=config,
        problem_name=PROBLEM_NAME,
        pde_residual_fn=burgers_residual,
        exact_solution_fn=exact_solution,
    )
