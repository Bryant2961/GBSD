# coding = utf-8
"""
Shared pipeline runner for all GBSD problem types.

Called by each problem module (laplace.py, burgers_inverse.py, poisson.py).
Orchestrates the 5-stage pipeline:
  Stage 1: Teacher PINN training
  Stage 2: Student distillation (or skip if vanilla)
  Stage 3: Structure discovery
  Stage 4: Structured reconstruction
  Stage 5: Evaluation
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from Module.Training import BayesianPsiNNTrainer
from Module.StructureDiscovery import StructureDiscovery
from Module.StructuredPINN import build_structured_pinn
from Module.ReconstructionTrainer import ReconstructionTrainer, get_boundary_fn
from Module.Evaluation import evaluate_reconstruction

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def run_pipeline(config, problem_name: str, pde_residual_fn: Callable,
                 exact_solution_fn: Optional[Callable] = None) -> dict:
    """
    Unified 5-stage pipeline.

    Args:
        config: argparse.Namespace with CLI parameters
        problem_name: e.g. 'Laplace', 'Burgers_inv', 'Poisson'
        pde_residual_fn: PDE residual callable(model, x) -> scalar
        exact_solution_fn: callable(x_eval) -> u_exact or None

    Returns:
        dict with pipeline results
    """
    student = config.student
    print_every = config.print_every

    print("\n" + "=" * 70)
    print("  BAYESIAN PSI-NN: COMPLETE PIPELINE")
    print("=" * 70)
    print(f"  Problem:  {problem_name}")
    print(f"  Student:  {student}")
    print(f"  Device:   {device}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1 + 2: Teacher training & student distillation
    # ------------------------------------------------------------------
    trainer = BayesianPsiNNTrainer(
        ques_name=problem_name,
        ini_num='EXP',
        student_type=student,
        heteroscedastic=getattr(config, 'heteroscedastic', False),
        dropout_rate=getattr(config, 'dropout_rate', 0.1),
        prior_sigma=getattr(config, 'prior_sigma', 1.0),
        kl_weight=getattr(config, 'kl_weight', 1e-4),
        l2_weight=getattr(config, 'l2_weight', 1e-3),
        grad_clip=1.0,
    )

    # Apply CLI overrides for epochs / learning rate
    if config.epochs_teacher is not None:
        trainer.train_steps = config.epochs_teacher
    if config.epochs_student is not None:
        # Compute ratio so that student_steps = epochs_student
        trainer.train_ratio = config.epochs_student / max(1, trainer.train_steps)
    if config.lr_teacher is not None:
        trainer.learning_rate = config.lr_teacher

    trainer.mesh_init()
    trainer.train_teacher(print_every=print_every)
    trainer.train_student(print_every=print_every)
    trainer.save_models()
    trainer.save_losses()

    # ------------------------------------------------------------------
    # Stage 3: Structure Discovery
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: Structure Discovery")
    print("=" * 60)

    # Use teacher in vanilla mode, student otherwise
    discover_model = trainer.teacher if student == 'vanilla' else trainer.student
    sd = StructureDiscovery(discover_model,
                            cluster_distance=config.cluster_distance)
    structure = sd.extract_structure(verbose=True)
    relation_matrices = sd.build_relation_matrix(structure)
    stats = sd.get_compression_stats(structure)
    print(f"\n  Overall compression: {stats['overall_compression']:.1f}x")

    # ------------------------------------------------------------------
    # Stage 4: Structured Reconstruction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Structured Reconstruction")
    print("=" * 60)

    dropout_for_recon = config.dropout_rate if config.bayesian_recon else 0.0
    structured_model = build_structured_pinn(
        structure=structure,
        relation_matrices=relation_matrices,
        reference_model=discover_model,
        dropout_rate=dropout_for_recon,
    )
    param_info = structured_model.count_parameters()
    print(f"  Trainable:  {param_info['trainable']}")
    print(f"  Original:   {param_info['original']}")
    print(f"  Compression: {param_info['compression_ratio']:.2f}x")

    # Collocation points
    grid_n = int(trainer.config.get('grid_node_num', 50))
    x_lin = torch.linspace(trainer.x_min, trainer.x_max, grid_n, device=device)
    y_lin = torch.linspace(trainer.y_min, trainer.y_max, grid_n, device=device)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing='ij')
    x_collocation = torch.stack([xx.reshape(-1), yy.reshape(-1)],
                                dim=1).requires_grad_(True)

    bc_fn = get_boundary_fn(problem_name, config=trainer.config, device=device)

    lr_recon = config.lr_recon if config.lr_recon is not None else 1e-3
    epochs_recon = config.epochs_recon if config.epochs_recon is not None else 5000

    recon_config = {
        'lr': lr_recon,
        'epochs': epochs_recon,
        'lr_step': max(1, epochs_recon // 3),
        'lr_gamma': 0.5,
        'lambda_pde': 1.0,
        'lambda_bc': 1.0,
        'lambda_data': 0.0,
        'lambda_distill': config.lambda_distill,
        'grad_clip': 1.0,
        'print_every': print_every,
        'seed': 1234,
    }

    recon_trainer = ReconstructionTrainer(
        structured_model=structured_model,
        pde_residual_fn=pde_residual_fn,
        boundary_loss_fn=bc_fn,
        teacher_model=trainer.teacher if config.lambda_distill > 0 else None,
        device=device,
        config=recon_config,
    )
    recon_trainer.set_collocation_points(x_collocation)
    history = recon_trainer.train(verbose=True)

    # Save checkpoint
    save_dir = f'./Results/{problem_name}_EXP/Models/'
    os.makedirs(save_dir, exist_ok=True)
    recon_trainer.save_checkpoint(
        os.path.join(save_dir, f'{problem_name}_structured.pth')
    )

    # ------------------------------------------------------------------
    # Stage 5: Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 5: Evaluation")
    print("=" * 60)

    eval_n = 80
    x_e = torch.linspace(trainer.x_min, trainer.x_max, eval_n, device=device)
    y_e = torch.linspace(trainer.y_min, trainer.y_max, eval_n, device=device)
    xx_e, yy_e = torch.meshgrid(x_e, y_e, indexing='ij')
    x_eval = torch.stack([xx_e.reshape(-1), yy_e.reshape(-1)], dim=1)

    u_exact = None
    if exact_solution_fn is not None:
        u_exact = exact_solution_fn(x_eval)

    include_uq = config.include_uq and dropout_for_recon > 0
    results = evaluate_reconstruction(
        pre_model=trainer.teacher,
        post_model=structured_model,
        pde_residual_fn=pde_residual_fn,
        x_eval=x_eval,
        u_exact=u_exact,
        include_uq=include_uq,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Problem:     {problem_name}")
    print(f"  Student:     {student}")
    print(f"  Compression: {param_info['compression_ratio']:.2f}x "
          f"({param_info['original']} -> {param_info['trainable']} params)")
    print(f"  Results in:  ./Results/{problem_name}_EXP/")
    print("=" * 70)

    return {
        'trainer': trainer,
        'structure': structure,
        'structured_model': structured_model,
        'eval_results': results,
        'param_info': param_info,
    }
