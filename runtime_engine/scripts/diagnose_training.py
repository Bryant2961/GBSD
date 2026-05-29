# coding = utf-8
"""
Diagnostic and Fix Script for Burgers Equation with VI-BNN

The original run showed training failure. This script provides:
1. Diagnosis of what went wrong
2. Fixed hyperparameters
3. Comparison between MC Dropout and VI-BNN
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from Module.Training import BayesianPsiNNTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_burgers_fixed():
    """
    Run Burgers equation with fixed hyperparameters for VI-BNN.
    
    Key fixes:
    1. Lower KL weight (1e-6 instead of 1e-4)
    2. Disable heteroscedastic mode initially
    3. More student training iterations
    4. Learning rate warmup
    """
    print("="*70)
    print("BURGERS EQUATION - FIXED VI-BNN TRAINING")
    print("="*70)
    
    # Fixed hyperparameters
    trainer = BayesianPsiNNTrainer(
        ques_name='Burgers_inv',
        ini_num='EXP_fixed',
        student_type='vi_bnn',
        heteroscedastic=False,      # FIX 1: Disable heteroscedastic initially
        prior_sigma=0.5,            # FIX 2: Tighter prior
        kl_weight=1e-6,             # FIX 3: Much lower KL weight (was 1e-4)
        l2_weight=1e-4
    )
    
    # Override training parameters
    trainer.train_steps = 15000     # More teacher iterations
    trainer.train_ratio = 1.0       # Equal student iterations
    trainer.learning_rate = 5e-4    # Slightly lower LR
    
    # Run training
    trainer.mesh_init()
    trainer.train_teacher(print_every=2000)
    trainer.train_student(print_every=1000)
    
    return trainer


def run_burgers_mc_dropout():
    """
    Run Burgers equation with MC Dropout (more stable alternative).
    """
    print("="*70)
    print("BURGERS EQUATION - MC DROPOUT (STABLE BASELINE)")
    print("="*70)
    
    trainer = BayesianPsiNNTrainer(
        ques_name='Burgers_inv',
        ini_num='EXP_mc',
        student_type='mc_dropout',
        heteroscedastic=False,
        dropout_rate=0.1,
        l2_weight=1e-3
    )
    
    trainer.mesh_init()
    trainer.train_teacher(print_every=2000)
    trainer.train_student(print_every=1000)
    
    return trainer


def diagnose_failure(results_dir='./Results/Burgers_inv_EXP/'):
    """
    Diagnose why the original VI-BNN training failed.
    """
    import pandas as pd
    
    print("\n" + "="*70)
    print("DIAGNOSIS OF TRAINING FAILURE")
    print("="*70)
    
    # Load losses
    teacher_loss = pd.read_csv(f'{results_dir}/Loss/Burgers_inv_EXP_teacher_loss.csv')
    student_loss = pd.read_csv(f'{results_dir}/Loss/Burgers_inv_EXP_student_loss.csv')
    
    print("\n1. TEACHER TRAINING")
    print(f"   Final loss: {teacher_loss['loss'].iloc[-1]:.2e}")
    print(f"   Final PDE loss: {teacher_loss['loss_f'].iloc[-1]:.2e}")
    print(f"   Final BC loss: {teacher_loss['loss_b'].iloc[-1]:.2e}")
    print(f"   Status: {'✅ Good' if teacher_loss['loss'].iloc[-1] < 1e-4 else '⚠️ May need more training'}")
    
    print("\n2. STUDENT (VI-BNN) TRAINING")
    print(f"   Final total loss: {student_loss['loss'].iloc[-1]:.2e}")
    print(f"   Final distill loss: {student_loss['loss_teach'].iloc[-1]:.2e}")
    print(f"   Final KL loss: {student_loss['loss_rgl'].iloc[-1]:.2e}")
    
    # Check for instability
    loss_std = np.std(student_loss['loss'].values[-1000:])
    loss_mean = np.mean(student_loss['loss'].values[-1000:])
    cv = loss_std / loss_mean  # Coefficient of variation
    
    print(f"\n   Loss stability (last 1000 iters):")
    print(f"   - Mean: {loss_mean:.2e}")
    print(f"   - Std: {loss_std:.2e}")
    print(f"   - CV (Std/Mean): {cv:.2f}")
    print(f"   - Status: {'❌ Unstable' if cv > 0.5 else '✅ Stable'}")
    
    # Check KL dominance
    kl_ratio = student_loss['loss_rgl'].iloc[-1] / student_loss['loss'].iloc[-1]
    print(f"\n3. KL DOMINANCE CHECK")
    print(f"   KL / Total ratio: {kl_ratio:.2%}")
    print(f"   Status: {'⚠️ KL may be too dominant' if kl_ratio > 0.3 else '✅ Balanced'}")
    
    print("\n4. RECOMMENDATIONS")
    if cv > 0.5:
        print("   - Reduce KL weight (try 1e-6 or 1e-7)")
        print("   - Try MC Dropout instead of VI-BNN")
        print("   - Increase student training iterations")
    if kl_ratio > 0.3:
        print("   - KL term is dominating - reduce kl_weight")
        print("   - Try tighter prior (prior_sigma=0.5)")


def compare_methods():
    """
    Compare MC Dropout vs VI-BNN on Burgers equation.
    """
    print("\n" + "="*70)
    print("METHOD COMPARISON: MC DROPOUT vs VI-BNN")
    print("="*70)
    
    results = {}
    
    # Run MC Dropout
    print("\n--- Running MC Dropout ---")
    trainer_mc = run_burgers_mc_dropout()
    
    # Evaluate MC Dropout
    trainer_mc.mesh_init()
    x_test = torch.cat([trainer_mc.x, trainer_mc.y], dim=1)
    preds_mc = trainer_mc.predict_with_uncertainty(x_test, n_samples=100)
    
    # Get exact solution
    x_np = x_test.detach().cpu().numpy()
    nu = 0.01 / np.pi
    exact = -np.sin(np.pi * x_np[:, 0]) * np.exp(-nu * np.pi**2 * x_np[:, 1])
    
    mean_mc = preds_mc['mean'].detach().cpu().numpy().flatten()
    std_mc = np.sqrt(preds_mc['variance'].detach().cpu().numpy().flatten())
    
    l2_mc = np.sqrt(np.mean((mean_mc - exact)**2))
    coverage_mc = np.mean(np.abs(mean_mc - exact) <= 2 * std_mc)
    
    results['MC Dropout'] = {
        'L2_error': l2_mc,
        'mean_uncertainty': std_mc.mean(),
        'coverage_95': coverage_mc
    }
    
    print(f"\nMC Dropout Results:")
    print(f"  L2 Error: {l2_mc:.4e}")
    print(f"  Mean Uncertainty: {std_mc.mean():.4e}")
    print(f"  95% Coverage: {coverage_mc:.2%}")
    
    # Run fixed VI-BNN
    print("\n--- Running Fixed VI-BNN ---")
    trainer_vi = run_burgers_fixed()
    
    # Evaluate VI-BNN
    preds_vi = trainer_vi.predict_with_uncertainty(x_test, n_samples=100)
    
    mean_vi = preds_vi['mean'].detach().cpu().numpy().flatten()
    std_vi = np.sqrt(preds_vi['variance'].detach().cpu().numpy().flatten())
    
    l2_vi = np.sqrt(np.mean((mean_vi - exact)**2))
    coverage_vi = np.mean(np.abs(mean_vi - exact) <= 2 * std_vi)
    
    results['VI-BNN (fixed)'] = {
        'L2_error': l2_vi,
        'mean_uncertainty': std_vi.mean(),
        'coverage_95': coverage_vi
    }
    
    print(f"\nVI-BNN (Fixed) Results:")
    print(f"  L2 Error: {l2_vi:.4e}")
    print(f"  Mean Uncertainty: {std_vi.mean():.4e}")
    print(f"  95% Coverage: {coverage_vi:.2%}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'L2 Error':<15} {'Mean Uncert.':<15} {'95% Coverage':<15}")
    print("-"*65)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['L2_error']:<15.4e} {metrics['mean_uncertainty']:<15.4e} {metrics['coverage_95']:<15.2%}")
    
    return results, trainer_mc, trainer_vi


def plot_comparison(trainer_mc, trainer_vi, save_dir='./Results/Burgers_comparison/'):
    """
    Create comparison plots between methods.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions
    trainer_mc.mesh_init()
    x_test = torch.cat([trainer_mc.x, trainer_mc.y], dim=1)
    
    preds_mc = trainer_mc.predict_with_uncertainty(x_test, n_samples=100)
    preds_vi = trainer_vi.predict_with_uncertainty(x_test, n_samples=100)
    
    # Exact solution
    x_np = x_test.detach().cpu().numpy()
    nu = 0.01 / np.pi
    exact = -np.sin(np.pi * x_np[:, 0]) * np.exp(-nu * np.pi**2 * x_np[:, 1])
    
    n = int(np.sqrt(len(x_np)))
    x1 = x_np[:, 0].reshape(n, n)
    x2 = x_np[:, 1].reshape(n, n)
    exact_2d = exact.reshape(n, n)
    
    mean_mc = preds_mc['mean'].detach().cpu().numpy().reshape(n, n)
    mean_vi = preds_vi['mean'].detach().cpu().numpy().reshape(n, n)
    std_mc = np.sqrt(preds_mc['variance'].detach().cpu().numpy()).reshape(n, n)
    std_vi = np.sqrt(preds_vi['variance'].detach().cpu().numpy()).reshape(n, n)
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Predictions
    im = axes[0, 0].contourf(x1, x2, exact_2d, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Reference Solution')
    plt.colorbar(im, ax=axes[0, 0])
    
    im = axes[0, 1].contourf(x1, x2, mean_mc, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('MC Dropout Prediction')
    plt.colorbar(im, ax=axes[0, 1])
    
    im = axes[0, 2].contourf(x1, x2, mean_vi, levels=50, cmap='RdBu_r')
    axes[0, 2].set_title('VI-BNN Prediction')
    plt.colorbar(im, ax=axes[0, 2])
    
    axes[0, 3].axis('off')
    
    # Row 2: Errors
    err_mc = np.abs(mean_mc - exact_2d)
    err_vi = np.abs(mean_vi - exact_2d)
    
    im = axes[1, 0].contourf(x1, x2, err_mc, levels=50, cmap='hot_r')
    axes[1, 0].set_title(f'MC Dropout Error\nL2={np.sqrt((err_mc**2).mean()):.4e}')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].contourf(x1, x2, err_vi, levels=50, cmap='hot_r')
    axes[1, 1].set_title(f'VI-BNN Error\nL2={np.sqrt((err_vi**2).mean()):.4e}')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].contourf(x1, x2, std_mc, levels=50, cmap='YlOrRd')
    axes[1, 2].set_title('MC Dropout Uncertainty')
    plt.colorbar(im, ax=axes[1, 2])
    
    im = axes[1, 3].contourf(x1, x2, std_vi, levels=50, cmap='YlOrRd')
    axes[1, 3].set_title('VI-BNN Uncertainty')
    plt.colorbar(im, ax=axes[1, 3])
    
    # Row 3: Calibration comparison
    # MC Dropout calibration
    err_mc_flat = err_mc.flatten()
    std_mc_flat = std_mc.flatten()
    within_2sigma_mc = err_mc_flat <= 2 * std_mc_flat
    
    err_vi_flat = err_vi.flatten()
    std_vi_flat = std_vi.flatten()
    within_2sigma_vi = err_vi_flat <= 2 * std_vi_flat
    
    axes[2, 0].scatter(std_mc_flat, err_mc_flat, alpha=0.3, s=10)
    max_val = max(std_mc_flat.max(), err_mc_flat.max())
    axes[2, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[2, 0].set_xlabel('Predicted Std')
    axes[2, 0].set_ylabel('Actual Error')
    axes[2, 0].set_title(f'MC Dropout Calibration\n95% Coverage: {within_2sigma_mc.mean():.1%}')
    
    axes[2, 1].scatter(std_vi_flat, err_vi_flat, alpha=0.3, s=10)
    max_val = max(std_vi_flat.max(), err_vi_flat.max())
    axes[2, 1].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[2, 1].set_xlabel('Predicted Std')
    axes[2, 1].set_ylabel('Actual Error')
    axes[2, 1].set_title(f'VI-BNN Calibration\n95% Coverage: {within_2sigma_vi.mean():.1%}')
    
    # Training history comparison
    axes[2, 2].semilogy(trainer_mc.student.iter_list, trainer_mc.student.loss_list, 
                        'b-', label='MC Dropout', alpha=0.7)
    axes[2, 2].semilogy(trainer_vi.student.iter_list, trainer_vi.student.loss_list, 
                        'r-', label='VI-BNN', alpha=0.7)
    axes[2, 2].set_xlabel('Iteration')
    axes[2, 2].set_ylabel('Student Loss')
    axes[2, 2].set_title('Student Training Comparison')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[2, 3].axis('off')
    summary_text = f"""
    COMPARISON SUMMARY
    ==================
    
    MC Dropout:
      L2 Error: {np.sqrt((err_mc**2).mean()):.4e}
      Mean Uncertainty: {std_mc.mean():.4e}
      95% Coverage: {within_2sigma_mc.mean():.1%}
    
    VI-BNN (Fixed):
      L2 Error: {np.sqrt((err_vi**2).mean()):.4e}
      Mean Uncertainty: {std_vi.mean():.4e}
      95% Coverage: {within_2sigma_vi.mean():.1%}
    
    Winner: {'MC Dropout' if np.sqrt((err_mc**2).mean()) < np.sqrt((err_vi**2).mean()) else 'VI-BNN'}
    """
    axes[2, 3].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    transform=axes[2, 3].transAxes, verticalalignment='center')
    
    for ax in axes.flat:
        if ax.get_xlabel():
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
    
    plt.suptitle('Burgers Equation: MC Dropout vs VI-BNN Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison figure saved to {save_dir}/method_comparison.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnose', action='store_true', help='Diagnose existing results')
    parser.add_argument('--fix', action='store_true', help='Run with fixed hyperparameters')
    parser.add_argument('--compare', action='store_true', help='Compare MC Dropout vs VI-BNN')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_failure()
    elif args.fix:
        trainer = run_burgers_fixed()
    elif args.compare:
        results, trainer_mc, trainer_vi = compare_methods()
        plot_comparison(trainer_mc, trainer_vi)
    else:
        # Default: run diagnosis
        diagnose_failure()
        print("\n\nTo fix the training, run:")
        print("  python fix_burgers.py --fix")
        print("\nTo compare methods, run:")
        print("  python fix_burgers.py --compare")
