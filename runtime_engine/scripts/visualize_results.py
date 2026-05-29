# coding = utf-8
"""
Visualization script for GBSD runtime diagnostics

Run after training to generate all paper figures:
    python visualize_results.py --problem Laplace --config EXP --student mc_dropout
    python visualize_results.py --problem Burgers_inv --config EXP --student vi_bnn
    python visualize_results.py --problem Poisson --config EXP --student mc_dropout

Or generate all at once:
    python visualize_results.py --all
"""
import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import modules
from Module.Training import BayesianPsiNNTrainer
import Module.PINN as PINN
import Module.Bayesian.Student_MCDropout as MCDropout
import Module.Bayesian.Student_VIBNN as VIBNN
import Module.UQ.UncertaintyEstimation as UE
import Module.StructureDiscovery as SD
import Module.Vis.Visualization as Vis

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_trained_models(ques_name: str, ini_num: str, student_type: str,
                        heteroscedastic: bool = False):
    """
    Load trained teacher and student models.
    
    Args:
        ques_name: Problem name (Laplace, Burgers_inv, Poisson)
        ini_num: Configuration number (EXP)
        student_type: 'mc_dropout' or 'vi_bnn'
        heteroscedastic: Whether heteroscedastic mode was used
    
    Returns:
        trainer: BayesianPsiNNTrainer with loaded models
    """
    # Create trainer (this sets up the network architectures)
    trainer = BayesianPsiNNTrainer(
        ques_name=ques_name,
        ini_num=ini_num,
        student_type=student_type,
        heteroscedastic=heteroscedastic
    )
    
    # Load saved weights
    model_dir = f'./Results/{ques_name}_{ini_num}/Models/'
    
    teacher_path = f'{model_dir}/{ques_name}_{ini_num}_teacher.pth'
    student_path = f'{model_dir}/{ques_name}_{ini_num}_student_{student_type}.pth'
    
    if os.path.exists(teacher_path):
        trainer.teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"Loaded teacher from {teacher_path}")
    else:
        print(f"Warning: Teacher model not found at {teacher_path}")
    
    if os.path.exists(student_path):
        trainer.student.load_state_dict(torch.load(student_path, map_location=device))
        print(f"Loaded student from {student_path}")
    else:
        print(f"Warning: Student model not found at {student_path}")
    
    # Load training losses for plotting
    loss_dir = f'./Results/{ques_name}_{ini_num}/Loss/'
    
    teacher_loss_path = f'{loss_dir}/{ques_name}_{ini_num}_teacher_loss.csv'
    student_loss_path = f'{loss_dir}/{ques_name}_{ini_num}_student_loss.csv'
    
    if os.path.exists(teacher_loss_path):
        df = pd.read_csv(teacher_loss_path)
        trainer.teacher.iter_list = df['iter'].tolist()
        trainer.teacher.loss_list = df['loss'].tolist()
        trainer.teacher.loss_f_list = df['loss_f'].tolist() if 'loss_f' in df else []
        trainer.teacher.loss_b_list = df['loss_b'].tolist() if 'loss_b' in df else []
        trainer.teacher.loss_rgl_list = df['loss_rgl'].tolist() if 'loss_rgl' in df else []
    
    if os.path.exists(student_loss_path):
        df = pd.read_csv(student_loss_path)
        trainer.student.iter_list = df['iter'].tolist()
        trainer.student.loss_list = df['loss'].tolist()
        trainer.student.loss_teach_list = df['loss_teach'].tolist() if 'loss_teach' in df else []
        trainer.student.loss_rgl_list = df['loss_rgl'].tolist() if 'loss_rgl' in df else []
    
    return trainer


def get_exact_solution(ques_name: str, x: torch.Tensor) -> np.ndarray:
    """
    Get exact solution for the given problem.
    """
    x_np = x.detach().cpu().numpy()
    x1, x2 = x_np[:, 0], x_np[:, 1]
    
    if 'Laplace' in ques_name:
        # u = x1^3 - 3*x1*x2^2
        return x1**3 - 3*x1*x2**2
    
    elif 'Burgers' in ques_name:
        # Burgers equation - numerical reference (simplified)
        # For visualization, we use the initial condition evolved
        # In practice, load from reference data
        nu = 0.01 / np.pi
        t = x2
        # Approximate solution (Cole-Hopf for small t)
        return -np.sin(np.pi * x1) * np.exp(-nu * np.pi**2 * t)
    
    elif 'Poisson' in ques_name:
        # Poisson with 4-frequency source
        u = np.zeros_like(x1)
        for k in range(1, 5):
            coef = 0.5 * ((-1)**(k+1)) / (k**2 * np.pi**2)
            u += coef * np.sin(k * np.pi * x1) * np.sin(k * np.pi * x2)
        return u
    
    else:
        return None


def generate_visualizations(ques_name: str, ini_num: str, student_type: str,
                            heteroscedastic: bool = False,
                            n_samples: int = 100):
    """
    Generate all visualizations for a trained model.
    """
    print(f"\n{'='*60}")
    print(f"Generating visualizations for {ques_name} ({student_type})")
    print(f"{'='*60}")
    
    # Load trained models
    trainer = load_trained_models(ques_name, ini_num, student_type, heteroscedastic)
    
    # Create output directory
    fig_dir = f'./Results/{ques_name}_{ini_num}/Figures/'
    os.makedirs(fig_dir, exist_ok=True)
    
    # Initialize mesh
    trainer.mesh_init()
    
    # Create test grid
    n_grid = 50
    x1 = torch.linspace(trainer.x_min, trainer.x_max, n_grid)
    x2 = torch.linspace(trainer.y_min, trainer.y_max, n_grid)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_test = torch.stack([X1.flatten(), X2.flatten()], dim=1).float().to(device)
    
    # Get predictions with uncertainty
    print("\nComputing predictions with uncertainty...")
    trainer.student.eval()
    predictions = trainer.predict_with_uncertainty(x_test, n_samples=n_samples)
    
    # Get exact solution
    exact = get_exact_solution(ques_name, x_test)
    
    # Discover structure
    print("\nDiscovering network structure...")
    structure = trainer.discover_structure(cluster_distance=0.1)
    relation_matrices = trainer.get_relation_matrices(structure)
    
    # Reshape for plotting
    n = n_grid
    x1_2d = X1.numpy()
    x2_2d = X2.numpy()
    
    mean = predictions['mean'].detach().cpu().numpy().reshape(n, n)
    epistemic = np.sqrt(predictions['epistemic'].detach().cpu().numpy()).reshape(n, n)
    aleatoric = np.sqrt(predictions['aleatoric'].detach().cpu().numpy()).reshape(n, n)
    total = np.sqrt(predictions['variance'].detach().cpu().numpy()).reshape(n, n)
    
    exact_2d = exact.reshape(n, n) if exact is not None else None
    
    # =========================================================================
    # FIGURE 1: Training History
    # =========================================================================
    print("\n[1/8] Generating training history plot...")
    fig = Vis.plot_training_history(trainer, save_path=f'{fig_dir}/01_training_history.png')
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Numerical Results (Paper Fig. 4 style)
    # =========================================================================
    print("[2/8] Generating numerical results plot...")
    
    # Get teacher prediction
    trainer.teacher.eval()
    with torch.no_grad():
        u_teacher = trainer.teacher(x_test).cpu().numpy().reshape(n, n)
    
    predictions_dict = {
        'Teacher (PINN)': u_teacher,
        f'Student ({student_type})': mean
    }
    
    loss_histories = {}
    if trainer.teacher.loss_list:
        loss_histories['Teacher'] = trainer.teacher.loss_list
    if trainer.student.loss_list:
        loss_histories['Student'] = trainer.student.loss_list
    
    fig = Vis.plot_numerical_results(
        x1_2d, x2_2d, exact_2d if exact_2d is not None else mean,
        predictions_dict, loss_histories,
        problem_name=ques_name,
        save_path=f'{fig_dir}/02_numerical_results.png'
    )
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Uncertainty Decomposition
    # =========================================================================
    print("[3/8] Generating uncertainty decomposition plot...")
    fig = Vis.plot_uncertainty_decomposition(
        x1_2d, x2_2d, mean, epistemic, aleatoric, total, exact_2d,
        save_path=f'{fig_dir}/03_uncertainty_decomposition.png'
    )
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Weight Clusters (Paper Fig. 5b style)
    # =========================================================================
    print("[4/8] Generating weight clustering plot...")
    fig = Vis.plot_weight_clusters(structure, save_path=f'{fig_dir}/04_weight_clusters.png')
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Network Structure with Relation Matrices
    # =========================================================================
    print("[5/8] Generating network structure plot...")
    fig = Vis.plot_network_structure(
        structure, relation_matrices,
        save_path=f'{fig_dir}/05_network_structure.png'
    )
    plt.close()
    
    # =========================================================================
    # FIGURE 6: Calibration Curve
    # =========================================================================
    if exact_2d is not None:
        print("[6/8] Generating calibration plot...")
        actual_errors = np.abs(mean - exact_2d).flatten()
        predicted_std = total.flatten()
        fig = Vis.plot_calibration(
            predicted_std, actual_errors,
            save_path=f'{fig_dir}/06_calibration.png'
        )
        plt.close()
    
    # =========================================================================
    # FIGURE 7: MC Samples at Selected Points
    # =========================================================================
    print("[7/8] Generating MC samples distribution plot...")
    
    # Select a few interesting points
    samples = predictions['samples'].detach().cpu().numpy()  # (n_samples, n_points)
    
    # Point at center
    center_idx = n * n // 2
    center_samples = samples[:, center_idx, 0]
    center_mean = mean.flatten()[center_idx]
    center_std = total.flatten()[center_idx]
    center_exact = exact_2d.flatten()[center_idx] if exact_2d is not None else None
    
    fig = Vis.plot_mc_samples(
        center_samples, center_mean, center_std, center_exact,
        point_label="center (0, 0)",
        save_path=f'{fig_dir}/07_mc_samples_center.png'
    )
    plt.close()
    
    # =========================================================================
    # FIGURE 8: 1D Slice with Confidence Intervals
    # =========================================================================
    print("[8/8] Generating confidence interval plot...")
    
    # Take a slice at x2 = 0 (or middle of domain)
    mid_idx = n // 2
    x_slice = x1.numpy()
    mean_slice = mean[:, mid_idx]
    std_slice = total[:, mid_idx]
    exact_slice = exact_2d[:, mid_idx] if exact_2d is not None else None
    samples_slice = samples[:, mid_idx::n, 0]  # Take every n-th point along x1
    
    fig = Vis.plot_confidence_intervals(
        x_slice, mean_slice, std_slice, exact_slice, samples_slice,
        save_path=f'{fig_dir}/08_confidence_intervals.png'
    )
    plt.close()
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    if exact_2d is not None:
        l2_error = np.sqrt(np.mean((mean - exact_2d)**2))
        max_error = np.max(np.abs(mean - exact_2d))
        print(f"L2 Error (Student): {l2_error:.6e}")
        print(f"Max Error (Student): {max_error:.6e}")
        
        l2_teacher = np.sqrt(np.mean((u_teacher - exact_2d)**2))
        print(f"L2 Error (Teacher): {l2_teacher:.6e}")
    
    print(f"Mean Epistemic Uncertainty: {epistemic.mean():.6e}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric.mean():.6e}")
    print(f"Mean Total Uncertainty: {total.mean():.6e}")
    
    # Compression statistics
    stats = trainer.structure_discovery.get_compression_stats(structure)
    print(f"\nCompression: {stats['total_original_params']} params -> "
          f"{stats['total_cluster_centers']} clusters "
          f"({stats['overall_compression']:.1f}x)")
    
    print(f"\nAll figures saved to: {fig_dir}")
    
    return trainer, predictions, structure


def compare_all_models():
    """
    Generate comparison figures across all trained models.
    """
    print("\n" + "="*70)
    print("GENERATING MODEL COMPARISON")
    print("="*70)
    
    results = {}
    
    # Define experiments
    experiments = [
        ('Laplace', 'EXP', 'mc_dropout', False),
        ('Burgers_inv', 'EXP', 'vi_bnn', True),
        ('Poisson', 'EXP', 'mc_dropout', False),
    ]
    
    for ques_name, ini_num, student_type, hetero in experiments:
        model_dir = f'./Results/{ques_name}_{ini_num}/Models/'
        if os.path.exists(model_dir):
            try:
                trainer, predictions, structure = generate_visualizations(
                    ques_name, ini_num, student_type, hetero
                )
                
                # Collect metrics
                n = int(np.sqrt(predictions['mean'].shape[0]))
                mean = predictions['mean'].detach().cpu().numpy().reshape(n, n)
                total = np.sqrt(predictions['variance'].detach().cpu().numpy()).reshape(n, n)
                
                x_test = torch.stack([trainer.x.flatten(), trainer.y.flatten()], dim=1)
                exact = get_exact_solution(ques_name, x_test)
                exact_2d = exact.reshape(n, n) if exact is not None else mean
                
                l2_error = np.sqrt(np.mean((mean - exact_2d)**2))
                
                results[f'{ques_name} ({student_type})'] = {
                    'L2_error': l2_error,
                    'mean_uncertainty': total.mean(),
                    'compression': trainer.structure_discovery.get_compression_stats(structure)['overall_compression']
                }
            except Exception as e:
                print(f"Warning: Could not process {ques_name}: {e}")
    
    # Generate comparison plot
    if results:
        os.makedirs('./Results/Comparison/', exist_ok=True)
        fig = Vis.plot_model_comparison(
            results,
            metrics=['L2_error', 'mean_uncertainty', 'compression'],
            save_path='./Results/Comparison/model_comparison.png'
        )
        plt.close()
        print("\nComparison figure saved to ./Results/Comparison/model_comparison.png")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate visualizations for GBSD runtime diagnostics')
    parser.add_argument('--problem', type=str, default='Laplace',
                        choices=['Laplace', 'Burgers_inv', 'Poisson'],
                        help='Problem name')
    parser.add_argument('--config', type=str, default='EXP',
                        help='Configuration number')
    parser.add_argument('--student', type=str, default='mc_dropout',
                        choices=['mc_dropout', 'vi_bnn'],
                        help='Student type')
    parser.add_argument('--heteroscedastic', action='store_true',
                        help='Whether heteroscedastic mode was used')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of MC samples for uncertainty')
    parser.add_argument('--all', action='store_true',
                        help='Generate visualizations for all trained models')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models()
    else:
        generate_visualizations(
            args.problem,
            args.config,
            args.student,
            args.heteroscedastic,
            args.n_samples
        )
