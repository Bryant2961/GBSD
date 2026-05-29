# coding = utf-8
"""
Visualization module for GBSD runtime diagnostics

Comprehensive plotting utilities matching the paper figures:
- Fig. 1: Regularization conflict visualization
- Fig. 4: Numerical results (field predictions + errors)
- Fig. 5: Parameter evolution and clustering
- Fig. 6: Prediction comparison by iterations
- Fig. 7: Control parameter transfer
- Fig. 8: Flow field results

Plus additional Bayesian-specific visualizations:
- Uncertainty maps (epistemic vs aleatoric)
- Confidence intervals
- MC sample distributions
- Calibration curves
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
from typing import Dict, Optional, List, Tuple, Union


# Paper-quality figure settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
})


# =============================================================================
# PAPER FIGURE 1: Regularization Conflict
# =============================================================================

def plot_regularization_conflict(loss_histories: Dict[str, List[float]],
                                  save_path: Optional[str] = None,
                                  figsize: Tuple = (8, 6)):
    """
    Shows how regularization conflicts with PINN training (Paper Figure 1).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'PINN': '#2E86AB', 'PINN-L1': '#A23B72', 
              'PINN-L2': '#F18F01', 'PINN-GrOWL': '#C73E1D',
              'GBSD': '#4CAF50'}
    linestyles = {'PINN': '-', 'PINN-L1': '--', 'PINN-L2': '-.', 
                  'PINN-GrOWL': ':', 'GBSD': '-'}
    
    for name, losses in loss_histories.items():
        ax.semilogy(losses, label=name, color=colors.get(name, '#333'),
                   linestyle=linestyles.get(name, '-'), linewidth=2)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Effect of Regularization on PINN Training')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# PAPER FIGURE 4: Numerical Results
# =============================================================================

def plot_numerical_results(x: np.ndarray, y: np.ndarray,
                           u_exact: np.ndarray,
                           predictions: Dict[str, np.ndarray],
                           loss_histories: Dict[str, List[float]] = None,
                           problem_name: str = "PDE",
                           save_path: Optional[str] = None,
                           figsize: Tuple = (18, 8)):
    """
    Paper Figure 4 style: MSE propagation, exact solution, predictions, errors.
    """
    n_models = len(predictions)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, n_models + 2, width_ratios=[1.2, 1] + [1]*n_models,
                          hspace=0.3, wspace=0.25)
    
    # MSE propagation
    ax_loss = fig.add_subplot(gs[:, 0])
    if loss_histories:
        colors = ['#2E86AB', '#F18F01', '#4CAF50', '#E63946']
        for i, (name, losses) in enumerate(loss_histories.items()):
            ax_loss.semilogy(losses, label=name, color=colors[i % len(colors)], linewidth=2)
        ax_loss.set_xlabel('Iterations')
        ax_loss.set_ylabel('MSE')
        ax_loss.set_title('MSE propagation')
        ax_loss.legend(fontsize=9)
        ax_loss.grid(True, alpha=0.3)
    
    # Exact solution
    ax_exact = fig.add_subplot(gs[0, 1])
    im = ax_exact.contourf(x, y, u_exact, levels=50, cmap='RdBu_r')
    ax_exact.set_title('Exact')
    ax_exact.set_xlabel('$x_1$'); ax_exact.set_ylabel('$x_2$')
    plt.colorbar(im, ax=ax_exact, fraction=0.046)
    
    # Model predictions and errors
    for i, (name, pred) in enumerate(predictions.items()):
        ax_pred = fig.add_subplot(gs[0, i + 2])
        im = ax_pred.contourf(x, y, pred, levels=50, cmap='RdBu_r')
        ax_pred.set_title(name)
        ax_pred.set_xlabel('$x_1$')
        plt.colorbar(im, ax=ax_pred, fraction=0.046)
        
        ax_err = fig.add_subplot(gs[1, i + 2])
        err = np.abs(pred - u_exact)
        im_err = ax_err.contourf(x, y, err, levels=50, cmap='hot_r')
        ax_err.set_title(f'{name} Error')
        ax_err.set_xlabel('$x_1$')
        plt.colorbar(im_err, ax=ax_err, fraction=0.046)
    
    plt.suptitle(f'{problem_name} - Numerical Results', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# PAPER FIGURE 5: Parameter Evolution and Clustering
# =============================================================================

def plot_parameter_evolution(param_history: np.ndarray,
                             loss_history: np.ndarray,
                             save_path: Optional[str] = None,
                             figsize: Tuple = (12, 5)):
    """
    Paper Figure 5a style: Parameter evolution during training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_iters, n_params = param_history.shape
    cmap = plt.cm.viridis
    
    for i in range(min(n_params, 50)):
        ax1.plot(param_history[:, i], color=cmap(i/n_params), alpha=0.5, linewidth=0.8)
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Parameter values')
    ax1.set_title('Parameter Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Loss curve
    ax2.semilogy(loss_history, 'b-', linewidth=2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    # Inset: final distribution
    ax_inset = ax1.inset_axes([0.65, 0.6, 0.3, 0.35])
    ax_inset.hist(param_history[-1, :], bins=30, color='steelblue', edgecolor='white')
    ax_inset.set_title('Final Dist.', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_weight_clusters(structure: Dict,
                         save_path: Optional[str] = None,
                         figsize: Tuple = (14, 10)):
    """
    Paper Figure 5b style: Weight clustering visualization.
    """
    n_layers = len(structure)
    fig, axes = plt.subplots(n_layers, 2, figsize=figsize)
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for row, (layer_name, info) in enumerate(structure.items()):
        centers = info['cluster_centers']
        labels = info['labels']
        signs = info['signs']
        
        weight_values = centers[labels] * signs
        
        # Scatter plot
        ax = axes[row, 0]
        pos_mask = signs > 0
        neg_mask = signs < 0
        
        ax.scatter(np.where(pos_mask)[0], np.abs(weight_values[pos_mask]), 
                  c='#2E86AB', label='Positive', alpha=0.6, s=30)
        ax.scatter(np.where(neg_mask)[0], np.abs(weight_values[neg_mask]), 
                  c='#E63946', label='Negative', alpha=0.6, s=30)
        
        for center in centers:
            ax.axhline(y=center, color='#4CAF50', linestyle='--', alpha=0.7)
        
        ax.set_ylabel('|Weight|')
        ax.set_title(f'{layer_name} ({len(centers)} clusters)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Histogram
        ax_hist = axes[row, 1]
        ax_hist.hist(np.abs(weight_values), bins=30, color='steelblue', 
                    edgecolor='white', alpha=0.7, orientation='horizontal')
        for center in centers:
            ax_hist.axhline(y=center, color='#4CAF50', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Count')
        ax_hist.set_title('Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# PAPER FIGURE 6: Prediction Evolution
# =============================================================================

def plot_prediction_by_iterations(x_slice: np.ndarray,
                                   predictions_by_iter: Dict[int, np.ndarray],
                                   exact: np.ndarray,
                                   model_name: str = "Model",
                                   save_path: Optional[str] = None,
                                   figsize: Tuple = (10, 6)):
    """
    Paper Figure 6 style: Prediction evolution across iterations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    cmap = plt.cm.viridis
    iterations = sorted(predictions_by_iter.keys())
    n_iters = len(iterations)
    
    # Line plot
    ax1.plot(x_slice, exact, 'r--', linewidth=2, label='Exact')
    for i, it in enumerate(iterations):
        ax1.plot(x_slice, predictions_by_iter[it], color=cmap(i/n_iters), 
                alpha=0.7, label=f'Iter={it}' if i % 3 == 0 else '')
    
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$u$')
    ax1.set_title(f'{model_name} Evolution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2D heatmap
    pred_matrix = np.array([predictions_by_iter[it] for it in iterations])
    im = ax2.imshow(pred_matrix.T, aspect='auto', cmap='RdBu_r',
                   extent=[iterations[0], iterations[-1], x_slice.min(), x_slice.max()],
                   origin='lower')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('$x$')
    ax2.set_title('Temporal Evolution')
    plt.colorbar(im, ax=ax2, label='$u$')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# PAPER FIGURE 7: Parameter Transfer
# =============================================================================

def plot_parameter_trajectories(param_trajectories: Dict[str, np.ndarray],
                                 true_value: float,
                                 param_name: str = "λ",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple = (8, 5)):
    """
    Paper Figure 7 style: Parameter estimation trajectories.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'PINN': '#2E86AB', 'PINN-post': '#F18F01', 
              'Structured Candidate': '#4CAF50', 'GBSD': '#9C27B0'}
    
    for model_name, trajectory in param_trajectories.items():
        ax.semilogx(np.arange(1, len(trajectory)+1), trajectory, 
                   label=model_name, color=colors.get(model_name, '#333'), linewidth=2)
    
    ax.axhline(y=true_value, color='red', linestyle='--', linewidth=2, label='True value')
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel(param_name)
    ax.set_title(f'Parameter Estimation: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# BAYESIAN-SPECIFIC: Uncertainty Decomposition
# =============================================================================

def plot_uncertainty_decomposition(x: np.ndarray, y: np.ndarray,
                                    mean: np.ndarray,
                                    epistemic: np.ndarray,
                                    aleatoric: np.ndarray,
                                    total: np.ndarray,
                                    exact: np.ndarray = None,
                                    save_path: Optional[str] = None,
                                    figsize: Tuple = (16, 10)):
    """
    Comprehensive uncertainty decomposition visualization.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(x, y, mean, levels=50, cmap='RdBu_r')
    ax1.set_title('Predictive Mean')
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    if exact is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(x, y, exact, levels=50, cmap='RdBu_r')
        ax2.set_title('Reference Solution')
        ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        error = np.abs(mean - exact)
        im3 = ax3.contourf(x, y, error, levels=50, cmap='hot_r')
        ax3.set_title('Absolute Error')
        ax3.set_xlabel('$x_1$'); ax3.set_ylabel('$x_2$')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[0, 3])
        norm_error = error / (total + 1e-10)
        im4 = ax4.contourf(x, y, norm_error, levels=50, cmap='RdYlGn_r')
        ax4.set_title('Error / Uncertainty')
        ax4.set_xlabel('$x_1$'); ax4.set_ylabel('$x_2$')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Uncertainties
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.contourf(x, y, total, levels=50, cmap='YlOrRd')
    ax5.set_title('Total Uncertainty (σ)')
    ax5.set_xlabel('$x_1$'); ax5.set_ylabel('$x_2$')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.contourf(x, y, epistemic, levels=50, cmap='YlOrRd')
    ax6.set_title('Epistemic Uncertainty')
    ax6.set_xlabel('$x_1$'); ax6.set_ylabel('$x_2$')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.contourf(x, y, aleatoric, levels=50, cmap='YlOrRd')
    ax7.set_title('Aleatoric Uncertainty')
    ax7.set_xlabel('$x_1$'); ax7.set_ylabel('$x_2$')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[1, 3])
    ratio = epistemic / (total + 1e-10)
    im8 = ax8.contourf(x, y, ratio, levels=50, cmap='coolwarm', vmin=0, vmax=1)
    ax8.set_title('Epistemic / Total Ratio')
    ax8.set_xlabel('$x_1$'); ax8.set_ylabel('$x_2$')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    plt.suptitle('Uncertainty Decomposition Analysis', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# BAYESIAN-SPECIFIC: MC Samples Distribution
# =============================================================================

def plot_mc_samples(samples: np.ndarray,
                    mean: float,
                    std: float,
                    exact: float = None,
                    point_label: str = "x",
                    save_path: Optional[str] = None,
                    figsize: Tuple = (10, 5)):
    """
    Visualize MC sample distribution at a specific point.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with Gaussian fit
    ax1.hist(samples, bins=30, density=True, color='steelblue', 
            edgecolor='white', alpha=0.7, label='MC samples')
    
    x_range = np.linspace(samples.min(), samples.max(), 100)
    gaussian = np.exp(-0.5*((x_range-mean)/std)**2) / (std*np.sqrt(2*np.pi))
    ax1.plot(x_range, gaussian, 'r-', linewidth=2, 
            label=f'Gaussian\nμ={mean:.4f}\nσ={std:.4f}')
    
    ax1.axvline(x=mean, color='red', linestyle='--', linewidth=1.5)
    ax1.axvspan(mean-2*std, mean+2*std, alpha=0.2, color='red', label='95% CI')
    
    if exact is not None:
        ax1.axvline(x=exact, color='green', linestyle='-', linewidth=2, 
                   label=f'Exact={exact:.4f}')
    
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Density')
    ax1.set_title(f'MC Distribution at {point_label}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([samples], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    if exact is not None:
        ax2.axhline(y=exact, color='green', linestyle='-', linewidth=2, label='Exact')
    ax2.axhline(y=mean, color='red', linestyle='--', linewidth=1.5, label='Mean')
    ax2.set_ylabel('Prediction')
    ax2.set_title('Box Plot')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# BAYESIAN-SPECIFIC: Confidence Intervals
# =============================================================================

def plot_confidence_intervals(x: np.ndarray,
                               mean: np.ndarray,
                               std: np.ndarray,
                               exact: np.ndarray = None,
                               samples: np.ndarray = None,
                               save_path: Optional[str] = None,
                               figsize: Tuple = (12, 6)):
    """
    Plot 1D predictions with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 95% CI
    ax.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.3, 
                    color='steelblue', label='95% CI')
    # 68% CI
    ax.fill_between(x, mean - std, mean + std, alpha=0.5, 
                    color='steelblue', label='68% CI')
    
    # Plot samples
    if samples is not None:
        for i in range(min(samples.shape[0], 20)):
            ax.plot(x, samples[i], 'b-', alpha=0.1, linewidth=0.5)
    
    ax.plot(x, mean, 'b-', linewidth=2, label='Mean')
    
    if exact is not None:
        ax.plot(x, exact, 'r--', linewidth=2, label='Exact')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    ax.set_title('Predictions with Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# BAYESIAN-SPECIFIC: Calibration Curve
# =============================================================================

def plot_calibration(predicted_std: np.ndarray,
                     actual_errors: np.ndarray,
                     n_bins: int = 10,
                     save_path: Optional[str] = None,
                     figsize: Tuple = (10, 5)):
    """
    Plot uncertainty calibration curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bin by predicted uncertainty
    bin_edges = np.percentile(predicted_std, np.linspace(0, 100, n_bins + 1))
    bin_centers, rmse_per_bin, coverage = [], [], []
    
    for i in range(n_bins):
        mask = (predicted_std >= bin_edges[i]) & (predicted_std < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(predicted_std[mask].mean())
            rmse_per_bin.append(np.sqrt((actual_errors[mask]**2).mean()))
            within_2sigma = actual_errors[mask] <= 2 * predicted_std[mask]
            coverage.append(within_2sigma.mean())
    
    bin_centers = np.array(bin_centers)
    rmse_per_bin = np.array(rmse_per_bin)
    coverage = np.array(coverage)
    
    # Calibration plot
    ax1.scatter(bin_centers, rmse_per_bin, s=100, c='steelblue', edgecolors='white')
    max_val = max(bin_centers.max(), rmse_per_bin.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('Predicted Std')
    ax1.set_ylabel('Actual RMSE')
    ax1.set_title('Calibration Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Coverage plot
    ax2.bar(range(len(coverage)), coverage, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Expected 95%')
    ax2.set_xlabel('Uncertainty Bin')
    ax2.set_ylabel('Coverage')
    ax2.set_title('Coverage Analysis')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def plot_model_comparison(results: Dict[str, Dict],
                          metrics: List[str] = ['L2_error', 'mean_uncertainty', 'time'],
                          save_path: Optional[str] = None,
                          figsize: Tuple = (12, 4)):
    """
    Bar chart comparing models across metrics.
    """
    models = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor='white')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# NETWORK STRUCTURE VISUALIZATION
# =============================================================================

def plot_network_structure(structure: Dict,
                           relation_matrices: Dict = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple = (14, 8)):
    """
    Visualize discovered network structure with relation matrices.
    """
    n_layers = len(structure)
    n_rows = 2 if relation_matrices else 1
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_layers, hspace=0.4)
    
    for i, (layer_name, info) in enumerate(structure.items()):
        ax1 = fig.add_subplot(gs[0, i])
        
        centers = info['cluster_centers']
        labels = info['labels']
        signs = info['signs']
        shape = info['original_shape']
        
        weights = (centers[labels] * signs).reshape(shape)
        
        im1 = ax1.imshow(weights, cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'{layer_name}\n{shape}')
        ax1.set_xlabel('Input')
        ax1.set_ylabel('Output')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        if relation_matrices and layer_name in relation_matrices:
            ax2 = fig.add_subplot(gs[1, i])
            R = relation_matrices[layer_name]
            if isinstance(R, torch.Tensor):
                R = R.numpy()
            
            im2 = ax2.imshow(R, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax2.set_title(f'R: {R.shape}')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Param')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    plt.suptitle('Discovered Network Structure', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# TRAINING HISTORY
# =============================================================================

def plot_training_history(trainer,
                          save_path: Optional[str] = None,
                          figsize: Tuple = (14, 5)):
    """
    Plot training history for teacher and student.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Teacher
    if hasattr(trainer, 'teacher') and len(trainer.teacher.iter_list) > 0:
        ax = axes[0]
        ax.semilogy(trainer.teacher.iter_list, trainer.teacher.loss_list, 'b-', lw=2, label='Total')
        if trainer.teacher.loss_f_list:
            ax.semilogy(trainer.teacher.iter_list, trainer.teacher.loss_f_list, 'r--', lw=1.5, label='PDE')
        if trainer.teacher.loss_b_list:
            ax.semilogy(trainer.teacher.iter_list, trainer.teacher.loss_b_list, 'g--', lw=1.5, label='BC')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Teacher (PINN)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Student
    if hasattr(trainer, 'student') and len(trainer.student.iter_list) > 0:
        ax = axes[1]
        ax.semilogy(trainer.student.iter_list, trainer.student.loss_list, 'b-', lw=2, label='Total')
        if trainer.student.loss_teach_list:
            ax.semilogy(trainer.student.iter_list, trainer.student.loss_teach_list, 'r--', lw=1.5, label='Distill')
        if trainer.student.loss_rgl_list:
            ax.semilogy(trainer.student.iter_list, trainer.student.loss_rgl_list, 'g--', lw=1.5, label='Reg/KL')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'Student ({getattr(trainer, "student_type", "BNN")})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Combined
    ax = axes[2]
    if hasattr(trainer, 'teacher') and len(trainer.teacher.iter_list) > 0:
        ax.semilogy(trainer.teacher.iter_list, trainer.teacher.loss_list, 'b-', lw=2, label='Teacher')
    if hasattr(trainer, 'student') and len(trainer.student.iter_list) > 0:
        offset = trainer.teacher.iter_list[-1] if trainer.teacher.iter_list else 0
        student_iters = [i + offset for i in trainer.student.iter_list]
        ax.semilogy(student_iters, trainer.student.loss_list, 'r-', lw=2, label='Student')
        ax.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Combined')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(trainer, x_test, predictions, structure,
                    exact=None, save_dir='./figures'):
    """
    Generate complete set of figures for paper.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    x_np = x_test.detach().cpu().numpy()
    n = int(np.sqrt(len(x_np)))
    x1 = x_np[:, 0].reshape(n, n)
    x2 = x_np[:, 1].reshape(n, n)
    
    mean = predictions['mean'].detach().cpu().numpy().reshape(n, n)
    epistemic = np.sqrt(predictions['epistemic'].detach().cpu().numpy()).reshape(n, n)
    aleatoric = np.sqrt(predictions['aleatoric'].detach().cpu().numpy()).reshape(n, n)
    total = np.sqrt(predictions['variance'].detach().cpu().numpy()).reshape(n, n)
    
    if exact is not None:
        exact_2d = exact.reshape(n, n) if exact.ndim == 1 else exact
    else:
        exact_2d = None
    
    print("Generating figures...")
    
    plot_training_history(trainer, f'{save_dir}/01_training_history.png')
    plt.close()
    
    plot_uncertainty_decomposition(x1, x2, mean, epistemic, aleatoric, total, exact_2d,
                                   f'{save_dir}/02_uncertainty.png')
    plt.close()
    
    plot_weight_clusters(structure, f'{save_dir}/03_clusters.png')
    plt.close()
    
    R = trainer.get_relation_matrices(structure)
    plot_network_structure(structure, R, f'{save_dir}/04_structure.png')
    plt.close()
    
    if exact_2d is not None:
        errors = np.abs(mean - exact_2d).flatten()
        stds = total.flatten()
        plot_calibration(stds, errors, save_path=f'{save_dir}/05_calibration.png')
        plt.close()
    
    print(f"Figures saved to {save_dir}/")
    return save_dir
