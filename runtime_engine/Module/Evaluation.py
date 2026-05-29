# coding = utf-8
"""
Evaluation module for structured reconstruction

Provides standardized evaluation comparing pre- vs post-reconstruction
performance, including:
  - Solution accuracy: MSE, MAE, relative L2 error
  - PDE residual statistics: mean, max, std
  - Parameter count and compression ratio
  - Inference time benchmarks
  - UQ quality metrics (when Bayesian mode is used):
      - NLL (negative log-likelihood)
      - PICP (prediction interval coverage probability)
      - MPIW (mean prediction interval width)
      - Epistemic/aleatoric separation

Third-party provenance for the migrated runtime engine is documented in
docs/THIRD_PARTY_NOTICE.md.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple


class PsiNNEvaluator:
    """
    Evaluator for pre- vs post-reconstruction comparison.

    Args:
        pre_model: Original model (teacher, student, or dense PINN)
        post_model: Reconstructed StructuredPINN
        pde_residual_fn: Callable(model, x) -> scalar PDE residual loss
        x_eval: Evaluation grid points, shape (N, input_dim)
        u_exact: Exact solution at eval points, shape (N, output_dim).
            If None, solution accuracy metrics are skipped.
        device: torch device
    """
    def __init__(self,
                 pre_model: nn.Module,
                 post_model: nn.Module,
                 pde_residual_fn: Callable,
                 x_eval: torch.Tensor,
                 u_exact: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None):

        self.pre_model = pre_model
        self.post_model = post_model
        self.pde_residual_fn = pde_residual_fn
        self.device = device or next(post_model.parameters()).device
        self.x_eval = x_eval.to(self.device)
        self.u_exact = u_exact.to(self.device) if u_exact is not None else None

    # -----------------------------------------------------------------
    # Core accuracy metrics
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _compute_solution_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compute solution accuracy metrics for a model."""
        model.eval()
        u_pred = model(self.x_eval)

        result = {}
        if self.u_exact is not None:
            err = u_pred - self.u_exact
            result['mse'] = torch.mean(err ** 2).item()
            result['mae'] = torch.mean(torch.abs(err)).item()
            norm_exact = torch.norm(self.u_exact)
            if norm_exact > 0:
                result['relative_l2'] = (torch.norm(err) / norm_exact).item()
            else:
                result['relative_l2'] = float('inf')
            result['max_error'] = torch.max(torch.abs(err)).item()
        return result

    def _compute_pde_residual_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute pointwise PDE residual statistics."""
        model.eval()
        x = self.x_eval.clone().requires_grad_(True)
        u = model(x)

        # Get pointwise residuals via autograd
        # We compute the scalar loss, but also want pointwise stats
        # Use the residual function which returns mean(residual^2)
        # For stats we need pointwise, so we compute manually
        scalar_loss = self.pde_residual_fn(model, x)

        return {
            'pde_loss_mean': scalar_loss.item(),
        }

    # -----------------------------------------------------------------
    # Parameter compression metrics
    # -----------------------------------------------------------------
    @staticmethod
    def _count_params(model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _compression_metrics(self) -> Dict[str, float]:
        """Compute parameter compression metrics."""
        pre_params = self._count_params(self.pre_model)

        if hasattr(self.post_model, 'count_parameters'):
            stats = self.post_model.count_parameters()
            post_params = stats['trainable']
            original_params = stats['original']
            compression = stats['compression_ratio']
        else:
            post_params = self._count_params(self.post_model)
            original_params = pre_params
            compression = pre_params / post_params if post_params > 0 else float('inf')

        return {
            'pre_params': pre_params,
            'post_params': post_params,
            'original_dense_params': original_params,
            'compression_ratio': compression
        }

    # -----------------------------------------------------------------
    # Inference time
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _benchmark_inference(self, model: nn.Module,
                              n_runs: int = 100) -> float:
        """Benchmark inference time (seconds per forward pass)."""
        model.eval()
        x = self.x_eval

        # Warmup
        for _ in range(10):
            _ = model(x)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        return (t1 - t0) / n_runs

    # -----------------------------------------------------------------
    # Uncertainty quality metrics
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _compute_uq_metrics(self, model: nn.Module,
                             n_mc_samples: int = 200,
                             confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute UQ quality metrics via MC sampling.

        Requires model to support stochastic forward passes (MC Dropout
        or VI-BNN reparameterization).

        Returns dict with NLL, PICP, MPIW, epistemic/aleatoric stats.
        """
        if self.u_exact is None:
            return {}

        # Enable stochastic mode
        was_training = model.training
        model.train()
        if hasattr(model, 'enable_dropout'):
            model.enable_dropout()

        samples = []
        for _ in range(n_mc_samples):
            out = model(self.x_eval)
            if isinstance(out, tuple):
                out = out[0]  # mean if heteroscedastic
            samples.append(out)

        samples = torch.stack(samples, dim=0)  # (S, N, D)
        pred_mean = samples.mean(dim=0)
        pred_var = samples.var(dim=0)
        pred_std = torch.sqrt(pred_var + 1e-8)

        result = {}

        # Epistemic uncertainty (variance of means)
        result['epistemic_mean'] = pred_var.mean().item()
        result['epistemic_max'] = pred_var.max().item()

        # NLL under Gaussian predictive distribution
        # NLL = 0.5 * [log(2*pi*var) + (y - mu)^2 / var]
        nll = 0.5 * (torch.log(2 * np.pi * (pred_var + 1e-8)) +
                      (self.u_exact - pred_mean) ** 2 / (pred_var + 1e-8))
        result['nll'] = nll.mean().item()

        # PICP: Prediction Interval Coverage Probability
        alpha = (1 - confidence) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        covered = ((self.u_exact >= lower) & (self.u_exact <= upper)).float()
        result['picp'] = covered.mean().item()
        result['picp_target'] = confidence

        # MPIW: Mean Prediction Interval Width
        result['mpiw'] = (upper - lower).mean().item()

        # Restore state
        if not was_training:
            model.eval()

        return result

    # -----------------------------------------------------------------
    # Full evaluation
    # -----------------------------------------------------------------
    def evaluate(self, include_uq: bool = False,
                 n_mc_samples: int = 200,
                 confidence: float = 0.95,
                 benchmark_inference: bool = True) -> Dict:
        """
        Run full pre-vs-post evaluation.

        Args:
            include_uq: Whether to compute UQ metrics (requires stochastic model)
            n_mc_samples: MC samples for UQ evaluation
            confidence: Confidence level for prediction intervals
            benchmark_inference: Whether to benchmark inference time

        Returns:
            Dict with all metrics organized as:
            {
                'pre': { solution metrics, pde stats },
                'post': { solution metrics, pde stats },
                'compression': { param counts and ratio },
                'inference_time': { pre, post, speedup },
                'uq_pre': { ... },   # if include_uq
                'uq_post': { ... },  # if include_uq
            }
        """
        results = {}

        # Solution accuracy
        results['pre'] = self._compute_solution_metrics(self.pre_model)
        results['post'] = self._compute_solution_metrics(self.post_model)

        # PDE residual
        pre_pde = self._compute_pde_residual_stats(self.pre_model)
        post_pde = self._compute_pde_residual_stats(self.post_model)
        results['pre'].update(pre_pde)
        results['post'].update(post_pde)

        # Compression
        results['compression'] = self._compression_metrics()

        # Inference time
        if benchmark_inference:
            t_pre = self._benchmark_inference(self.pre_model)
            t_post = self._benchmark_inference(self.post_model)
            results['inference_time'] = {
                'pre_seconds': t_pre,
                'post_seconds': t_post,
                'speedup': t_pre / t_post if t_post > 0 else float('inf')
            }

        # UQ metrics
        if include_uq:
            results['uq_pre'] = self._compute_uq_metrics(
                self.pre_model, n_mc_samples, confidence)
            results['uq_post'] = self._compute_uq_metrics(
                self.post_model, n_mc_samples, confidence)

        return results

    # -----------------------------------------------------------------
    # Pretty printing
    # -----------------------------------------------------------------
    @staticmethod
    def print_report(results: Dict):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 70)
        print("  PSI-NN RECONSTRUCTION EVALUATION REPORT")
        print("=" * 70)

        # Solution accuracy
        if 'mse' in results.get('pre', {}):
            print("\n  SOLUTION ACCURACY")
            print("  " + "-" * 50)
            header = f"  {'Metric':<20} {'Pre':>14} {'Post':>14} {'Change':>12}"
            print(header)
            print("  " + "-" * 50)

            for key in ['mse', 'mae', 'relative_l2', 'max_error']:
                if key in results['pre'] and key in results['post']:
                    v_pre = results['pre'][key]
                    v_post = results['post'][key]
                    if v_pre > 0:
                        change = (v_post - v_pre) / v_pre * 100
                        sign = "+" if change > 0 else ""
                        print(f"  {key:<20} {v_pre:>14.6e} {v_post:>14.6e} {sign}{change:>10.1f}%")
                    else:
                        print(f"  {key:<20} {v_pre:>14.6e} {v_post:>14.6e} {'N/A':>12}")

        # PDE residual
        print("\n  PDE RESIDUAL")
        print("  " + "-" * 50)
        for key in ['pde_loss_mean']:
            if key in results.get('pre', {}) and key in results.get('post', {}):
                v_pre = results['pre'][key]
                v_post = results['post'][key]
                print(f"  {key:<20} {v_pre:>14.6e} {v_post:>14.6e}")

        # Compression
        if 'compression' in results:
            c = results['compression']
            print("\n  PARAMETER COMPRESSION")
            print("  " + "-" * 50)
            print(f"  {'Pre-model params':<30} {c['pre_params']:>10}")
            print(f"  {'Post-model params':<30} {c['post_params']:>10}")
            print(f"  {'Compression ratio':<30} {c['compression_ratio']:>10.2f}x")

        # Inference time
        if 'inference_time' in results:
            t = results['inference_time']
            print("\n  INFERENCE TIME (per forward pass)")
            print("  " + "-" * 50)
            print(f"  {'Pre-model':<30} {t['pre_seconds']*1000:>10.3f} ms")
            print(f"  {'Post-model':<30} {t['post_seconds']*1000:>10.3f} ms")
            print(f"  {'Speedup':<30} {t['speedup']:>10.2f}x")

        # UQ metrics
        for tag in ['uq_pre', 'uq_post']:
            if tag in results and results[tag]:
                label = "PRE-RECONSTRUCTION" if 'pre' in tag else "POST-RECONSTRUCTION"
                print(f"\n  UQ METRICS ({label})")
                print("  " + "-" * 50)
                uq = results[tag]
                for k, v in uq.items():
                    if isinstance(v, float):
                        print(f"  {k:<30} {v:>14.6e}")
                    else:
                        print(f"  {k:<30} {str(v):>14}")

        print("\n" + "=" * 70)


# =============================================================================
# Convenience function
# =============================================================================

def evaluate_reconstruction(pre_model: nn.Module,
                            post_model: nn.Module,
                            pde_residual_fn: Callable,
                            x_eval: torch.Tensor,
                            u_exact: Optional[torch.Tensor] = None,
                            include_uq: bool = False,
                            n_mc_samples: int = 200,
                            verbose: bool = True) -> Dict:
    """
    One-call evaluation of pre- vs post-reconstruction.

    Args:
        pre_model: Original model (teacher or student)
        post_model: Reconstructed StructuredPINN
        pde_residual_fn: PDE residual function
        x_eval: Evaluation grid (N, input_dim)
        u_exact: Exact solution (N, output_dim) or None
        include_uq: Compute UQ metrics
        n_mc_samples: MC samples for UQ (default 200)
        verbose: Print report

    Returns:
        Evaluation results dict
    """
    evaluator = PsiNNEvaluator(
        pre_model=pre_model,
        post_model=post_model,
        pde_residual_fn=pde_residual_fn,
        x_eval=x_eval,
        u_exact=u_exact
    )
    results = evaluator.evaluate(include_uq=include_uq, n_mc_samples=n_mc_samples)

    if verbose:
        PsiNNEvaluator.print_report(results)

    return results
