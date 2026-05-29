# coding = utf-8
"""
Uncertainty Estimation Module

Provides Monte Carlo sampling for predictive uncertainty estimation
from Bayesian neural networks (MC Dropout or VI-BNN).

FIXES in v1.2:
1. Fixed device handling for output tensors
2. Fixed aleatoric uncertainty handling when heteroscedastic=False
3. Added proper model state restoration
"""
import torch
import torch.nn as nn
from typing import Dict


class UncertaintyEstimator:
    """
    Estimates predictive uncertainty using Monte Carlo sampling.
    
    Supports both MC Dropout and VI-BNN models.
    
    Args:
        model: Bayesian neural network (MCDropout or VIBNN)
        n_samples: Number of Monte Carlo samples (default 100)
    """
    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        self.heteroscedastic = getattr(model, 'heteroscedastic', False)
        
        # Check model type
        model_name = model.__class__.__name__
        self.is_mc_dropout = 'MCDropout' in str(type(model).__module__) or hasattr(model, 'enable_dropout')
        self.is_vi_bnn = 'VIBNN' in str(type(model).__module__) or hasattr(model, 'get_kl_divergence')
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute predictive statistics using MC sampling.
        
        Returns both:
        - 'mean': Deterministic forward pass (dropout OFF) — accurate point estimate
        - 'mc_mean': Average of MC samples (dropout ON) — smoother, lower amplitude
        - 'std': MC predictive standard deviation — uncertainty estimate
        """
        was_training = self.model.training
        device = x.device
        
        # 1. Deterministic prediction (dropout OFF) — accurate point estimate
        self.model.eval()
        det_output = self.model(x)
        if self.heteroscedastic:
            det_mean = det_output[0]
        else:
            det_mean = det_output
        
        # 2. MC sampling (dropout ON) — for uncertainty estimation
        self.model.train()
        if self.is_mc_dropout and hasattr(self.model, 'enable_dropout'):
            self.model.enable_dropout()
        
        samples = []
        aleatoric_samples = []
        
        for _ in range(self.n_samples):
            if self.heteroscedastic:
                mean, log_var = self.model(x)
                samples.append(mean)
                aleatoric_samples.append(torch.exp(log_var))
            else:
                out = self.model(x)
                samples.append(out)
        
        samples = torch.stack(samples, dim=0)
        mc_mean = samples.mean(dim=0)
        epistemic_var = samples.var(dim=0)
        
        results = {
            'mean': det_mean,       # Deterministic (accurate, no amplitude compression)
            'mc_mean': mc_mean,     # MC average (for reference)
            'epistemic': epistemic_var,
            'samples': samples
        }
        
        if self.heteroscedastic and len(aleatoric_samples) > 0:
            aleatoric_samples = torch.stack(aleatoric_samples, dim=0)
            aleatoric_var = aleatoric_samples.mean(dim=0)
            results['aleatoric'] = aleatoric_var
            results['variance'] = epistemic_var + aleatoric_var
        else:
            results['variance'] = epistemic_var
            results['aleatoric'] = torch.zeros_like(epistemic_var)
        
        results['std'] = torch.sqrt(results['variance'])
        
        if not was_training:
            self.model.eval()
            
        return results
    
    def get_confidence_interval(self, x: torch.Tensor, 
                                 confidence: float = 0.95) -> Dict[str, torch.Tensor]:
        """
        Compute confidence intervals for predictions.
        
        Args:
            x: Input tensor
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary with 'mean', 'lower', 'upper' bounds
        """
        predictions = self.predict(x)
        samples = predictions['samples']
        
        # Compute percentiles
        alpha = (1 - confidence) / 2
        
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        
        return {
            'mean': predictions['mean'],
            'lower': lower,
            'upper': upper,
            'confidence': confidence
        }


def estimate_uncertainty(model: nn.Module, x: torch.Tensor, 
                         n_samples: int = 100) -> Dict[str, torch.Tensor]:
    """
    Convenience function for uncertainty estimation.
    
    Args:
        model: Bayesian neural network
        x: Input tensor
        n_samples: Number of MC samples
        
    Returns:
        Dictionary with uncertainty estimates
    """
    estimator = UncertaintyEstimator(model, n_samples)
    return estimator.predict(x)
