# coding = utf-8
"""
Optimized training module for experimental GBSD runtime paths

Advanced training techniques:
1. Learning rate scheduling (warmup + cosine annealing)
2. Adaptive loss weighting (NTK-based or gradient-based)
3. Causal training for time-dependent PDEs
4. Residual-based adaptive refinement (RAR)
5. L-BFGS fine-tuning
6. Gradient clipping for stability
7. Mixed precision training (optional)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import pandas as pd
import os
import time
from typing import Callable, Dict, Optional, Tuple, List

# Import modules
import Module.Optim.OptimizedPINN as OptPINN
import Module.PINN as PINN
import Module.Bayesian.Student_MCDropout as MCDropout
import Module.Bayesian.Student_VIBNN as VIBNN
import Module.UQ.UncertaintyEstimation as UE
import Module.StructureDiscovery as SD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int,
                                 min_lr_ratio: float = 0.01):
    """
    Warmup + Cosine annealing scheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Adaptive Loss Weighting
# =============================================================================

class AdaptiveLossWeighting:
    """
    Adaptive loss weighting based on gradient statistics.
    
    Methods:
    - 'gradnorm': GradNorm algorithm
    - 'uncertainty': Uncertainty weighting (Kendall et al.)
    - 'ntk': Neural Tangent Kernel-based weighting
    - 'rms': RMS balancing
    """
    def __init__(self, num_losses: int, method: str = 'rms', 
                 alpha: float = 0.9, device: torch.device = device):
        self.num_losses = num_losses
        self.method = method
        self.alpha = alpha  # EMA coefficient
        self.device = device
        
        # Initialize weights
        self.weights = torch.ones(num_losses, device=device)
        
        # Running statistics for RMS balancing
        self.running_loss = torch.ones(num_losses, device=device)
        
        # Learnable parameters for uncertainty weighting
        if method == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(num_losses, device=device))
    
    def update(self, losses: List[torch.Tensor], model: nn.Module = None) -> torch.Tensor:
        """Update weights and return weighted sum of losses."""
        losses_tensor = torch.stack([l.detach() for l in losses])
        
        if self.method == 'rms':
            # RMS balancing: weight inversely proportional to loss magnitude
            self.running_loss = self.alpha * self.running_loss + (1 - self.alpha) * losses_tensor
            self.weights = 1.0 / (self.running_loss + 1e-8)
            self.weights = self.weights / self.weights.sum() * self.num_losses
        
        elif self.method == 'uncertainty':
            # Uncertainty weighting: L = sum(exp(-s_i) * L_i + s_i)
            precisions = torch.exp(-self.log_vars)
            weighted_losses = precisions * losses_tensor + self.log_vars
            return weighted_losses.sum()
        
        elif self.method == 'max':
            # Simple max-normalization
            max_loss = losses_tensor.max()
            self.weights = max_loss / (losses_tensor + 1e-8)
        
        # Compute weighted loss
        weighted_loss = sum(w * l for w, l in zip(self.weights, losses))
        return weighted_loss
    
    def get_weights(self) -> torch.Tensor:
        return self.weights.detach()


# =============================================================================
# Causal Training for Time-Dependent PDEs
# =============================================================================

class CausalTraining:
    """
    Causal training that respects temporal causality.
    Early time steps are weighted more heavily.
    
    Reference: Wang et al., "Respecting Causality is All You Need for 
               Training Physics-informed Neural Networks"
    """
    def __init__(self, t_min: float, t_max: float, n_chunks: int = 10,
                 epsilon: float = 0.1, device: torch.device = device):
        self.t_min = t_min
        self.t_max = t_max
        self.n_chunks = n_chunks
        self.epsilon = epsilon
        self.device = device
        
        # Time boundaries
        self.t_boundaries = torch.linspace(t_min, t_max, n_chunks + 1, device=device)
        
        # Causal weights (start with uniform, then update)
        self.weights = torch.ones(n_chunks, device=device)
    
    def get_causal_weights(self, t: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        Compute causal weights for each point based on cumulative residuals.
        Points at early times should have lower accumulated error.
        """
        weights = torch.ones_like(t).squeeze()
        
        # Compute cumulative residual for each time chunk
        for i in range(self.n_chunks):
            t_low = self.t_boundaries[i]
            t_high = self.t_boundaries[i + 1]
            
            mask = (t >= t_low) & (t < t_high)
            
            if i > 0:
                # Weight based on cumulative residual from previous chunks
                prev_mask = t < t_low
                if prev_mask.any():
                    cumulative_residual = residuals[prev_mask.squeeze()].mean()
                    self.weights[i] = torch.exp(-self.epsilon * cumulative_residual)
            
            weights[mask.squeeze()] = self.weights[i]
        
        return weights
    
    def weighted_loss(self, t: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """Compute causally-weighted loss."""
        weights = self.get_causal_weights(t, residuals.detach())
        return (weights * residuals).mean()


# =============================================================================
# Residual-based Adaptive Refinement
# =============================================================================

class ResidualAdaptiveRefinement:
    """
    Add collocation points where PDE residual is high.
    
    Reference: Lu et al., "DeepXDE: A Deep Learning Library for Solving 
               Differential Equations"
    """
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]],
                 initial_points: int = 1000,
                 refine_points: int = 100,
                 refine_threshold: float = 0.9,  # Top 10% residuals
                 device: torch.device = device):
        self.domain_bounds = domain_bounds
        self.initial_points = initial_points
        self.refine_points = refine_points
        self.refine_threshold = refine_threshold
        self.device = device
        
        # Initialize points
        self.points = self._sample_points(initial_points)
    
    def _sample_points(self, n: int) -> torch.Tensor:
        """Sample points uniformly in domain."""
        points = []
        for key in sorted(self.domain_bounds.keys()):
            low, high = self.domain_bounds[key]
            points.append(torch.rand(n, 1, device=self.device) * (high - low) + low)
        return torch.cat(points, dim=1)
    
    def refine(self, model: nn.Module, pde_residual_fn: Callable) -> None:
        """Add points where residual is high."""
        model.eval()
        
        with torch.no_grad():
            # Compute residuals at current points
            residuals = pde_residual_fn(model, self.points)
            
            if isinstance(residuals, torch.Tensor) and residuals.numel() > 1:
                # Find high-residual regions
                threshold = torch.quantile(residuals.abs(), self.refine_threshold)
                high_residual_mask = residuals.abs() > threshold
                
                if high_residual_mask.any():
                    # Sample new points near high-residual points
                    high_residual_points = self.points[high_residual_mask.squeeze()]
                    
                    # Add noise to create nearby points
                    noise_scale = 0.1 * torch.tensor([
                        self.domain_bounds[k][1] - self.domain_bounds[k][0] 
                        for k in sorted(self.domain_bounds.keys())
                    ], device=self.device)
                    
                    n_new = min(self.refine_points, len(high_residual_points))
                    idx = torch.randperm(len(high_residual_points))[:n_new]
                    new_points = high_residual_points[idx] + torch.randn(n_new, 2, device=self.device) * noise_scale
                    
                    # Clip to domain
                    for i, key in enumerate(sorted(self.domain_bounds.keys())):
                        low, high = self.domain_bounds[key]
                        new_points[:, i] = new_points[:, i].clamp(low, high)
                    
                    self.points = torch.cat([self.points, new_points], dim=0)
        
        model.train()
    
    def get_points(self) -> torch.Tensor:
        return self.points


# =============================================================================
# Optimized Trainer
# =============================================================================

class OptimizedBayesianPsiNNTrainer:
    """
    Optimized trainer with advanced PINN techniques.
    """
    def __init__(self,
                 ques_name: str = 'Laplace',
                 ini_num: str = 'EXP',
                 student_type: str = 'mc_dropout',
                 heteroscedastic: bool = False,
                 # Network architecture
                 hidden_dims: List[int] = [64, 64, 64],
                 use_fourier: bool = True,
                 fourier_dim: int = 64,
                 fourier_scale: float = 2.0,
                 activation: str = 'tanh',
                 # Training parameters
                 train_steps: int = 15000,
                 student_steps: int = 10000,
                 learning_rate: float = 1e-3,
                 warmup_steps: int = 1000,
                 use_lbfgs: bool = True,
                 lbfgs_steps: int = 500,
                 # Advanced techniques
                 use_adaptive_weights: bool = True,
                 adaptive_method: str = 'rms',
                 use_causal: bool = True,
                 use_rar: bool = True,
                 rar_interval: int = 2000,
                 grad_clip: float = 1.0,
                 # Bayesian parameters
                 dropout_rate: float = 0.1,
                 prior_sigma: float = 1.0,
                 kl_weight: float = 1e-5,
                 l2_weight: float = 1e-4,
                 # Other
                 save_dir: str = './Results'):
        
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.student_type = student_type
        self.heteroscedastic = heteroscedastic
        
        # Network settings
        self.hidden_dims = hidden_dims
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_scale = fourier_scale
        self.activation = activation
        
        # Training settings
        self.train_steps = train_steps
        self.student_steps = student_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.use_lbfgs = use_lbfgs
        self.lbfgs_steps = lbfgs_steps
        
        # Advanced techniques
        self.use_adaptive_weights = use_adaptive_weights
        self.adaptive_method = adaptive_method
        self.use_causal = use_causal
        self.use_rar = use_rar
        self.rar_interval = rar_interval
        self.grad_clip = grad_clip
        
        # Bayesian settings
        self.dropout_rate = dropout_rate
        self.prior_sigma = prior_sigma
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        
        # Paths
        self.save_dir = f'{save_dir}/{ques_name}_{ini_num}'
        os.makedirs(f'{self.save_dir}/Models', exist_ok=True)
        os.makedirs(f'{self.save_dir}/Loss', exist_ok=True)
        os.makedirs(f'{self.save_dir}/Figures', exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Build networks
        self._build_networks()
        
        # Initialize helper classes
        self._init_helpers()
    
    def _load_config(self):
        """Load problem configuration from key-value CSV format."""
        config_path = f'./Config/{self.ques_name}_{self.ini_num}.csv'
        if os.path.exists(config_path):
            config = pd.read_csv(config_path)
            # Convert key-value CSV to dictionary
            config_dict = dict(zip(config['key'], config['value']))
            
            self.x_min = float(config_dict.get('x_min', -1))
            self.x_max = float(config_dict.get('x_max', 1))
            self.y_min = float(config_dict.get('y_min', -1))
            self.y_max = float(config_dict.get('y_max', 1))
            self.grid_num = int(float(config_dict.get('grid_node_num', 50)))
            self.bun_num = int(float(config_dict.get('bun_node_num', 50)))
            self.node_num = int(float(config_dict.get('node_num', 32)))
        else:
            # Defaults
            self.x_min, self.x_max = -1, 1
            self.y_min, self.y_max = 0 if 'Burgers' in self.ques_name else -1, 1
            self.grid_num = 50
            self.bun_num = 50
            self.node_num = 32
    
    def _build_networks(self):
        """Build teacher and student networks."""
        output_dim = 2 if self.heteroscedastic else 1
        
        # Teacher: Optimized PINN
        self.teacher = OptPINN.OptimizedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_fourier=self.use_fourier,
            fourier_dim=self.fourier_dim,
            fourier_scale=self.fourier_scale
        ).to(device)
        
        # Build layers list for student: [input_dim, hidden1, hidden2, hidden3, output_dim]
        student_layers = [2] + [self.node_num] * 3 + [output_dim]
        
        # Student: Bayesian network
        if self.student_type == 'mc_dropout':
            self.student = MCDropout.Net(
                layers=student_layers,
                dropout_rate=self.dropout_rate,
                heteroscedastic=self.heteroscedastic
            ).to(device)
        else:  # vi_bnn
            self.student = VIBNN.Net(
                layers=student_layers,
                prior_sigma=self.prior_sigma,
                heteroscedastic=self.heteroscedastic
            ).to(device)
        
        # Uncertainty estimator (takes model and optional n_samples)
        self.uncertainty_estimator = UE.UncertaintyEstimator(self.student)
        
        # Structure discovery (takes model and optional cluster_distance)
        self.structure_discovery = SD.StructureDiscovery(self.student)
    
    def _init_helpers(self):
        """Initialize helper classes for advanced training."""
        # Adaptive loss weighting
        if self.use_adaptive_weights:
            self.loss_weighter = AdaptiveLossWeighting(3, self.adaptive_method)
        
        # Causal training (for time-dependent PDEs)
        if self.use_causal and 'Burgers' in self.ques_name:
            self.causal_trainer = CausalTraining(self.y_min, self.y_max, n_chunks=10)
        else:
            self.causal_trainer = None
        
        # Residual adaptive refinement
        if self.use_rar:
            self.rar = ResidualAdaptiveRefinement(
                {'x': (self.x_min, self.x_max), 't': (self.y_min, self.y_max)},
                initial_points=self.grid_num ** 2
            )
    
    def mesh_init(self):
        """Initialize mesh grid."""
        x = torch.linspace(self.x_min, self.x_max, self.grid_num)
        y = torch.linspace(self.y_min, self.y_max, self.grid_num)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        self.x = X.reshape(-1, 1).float().to(device).requires_grad_(True)
        self.y = Y.reshape(-1, 1).float().to(device).requires_grad_(True)
    
    # =========================================================================
    # PDE Residuals
    # =========================================================================
    
    def burgers_residual(self, model: nn.Module, x: torch.Tensor, 
                         return_pointwise: bool = False) -> torch.Tensor:
        """Burgers equation residual."""
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_spatial = u_x[:, 0:1]
        u_temporal = u_x[:, 1:2]
        
        u_xx = torch.autograd.grad(u_spatial, x, grad_outputs=torch.ones_like(u_spatial),
                                   create_graph=True, retain_graph=True)[0][:, 0:1]
        
        nu = 0.01 / np.pi
        residual = u_temporal + u * u_spatial - nu * u_xx
        
        if return_pointwise:
            return residual ** 2
        return torch.mean(residual ** 2)
    
    def laplace_residual(self, model: nn.Module, x: torch.Tensor,
                         return_pointwise: bool = False) -> torch.Tensor:
        """Laplace equation residual."""
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_x1, u_x2 = u_x[:, 0:1], u_x[:, 1:2]
        
        u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1),
                                      create_graph=True, retain_graph=True)[0][:, 0:1]
        u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2),
                                      create_graph=True, retain_graph=True)[0][:, 1:2]
        
        residual = u_x1x1 + u_x2x2
        
        if return_pointwise:
            return residual ** 2
        return torch.mean(residual ** 2)
    
    def poisson_residual(self, model: nn.Module, x: torch.Tensor,
                         return_pointwise: bool = False) -> torch.Tensor:
        """Poisson equation residual."""
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_x1, u_x2 = u_x[:, 0:1], u_x[:, 1:2]
        
        u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1),
                                      create_graph=True, retain_graph=True)[0][:, 0:1]
        u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2),
                                      create_graph=True, retain_graph=True)[0][:, 1:2]
        
        # Source term: sum of sin functions
        x1, x2 = x[:, 0:1], x[:, 1:2]
        f = sum(0.5 * ((-1)**(k+1)) * torch.sin(k * np.pi * x1) * torch.sin(k * np.pi * x2)
                for k in range(1, 5))
        
        residual = u_x1x1 + u_x2x2 + f
        
        if return_pointwise:
            return residual ** 2
        return torch.mean(residual ** 2)
    
    def get_residual_fn(self):
        """Get appropriate residual function."""
        if 'Burgers' in self.ques_name:
            return self.burgers_residual
        elif 'Laplace' in self.ques_name:
            return self.laplace_residual
        elif 'Poisson' in self.ques_name:
            return self.poisson_residual
        else:
            raise ValueError(f"Unknown problem: {self.ques_name}")
    
    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    
    def compute_bc_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute boundary and initial condition losses separately."""
        loss_bc = torch.tensor(0., device=device)
        loss_ic = torch.tensor(0., device=device)
        
        # Boundary points
        y_b = torch.linspace(self.y_min, self.y_max, self.bun_num).float().to(device).reshape(-1, 1)
        x_left = torch.full_like(y_b, self.x_min)
        x_right = torch.full_like(y_b, self.x_max)
        
        x_d = torch.linspace(self.x_min, self.x_max, self.bun_num).float().to(device).reshape(-1, 1)
        y_bottom = torch.full_like(x_d, self.y_min)
        y_top = torch.full_like(x_d, self.y_max)
        
        if 'Burgers' in self.ques_name:
            # Left BC: u(-1, t) = 0
            u_left = self.teacher(torch.cat([x_left, y_b], dim=1))
            loss_bc += torch.mean(u_left ** 2)
            
            # Right BC: u(1, t) = 0
            u_right = self.teacher(torch.cat([x_right, y_b], dim=1))
            loss_bc += torch.mean(u_right ** 2)
            
            # IC: u(x, 0) = -sin(πx)
            u_init = self.teacher(torch.cat([x_d, y_bottom], dim=1))
            u_init_exact = -torch.sin(np.pi * x_d)
            loss_ic = torch.mean((u_init - u_init_exact) ** 2)
            
        elif 'Laplace' in self.ques_name:
            # All boundaries: u = x^3 - 3xy^2
            for x_pts, y_pts in [(x_left, y_b), (x_right, y_b), 
                                  (x_d, y_bottom), (x_d, y_top)]:
                u = self.teacher(torch.cat([x_pts, y_pts], dim=1))
                u_exact = x_pts**3 - 3*x_pts*y_pts**2
                loss_bc += torch.mean((u - u_exact) ** 2)
            loss_ic = loss_bc  # No separate IC for Laplace
            
        elif 'Poisson' in self.ques_name:
            # Zero Dirichlet BC
            all_x = torch.cat([x_left, x_right, x_d, x_d], dim=0)
            all_y = torch.cat([y_b, y_b, y_bottom, y_top], dim=0)
            u_bc = self.teacher(torch.cat([all_x, all_y], dim=1))
            loss_bc = torch.mean(u_bc ** 2)
            loss_ic = loss_bc
        
        return loss_bc, loss_ic
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def train_teacher(self, print_every: int = 1000):
        """Train teacher PINN with advanced techniques."""
        print("\n" + "="*60)
        print("PHASE 1: Training Optimized Teacher PINN")
        print("="*60)
        
        self.mesh_init()
        
        # Optimizer
        optimizer = optim.Adam(self.teacher.parameters(), lr=self.learning_rate)
        scheduler = get_warmup_cosine_scheduler(optimizer, self.warmup_steps, self.train_steps)
        
        # Get residual function
        residual_fn = self.get_residual_fn()
        
        start_time = time.time()
        
        for epoch in range(1, self.train_steps + 1):
            self.teacher.train()
            optimizer.zero_grad()
            
            # Get collocation points (with RAR if enabled)
            if self.use_rar:
                x_coll = self.rar.get_points()
            else:
                x_coll = torch.cat([self.x, self.y], dim=1)
            
            # PDE residual
            if self.causal_trainer and 'Burgers' in self.ques_name:
                residuals = residual_fn(self.teacher, x_coll, return_pointwise=True)
                t_vals = x_coll[:, 1:2]
                loss_f = self.causal_trainer.weighted_loss(t_vals, residuals)
            else:
                loss_f = residual_fn(self.teacher, x_coll)
            
            # Boundary conditions
            loss_bc, loss_ic = self.compute_bc_loss()
            
            # Regularization
            loss_rgl = sum(torch.norm(p, p=2) for p in self.teacher.parameters()) * self.l2_weight
            
            # Combine losses
            if self.use_adaptive_weights:
                loss = self.loss_weighter.update([loss_f, loss_bc, loss_ic])
            else:
                # Weight IC more heavily for Burgers
                ic_weight = 10.0 if 'Burgers' in self.ques_name else 1.0
                loss = loss_f + loss_bc + ic_weight * loss_ic + loss_rgl
            
            # Backward pass with gradient clipping
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # RAR refinement
            if self.use_rar and epoch % self.rar_interval == 0:
                self.rar.refine(self.teacher, lambda m, x: residual_fn(m, x, True))
            
            # Logging
            self.teacher.iter_list.append(epoch)
            self.teacher.loss_list.append(loss.item())
            self.teacher.loss_f_list.append(loss_f.item())
            self.teacher.loss_b_list.append((loss_bc + loss_ic).item())
            self.teacher.loss_rgl_list.append(loss_rgl.item())
            
            if epoch % print_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}/{self.train_steps} | Loss: {loss.item():.3e} | "
                      f"PDE: {loss_f.item():.3e} | BC/IC: {(loss_bc+loss_ic).item():.3e} | LR: {lr:.2e}")
        
        # L-BFGS fine-tuning
        if self.use_lbfgs:
            print("\nL-BFGS Fine-tuning...")
            self._lbfgs_finetune(residual_fn)
        
        elapsed = time.time() - start_time
        print(f"\nTeacher training complete. Time: {elapsed:.2f}s")
        
        # Save
        torch.save(self.teacher.state_dict(), 
                   f'{self.save_dir}/Models/{self.ques_name}_{self.ini_num}_teacher.pth')
    
    def _lbfgs_finetune(self, residual_fn: Callable):
        """Fine-tune with L-BFGS optimizer."""
        optimizer = optim.LBFGS(
            self.teacher.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        x_coll = torch.cat([self.x, self.y], dim=1)
        
        for i in range(self.lbfgs_steps // 20):
            def closure():
                optimizer.zero_grad()
                loss_f = residual_fn(self.teacher, x_coll)
                loss_bc, loss_ic = self.compute_bc_loss()
                ic_weight = 10.0 if 'Burgers' in self.ques_name else 1.0
                loss = loss_f + loss_bc + ic_weight * loss_ic
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            if i % 5 == 0:
                print(f"  L-BFGS iter {i*20}: Loss = {loss.item():.3e}")
    
    def train_student(self, print_every: int = 1000):
        """Train Bayesian student via distillation."""
        print("\n" + "="*60)
        print(f"PHASE 2: Training Bayesian Student ({self.student_type})")
        print("="*60)
        
        optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate)
        scheduler = get_warmup_cosine_scheduler(optimizer, self.warmup_steps // 2, self.student_steps)
        
        self.teacher.eval()
        x_train = torch.cat([self.x, self.y], dim=1)
        
        with torch.no_grad():
            teacher_output = self.teacher(x_train)
        
        start_time = time.time()
        
        for epoch in range(1, self.student_steps + 1):
            self.student.train()
            optimizer.zero_grad()
            
            # Student prediction
            if self.heteroscedastic:
                # Net returns tuple (mean, log_var) when heteroscedastic=True
                student_mean, student_log_var = self.student(x_train)
                var = torch.exp(student_log_var) + 1e-6
                loss_distill = 0.5 * ((teacher_output - student_mean)**2 / var + student_log_var).mean()
            else:
                student_output = self.student(x_train)
                loss_distill = F.mse_loss(student_output, teacher_output)
            
            # Regularization / KL
            if self.student_type == 'vi_bnn':
                loss_reg = self.student.get_kl_divergence() * self.kl_weight
            else:
                loss_reg = sum(torch.norm(p, p=2) for p in self.student.parameters()) * self.l2_weight
            
            loss = loss_distill + loss_reg
            
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()
            
            # Logging
            self.student.iter_list.append(epoch)
            self.student.loss_list.append(loss.item())
            self.student.loss_teach_list.append(loss_distill.item())
            self.student.loss_rgl_list.append(loss_reg.item())
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{self.student_steps} | Loss: {loss.item():.3e} | "
                      f"Distill: {loss_distill.item():.3e} | Reg: {loss_reg.item():.3e}")
        
        elapsed = time.time() - start_time
        print(f"\nStudent training complete. Time: {elapsed:.2f}s")
        
        # Save
        torch.save(self.student.state_dict(),
                   f'{self.save_dir}/Models/{self.ques_name}_{self.ini_num}_student_{self.student_type}.pth')
        
        # Save losses
        self._save_losses()
    
    def _save_losses(self):
        """Save training losses to CSV."""
        # Teacher
        pd.DataFrame({
            'iter': self.teacher.iter_list,
            'loss': self.teacher.loss_list,
            'loss_f': self.teacher.loss_f_list,
            'loss_b': self.teacher.loss_b_list,
            'loss_rgl': self.teacher.loss_rgl_list
        }).to_csv(f'{self.save_dir}/Loss/{self.ques_name}_{self.ini_num}_teacher_loss.csv', index=False)
        
        # Student
        pd.DataFrame({
            'iter': self.student.iter_list,
            'loss': self.student.loss_list,
            'loss_teach': self.student.loss_teach_list,
            'loss_rgl': self.student.loss_rgl_list
        }).to_csv(f'{self.save_dir}/Loss/{self.ques_name}_{self.ini_num}_student_loss.csv', index=False)
    
    def discover_structure(self, **kwargs):
        """Discover network structure."""
        print("\n" + "="*60)
        print("PHASE 3: Structure Discovery")
        print("="*60)
        return self.structure_discovery.extract_structure(**kwargs)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Get predictions with uncertainty."""
        # Create estimator with specified n_samples
        estimator = UE.UncertaintyEstimator(self.student, n_samples=n_samples)
        return estimator.predict(x)
    
    def get_relation_matrices(self, structure: Dict):
        """Get relation matrices from structure."""
        return self.structure_discovery.build_relation_matrix(structure)
    
    def workflow(self, print_every: int = 1000):
        """Run complete training workflow."""
        self.train_teacher(print_every)
        self.train_student(print_every)
        structure = self.discover_structure()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.save_dir}")
        
        return structure
