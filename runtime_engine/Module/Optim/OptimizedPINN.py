# coding = utf-8
"""
Optimized PINN module for experimental GBSD runtime paths

Advanced techniques for improved training:
1. Fourier Feature Embeddings (reduce spectral bias)
2. Modified MLP with residual connections
3. Adaptive activation functions
4. Better weight initialization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FourierFeatureEmbedding(nn.Module):
    """
    Fourier feature embedding to help networks learn high-frequency functions.
    Maps input x to [sin(2πBx), cos(2πBx)] where B is a learnable or fixed matrix.
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn High Frequency 
               Functions in Low Dimensional Domains", NeurIPS 2020
    """
    def __init__(self, input_dim: int, embedding_dim: int = 256, 
                 scale: float = 1.0, learnable: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        
        # Initialize frequency matrix
        B = torch.randn(input_dim, embedding_dim) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    @property
    def output_dim(self) -> int:
        return 2 * self.embedding_dim


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function with learnable parameters.
    f(x) = a * tanh(b * x) where a, b are learnable.
    
    Reference: Jagtap et al., "Adaptive activation functions accelerate 
               convergence in deep and physics-informed neural networks", JCP 2020
    """
    def __init__(self, init_a: float = 1.0, init_b: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(self.b * x)


class SirenActivation(nn.Module):
    """
    SIREN activation: sin(ωx)
    Good for learning implicit representations.
    
    Reference: Sitzmann et al., "Implicit Neural Representations with 
               Periodic Activation Functions", NeurIPS 2020
    """
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, dim: int, activation: nn.Module = None):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation or nn.Tanh()
        
        # Initialize for residual learning
        nn.init.xavier_normal_(self.linear1.weight, gain=0.1)
        nn.init.xavier_normal_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)


class OptimizedPINN(nn.Module):
    """
    Optimized Physics-Informed Neural Network.
    
    Features:
    - Optional Fourier feature embedding
    - Adaptive or standard activations
    - Optional residual connections
    - Proper weight initialization
    """
    def __init__(self, 
                 input_dim: int = 2,
                 output_dim: int = 1,
                 hidden_dims: List[int] = [64, 64, 64, 64],
                 activation: str = 'tanh',  # 'tanh', 'adaptive', 'siren'
                 use_fourier: bool = False,
                 fourier_dim: int = 128,
                 fourier_scale: float = 1.0,
                 use_residual: bool = False):
        super().__init__()
        
        self.use_fourier = use_fourier
        self.use_residual = use_residual
        
        # Fourier embedding
        if use_fourier:
            self.fourier = FourierFeatureEmbedding(input_dim, fourier_dim, fourier_scale)
            current_dim = self.fourier.output_dim
        else:
            self.fourier = None
            current_dim = input_dim
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Activation function
        def get_activation():
            if activation == 'adaptive':
                return AdaptiveActivation()
            elif activation == 'siren':
                return SirenActivation()
            else:
                return nn.Tanh()
        
        # Input layer
        self.input_layer = nn.Linear(current_dim, hidden_dims[0])
        self._init_weights(self.input_layer, activation == 'siren')
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.layers.append(ResidualBlock(hidden_dims[i], get_activation()))
            else:
                layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                self._init_weights(layer, activation == 'siren')
                self.layers.append(layer)
                self.layers.append(get_activation())
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._init_weights(self.output_layer, activation == 'siren', is_output=True)
        
        # Final activation for hidden layers
        self.hidden_activation = get_activation()
        
        # Training history
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_rgl_list = []
    
    def _init_weights(self, layer: nn.Linear, is_siren: bool = False, is_output: bool = False):
        """Initialize weights properly."""
        if is_siren:
            # SIREN initialization
            fan_in = layer.weight.shape[1]
            if is_output:
                bound = np.sqrt(6 / fan_in)
            else:
                bound = np.sqrt(6 / fan_in) / 30  # omega = 30
            nn.init.uniform_(layer.weight, -bound, bound)
        else:
            # Xavier initialization
            nn.init.xavier_normal_(layer.weight)
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier embedding
        if self.use_fourier:
            x = self.fourier(x)
        
        # Input layer
        x = self.hidden_activation(self.input_layer(x))
        
        # Hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        return self.output_layer(x)


class Net(OptimizedPINN):
    """Alias for backward compatibility."""
    def __init__(self, node_num: int = 32, input_dim: int = 2, output_dim: int = 1,
                 use_fourier: bool = False, activation: str = 'tanh'):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[node_num, node_num],
            activation=activation,
            use_fourier=use_fourier
        )
