# coding = utf-8
"""
MC Dropout Bayesian Student Network

Implements Monte Carlo Dropout for uncertainty estimation.
Dropout is kept ON during inference to enable sampling.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union, Tuple
from Module import PoissonTools as PT


class Net(nn.Module):
    """
    MC Dropout Bayesian student network.
    
    Dropout is kept ON during inference to enable Monte Carlo sampling
    for uncertainty estimation.
    
    Args:
        layers: List of layer sizes [input, hidden1, ..., hiddenN, output]
        dropout_rate: Dropout probability (default 0.15)
        heteroscedastic: If True, outputs both mean and log_variance
    """
    def __init__(self, layers, dropout_rate: float = 0.15,
                 heteroscedastic: bool = False,
                 use_fourier_features: bool = False,
                 fourier_modes: int = 4,
                 hard_bc: bool = False):
        super(Net, self).__init__()
        self.depth = len(layers) - 1
        self.dropout_rate = dropout_rate
        self.heteroscedastic = heteroscedastic
        self.use_fourier_features = use_fourier_features
        self.fourier_modes = int(fourier_modes)
        self.hard_bc = hard_bc
        self.activation = nn.Tanh
        self.output_dim = layers[-1]
        
        # Build network layers
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                (f'layer_{i}', nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append((f'activation_{i}', self.activation()))
            layer_list.append((f'dropout_{i}', nn.Dropout(p=dropout_rate)))
        
        # Output layer (no dropout after final layer)
        if heteroscedastic:
            layer_list.append(
                (f'layer_{self.depth - 1}', nn.Linear(layers[-2], layers[-1] * 2))
            )
        else:
            layer_list.append(
                (f'layer_{self.depth - 1}', nn.Linear(layers[-2], layers[-1]))
            )
        
        self.layers = nn.Sequential(OrderedDict(layer_list))
        
        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.loss_d_list = []
        self.loss_f_list = []
        self.loss_b_list = []

    def set_dropout_rate(self, rate: float):
        """Change dropout rate for all dropout layers.
        
        Use low rate during training for accurate distillation,
        then restore full rate for MC inference.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            If heteroscedastic=False: predictions of shape (batch, output_dim)
            If heteroscedastic=True: (mean, log_var) each of shape (batch, output_dim)
        """
        raw_x = x
        if self.use_fourier_features:
            x = PT.encode_fourier(x, modes=self.fourier_modes)
        out = self.layers(x)
        
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            if self.hard_bc:
                mean = PT.apply_zero_dirichlet_hard_bc(raw_x, mean)
            return mean, log_var
        if self.hard_bc:
            out = PT.apply_zero_dirichlet_hard_bc(raw_x, out)
        return out
    
    def enable_dropout(self):
        """Enable dropout for MC sampling during inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def get_deterministic_weights(self):
        """
        Get deterministic weights for clustering.
        For MC Dropout, these are just the trained weights.
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.clone()
        return weights


class NetWithNodeNum(nn.Module):
    """
    MC Dropout network initialized with node_num for runtime interface compatibility.
    
    Args:
        node_num: Number of nodes per hidden layer
        output_num: Number of output dimensions
        dropout_rate: Dropout probability
        heteroscedastic: If True, outputs both mean and log_variance
    """
    def __init__(self, node_num: int, output_num: int = 1,
                 dropout_rate: float = 0.15, heteroscedastic: bool = False,
                 use_fourier_features: bool = False,
                 fourier_modes: int = 4,
                 hard_bc: bool = False):
        super(NetWithNodeNum, self).__init__()
        
        # Default 3 hidden layers
        input_dim = PT.fourier_feature_dim(2, fourier_modes) if use_fourier_features else 2
        layers = [input_dim, node_num, node_num, node_num, output_num]
        
        self.depth = len(layers) - 1
        self.dropout_rate = dropout_rate
        self.heteroscedastic = heteroscedastic
        self.use_fourier_features = use_fourier_features
        self.fourier_modes = int(fourier_modes)
        self.hard_bc = hard_bc
        self.activation = nn.Tanh
        self.output_dim = output_num
        self.node_num = node_num
        
        # Build network layers
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                (f'layer_{i}', nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append((f'activation_{i}', self.activation()))
            layer_list.append((f'dropout_{i}', nn.Dropout(p=dropout_rate)))
        
        # Output layer
        if heteroscedastic:
            layer_list.append(
                (f'layer_{self.depth - 1}', nn.Linear(layers[-2], layers[-1] * 2))
            )
        else:
            layer_list.append(
                (f'layer_{self.depth - 1}', nn.Linear(layers[-2], layers[-1]))
            )
        
        self.layers = nn.Sequential(OrderedDict(layer_list))
        
        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.loss_d_list = []
        self.loss_f_list = []
        self.loss_b_list = []

    def forward(self, x):
        raw_x = x
        if self.use_fourier_features:
            x = PT.encode_fourier(x, modes=self.fourier_modes)
        out = self.layers(x)
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            if self.hard_bc:
                mean = PT.apply_zero_dirichlet_hard_bc(raw_x, mean)
            return mean, log_var
        if self.hard_bc:
            out = PT.apply_zero_dirichlet_hard_bc(raw_x, out)
        return out
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def set_dropout_rate(self, rate: float):
        """Change dropout rate for all dropout layers."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
    
    def get_deterministic_weights(self):
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.clone()
        return weights
