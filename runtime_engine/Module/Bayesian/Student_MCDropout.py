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


class Net(nn.Module):
    """
    MC Dropout Bayesian student network.
    
    Dropout is kept ON during inference to enable Monte Carlo sampling
    for uncertainty estimation.
    
    Args:
        layers: List of layer sizes [input, hidden1, ..., hiddenN, output]
        dropout_rate: Dropout probability (default 0.1)
        heteroscedastic: If True, outputs both mean and log_variance
    """
    def __init__(self, layers, dropout_rate: float = 0.1, 
                 heteroscedastic: bool = False):
        super(Net, self).__init__()
        self.depth = len(layers) - 1
        self.dropout_rate = dropout_rate
        self.heteroscedastic = heteroscedastic
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
            # Two outputs: mean and log_variance
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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            If heteroscedastic=False: predictions of shape (batch, output_dim)
            If heteroscedastic=True: (mean, log_var) each of shape (batch, output_dim)
        """
        out = self.layers(x)
        
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            return mean, log_var
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
                 dropout_rate: float = 0.1, heteroscedastic: bool = False):
        super(NetWithNodeNum, self).__init__()
        
        # Default 3 hidden layers
        layers = [2, node_num, node_num, node_num, output_num]
        
        self.depth = len(layers) - 1
        self.dropout_rate = dropout_rate
        self.heteroscedastic = heteroscedastic
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

    def forward(self, x):
        out = self.layers(x)
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            return mean, log_var
        return out
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def get_deterministic_weights(self):
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.clone()
        return weights
