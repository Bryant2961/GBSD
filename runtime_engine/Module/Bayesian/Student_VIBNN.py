# coding = utf-8
"""
Variational Inference Bayesian Student Network (VI-BNN)

Implements weight uncertainty using variational inference.
Each weight has learnable mean (mu) and variance (via rho parameter).

FIXES in v1.2:
1. get_kl_divergence() now always returns a tensor (not float)
2. Added get_deterministic_weights() for compatibility with StructureDiscovery
3. Improved numerical stability in KL computation
4. Added device handling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict


class VariationalLinear(nn.Module):
    """
    Variational Bayesian Linear Layer.
    
    Implements weight uncertainty using the reparameterization trick.
    Weights are sampled from N(mu, sigma) where sigma = log(1 + exp(rho)).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        prior_sigma: Prior standard deviation for weights
    """
    def __init__(self, in_features: int, out_features: int, 
                 prior_sigma: float = 1.0):
        super(VariationalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
        # For storing KL divergence - FIXED: always use tensor
        self._kl_div = None
    
    def reset_parameters(self):
        """Initialize variational parameters."""
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3.0)  # Small initial variance
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)
    
    @property
    def kl_div(self) -> torch.Tensor:
        """Return KL divergence as tensor."""
        if self._kl_div is None:
            return torch.tensor(0.0, device=self.weight_mu.device)
        return self._kl_div
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        Uses reparameterization trick: w = mu + sigma * epsilon
        """
        if self.training:
            # Sample weights
            weight_sigma = F.softplus(self.weight_rho)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            # Sample bias
            bias_sigma = F.softplus(self.bias_rho)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon
            
            # Compute KL divergence - FIXED: always store as tensor
            self._kl_div = self._compute_kl(
                self.weight_mu, weight_sigma, 
                self.bias_mu, bias_sigma
            )
        else:
            # Use mean for deterministic inference
            weight = self.weight_mu
            bias = self.bias_mu
            # FIXED: Return zero tensor instead of float
            self._kl_div = torch.tensor(0.0, device=self.weight_mu.device)
        
        return F.linear(x, weight, bias)
    
    def _compute_kl(self, w_mu, w_sigma, b_mu, b_sigma) -> torch.Tensor:
        """
        Compute KL divergence from posterior to prior.
        KL(q(w) || p(w)) where q ~ N(mu, sigma^2) and p ~ N(0, prior_sigma^2)
        
        FIXED: Added numerical stability with clamping
        """
        prior_var = self.prior_sigma ** 2
        
        # Clamp sigma for numerical stability
        w_sigma_clamped = w_sigma.clamp(min=1e-8)
        b_sigma_clamped = b_sigma.clamp(min=1e-8)
        
        # KL for weights
        kl_weights = 0.5 * torch.sum(
            (w_sigma_clamped ** 2 + w_mu ** 2) / prior_var - 1 
            - 2 * torch.log(w_sigma_clamped / self.prior_sigma)
        )
        
        # KL for biases
        kl_biases = 0.5 * torch.sum(
            (b_sigma_clamped ** 2 + b_mu ** 2) / prior_var - 1 
            - 2 * torch.log(b_sigma_clamped / self.prior_sigma)
        )
        
        return kl_weights + kl_biases


class Net(nn.Module):
    """
    Variational Inference Bayesian Neural Network (VI-BNN).
    
    Each linear layer has learnable mean and variance parameters.
    Training minimizes ELBO = reconstruction_loss + KL_divergence.
    
    Args:
        layers: List of layer sizes [input, hidden1, ..., hiddenN, output]
        prior_sigma: Prior standard deviation for weights
        heteroscedastic: If True, outputs both mean and log_variance
    """
    def __init__(self, layers, prior_sigma: float = 1.0,
                 heteroscedastic: bool = False):
        super(Net, self).__init__()
        self.depth = len(layers) - 1
        self.prior_sigma = prior_sigma
        self.heteroscedastic = heteroscedastic
        self.activation = nn.Tanh()
        self.output_dim = layers[-1]
        
        # Build variational layers
        self.variational_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            self.variational_layers.append(
                VariationalLinear(layers[i], layers[i + 1], prior_sigma)
            )
        
        # Output layer
        if heteroscedastic:
            self.variational_layers.append(
                VariationalLinear(layers[-2], layers[-1] * 2, prior_sigma)
            )
        else:
            self.variational_layers.append(
                VariationalLinear(layers[-2], layers[-1], prior_sigma)
            )
        
        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.loss_d_list = []
        self.kl_list = []

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through variational layers."""
        for i, layer in enumerate(self.variational_layers[:-1]):
            x = self.activation(layer(x))
        
        out = self.variational_layers[-1](x)
        
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            return mean, log_var
        return out
    
    def get_kl_divergence(self) -> torch.Tensor:
        """
        Sum KL divergence from all variational layers.
        FIXED: Always returns tensor, handles eval mode correctly.
        """
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.variational_layers:
            kl = kl + layer.kl_div
        return kl
    
    def get_mean_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior mean weights for clustering.
        Returns the mu parameters of each variational layer.
        """
        weights = {}
        for i, layer in enumerate(self.variational_layers):
            weights[f'layer_{i}.weight'] = layer.weight_mu.data.clone()
            weights[f'layer_{i}.bias'] = layer.bias_mu.data.clone()
        return weights
    
    def get_deterministic_weights(self) -> Dict[str, torch.Tensor]:
        """
        ADDED: Alias for get_mean_weights() for compatibility with StructureDiscovery.
        """
        return self.get_mean_weights()


class NetWithNodeNum(nn.Module):
    """
    VI-BNN network initialized with node_num for runtime interface compatibility.
    
    Args:
        node_num: Number of nodes per hidden layer
        output_num: Number of output dimensions
        prior_sigma: Prior standard deviation
        heteroscedastic: If True, outputs both mean and log_variance
    """
    def __init__(self, node_num: int, output_num: int = 1,
                 prior_sigma: float = 1.0, heteroscedastic: bool = False):
        super(NetWithNodeNum, self).__init__()
        
        # Default 3 hidden layers
        layers = [2, node_num, node_num, node_num, output_num]
        
        self.depth = len(layers) - 1
        self.prior_sigma = prior_sigma
        self.heteroscedastic = heteroscedastic
        self.activation = nn.Tanh()
        self.output_dim = output_num
        self.node_num = node_num
        
        # Build variational layers
        self.variational_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            self.variational_layers.append(
                VariationalLinear(layers[i], layers[i + 1], prior_sigma)
            )
        
        # Output layer
        if heteroscedastic:
            self.variational_layers.append(
                VariationalLinear(layers[-2], layers[-1] * 2, prior_sigma)
            )
        else:
            self.variational_layers.append(
                VariationalLinear(layers[-2], layers[-1], prior_sigma)
            )
        
        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.loss_d_list = []
        self.kl_list = []

    def forward(self, x):
        for i, layer in enumerate(self.variational_layers[:-1]):
            x = self.activation(layer(x))
        out = self.variational_layers[-1](x)
        
        if self.heteroscedastic:
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            return mean, log_var
        return out
    
    def get_kl_divergence(self):
        """FIXED: Always returns tensor."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.variational_layers:
            kl = kl + layer.kl_div
        return kl
    
    def get_mean_weights(self):
        weights = {}
        for i, layer in enumerate(self.variational_layers):
            weights[f'layer_{i}.weight'] = layer.weight_mu.data.clone()
            weights[f'layer_{i}.bias'] = layer.bias_mu.data.clone()
        return weights
    
    def get_deterministic_weights(self):
        """ADDED: Alias for compatibility."""
        return self.get_mean_weights()
