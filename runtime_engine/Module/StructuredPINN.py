# coding = utf-8
"""
Structured PINN Module (Stage-3 Reconstruction)

Implements a compact structured PINN where each layer stores only cluster
centers as trainable parameters. Full dense weights are generated on-the-fly
via: W_full = R @ diag(centers), using a frozen relation matrix formulation.

Supports two modes:
  Mode 1 (Deterministic): centers are nn.Parameter tensors
  Mode 2 (Bayesian MC Dropout): adds dropout between structured layers

Third-party provenance for the migrated runtime engine is documented in
docs/THIRD_PARTY_NOTICE.md.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
from Module import PoissonTools as PT


class StructuredLinear(nn.Module):
    """
    A structured linear layer that stores only cluster centers and
    reconstructs full weights on-the-fly via a frozen relation matrix R.

    Forward: y = (R @ centers) reshaped to (out, in) applied to x, plus bias.

    The relation matrix R is frozen (not trainable). Only centers (and bias)
    receive gradients.

    Args:
        R: Relation matrix of shape (n_params, n_clusters). Frozen.
        centers_init: Initial cluster center values, shape (n_clusters,).
        original_shape: Original weight matrix shape (out_features, in_features).
        bias_init: Optional initial bias values, shape (out_features,).
        has_bias: Whether to include bias term.
    """
    def __init__(self, R: torch.Tensor, centers_init: torch.Tensor,
                 original_shape: Tuple[int, int],
                 bias_init: Optional[torch.Tensor] = None,
                 has_bias: bool = True):
        super().__init__()

        self.original_shape = original_shape
        out_features, in_features = original_shape
        self.out_features = out_features
        self.in_features = in_features

        # Relation matrix: FROZEN (encodes structure)
        self.register_buffer('R', R.clone().detach())

        # Trainable cluster centers
        self.centers = nn.Parameter(centers_init.clone().detach().float())

        # Bias
        if has_bias and bias_init is not None:
            self.bias = nn.Parameter(bias_init.clone().detach().float())
        elif has_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix: W = reshape(R @ centers)."""
        flat_w = self.R @ self.centers
        return flat_w.reshape(self.original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.reconstruct_weight()
        out = torch.nn.functional.linear(x, W, self.bias)
        return out

    def count_trainable(self) -> int:
        """Count trainable parameters in this layer."""
        n = self.centers.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    def count_original(self) -> int:
        """Count original (dense) parameters."""
        n = self.original_shape[0] * self.original_shape[1]
        if self.bias is not None:
            n += self.original_shape[0]
        return n


class StructuredPINN(nn.Module):
    """
    Compact structured PINN with true parameter sharing via relation matrices.

    Each hidden layer uses a StructuredLinear layer. Full dense weights are
    generated on-the-fly from cluster centers and the frozen R matrix.

    Args:
        structure: Dict from StructureDiscovery.extract_structure()
        relation_matrices: Dict from StructureDiscovery.build_relation_matrix()
        reference_model: The original student/teacher model (for biases and arch)
        activation: Activation function class (default: nn.Tanh)
        dropout_rate: If > 0, add MC Dropout for Bayesian mode (Mode 2)
    """
    def __init__(self, structure: Dict, relation_matrices: Dict[str, torch.Tensor],
                 reference_model: nn.Module,
                 activation: type = nn.Tanh,
                 dropout_rate: float = 0.0,
                 use_fourier_features: Optional[bool] = None,
                 fourier_modes: Optional[int] = None,
                 hard_bc: Optional[bool] = None,
                 residual_branch: bool = False,
                 residual_alpha: float = 0.1,
                 residual_width: int = 32):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.activation_cls = activation
        self._structure = structure
        self._layer_order = []  # track layer ordering
        self.use_fourier_features = (
            getattr(reference_model, 'use_fourier_features', False)
            if use_fourier_features is None else use_fourier_features
        )
        self.fourier_modes = int(
            getattr(reference_model, 'fourier_modes', 4)
            if fourier_modes is None else fourier_modes
        )
        self.hard_bc = (
            getattr(reference_model, 'hard_bc', False)
            if hard_bc is None else hard_bc
        )
        self.raw_input_dim = 2

        # Extract architecture info from reference model
        ref_state = reference_model.state_dict()
        device = next(reference_model.parameters()).device

        # Identify linear layer names in order
        weight_names = sorted([k for k in ref_state.keys() if 'weight' in k and 'layer' in k],
                              key=lambda s: int(s.split('layer_')[1].split('.')[0]))
        bias_names = sorted([k for k in ref_state.keys() if 'bias' in k and 'layer' in k],
                            key=lambda s: int(s.split('layer_')[1].split('.')[0]))

        # Map weight names in ref_state to structure keys
        # Structure keys may differ slightly; build a mapping
        struct_key_map = {}
        for wn in weight_names:
            # Try direct match first
            for sk in structure.keys():
                if wn == sk or wn in sk or sk in wn:
                    struct_key_map[wn] = sk
                    break
            # Fallback: match by layer index
            if wn not in struct_key_map:
                for sk in structure.keys():
                    if self._extract_layer_idx(wn) == self._extract_layer_idx(sk):
                        struct_key_map[wn] = sk
                        break

        # Build structured layers
        layers = OrderedDict()
        n_layers = len(weight_names)

        for i, wn in enumerate(weight_names):
            layer_idx = self._extract_layer_idx(wn)

            # Get bias
            bn = None
            for b in bias_names:
                if self._extract_layer_idx(b) == layer_idx:
                    bn = b
                    break

            if wn in struct_key_map:
                sk = struct_key_map[wn]
                R = relation_matrices[sk].to(device)
                centers_np = structure[sk]['cluster_centers']
                centers_init = torch.tensor(centers_np, dtype=torch.float32, device=device)
                orig_shape = tuple(ref_state[wn].shape)

                bias_init = ref_state[bn].to(device) if bn is not None else None

                layers[f'structured_{layer_idx}'] = StructuredLinear(
                    R=R,
                    centers_init=centers_init,
                    original_shape=orig_shape,
                    bias_init=bias_init,
                    has_bias=(bn is not None)
                )
            else:
                # Fallback: standard linear (should not happen if structure covers all)
                W = ref_state[wn].to(device)
                bias_val = ref_state[bn].to(device) if bn is not None else None
                lin = nn.Linear(W.shape[1], W.shape[0], bias=(bn is not None))
                lin.weight.data.copy_(W)
                if bias_val is not None:
                    lin.bias.data.copy_(bias_val)
                layers[f'linear_{layer_idx}'] = lin

            self._layer_order.append(layer_idx)

            # Add activation (not after last layer)
            if i < n_layers - 1:
                layers[f'activation_{layer_idx}'] = activation()
                if dropout_rate > 0:
                    layers[f'dropout_{layer_idx}'] = nn.Dropout(p=dropout_rate)

        self.layers = nn.Sequential(layers)

        self.use_residual_branch = bool(residual_branch)
        if self.use_residual_branch:
            out_dim = ref_state[weight_names[-1]].shape[0]
            self.residual_alpha = nn.Parameter(
                torch.tensor(float(residual_alpha), dtype=torch.float32, device=device)
            )
            self.residual_branch = nn.Sequential(
                nn.Linear(self.raw_input_dim, int(residual_width)),
                activation(),
                nn.Linear(int(residual_width), out_dim),
            ).to(device)
        else:
            self.register_parameter('residual_alpha', None)
            self.residual_branch = None

        # Training history
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_distill_list = []

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        """Extract layer index from parameter name like 'layers.layer_2.weight'."""
        import re
        match = re.search(r'layer_(\d+)', name)
        if match:
            return int(match.group(1))
        return -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        if self.use_fourier_features:
            x = PT.encode_fourier(x, modes=self.fourier_modes)
        out = self.layers(x)
        if self.residual_branch is not None:
            out = out + self.residual_alpha * self.residual_branch(raw_x)
        if self.hard_bc:
            out = PT.apply_zero_dirichlet_hard_bc(raw_x, out)
        return out

    def enable_dropout(self):
        """Enable dropout for MC sampling during inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def set_dropout_rate(self, rate: float):
        """Set all dropout probabilities without changing learned weights."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(rate)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters vs original dense parameters.

        Returns:
            Dict with 'trainable', 'original', 'compression_ratio',
            and per-layer breakdown.
        """
        total_trainable = 0
        total_original = 0
        per_layer = {}

        for name, module in self.layers.named_modules():
            if isinstance(module, StructuredLinear):
                t = module.count_trainable()
                o = module.count_original()
                total_trainable += t
                total_original += o
                per_layer[name] = {
                    'trainable': t,
                    'original': o,
                    'compression': o / t if t > 0 else float('inf')
                }
            elif isinstance(module, nn.Linear):
                n = sum(p.numel() for p in module.parameters())
                total_trainable += n
                total_original += n
                per_layer[name] = {
                    'trainable': n,
                    'original': n,
                    'compression': 1.0
                }

        if self.residual_branch is not None:
            n = sum(p.numel() for p in self.residual_branch.parameters())
            n += 1  # residual_alpha
            total_trainable += n
            per_layer['residual_branch'] = {
                'trainable': n,
                'original': 0,
                'compression': 1.0
            }

        ratio = total_original / total_trainable if total_trainable > 0 else float('inf')

        return {
            'trainable': total_trainable,
            'original': total_original,
            'compression_ratio': ratio,
            'per_layer': per_layer
        }

    def get_reconstructed_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get dense weight matrices (reconstructed from centers) as a state dict."""
        result = {}
        for name, module in self.layers.named_modules():
            if isinstance(module, StructuredLinear):
                result[f'{name}.weight'] = module.reconstruct_weight().detach()
                if module.bias is not None:
                    result[f'{name}.bias'] = module.bias.detach()
        return result


def build_structured_pinn(structure: Dict,
                          relation_matrices: Dict[str, torch.Tensor],
                          reference_model: nn.Module,
                          dropout_rate: float = 0.0,
                          residual_branch: bool = False,
                          residual_alpha: float = 0.1,
                          residual_width: int = 32) -> StructuredPINN:
    """
    Convenience function to build a StructuredPINN from structure discovery results.

    Args:
        structure: From StructureDiscovery.extract_structure()
        relation_matrices: From StructureDiscovery.build_relation_matrix()
        reference_model: Original student (or teacher) model
        dropout_rate: If > 0, enables MC Dropout mode

    Returns:
        StructuredPINN instance on same device as reference_model
    """
    device = next(reference_model.parameters()).device
    model = StructuredPINN(
        structure=structure,
        relation_matrices=relation_matrices,
        reference_model=reference_model,
        dropout_rate=dropout_rate,
        residual_branch=residual_branch,
        residual_alpha=residual_alpha,
        residual_width=residual_width
    ).to(device)

    return model
