# coding = utf-8
"""
Structure-Discovery Baselines (JCP review item VI)
====================================================

The proposed method (`Module/StructureDiscovery.StructureDiscovery`) clusters
absolute weight values via HAC. A JCP reviewer can plausibly call that a
"magnitude-clustering heuristic" unless the paper also reports the same
reconstruction pipeline with a non-HAC structure builder. This module
provides three such baselines that produce a structure dict in exactly the
format `Module.StructuredPINN.build_structured_pinn` already consumes, so
the reconstruction trainer is reused unchanged.

Baselines
---------
1. `random_clustering(weights, n_clusters_per_layer, seed)`
   Each weight is assigned to a cluster uniformly at random. Same
   parameter budget as HAC. Tests whether the HAC topology matters.

2. `magnitude_clustering(weights, n_clusters_per_layer)`
   Weights are sorted by |w| and split into `n_clusters` equal-quantile
   bins. Same parameter budget as HAC. Tests whether the spatial pattern
   of clusters (vs. a pure-magnitude partition) matters.

3. `low_rank_svd(weights, n_clusters_per_layer)`
   For each weight matrix, keep the top-K singular components where K
   matches the cluster count. The trainable parameters are the K
   singular values. Note: this is a STRICTLY more restrictive
   parameterization than HAC, so it is included as a "weaker baseline"
   reference rather than a head-to-head challenger.

4. `identity_no_compression(weights)`
   Each parameter is its own cluster. No compression. This is a sanity
   check: the structured trainer with this structure should match the
   dense student up to optimization noise.

All four produce `(structure, relation_matrices)` pairs that drop into
`build_structured_pinn(structure, relation_matrices, reference_model, ...)`.

Usage
-----
    from Module.StructureBaselines import build_baseline_structure

    structure, R_dict = build_baseline_structure(
        weights, n_clusters_per_layer, variant='random', seed=0)
    model = SP.build_structured_pinn(structure, R_dict, reference_model)

The companion runner script is
`Supplementary experiment/structure_baseline_ablation.py`.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _signs_from_weight(weight: torch.Tensor) -> np.ndarray:
    return np.sign(weight.detach().cpu().numpy().flatten())


# ---------------------------------------------------------------------------
# Random clustering
# ---------------------------------------------------------------------------

def random_clustering(weights: torch.Tensor, n_clusters: int,
                      seed: int = 0) -> Dict:
    flat = weights.detach().cpu().numpy().flatten()
    n_params = flat.size
    n_clusters = max(1, min(int(n_clusters), n_params))
    rng = np.random.default_rng(seed)

    # Random assignment, but each cluster gets at least one element.
    perm = rng.permutation(n_params)
    labels = np.empty(n_params, dtype=int)
    labels[perm[:n_clusters]] = np.arange(n_clusters)
    if n_params > n_clusters:
        labels[perm[n_clusters:]] = rng.integers(0, n_clusters,
                                                 size=n_params - n_clusters)
    abs_w = np.abs(flat)
    centers = np.array([abs_w[labels == k].mean() if (labels == k).any() else 0.0
                        for k in range(n_clusters)])
    return {
        'cluster_centers': centers,
        'labels': labels,
        'signs': np.sign(flat),
        'original_shape': tuple(weights.shape),
        'n_clusters': int(n_clusters),
        'n_params': int(n_params),
        'method': 'random_clustering',
    }


# ---------------------------------------------------------------------------
# Magnitude clustering (equal-quantile bins)
# ---------------------------------------------------------------------------

def magnitude_clustering(weights: torch.Tensor, n_clusters: int) -> Dict:
    flat = weights.detach().cpu().numpy().flatten()
    n_params = flat.size
    n_clusters = max(1, min(int(n_clusters), n_params))
    abs_w = np.abs(flat)
    # Quantile bin edges.
    quantiles = np.quantile(abs_w, np.linspace(0.0, 1.0, n_clusters + 1))
    # Strictly increasing edges; nudge ties so bins do not collapse.
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-12
    labels = np.clip(np.searchsorted(quantiles[1:-1], abs_w), 0, n_clusters - 1)
    centers = np.array([abs_w[labels == k].mean()
                        if (labels == k).any() else float(quantiles[k])
                        for k in range(n_clusters)])
    return {
        'cluster_centers': centers,
        'labels': labels,
        'signs': np.sign(flat),
        'original_shape': tuple(weights.shape),
        'n_clusters': int(n_clusters),
        'n_params': int(n_params),
        'method': 'magnitude_clustering',
    }


def _structure_to_R(structure: Dict, device: torch.device) -> torch.Tensor:
    n_params = int(structure['n_params'])
    n_clusters = int(structure['n_clusters'])
    R = np.zeros((n_params, n_clusters), dtype=np.float32)
    labels = structure['labels']
    signs = structure['signs']
    for i, (lab, sg) in enumerate(zip(labels, signs)):
        R[i, int(lab)] = float(sg)
    return torch.tensor(R, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Low-rank SVD parameterization
# ---------------------------------------------------------------------------

def low_rank_svd(weights: torch.Tensor, n_components: int,
                 device: torch.device | None = None) -> Tuple[Dict, torch.Tensor]:
    """Express W as sum_k s_k * (u_k v_k^T) with K trainable singular values.

    This parameterization is more restrictive than HAC clustering (it can
    only represent matrices spanned by the K rank-1 components). It is
    included as a 'weaker baseline' reference.
    """
    if device is None:
        device = weights.device
    W = weights.detach().to(device).float()
    if W.dim() != 2:
        raise ValueError(f'low_rank_svd expects 2D weights, got {W.shape}')
    out_features, in_features = W.shape
    n_params = out_features * in_features
    n_components = max(1, min(int(n_components), min(out_features, in_features)))

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_k = U[:, :n_components]                                # (out, K)
    Vh_k = Vh[:n_components, :]                              # (K, in)
    S_k = S[:n_components]                                   # (K,)

    # R[:, k] = vec(u_k * v_k^T)  (column-major to match reshape(out, in)).
    R = torch.zeros((n_params, n_components), dtype=torch.float32, device=device)
    for k in range(n_components):
        outer = torch.outer(U_k[:, k], Vh_k[k, :])           # (out, in)
        R[:, k] = outer.reshape(-1)

    structure = {
        'cluster_centers': S_k.cpu().numpy(),
        'labels': np.zeros(n_params, dtype=int),  # unused for low-rank
        'signs': np.ones(n_params, dtype=int),    # unused (sign in R)
        'original_shape': (out_features, in_features),
        'n_clusters': int(n_components),
        'n_params': int(n_params),
        'method': 'low_rank_svd',
    }
    return structure, R


# ---------------------------------------------------------------------------
# Identity (no compression, no sharing)
# ---------------------------------------------------------------------------

def identity_no_compression(weights: torch.Tensor) -> Dict:
    flat = weights.detach().cpu().numpy().flatten()
    n = flat.size
    return {
        'cluster_centers': np.abs(flat),
        'labels': np.arange(n),
        'signs': np.sign(flat),
        'original_shape': tuple(weights.shape),
        'n_clusters': int(n),
        'n_params': int(n),
        'method': 'identity_no_compression',
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_baseline_structure(weights_dict: Dict[str, torch.Tensor],
                             n_clusters_per_layer: Dict[str, int],
                             variant: str,
                             seed: int = 0,
                             device: torch.device | None = None
                             ) -> Tuple[Dict[str, Dict], Dict[str, torch.Tensor]]:
    """Build a baseline (structure, R) pair from a dense model's weight dict.

    Args:
        weights_dict: from `StructureDiscovery.get_weights_for_clustering`
                      or `{name: param.data}` for any nn.Module. Only entries
                      with 'weight' in the name are clustered.
        n_clusters_per_layer: map from layer name to target cluster count.
                              Match the HAC reference so parameter budgets
                              are comparable.
        variant: 'random', 'magnitude', 'low_rank', or 'identity'.
        seed: used by the 'random' variant.
        device: where to place the relation matrices.

    Returns:
        (structure_dict, relation_matrices_dict) consumable by
        Module.StructuredPINN.build_structured_pinn.
    """
    structure = {}
    R_dict = {}
    if device is None:
        # Pick any weight tensor's device.
        for w in weights_dict.values():
            device = w.device
            break
    device = device or torch.device('cpu')

    for name, weight in weights_dict.items():
        if 'weight' not in name:
            continue
        n_k = int(n_clusters_per_layer.get(name, weight.numel()))

        if variant == 'random':
            sub = random_clustering(weight, n_k, seed=seed)
            R = _structure_to_R(sub, device)
        elif variant == 'magnitude':
            sub = magnitude_clustering(weight, n_k)
            R = _structure_to_R(sub, device)
        elif variant == 'low_rank':
            sub, R = low_rank_svd(weight, n_k, device=device)
        elif variant == 'identity':
            sub = identity_no_compression(weight)
            R = _structure_to_R(sub, device)
        else:
            raise ValueError(f'Unknown variant: {variant}')

        structure[name] = sub
        R_dict[name] = R

    return structure, R_dict
