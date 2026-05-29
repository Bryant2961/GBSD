# coding = utf-8
"""
Structure discovery module for GBSD

Performs relation-matrix structure discovery using posterior mean weights
from Bayesian neural networks. Implements hierarchical agglomerative
clustering (HAC) and relation matrix construction.

FIXES in v1.2:
1. Fixed model type detection to handle both methods
2. Improved handling of edge cases in clustering
3. Fixed device handling for relation matrices
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class StructureDiscovery:
    """
    Relation-matrix structure discovery using Bayesian student weights.
    
    For Bayesian networks, clustering is performed on:
    - MC Dropout: deterministic weights
    - VI-BNN: posterior mean weights (E[w])
    
    This ensures stable clusters despite weight uncertainty.
    
    Args:
        model: Bayesian neural network (MC Dropout or VI-BNN)
        cluster_distance: Distance threshold for HAC clustering (default 0.1)
        cluster_mode: 'absolute' (default, original behavior) or 'relative'.
            - 'absolute': cut tree at t=cluster_distance
            - 'relative': cut tree at t=cluster_distance * max(|weights|) per layer,
              so threshold scales with layer weight magnitude (fixes cluster
              collapse when all weights are small).
    """
    def __init__(self, model: nn.Module, cluster_distance: float = 0.1,
                 cluster_mode: str = 'absolute'):
        self.model = model
        self.cluster_distance = cluster_distance
        self.cluster_mode = cluster_mode

        # FIXED: Check model type more robustly
        self.is_vi_bnn = hasattr(model, 'get_mean_weights')
        self.is_mc_dropout = hasattr(model, 'get_deterministic_weights')
        
    def get_weights_for_clustering(self) -> Dict[str, torch.Tensor]:
        """
        Get weights to use for clustering.
        Uses posterior mean for VI-BNN, deterministic weights for MC Dropout.
        
        FIXED: Now handles both methods consistently
        """
        # Try get_mean_weights first (VI-BNN)
        if hasattr(self.model, 'get_mean_weights'):
            return self.model.get_mean_weights()
        # Then try get_deterministic_weights (MC Dropout)
        elif hasattr(self.model, 'get_deterministic_weights'):
            return self.model.get_deterministic_weights()
        else:
            # Standard network - use regular weights
            weights = {}
            for name, param in self.model.named_parameters():
                weights[name] = param.data.clone()
            return weights
    
    def cluster_layer_weights(self, weights: torch.Tensor, 
                              layer_name: str) -> Dict:
        """
        Perform hierarchical agglomerative clustering on absolute weight values.
        
        Relation-matrix clustering procedure:
        1. Take absolute values
        2. Apply HAC clustering
        3. Replace weights with cluster centers
        4. Preserve sign information
        
        Args:
            weights: Weight tensor to cluster
            layer_name: Name of the layer (for logging)
            
        Returns:
            Dictionary with clustering results
        """
        weights_np = weights.detach().cpu().numpy().flatten()
        abs_weights = np.abs(weights_np)
        signs = np.sign(weights_np)
        
        # FIXED: Handle edge case of all zeros
        if np.allclose(abs_weights, 0, atol=1e-10):
            return {
                'cluster_centers': np.array([0.0]),
                'labels': np.zeros(len(weights_np), dtype=int),
                'signs': signs,
                'original_shape': weights.shape,
                'n_clusters': 1,
                'n_params': len(weights_np)
            }
        
        # FIXED: Handle edge case of single parameter
        if len(abs_weights) == 1:
            return {
                'cluster_centers': abs_weights.copy(),
                'labels': np.array([0]),
                'signs': signs,
                'original_shape': weights.shape,
                'n_clusters': 1,
                'n_params': 1
            }
        
        # Reshape for clustering (each weight is a 1D feature)
        abs_weights_2d = abs_weights.reshape(-1, 1)
        
        # Compute pairwise distances and perform clustering
        try:
            distances = pdist(abs_weights_2d, metric='euclidean')
            
            # FIXED: Handle case where all distances are zero
            if np.allclose(distances, 0):
                return {
                    'cluster_centers': np.array([abs_weights.mean()]),
                    'labels': np.zeros(len(weights_np), dtype=int),
                    'signs': signs,
                    'original_shape': weights.shape,
                    'n_clusters': 1,
                    'n_params': len(weights_np)
                }
            
            # Hierarchical clustering
            linkage_matrix = linkage(distances, method='ward')

            # Determine threshold
            if self.cluster_mode == 'relative':
                # Scale threshold by layer weight magnitude (max |w|)
                # This prevents cluster collapse when all weights are small
                w_scale = float(np.max(abs_weights))
                if w_scale <= 0:
                    w_scale = 1.0
                threshold = self.cluster_distance * w_scale
            else:
                # 'absolute' (original behavior)
                threshold = self.cluster_distance

            # Cut tree at specified distance
            labels = fcluster(linkage_matrix, t=threshold,
                             criterion='distance')
        except Exception as e:
            # Fallback: single cluster
            print(f"Warning: Clustering failed for {layer_name}: {e}. Using single cluster.")
            labels = np.ones(len(weights_np), dtype=int)
        
        # Compute cluster centers
        unique_labels = np.unique(labels)
        cluster_centers = []
        for label in unique_labels:
            mask = labels == label
            center = np.mean(abs_weights[mask])
            cluster_centers.append(center)
        cluster_centers = np.array(cluster_centers)
        
        return {
            'cluster_centers': cluster_centers,
            'labels': labels - 1,  # Convert to 0-indexed
            'signs': signs,
            'original_shape': weights.shape,
            'n_clusters': len(cluster_centers),
            'n_params': len(weights_np)
        }
    
    def extract_structure(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Extract network structure by clustering all weight matrices.
        
        Args:
            verbose: Print clustering summary
            
        Returns:
            Dictionary mapping layer names to clustering results
        """
        weights = self.get_weights_for_clustering()
        structure = {}
        
        total_params = 0
        total_clusters = 0
        
        for name, weight_tensor in weights.items():
            if 'weight' in name:  # Only cluster weight matrices, not biases
                structure[name] = self.cluster_layer_weights(weight_tensor, name)
                total_params += structure[name]['n_params']
                total_clusters += structure[name]['n_clusters']
                
                if verbose:
                    info = structure[name]
                    compression = info['n_params'] / info['n_clusters'] if info['n_clusters'] > 0 else 0
                    print(f"  {name}: {info['n_params']} params -> {info['n_clusters']} clusters "
                          f"(compression: {compression:.1f}x)")
        
        if verbose:
            overall_compression = total_params / total_clusters if total_clusters > 0 else 0
            print(f"\n  Overall: {total_params} params -> {total_clusters} clusters "
                  f"(compression: {overall_compression:.1f}x)")
        
        return structure
    
    def build_relation_matrix(self, structure: Dict) -> Dict[str, torch.Tensor]:
        """
        Build relation matrices R for each layer.
        
        The relation matrix encodes:
        - Parameter sharing (same rows in R)
        - Sign reversal (rows with -1)
        - Permutation (swapped rows)
        
        Args:
            structure: Dictionary from extract_structure()
            
        Returns:
            Dictionary mapping layer names to relation matrices
        """
        relation_matrices = {}
        
        # Get device from model
        device = next(self.model.parameters()).device
        
        for layer_name, cluster_info in structure.items():
            labels = cluster_info['labels']
            signs = cluster_info['signs']
            n_clusters = len(cluster_info['cluster_centers'])
            n_params = len(labels)
            
            # Build relation matrix
            R = np.zeros((n_params, n_clusters))
            for i, (label, sign) in enumerate(zip(labels, signs)):
                R[i, label] = sign

            rank = int(np.linalg.matrix_rank(R)) if R.size else 0
            cluster_info['relation_rank'] = rank
            if rank < n_clusters:
                print(f"  WARNING: {layer_name} relation matrix rank "
                      f"{rank}/{n_clusters}; structure may be too restrictive.")
            
            # FIXED: Create tensor on correct device
            relation_matrices[layer_name] = torch.tensor(R, dtype=torch.float32, device=device)
        
        return relation_matrices
    
    def reconstruct_weights(self, structure: Dict, 
                           trainable_params: Optional[Dict[str, torch.Tensor]] = None
                           ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct full weight matrices from cluster centers and relation matrices.
        
        theta = R @ diag(trainable_cluster_centers)
        
        Args:
            structure: Dictionary from extract_structure()
            trainable_params: Optional new trainable parameters (cluster centers)
            
        Returns:
            Dictionary mapping layer names to reconstructed weight tensors
        """
        relation_matrices = self.build_relation_matrix(structure)
        reconstructed = {}
        
        # Get device from model
        device = next(self.model.parameters()).device
        
        for layer_name, cluster_info in structure.items():
            R = relation_matrices[layer_name]
            
            if trainable_params is not None and layer_name in trainable_params:
                centers = trainable_params[layer_name]
            else:
                centers = torch.tensor(cluster_info['cluster_centers'], 
                                       dtype=torch.float32, device=device)
            
            # Reconstruct: each parameter = sign * cluster_center
            flat_weights = R @ centers
            reconstructed[layer_name] = flat_weights.reshape(cluster_info['original_shape'])
        
        return reconstructed
    
    def get_compression_stats(self, structure: Dict) -> Dict:
        """
        Get compression statistics for the discovered structure.
        
        Returns:
            Dictionary with compression statistics
        """
        stats = {
            'layers': {},
            'total_original_params': 0,
            'total_cluster_centers': 0
        }
        
        for name, info in structure.items():
            stats['layers'][name] = {
                'original_params': info['n_params'],
                'cluster_centers': info['n_clusters'],
                'relation_rank': info.get('relation_rank', None),
                'compression_ratio': info['n_params'] / info['n_clusters'] if info['n_clusters'] > 0 else 0
            }
            stats['total_original_params'] += info['n_params']
            stats['total_cluster_centers'] += info['n_clusters']
        
        stats['overall_compression'] = (stats['total_original_params'] / 
                                        stats['total_cluster_centers'] 
                                        if stats['total_cluster_centers'] > 0 else 0)
        
        return stats


def discover_structure(model: nn.Module, cluster_distance: float = 0.1,
                       verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Convenience function for structure discovery.
    
    Args:
        model: Bayesian neural network
        cluster_distance: HAC clustering threshold
        verbose: Print summary
        
    Returns:
        Tuple of (structure_dict, relation_matrices_dict)
    """
    discovery = StructureDiscovery(model, cluster_distance)
    structure = discovery.extract_structure(verbose=verbose)
    relation_matrices = discovery.build_relation_matrix(structure)
    return structure, relation_matrices
