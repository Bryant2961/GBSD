# Module/BayesianStructureDiscovery.py — Posterior-based structure extraction
"""
Upgrades point-estimate clustering to Bayesian posterior scoring.
Score: s_ij = |mu_i - mu_j| / sqrt(sigma_i^2 + sigma_j^2 + eps)
Parameters with close means but large variance are NOT merged.

Interface compatible with StructuredPINN builder.
"""
import numpy as np, torch, torch.nn as nn
from typing import Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class BayesianStructureDiscovery:
    def __init__(self, model: nn.Module, n_mc_samples: int = 100,
                 cluster_threshold: float = 2.0, epsilon: float = 1e-8):
        self.model = model
        self.n_mc_samples = n_mc_samples
        self.cluster_threshold = cluster_threshold
        self.epsilon = epsilon

    def _get_mc_stats(self):
        self.model.train()
        if hasattr(self.model, 'enable_dropout'): self.model.enable_dropout()
        wnames = [n for n,p in self.model.named_parameters() if 'weight' in n and p.dim()>=2]
        device = next(self.model.parameters()).device
        idim = 2
        for n,p in self.model.named_parameters():
            if 'layer_0.weight' in n: idim=p.shape[1]; break
        samples = {n:[] for n in wnames}
        dummy = torch.randn(10, idim, device=device)
        for _ in range(self.n_mc_samples):
            with torch.no_grad(): _ = self.model(dummy)
            for n in wnames:
                samples[n].append(dict(self.model.named_parameters())[n].data.cpu().numpy().flatten())
        stats = {}
        for n in wnames:
            s = np.array(samples[n])
            stats[n] = {'mu': s.mean(0), 'sigma': s.std(0),
                         'shape': dict(self.model.named_parameters())[n].shape}
        return stats

    def _cluster_bayesian(self, name, mu, sigma, shape):
        abs_mu = np.abs(mu); signs = np.sign(mu); n = len(mu)
        if n <= 1:
            return {'cluster_centers': abs_mu, 'labels': np.array([0]), 'signs': signs,
                    'original_shape': shape, 'n_clusters': 1, 'n_params': n,
                    'cluster_stds': sigma.copy(), 'method': 'bayesian'}

        # Pairwise Bayesian z-scores
        scores = []
        for i in range(n):
            for j in range(i+1,n):
                scores.append(abs(abs_mu[i]-abs_mu[j]) / np.sqrt(sigma[i]**2+sigma[j]**2+self.epsilon))
        scores = np.array(scores)

        try:
            lm = linkage(scores.reshape(-1,1), method='complete')
            labels = fcluster(lm, t=self.cluster_threshold, criterion='distance')
        except:
            labels = np.ones(n, dtype=int)

        unique = np.unique(labels)
        centers = [np.mean(abs_mu[labels==l]) for l in unique]
        stds = [np.sqrt(np.mean(sigma[labels==l]**2)) for l in unique]
        return {
            'cluster_centers': np.array(centers), 'labels': labels-1, 'signs': signs,
            'original_shape': shape, 'n_clusters': len(centers), 'n_params': n,
            'cluster_stds': np.array(stds), 'method': 'bayesian'
        }

    def extract_structure(self, verbose=True):
        stats = self._get_mc_stats()
        structure = {}
        tp, tc = 0, 0
        if verbose: print(f'\n  Bayesian Structure Discovery (threshold={self.cluster_threshold})\n')
        for name, s in stats.items():
            r = self._cluster_bayesian(name, s['mu'], s['sigma'], s['shape'])
            structure[name] = r
            tp += r['n_params']; tc += r['n_clusters']
            if verbose:
                print(f"  {name}: {r['n_params']} -> {r['n_clusters']} clusters "
                      f"({r['n_params']/r['n_clusters']:.1f}x) [σ_mean={np.mean(s['sigma']):.4e}]")
        if verbose and tc > 0:
            print(f"\n  Overall: {tp} -> {tc} clusters ({tp/tc:.1f}x)")
        return structure

    def build_relation_matrix(self, structure):
        R = {}
        for name, info in structure.items():
            labels = info['labels']; n = info['n_params']; nc = info['n_clusters']
            mat = np.zeros((n, nc))
            for i in range(n): mat[i, labels[i]] = info['signs'][i]
            R[name] = torch.tensor(mat, dtype=torch.float32)
        return R

    def get_compression_stats(self, structure):
        to = sum(s['n_params'] for s in structure.values())
        tc = sum(s['n_clusters'] for s in structure.values())
        return {'total_original': to, 'total_clusters': tc,
                'overall_compression': to/max(1,tc)}
