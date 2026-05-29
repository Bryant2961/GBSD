# plot_posterior_structure.py — Figure 6: Posterior structure extraction
"""
Loads Bayesian student, estimates per-weight posterior stats via MC sampling,
performs Bayesian clustering, and generates visualization figures.

Usage: python plot_posterior_structure.py --case Laplace --device cpu
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = './results/figures'


def extract_structure_from_posterior(mu, sigma, threshold=2.0, epsilon=1e-8):
    """
    Bayesian structure extraction.
    Merge score: s_ij = |mu_i - mu_j| / sqrt(sigma_i^2 + sigma_j^2 + eps)
    Greedy agglomerative: merge clusters with score < threshold.

    Returns: labels, cluster_means, cluster_stds
    """
    n = len(mu)
    abs_mu = np.abs(mu)
    labels = np.arange(n)

    merged = True
    while merged:
        merged = False
        unique = np.unique(labels)
        for li in unique:
            for lj in unique:
                if li >= lj: continue
                idx_i = np.where(labels == li)[0]
                idx_j = np.where(labels == lj)[0]
                mu_i = np.mean(abs_mu[idx_i])
                mu_j = np.mean(abs_mu[idx_j])
                sig_i = np.sqrt(np.mean(sigma[idx_i] ** 2))
                sig_j = np.sqrt(np.mean(sigma[idx_j] ** 2))
                s = abs(mu_i - mu_j) / np.sqrt(sig_i**2 + sig_j**2 + epsilon)
                if s < threshold:
                    labels[labels == lj] = li
                    merged = True

    unique = np.unique(labels)
    relabel = {old: new for new, old in enumerate(unique)}
    labels = np.array([relabel[l] for l in labels])
    nc = len(unique)
    c_means = np.array([np.mean(abs_mu[labels == k]) for k in range(nc)])
    c_stds = np.array([np.sqrt(np.mean(sigma[labels == k] ** 2)) for k in range(nc)])
    return labels, c_means, c_stds


def get_mc_weight_stats(model, n_samples=100):
    """Estimate per-weight posterior mean and std via MC forward passes."""
    import torch
    device = next(model.parameters()).device
    model.train()
    if hasattr(model, 'enable_dropout'):
        model.enable_dropout()

    weight_names = [n for n, p in model.named_parameters() if 'weight' in n and p.dim() >= 2]
    input_dim = 2
    for n, p in model.named_parameters():
        if 'layer_0.weight' in n: input_dim = p.shape[1]; break

    samples = {n: [] for n in weight_names}
    dummy = torch.randn(10, input_dim, device=device)
    for _ in range(n_samples):
        with torch.no_grad():
            _ = model(dummy)
        for n in weight_names:
            p = dict(model.named_parameters())[n]
            samples[n].append(p.data.cpu().numpy().flatten())

    stats = {}
    for n in weight_names:
        s = np.array(samples[n])
        stats[n] = {'mu': s.mean(axis=0), 'sigma': s.std(axis=0),
                     'shape': dict(model.named_parameters())[n].shape}
    return stats


def plot_posterior_structure(case, device_str='cpu'):
    os.makedirs(FIG_DIR, exist_ok=True)
    import torch

    model_path = f'./Results/{case}_EXP/Models/{case}_EXP_Student_MCDropout_student.pth'
    if not os.path.isfile(model_path):
        print(f'  No Bayesian student model for {case}'); return

    device = torch.device(device_str)
    from Module.Student_MCDropout import Net
    from utils.posterior_predict import _load_config, _build_layer
    config = _load_config(case)
    layer = _build_layer(config, student=False)
    dr = float(config.get('dropout_rate', 0.15))
    model = Net(layer, dropout_rate=dr).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    stats = get_mc_weight_stats(model, n_samples=100)

    # ---- Figure 6a/6b: per-layer posterior mean and std ----
    n_layers = len(stats)
    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 6))
    if n_layers == 1: axes = axes.reshape(2, 1)

    for idx, (name, s) in enumerate(stats.items()):
        mu, sigma = s['mu'], s['sigma']
        order = np.argsort(np.abs(mu))
        axes[0, idx].bar(range(len(mu)), mu[order], color='steelblue', alpha=0.7, width=1.0)
        axes[0, idx].set_title(f'{name}\nPosterior Mean', fontsize=9)
        axes[1, idx].bar(range(len(sigma)), sigma[order], color='coral', alpha=0.7, width=1.0)
        axes[1, idx].set_title('Posterior Std', fontsize=9)

    fig.suptitle(f'{case} — Posterior Weight Statistics', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ['png', 'pdf']:
        fig.savefig(f'{FIG_DIR}/fig_posterior_mean_evolution_{case.lower()}.{ext}', dpi=200)
        fig.savefig(f'{FIG_DIR}/fig_posterior_variance_evolution_{case.lower()}.{ext}', dpi=200)
    plt.close(fig)

    # ---- Figure 6c: Bayesian structure extraction on largest layer ----
    biggest = max(stats.keys(), key=lambda k: len(stats[k]['mu']))
    mu = stats[biggest]['mu']
    sigma = stats[biggest]['sigma']
    labels, c_means, c_stds = extract_structure_from_posterior(np.abs(mu), sigma, threshold=2.0)
    nc = len(np.unique(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cmap = plt.cm.Set3(np.linspace(0, 1, max(nc, 3)))

    # Panel 1: parameters colored by cluster
    for k in range(nc):
        mask = labels == k
        axes[0].scatter(np.where(mask)[0], np.abs(mu[mask]), c=[cmap[k % len(cmap)]],
                        s=8, label=f'C{k}' if k < 8 else None)
    axes[0].set_xlabel('Parameter index'); axes[0].set_ylabel('|Posterior mean|')
    axes[0].set_title(f'Bayesian Clustering ({nc} clusters)')
    if nc <= 8: axes[0].legend(fontsize=7, ncol=2)

    # Panel 2: cluster centers with std error bars
    axes[1].errorbar(range(nc), c_means, yerr=2*c_stds, fmt='o', capsize=3, color='steelblue', ms=5)
    axes[1].set_xlabel('Cluster ID'); axes[1].set_ylabel('Center ± 2σ')
    axes[1].set_title('Cluster Centers + Credible Intervals')

    # Panel 3: std per parameter, colored by cluster
    for k in range(nc):
        mask = labels == k
        axes[2].scatter(np.where(mask)[0], sigma[mask], c=[cmap[k % len(cmap)]], s=8)
    axes[2].set_xlabel('Parameter index'); axes[2].set_ylabel('Posterior std')
    axes[2].set_title('Posterior Std by Cluster')

    fig.suptitle(f'{case} — {biggest}: Bayesian Structure Extraction', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ['png', 'pdf']:
        fig.savefig(f'{FIG_DIR}/fig_posterior_structure_extraction_{case.lower()}.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 6: {case} ({nc} Bayesian clusters in {biggest})')

    # Figure-only diagnostic: do not write posterior-structure cache into Results/raw.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='Laplace')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    plot_posterior_structure(args.case, args.device)
