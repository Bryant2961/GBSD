# plot_transfer_generalization.py — Figure 9: Cross-problem method comparison
"""Reads per-case metrics CSVs. Usage: python plot_transfer_generalization.py"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG_DIR = './results/figures'; MET_DIR = './results/metrics'

def plot():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    for f in os.listdir(MET_DIR):
        if f.endswith('_metrics.csv') and not f.startswith('calibration') and not f.startswith('parameter'):
            try: rows.append(pd.read_csv(f'{MET_DIR}/{f}'))
            except: pass
    if not rows: print('  No metrics CSVs found'); return
    df = pd.concat(rows, ignore_index=True)
    mcols = [c for c in ['l2_error','mae','rmse','nll','coverage_95','avg_interval_width_95','calibration_error']
             if c in df.columns and df[c].notna().any()]
    if not mcols: return
    cases = df['case'].unique(); methods = df['method'].unique()
    nm = len(mcols)
    fig, axes = plt.subplots(1, nm, figsize=(4*nm, 4.5))
    if nm == 1: axes = [axes]
    x = np.arange(len(cases)); w = 0.8/len(methods)
    colors = plt.cm.Set2(np.linspace(0,0.8,len(methods)))
    for mi, metric in enumerate(mcols):
        for j, method in enumerate(methods):
            vals = [df[(df['case']==c)&(df['method']==method)][metric].mean()
                    if len(df[(df['case']==c)&(df['method']==method)])>0 else 0 for c in cases]
            off = (j - len(methods)/2 + 0.5)*w
            axes[mi].bar(x+off, vals, w, label=method if mi==0 else '', color=colors[j], alpha=0.85)
        axes[mi].set_xticks(x); axes[mi].set_xticklabels(cases, rotation=30, ha='right', fontsize=8)
        axes[mi].set_title(metric.replace('_',' ').title(), fontsize=10); axes[mi].grid(True, axis='y', alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle('Cross-Problem Comparison', fontsize=12, y=1.02); fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_transfer_generalization_uncertainty.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    df.to_csv(f'{MET_DIR}/transfer_generalization_metrics.csv', index=False)
    # Deduplicate: keep last occurrence of each (case, method, seed) combo
    df_dedup = df.drop_duplicates(subset=['case', 'method', 'seed'], keep='last')
    if len(df_dedup) < len(df):
        print(f'  Deduplicated: {len(df)} → {len(df_dedup)} rows')
        df_dedup.to_csv(f'{MET_DIR}/transfer_generalization_metrics.csv', index=False)
    print('  Fig 9 saved')

if __name__ == '__main__': plot()
