# plot_framework_diagram.py -- Figure 1: GBSD framework overview
"""Pure matplotlib diagram. No training data needed. Usage: python plot_framework_diagram.py"""
import os; import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = './results/figures'

def draw_framework():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 5.5)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.5); ax.axis('off')
    stages = [
        (1.5, 2.8, 'Teacher\nPINN', '#4A90D9'),
        (4.2, 2.8, 'Dense Bayesian\nStudent\n(MC Dropout)', '#E8913A'),
        (7.0, 4.0, 'Structure\nDiscovery\n(HAC)', '#6CB44C'),
        (7.0, 1.5, 'Empirical UQ\nCalibration', '#9B59B6'),
        (10.0, 2.8, 'Structured\nCandidate', '#E74C3C'),
        (12.8, 3.8, 'Guarded\nFinal Source', '#2C3E50'),
        (12.8, 1.8, 'Calibrated\nUncertainty', '#2C3E50'),
    ]
    for x,y,txt,fc in stages:
        bbox = FancyBboxPatch((x-1.1,y-0.7), 2.2, 1.4, boxstyle='round,pad=0.15',
                              facecolor=fc, edgecolor='#333', lw=1.5, alpha=0.9)
        ax.add_patch(bbox)
        ax.text(x, y, txt, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
    arrows = [((2.6,2.8),(3.1,2.8)), ((5.3,3.3),(5.9,3.7)), ((5.3,2.3),(5.9,1.8)),
              ((8.1,3.7),(8.9,3.1)), ((8.1,1.8),(8.9,2.5)),
              ((11.1,3.3),(11.7,3.6)), ((11.1,2.3),(11.7,2.0))]
    for (x1,y1),(x2,y2) in arrows:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    for x,lbl in [(1.5,'Stage 1'),(4.2,'Stage 2'),(7.0,'Stage 3'),(10.0,'Stage 4'),(12.8,'Output')]:
        ax.text(x, 5.2, lbl, ha='center', fontsize=8, color='gray')
    ax.annotate('', xy=(4.2,5.0), xytext=(1.5,5.0),
                arrowprops=dict(arrowstyle='->', color='#E8913A', lw=1.5, ls='--'))
    ax.text(2.85, 5.2, 'Teacher-to-student distillation', fontsize=7, ha='center', color='#E8913A', fontstyle='italic')
    fig.suptitle('Guarded Bayesian Structured Distillation (GBSD)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    for ext in ['png','pdf']:
        fig.savefig(f'{FIG_DIR}/fig_bayesian_framework.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig); print(f'  Fig 1 → {FIG_DIR}/fig_bayesian_framework.{{png,pdf}}')

if __name__ == '__main__': draw_framework()
