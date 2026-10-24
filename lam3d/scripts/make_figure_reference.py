"""Extra figure: distance to the analytic solution Sigma_exact = J_v S0 J_v^T.

Reads /tmp/cov_reference.json (eval_cov_reference.py). Each pixel metric gets
its own zoomed axis because the method spread (0.02 dB / 0.001 / 0.0007) is
invisible on a shared 0-24 axis. The bottom-right panel shows the per-vector
PSNR, which is the evidence that the tiny ordering is nevertheless stable.

Run:  python scripts/make_figure_reference.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = '/workspace/lam3d/docs/images'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})

MS = ['lam', 'tga', 'rot_sync', 'vertex_then_polar']
LBL = ['LAM (baseline)', 'TGA (Jacobian)', 'RotSync', 'VertexThenPolar']
SHORT = ['LAM', 'TGA', 'RotSync', 'VTP']
COL = ['#4C78A8', '#E45756', '#54A24B', '#B279A2']
R = json.load(open('/tmp/cov_reference.json'))
AGG = R['aggregate']
V = R['vectors']

fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
fig.subplots_adjust(hspace=0.75, wspace=0.32)
specs = [('psnr', 'PSNR vs analytic solution (dB, higher better)', axes[0][0]),
         ('ssim', 'SSIM vs analytic solution (higher better)', axes[0][1]),
         ('lpips', 'LPIPS vs analytic solution (lower better)', axes[1][0])]
for key, title, ax in specs:
    vals = [AGG[m][key] for m in MS]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 2.5, hi * 1e-4)
    ax.bar(range(4), vals, 0.6, color=COL)
    ax.set_xticks(range(4)); ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title + '\nspread ' + ('%.1e' % (hi - lo)), fontsize=8.5)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    for i, v in enumerate(vals):
        ax.text(i, v, ('%.2f' % v) if key == 'psnr' else ('%.4f' % v),
                ha='center', va='bottom', fontsize=6.5)

ax = axes[1][1]
xv = np.arange(len(V))
for i, m in enumerate(MS):
    ax.plot(xv, [R['per'][v][m]['psnr'] for v in V], 'o-', ms=3.5, lw=1.4,
            label=LBL[i], color=COL[i])
ax.set_xticks(xv); ax.set_xticklabels(V, rotation=25, fontsize=8)
ax.set_ylabel('PSNR vs analytic solution (dB)')
ax.set_title('Per-vector PSNR: the 0.02 dB ordering is stable\n'
             'across all vectors (LAM is below in 6/7)', fontsize=8.5)
ax.legend(fontsize=7, frameon=False, loc='lower right')

fig.savefig(os.path.join(OUT, 'cov_metric_reference.png'))
plt.close(fig)
print('saved cov_metric_reference.png')
for v, b in R['analytic_box'].items():
    print('  %-10s max-ratio p50 %.2f  >1.35 %.1f%%  >10 %.1f%%  max %.0f'
          % (v, b['max_axis_ratio_p50'], 100 * b['frac_gt_1p35'],
             100 * b['frac_gt_10'], b['max_axis_ratio_max']))