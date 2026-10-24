"""Regenerate the four metric figures in docs/images/ from the JSON reports.

Inputs (produced by the eval scripts in this directory):
  /tmp/cov_metrics.json   -> eval_cov_methods.py
  /tmp/cov_offline.json   -> eval_cov_offline.py
  /tmp/cov_dynamics.json  -> eval_cov_dynamics.py

All in-figure text is English on purpose: the remote host has no CJK font
(fc-list is empty for CJK), so Chinese labels render as tofu boxes.

Run:  python scripts/make_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/workspace/lam3d/docs/images"
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

MS  = ["lam", "tga", "rot_sync", "vertex_then_polar"]
LBL = ["LAM (baseline)", "TGA (Jacobian)", "RotSync", "VertexThenPolar"]
COL = ["#4C78A8", "#E45756", "#54A24B", "#B279A2"]

M  = json.load(open("/tmp/cov_metrics.json"))
O  = json.load(open("/tmp/cov_offline.json"))
D  = json.load(open("/tmp/cov_dynamics.json"))
PRES = ["smile","bigSmile","laugh","surprise","angry","sad","kiss","mouthOpen","blink"]

def leg(ax, **kw):
    ax.legend(fontsize=7.5, frameon=False, **kw)

# ---- Fig 1: offline decisive metrics -----------------------------------------
VECS = ["ones","half","extreme","rand","rand_sparse"]
VLAB = ["all-1.0","all-0.5","all-2.0","random","sparse rand"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
x = np.arange(len(VECS)); w = 0.2
for i, m in enumerate(MS):
    axes[0].bar(x + (i-1.5)*w, [O["per"][m][v]["rigid_err"]["p50"] for v in VECS],
                w, label=LBL[i], color=COL[i])
axes[0].set_xticks(x); axes[0].set_xticklabels(VLAB, fontsize=8)
axes[0].set_ylabel("one-ring rigid error p50 (deg)")
axes[0].set_title("Rotation fidelity vs one-ring vertex motion\n(lower is better)", fontsize=9)
leg(axes[0])
for i, m in enumerate(MS):
    axes[1].bar(x + (i-1.5)*w, [100*O["per"][m][v]["false_move_frac"] for v in VECS],
                w, label=LBL[i], color=COL[i])
axes[1].set_xticks(x); axes[1].set_xticklabels(VLAB, fontsize=8)
axes[1].set_ylabel("false-motion fraction of gaussians (%)")
axes[1].set_title("Rotated >20 deg while local motion <5 deg\n(TGA wins the wrong prize here)", fontsize=9)
axes[1].set_yscale("log")
fig.savefig(os.path.join(OUT, "cov_metric_offline.png")); plt.close(fig)

# ---- Fig 2: dynamics ---------------------------------------------------------
RP = ["jawOpen","mouthSmileLeft","browInnerUp","eyeBlinkLeft"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
x = np.arange(len(RP))
for i, m in enumerate(MS):
    axes[0].bar(x + (i-1.5)*w, [D["ramps"][r][m]["rot_roughness"] for r in RP],
                w, label=LBL[i], color=COL[i])
axes[0].set_xticks(x); axes[0].set_xticklabels(RP, rotation=12, fontsize=8)
axes[0].set_ylabel("2nd difference / mean step")
axes[0].set_title("Temporal roughness over a 24-step ramp\n(lower is smoother)", fontsize=9)
leg(axes[0])
for i, m in enumerate(MS):
    axes[1].bar(x + (i-1.5)*w, [D["ramps"][r][m]["rot_step_mean_deg"] for r in RP],
                w, label=LBL[i], color=COL[i])
axes[1].set_xticks(x); axes[1].set_xticklabels(RP, rotation=12, fontsize=8)
axes[1].set_ylabel("mean rotation per step (deg)")
axes[1].set_title("Absolute rotation magnitude\n(TGA moves ~10x more per step)", fontsize=9)
fig.savefig(os.path.join(OUT, "cov_metric_dynamics.png")); plt.close(fig)

# ---- Fig 3: pixel metrics are flat ------------------------------------------
def avg(key, view="front"):
    out = {}
    for m in MS:
        vals = [M["pixel"][v][view][m][key] for v in PRES if view in M["pixel"][v]]
        out[m] = sum(vals)/len(vals)
    return out

fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.0))
specs = [("psnr_vs_mesh", "PSNR vs mesh pseudo-GT (dB)", 3),
         ("ssim_vs_mesh", "SSIM vs mesh pseudo-GT", 5),
         ("lpips_vs_mesh", "LPIPS vs mesh pseudo-GT", 5)]
for ax, (k, t, dec) in zip(axes, specs):
    d = avg(k); vals = [d[m] for m in MS]
    lo, hi = min(vals), max(vals)
    pad = max((hi-lo)*4.0, 10**(-dec))
    ax.bar(range(4), vals, 0.6, color=COL)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["LAM","TGA","RotSync","VTP"], rotation=20, fontsize=8)
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_title("%s\nmethod spread = %.1e" % (t, hi-lo), fontsize=8.5)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
fig.savefig(os.path.join(OUT, "cov_metric_pixel_flat.png")); plt.close(fig)

# ---- Fig 4: identity (eig) + spatial jitter ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
x = np.arange(len(PRES))
for i, m in enumerate(MS):
    axes[0].semilogy(x, [M["geometry"][v][m]["eig_rel_err_p99"] for v in PRES],
                     "o-", ms=3.5, lw=1.4, label=LBL[i], color=COL[i])
axes[0].set_xticks(x); axes[0].set_xticklabels(PRES, rotation=38, fontsize=7)
axes[0].set_ylabel("p99 rel. error of eigenvalues")
axes[0].set_title("Axis-length preservation\n(RotSync/VTP are exactly 0 by construction)", fontsize=9)
leg(axes[0], loc="lower right")
for i, m in enumerate(MS):
    axes[1].plot(x, [M["geometry"][v][m]["ring_jitter_deg"]["mean"] for v in PRES],
                 "o-", ms=3.5, lw=1.4, label=LBL[i], color=COL[i])
axes[1].set_xticks(x); axes[1].set_xticklabels(PRES, rotation=38, fontsize=7)
axes[1].set_ylabel("mean one-ring rotation spread (deg)")
axes[1].set_title("Spatial coherence across neighbouring vertices\n(TGA is consistently rougher)", fontsize=9)
fig.savefig(os.path.join(OUT, "cov_metric_identity.png")); plt.close(fig)

print("saved 4 figures")
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(os.path.join(OUT, f)))

