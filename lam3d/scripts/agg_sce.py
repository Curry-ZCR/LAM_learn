#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把分片运行的 sce.json 合并成一份，并输出协议 §4/§6 需要的聚合表。"""
import json, sys
from pathlib import Path
import numpy as np

M = ["lam", "tga", "rot_sync", "vertex_then_polar"]
SH = {"lam": "M-LAM", "tga": "M-TGA", "rot_sync": "M-RS", "vertex_then_polar": "M-VTP"}
parts = sys.argv[1:] or ["/tmp/cov_v2_b/sce.json", "/tmp/cov_v2_b_prev.json"]
out = {"assets": [], "views": ["front", "yaw35"], "vectors": ["ones", "bigSmile"],
       "per": {}, "variance": {}}
for p in parts:
    if not Path(p).is_file():
        print("skip missing", p); continue
    d = json.loads(Path(p).read_text())
    for a, c in d["per"].items():
        out["per"][a] = c
        if a not in out["assets"]:
            out["assets"].append(a)
    out["variance"].update(d["variance"])
Path("/tmp/cov_v2/sce_merged.json").write_text(json.dumps(out, indent=1))
print("merged assets:", out["assets"], "conditions:", len(out["variance"]))

# --- 聚合：每方法每指标，跨 (资产 x 向量 x 视角) 的分布 ---
per = out["per"]
conds = [(a, k) for a in per for k in per[a]]
print("\nn_conditions =", len(conds), " n_assets =", len(per))
print("\n%-8s %-14s %10s %10s %10s %10s" % ("metric", "method", "mean", "p50", "p90", "worst"))
for metric in ("iou", "halo", "holes", "psnr_vs_mesh", "ssim_vs_mesh", "edge_mass"):
    for m in M:
        v = np.array([per[a][k][m][metric] for a, k in conds], np.float64)
        print("%-8s %-14s %10.5f %10.5f %10.5f %10.5f"
              % (metric, SH[m], v.mean(), np.median(v), np.percentile(v, 90), v.max() if metric != "iou" else v.min()))

# --- 配对差（同条件，方法一/方法二 为基准）+ 以资产为簇的 bootstrap ---
def cl_boot(d, nb=4000, seed=11):
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(nb, n))
    bm = d[idx].mean(1)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    return dict(mean=float(d.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                dz=float(d.mean() / sd) if sd > 1e-15 else float("nan"), n=int(n))

print("\n=== 配对效应（每资产先平均，再以资产为簇 bootstrap；基准=rot_sync）===")
stats = {}
for metric in ("iou", "halo", "holes", "psnr_vs_mesh", "ssim_vs_mesh"):
    for m in M:
        if m == "rot_sync":
            continue
        d = np.array([np.mean([per[a][k][m][metric] - per[a][k]["rot_sync"][metric]
                               for k in per[a]]) for a in per])
        s = cl_boot(d)
        stats["%s|%s_vs_rot_sync" % (metric, m)] = s
        print("%-14s %-6s d=%+.5f [%+.5f,%+.5f] dz=%+.2f n=%d"
              % (metric, SH[m], s["mean"], s["lo"], s["hi"], s["dz"], s["n"]))
print("\n=== 方法间跨度 / 视角间跨度（判别信噪比，越大越有判别力）===")
for metric in ("iou", "halo", "holes", "psnr_vs_mesh", "ssim_vs_mesh"):
    ratios = []
    for a in per:
        for vec in ("ones", "bigSmile"):
            k1, k2 = "%s|front" % vec, "%s|yaw35" % vec
            if k1 not in per[a] or k2 not in per[a]:
                continue
            ms = [per[a][k1][m][metric] for m in M]
            vs = [per[a][k2][m][metric] for m in M]
            ratios.append((max(ms) - min(ms)) / (abs(np.mean(ms) - np.mean(vs)) + 1e-12))
    print("  %-14s median %.4f  max %.4f  (>1 表示视角间差反而小于方法间差)" % (metric, np.median(ratios), np.max(ratios)))

# 方差分解按分区
print("\n=== 方差分解 E_var/E_sys（8~24 个条件平均）===")
V = out["variance"]
keys = sorted({k for c in V.values() for k in c})
for k in keys:
    rows = [V[c][k] for c in V if k in V[c]]
    r = np.mean([x["E_var"] / max(x["E_sys"], 1e-20) for x in rows])
    print("  %-15s ratio %.3e   (n_cond=%d)" % (k, r, len(rows)))
Path("/tmp/cov_v2/agg.json").write_text(json.dumps(stats, indent=1))
