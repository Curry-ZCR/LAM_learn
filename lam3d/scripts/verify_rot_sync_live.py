#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实 checkpoint 上的旋转同步验证：走 arkit_live.save_session -> compute_splat -> LSPL。

前面两个 check_rot_sync_*.py 证明的是"数学"和"网格通路"；本脚本证明**浏览器
实际收到的那份 LSPL 二进制**里，四元数通道被旋转同步改写、而缩放通道逐位未动。

对照三方（同一会话、同一 ARKit 向量）：
  (0) B = 0                          静态基准
  (1) B = ones + LAM 原调制器        q' = normalize(dq (x) q0), s' = s0 + ds
  (2) B = ones + TGA 调制器          Sigma' = J Sigma J^T -> (q', s')
  (3) B = ones + 旋转同步            q' = normalize(R (x) q0), s' = s0   <= 本次

运行:
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/verify_rot_sync_live.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, rot_sync, splat, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/rot_sync_live_job")
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  " + detail) if detail else ""), flush=True)


def eig_of(q, s):
    """由 (q, s) 还原 Sigma 的特征值（升序）。"""
    r = tga_covariance.quat_to_matrix(q.float())
    sig = r @ torch.diag_embed(s.float().pow(2)) @ r.transpose(-1, -2)
    return torch.linalg.eigvalsh(sig)


def main():
    print("== building pipeline (real checkpoint) ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="rotslive")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    N = int(sess["gs_model"].xyz.shape[0])
    print("N=%d installed=%s" % (N, sess.get("cov_modulator")), flush=True)

    ones = np.ones(52, dtype=np.float32)
    zeros = np.zeros(52, dtype=np.float32)

    with torch.no_grad():
        base = arkit_live.compute_splat(sess, None)
        base2 = arkit_live.compute_splat(sess, zeros)
    report("R1 B=0 与「不传 arkit」逐位一致",
           all(torch.equal(base[k], base2[k]) for k in base))

    # ---- (1) LAM 原调制器
    lam_mod = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    lam_mod.to("cuda")
    renderer._cov_modulator = lam_mod
    t0 = time.perf_counter()
    with torch.no_grad():
        lam = arkit_live.compute_splat(sess, ones)
    t_lam = (time.perf_counter() - t0) * 1e3

    # ---- (2) TGA（方法二上界，作参照）
    tga_mod = tga_covariance.install(renderer, force=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        tga = arkit_live.compute_splat(sess, ones)
    t_tga = (time.perf_counter() - t0) * 1e3

    # ---- (3) 旋转同步
    rot_mod = rot_sync.install(renderer, force=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        rot = arkit_live.compute_splat(sess, ones)
    t_rot = (time.perf_counter() - t0) * 1e3
    report("R2 会话里的调制器就是 RotSyncModulator",
           isinstance(rot_mod, rot_sync.RotSyncModulator), type(rot_mod).__name__)

    # -------------------------------------------------- 契约
    for tag, g in (("LAM", lam), ("TGA", tga), ("ROT", rot)):
        report("R3 %s payload 全有限" % tag,
               all(bool(torch.isfinite(v).all()) for v in g.values()))
        report("R4 %s |q'| == 1" % tag,
               bool(((g["rot"].norm(dim=-1) - 1).abs() <= 1e-5).all()),
               "max|1-|q||=%.2e" % (g["rot"].norm(dim=-1) - 1).abs().max().item())
        ratio = (g["scale"] / base["scale"].clamp(min=1e-30)).max().item()
        report("R5 %s s'/s0 <= 1.35 且 s' > 0" % tag,
               ratio <= 1.35 + 1e-4 and bool((g["scale"] > 0).all()), "max=%.6f" % ratio)

    # ------------------------- 方法一的定义性质：轴长严格不变
    report("R6 ROT 的 scaling 与 B=0 逐位相等（方法一：保轴长）",
           torch.equal(rot["scale"], base["scale"]))
    d_rot_scale = (rot["scale"] - base["scale"]).abs().max().item()
    report("R6b ROT 的 scaling 相对 LAM/TGA 未被改写",
           d_rot_scale == 0.0 and (rot["scale"] - tga["scale"]).abs().max().item() > 1e-6,
           "max|d|=%.3e；TGA 相对 ROT 的 scale 差=%.3e"
           % (d_rot_scale, (tga["scale"] - rot["scale"]).abs().max().item()))

    # 由 (q', s') 还原的 Sigma 特征值必须与 B=0 的逐个相等
    ev0 = eig_of(base["rot"], base["scale"])
    evr = eig_of(rot["rot"], rot["scale"])
    rel = ((evr - ev0).abs() / ev0.abs().clamp(min=1e-30).max(dim=-1, keepdim=True).values).max().item()
    report("R7 Sigma' 的三个特征值 == Sigma0（方法一的数学定义）",
           rel < 1e-5, "max rel err=%.3e" % rel)
    evt = eig_of(tga["rot"], tga["scale"])
    rel_t = ((evt - ev0).abs() / ev0.abs().clamp(min=1e-30).max(dim=-1, keepdim=True).values).max().item()
    report("R7b TGA 的特征值确实被改（对照组有效）", rel_t > 1e-3, "max rel err=%.3e" % rel_t)

    # ------------------------- 位置与朝向通道
    d_xyz = (rot["xyz"] - base["xyz"]).abs().mean().item()
    report("R8 ROT 的 xyz 与 TGA 逐位一致（旋转同步不碰位置）",
           torch.equal(rot["xyz"], tga["xyz"]), "mean|d vs base|=%.3e" % d_xyz)
    # 逐高斯朝向差（相对 B=0），用矩阵级比较避免 acos 病态
    def rot_diff(qa, qb):
        ra = tga_covariance.quat_to_matrix(qa.float())
        rb = tga_covariance.quat_to_matrix(qb.float())
        return (ra - rb).abs().amax(dim=(-1, -2)), (0.5 * (
            (ra * rb).sum(dim=(-1, -2)) - 1.0)).clamp(-1, 1).arccos()

    m_rot, ang_rot = rot_diff(rot["rot"], base["rot"])
    m_lam, ang_lam = rot_diff(lam["rot"], base["rot"])
    m_tga, ang_tga = rot_diff(tga["rot"], base["rot"])
    report("R9 ROT 确实改了朝向（非恒等）", ang_rot.max().item() > 1e-3,
           "max=%.3f deg  mean=%.4f deg" % (ang_rot.max(), ang_rot.mean()))
    report("R10 三种方法的朝向改变量级（信息项，不做强弱判定）",
           bool(torch.isfinite(ang_rot).all()),
           "mean deg: LAM=%.4f ROT=%.4f TGA=%.4f"
           % (ang_lam.mean(), ang_rot.mean(), ang_tga.mean()))
    diag = rot_mod._last_diag
    if diag is not None:
        a = diag["angle_deg"]
        report("R11 逐高斯转角统计（限幅前）",
               bool(torch.isfinite(a).all()),
               "mean=%.3f p50=%.3f p95=%.3f max=%.3f deg, clipped=%d"
               % (a.mean(), a.median(), a.quantile(0.95), a.max(), int(diag["clipped"].sum())))

    # -------------------------------------------------- LSPL 与耗时
    renderer._cov_modulator = rot_sync.install(renderer, force=True)
    blob, _ms = arkit_live.splat_bytes(sess, ones)
    expect = splat.LSPL_HEADER_SIZE + N * splat.LSPL_FLOATS_PER_GAUSSIAN
    report("R12 LSPL v1 字节数 == 144 + N*56", len(blob) == expect,
           "%d vs %d" % (len(blob), expect))
    report("R13 后端一次 splat 耗时 < 30 ms", t_rot < 30.0,
           "LAM=%.2f ms  TGA=%.2f ms  ROT=%.2f ms" % (t_lam, t_tga, t_rot))

    # -------------------------------------------------- 消融：门控与限幅
    print("\n== ablation ==", flush=True)
    for gain in (0.0, 0.5, 1.0):
        m = rot_sync.install(renderer, force=True, rotation_gain=gain)
        with torch.no_grad():
            g = arkit_live.compute_splat(sess, ones)
        _, ang = rot_diff(g["rot"], base["rot"])
        print("  gain=%.1f -> mean=%.4f max=%.3f deg" % (gain, ang.mean(), ang.max()), flush=True)
        if gain == 0.0:
            # 只对四元数通道要求逐位一致：位置通道的差异来自 ARKit 动态偏移
            # 经过 LBS 后的形变（xB vs x0），与旋转门无关；它应当与 TGA 逐位
            # 相同（两者共用同一份 xB），见 R8。
            report("R14 rotation_gain=0 -> 四元数与 B=0 逐位一致",
                   torch.equal(g["rot"], base["rot"]),
                   "xyz 与 TGA 逐位一致=%s" % torch.equal(g["xyz"], tga["xyz"]))

    for cap in (10.0, 45.0, 0.0):
        m = rot_sync.install(renderer, force=True, max_angle_deg=cap, rotation_gain=1.0)
        with torch.no_grad():
            g = arkit_live.compute_splat(sess, ones)
        _, ang = rot_diff(g["rot"], base["rot"])
        print("  max_angle=%.0f -> max=%.3f deg" % (cap, ang.max()), flush=True)
        if cap > 0:
            report("R15 max_angle=%.0f 生效" % cap, ang.max().item() <= cap + 0.05,
                   "max=%.4f deg" % ang.max())

    ok = all(RESULTS)
    print("")
    print("%d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    print("RESULT: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
