#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实 checkpoint 上的 TGA 协方差形变端到端验证（渲染对比，无 mock）。

对照三组，全部用同一个 canonical 高斯 + 同一台相机 + 同一条官方渲染路径
（renderer.forward_single_batch，即 gs_renderer.py:526-559 的 forward_single_view）：

  (0) B = 0            静态基准
  (1) B = ones, LAM 原调制器    面内 2x2 旋转 + 各向同性面积缩放
  (2) B = ones, TGA 调制器      Sigma' = J Sigma J^T，特征分解回写 (q', s')

验证：
  R1  两组都与静态基准不同（形变确实进了渲染）
  R2  两组之间也不同（TGA 带来了新的形变，不是同一张图）
  R3  TGA 路径逐点 s'/s0 <= 1.35（契约 A6）且 s' > 0
  R4  TGA 路径 |q'| == 1
  R5  全程无 NaN/Inf
  R6  TGA 调制耗时在同一量级（< 30 ms，契约 §8 C18 的预算）

运行:
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_render.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/tga_render_job")
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             ("  " + detail) if detail else ""))


def render(renderer, sess, arkit):
    """走官方 forward_single_batch（= 部署路径），返回 [H,W,3] float。"""
    gs = renderer.animate_gs_model(sess["gs_model"], sess["query_points"],
                                   sess["flame_data"],
                                   arkit_params=None if arkit is None else arkit)[0]
    dev = gs.xyz.device
    c2w = torch.from_numpy(np.asarray(sess["c2w"], dtype=np.float32))[None].to(dev)
    intr = torch.from_numpy(np.asarray(sess["intr"], dtype=np.float32))[None].to(dev)
    h, w = int(sess["render_hw"][0]), int(sess["render_hw"][1])
    bg = sess["bg"].reshape(1, 3).to(dev)
    out = renderer.forward_single_batch([gs], c2w, intr, h, w, bg)
    return out["comp_rgb"][0].clamp(0.0, 1.0), gs


def main():
    print("== building pipeline (real checkpoint) ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="tgacov")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    gs0 = sess["gs_model"]
    N = int(gs0.xyz.shape[0])
    print("N=%d render_hw=%s installed=%s"
          % (N, sess["render_hw"], sess.get("cov_modulator")), flush=True)

    ones = torch.ones(52, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        img0, _ = render(renderer, sess, None)

        # ---- (1) LAM 原调制器 ----
        lam_mod = tga_covariance._base_module().DynamicCovarianceModulator(
            renderer.flame_model)
        lam_mod.to("cuda")
        renderer._cov_modulator = lam_mod
        t0 = time.perf_counter()
        img1, gs1 = render(renderer, sess, ones)
        torch.cuda.synchronize()
        t_lam = (time.perf_counter() - t0) * 1e3

        # ---- (2) TGA 调制器（同一个 renderer，热切换）----
        tga_mod = tga_covariance.install(renderer, force=True)
        assert isinstance(tga_mod, tga_covariance.TgaCovarianceModulator)
        t0 = time.perf_counter()
        img2, gs2 = render(renderer, sess, ones)
        torch.cuda.synchronize()
        t_tga = (time.perf_counter() - t0) * 1e3

    d0a = (img1 - img0).abs()
    d0b = (img2 - img0).abs()
    dab = (img2 - img1).abs()
    report("R1a LAM 调制改变了画面 vs B=0", bool(d0a.mean() > 1e-4),
           "mean=%.5f max=%.4f changed=%.1f%%"
           % (d0a.mean(), d0a.max(), 100.0 * (d0a > 1e-6).float().mean()))
    report("R1b TGA 调制改变了画面 vs B=0", bool(d0b.mean() > 1e-4),
           "mean=%.5f max=%.4f changed=%.1f%%"
           % (d0b.mean(), d0b.max(), 100.0 * (d0b > 1e-6).float().mean()))
    report("R2 TGA 与 LAM 原调制器渲染结果不同（形变确实变了）",
           bool(dab.mean() > 1e-5),
           "mean=%.5f max=%.4f changed=%.1f%%"
           % (dab.mean(), dab.max(), 100.0 * (dab > 1e-6).float().mean()))

    # ---- 属性级检查 ----
    s0 = gs0.scaling.float()
    for tag, g in (("LAM", gs1), ("TGA", gs2)):
        s1 = g.scaling.float()
        q1 = g.rotation.float()
        finite = all(bool(torch.isfinite(getattr(g, k)).all())
                     for k in ("xyz", "rotation", "scaling", "opacity", "shs"))
        report("R5 %s 无 NaN/Inf" % tag, finite)
        report("R4 %s |q'| == 1" % tag,
               bool(((q1.norm(dim=-1) - 1).abs() <= 1e-5).all()),
               "max|1-|q||=%.3e" % (q1.norm(dim=-1) - 1).abs().max().item())
        if tag == "TGA":
            ratio = s1 / s0.clamp(min=1e-30)
            report("R3 TGA s'/s0 <= 1.35 且 > 0", bool(ratio.max() <= 1.35 + 1e-4)
                   and bool((s1 > 0).all()),
                   "max=%.6f min=%.4f" % (ratio.max(), ratio.min()))

    # ---- Sigma 的相对变化量：TGA 能表达方向性拉伸，LAM 只能各向同性缩放，
    # 所以同样的 ARKit 输入下 TGA 的协方差变化应当更大（用 median，避免被少数
    # 近退化的高斯主导）。
    def sig(g):
        R = tga_covariance.quat_to_matrix(g.rotation.float()).double()
        return R @ torch.diag_embed(g.scaling.double() ** 2) @ R.transpose(-1, -2)

    S0, S1, S2 = sig(gs0), sig(gs1), sig(gs2)

    def rel(a, b):
        return ((a - b).norm(dim=(-1, -2)) / b.norm(dim=(-1, -2)).clamp(min=1e-30))

    d_lam, d_tga = rel(S1, S0), rel(S2, S0)
    report("R6 协方差变化量 median(TGA) > median(LAM)（方向性 vs 各向同性）",
           bool(d_tga.median() > d_lam.median()),
           "median ||dSigma||/||Sigma||: LAM=%.4f TGA=%.4f (p95 %.3f / %.3f)"
           % (d_lam.median(), d_tga.median(),
              d_lam.quantile(0.95), d_tga.quantile(0.95)))

    report("R7 调制耗时 < 30 ms（契约 §8 C18）",
           t_tga < 30.0 and t_lam < 30.0,
           "LAM=%.2f ms  TGA=%.2f ms  (含 512^2 光栅化)" % (t_lam, t_tga))

    ok = all(RESULTS)
    print("")
    print("%d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    print("RESULT: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
