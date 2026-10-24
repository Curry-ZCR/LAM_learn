#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""旋转同步的性能基准 + R14 诊断（真实会话，无打点、独立预热）。

回答三个问题：
  1. 干净的 t_rot 是多少（不做任何 monkeypatch，避免 cuda 同步把固定开销算进去）。
  2. 增量按段拆开各占多少（离线重用同一份 x0/xB 反复计时）。
  3. R14 为什么 FAIL：gain=0 时到底哪些高斯的四元数与 B=0 不同。

运行（必须后台，>2 min）:
  cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/bench_rot_sync.py \
    > /tmp/bench_rot.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, rot_sync, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/bench_rot_job")
WARM = 6
REP = 30


def bench(fn, warm=WARM, rep=REP):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.mean()), float(np.percentile(ts, 90))


def gpu_bench(fn, warm=WARM, rep=REP):
    """纯 GPU 时间（Event 计时，不受 CPU 启动抖动影响）。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(rep):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.mean()), float(np.percentile(ts, 90))


def main():
    print("== build ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="benchrot")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    N = int(sess["gs_model"].xyz.shape[0])
    F = int(renderer.flame_model.faces_up.shape[0]) if hasattr(renderer.flame_model, "faces_up") else -1
    ones = np.ones(52, dtype=np.float32)
    zeros = np.zeros(52, dtype=np.float32)
    print("N=%d F=%d" % (N, F), flush=True)

    with torch.no_grad():
        base = arkit_live.compute_splat(sess, zeros)

    # ------------------------------------------------ 端到端
    lam_mod = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    lam_mod.to("cuda")
    renderer._cov_modulator = lam_mod
    m_lam = bench(lambda: arkit_live.compute_splat(sess, ones))

    tga_mod = tga_covariance.install(renderer, force=True)
    m_tga = bench(lambda: arkit_live.compute_splat(sess, ones))

    rot_mod = rot_sync.install(renderer, force=True)
    m_rot = bench(lambda: arkit_live.compute_splat(sess, ones))

    # 静态短路（B=0）作为基线
    m_base = bench(lambda: arkit_live.compute_splat(sess, zeros))

    print("\n== end-to-end compute_splat (median / mean / p90 ms) ==", flush=True)
    for tag, m in (("B=0 静态", m_base), ("LAM", m_lam), ("TGA", m_tga), ("ROT", m_rot)):
        print("  %-10s %7.2f  %7.2f  %7.2f" % (tag, m[0], m[1], m[2]), flush=True)
    print("  增量 ROT-B0 = %.2f ms, TGA-B0 = %.2f ms" % (m_rot[0] - m_base[0], m_tga[0] - m_base[0]),
          flush=True)

    # ------------------------------------------------ 抓一份 x0/xB
    captured = {}
    orig_rf = rot_sync.RotSyncModulator.rotation_field

    def spy(self, x0, xB):
        captured.setdefault("x0", x0.detach().clone())
        captured.setdefault("xB", xB.detach().clone())
        return orig_rf(self, x0, xB)

    rot_sync.RotSyncModulator.rotation_field = spy
    with torch.no_grad():
        arkit_live.compute_splat(sess, ones)
    rot_sync.RotSyncModulator.rotation_field = orig_rf
    x0 = captured["x0"][:1].contiguous()
    xB = captured["xB"][:1].contiguous()
    mod = rot_sync.install(renderer, force=True)
    print("\ncaptured x0 %s xB %s" % (tuple(x0.shape), tuple(xB.shape)), flush=True)

    # ------------------------------------------------ 分段（离线重用）
    with torch.no_grad():
        area = mod.face_areas(x0)
        Jf = mod.face_jacobians(x0, xB)
        Rf10 = rot_sync.polar_rotation(Jf, iters=10)
        Rf8 = rot_sync.polar_rotation(Jf, iters=8)
        Jv = mod.vertex_jacobians(x0, xB, Jf=Jf)
        qf = rot_sync.matrix_to_quat(Rf10)

    sect = []
    sect.append(("face_areas", gpu_bench(lambda: mod.face_areas(x0))))
    sect.append(("face_jacobians", gpu_bench(lambda: mod.face_jacobians(x0, xB))))
    sect.append(("polar(Jf) iters=10", gpu_bench(lambda: rot_sync.polar_rotation(Jf, iters=10))))
    sect.append(("polar(Jf) iters=8", gpu_bench(lambda: rot_sync.polar_rotation(Jf, iters=8))))
    sect.append(("polar(Jv) iters=10", gpu_bench(lambda: rot_sync.polar_rotation(Jv, iters=10))))
    sect.append(("vertex_jacobians", gpu_bench(lambda: mod.vertex_jacobians(x0, xB, Jf=Jf))))
    sect.append(("matrix_to_quat(39904)", gpu_bench(lambda: rot_sync.matrix_to_quat(Rf10))))
    sect.append(("matrix_to_quat(20018)", gpu_bench(lambda: rot_sync.matrix_to_quat(Jv))))
    sect.append(("rotation_field(face)", gpu_bench(lambda: mod.rotation_field(x0, xB))))
    print("\n== sections (GPU Event median / mean / p90 ms) ==", flush=True)
    for tag, m in sect:
        print("  %-24s %7.3f  %7.3f  %7.3f" % (tag, m[0], m[1], m[2]), flush=True)

    # 两次 polar 的差（iters 8 vs 10）
    d = (rot_sync.polar_rotation(Jf, iters=8) - Rf10).abs().max().item()
    print("  |polar iters8 - iters10| max = %.3e" % d, flush=True)

    # ------------------------------------------------ R14 诊断
    print("\n== R14 diagnosis (gain=0) ==", flush=True)
    g0 = rot_sync.install(renderer, force=True, rotation_gain=0.0)
    with torch.no_grad():
        out0 = arkit_live.compute_splat(sess, ones)
    q0, qb = out0["rot"], base["rot"]
    neq = (q0 != qb).any(dim=-1)
    print("  differing gaussians: %d / %d" % (int(neq.sum()), q0.shape[0]), flush=True)
    print("  max|dq| = %.3e" % (q0 - qb).abs().max().item(), flush=True)
    if bool(neq.any()):
        idx = neq.nonzero(as_tuple=False).squeeze(-1)[:5]
        for i in idx.tolist():
            print("   [%d] got %s  base %s" % (i, q0[i].tolist(), qb[i].tolist()), flush=True)
    # rotation_field 直接产出的 delta 是否严格单位
    with torch.no_grad():
        dq, diag = g0.rotation_field(x0, xB)
    ident = torch.zeros_like(dq)
    ident[..., 0] = 1.0
    bad = (dq != ident).any(dim=-1)
    print("  delta_q not exactly identity: %d / %d, max|d| = %.3e"
          % (int(bad.sum()), dq.shape[0] * dq.shape[1], (dq - ident).abs().max().item()), flush=True)
    if bool(bad.any()):
        b = bad.nonzero(as_tuple=False)[:5]
        for r, i in b.tolist():
            print("   [%d,%d] %s" % (r, i, dq[r, i].tolist()), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
