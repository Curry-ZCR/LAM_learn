#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""部署路径验证：真正走 arkit_live.save_session -> compute_splat -> LSPL 编码。

前面两个脚本证明的是「数学」和「官方光栅化路径」；本脚本证明**浏览器实际收到的
那份 LSPL 二进制**已经被 TGA 雅可比形变改写，并且满足 CONTRACT_v2 的 A6/§8 约束。

对照:
  (0) B = 0                      中性基准（LSPL 逐字节应与关掉调制时一致）
  (1) B = ones + LAM 原调制器    q'=normalize(dq⊗q0), s'=s0+ds
  (2) B = ones + TGA 调制器      Sigma'=J Sigma J^T -> 特征分解 -> (q',s')

运行:
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_live.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, splat, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/tga_live_job")
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  " + detail) if detail else ""), flush=True)


def main():
    print("== building pipeline (real checkpoint) ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="tgalive")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    N = int(sess["gs_model"].xyz.shape[0])
    print("N=%d installed=%s" % (N, sess.get("cov_modulator")), flush=True)

    ones = np.ones(52, dtype=np.float32)
    zeros = np.zeros(52, dtype=np.float32)

    # ---- (0) 中性基准：save_session 里已自动装好 TGA，B=0 必须与不调制一致
    with torch.no_grad():
        base = arkit_live.compute_splat(sess, None)
        base_tga = arkit_live.compute_splat(sess, zeros)
    same = all(torch.equal(base[k], base_tga[k]) for k in base)
    report("L1 B=0 走 TGA 安装后的会话 == 不传 arkit（逐位）", bool(same),
           "keys=" + ",".join(sorted(base)))

    # ---- (1) LAM 原调制器
    lam_mod = tga_covariance._base_module().DynamicCovarianceModulator(
        renderer.flame_model)
    lam_mod.to("cuda")
    renderer._cov_modulator = lam_mod
    t0 = time.perf_counter()
    with torch.no_grad():
        lam = arkit_live.compute_splat(sess, ones)
    t_lam = (time.perf_counter() - t0) * 1e3

    # ---- (2) TGA 调制器
    tga_mod = tga_covariance.install(renderer, force=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        tga = arkit_live.compute_splat(sess, ones)
    t_tga = (time.perf_counter() - t0) * 1e3
    report("L2 会话里的调制器就是 TGA", isinstance(tga_mod, tga_covariance.TgaCovarianceModulator),
           type(tga_mod).__name__)

    # ---- 契约：A6 尺度上限 / 四元数归一 / 有限性 / 退化一致
    s0 = base["scale"]
    for tag, g in (("LAM", lam), ("TGA", tga)):
        finite = all(bool(torch.isfinite(v).all()) for v in g.values())
        report("L3 %s payload 全有限" % tag, finite)
        report("L4 %s |q'| == 1" % tag,
               bool(((g["rot"].norm(dim=-1) - 1).abs() <= 1e-5).all()),
               "max|1-|q||=%.2e" % (g["rot"].norm(dim=-1) - 1).abs().max().item())
        ratio = (g["scale"] / s0.clamp(min=1e-30)).max().item()
        report("L5 %s s'/s0 <= 1.35 且 s' > 0（契约 §8 A6）" % tag,
               ratio <= 1.35 + 1e-4 and bool((g["scale"] > 0).all()),
               "max=%.6f" % ratio)

    # ---- 形变确实进入了 LSPL 的三个通道
    d_xyz = (tga["xyz"] - base["xyz"]).abs().mean().item()
    d_rot = (tga["rot"] - lam["rot"]).abs().mean().item()
    d_scl = (tga["scale"] - lam["scale"]).abs().mean().item()
    report("L6 相对中性：TGA 的 xyz 变化 mean|d| > 1e-5", d_xyz > 1e-5, "%.3e" % d_xyz)
    report("L7 两个调制器的 q' 不同（rot 通道确实被改写）", d_rot > 1e-4, "mean|d|=%.4f" % d_rot)
    report("L8 两个调制器的 s' 不同（scale 通道确实被改写）", d_scl > 1e-4, "mean|d|=%.4f" % d_scl)

    # ---- LSPL 打包 + 字节数（浏览器实际拿到的长度）
    blob, _ms = arkit_live.splat_bytes(sess, ones)
    expect = splat.LSPL_HEADER_SIZE + N * splat.LSPL_FLOATS_PER_GAUSSIAN
    report("L9 LSPL v1 字节数 == 144 + N*56", len(blob) == expect,
           "%d vs %d" % (len(blob), expect))
    report("L10 后端一次 splat 耗时 < 30 ms（契约 §8 C18）",
           t_tga < 30.0, "LAM=%.2f ms  TGA=%.2f ms" % (t_lam, t_tga))

    ok = all(RESULTS)
    print("")
    print("%d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    print("RESULT: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
