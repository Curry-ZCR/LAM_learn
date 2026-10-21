#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渲染 LAM 原调制器 vs TGA 雅可比调制器的对照图（真实 checkpoint）。

输出 /tmp/tga_compare.png：
  行 = {B=0 静态, B=ones LAM 原调制器, B=ones TGA, |TGA - LAM| x8}
  列 = {正面, 绕 y 偏转 35 度}

运行:
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/make_tga_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/tga_compare_job")
OUT = Path("/tmp/tga_compare.png")
SIZE = 512


def frame(sess, arkit, yaw):
    return arkit_live.render_frame(sess, arkit, yaw=yaw, size=SIZE)


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="tgacmp")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    ones = np.ones(52, dtype=np.float32)
    zeros = np.zeros(52, dtype=np.float32)

    rows = {0: [], 1: [], 2: [], 3: []}
    for yaw in (0.0, 0.610865):          # 0 deg / 35 deg
        base = frame(sess, zeros, yaw)
        lam_mod = tga_covariance._base_module().DynamicCovarianceModulator(
            renderer.flame_model)
        lam_mod.to("cuda")
        renderer._cov_modulator = lam_mod
        lam = frame(sess, ones, yaw)
        renderer._cov_modulator = tga_covariance.install(renderer, force=True)
        tga = frame(sess, ones, yaw)
        diff = np.clip(np.abs(tga.astype(np.float32) - lam.astype(np.float32)) * 8.0,
                       0, 255).astype(np.uint8)
        for k, v in enumerate((base, lam, tga, diff)):
            rows[k].append(v)

    grid = np.concatenate([np.concatenate(rows[k], axis=1) for k in range(4)], axis=0)
    from PIL import Image as PILImage
    PILImage.fromarray(grid).save(OUT)
    print("saved", OUT, grid.shape, flush=True)

    # 定量：改了几成像素
    for k, name in enumerate(("B=0", "LAM", "TGA", "diff")):
        pass
    b = rows[0][0].astype(np.float32)
    for k, name in ((1, "LAM"), (2, "TGA")):
        d = np.abs(rows[k][0].astype(np.float32) - b)
        print("%s vs B=0: mean=%.5f changed=%.1f%%"
              % (name, d.mean(), 100.0 * (d.sum(-1) > 1e-6).mean()), flush=True)
    d = np.abs(rows[2][0].astype(np.float32) - rows[1][0].astype(np.float32))
    print("TGA vs LAM: mean=%.5f changed=%.1f%%"
          % (d.mean(), 100.0 * (d.sum(-1) > 1e-6).mean()), flush=True)


if __name__ == "__main__":
    main()
