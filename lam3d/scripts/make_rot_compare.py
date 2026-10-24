#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渲染 LAM 原调制器 vs 旋转同步(ROT) vs TGA 的对照图（真实 checkpoint）。

输出 /tmp/rot_compare.png：
  行 = {B=0 静态, B=ones LAM 原调制器, B=ones ROT, B=ones TGA, |ROT-LAM| x8, |TGA-ROT| x8}
  列 = {正面, 绕 y 偏转 35 度}

运行（必须后台，约 150 s）：
    cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \\
      PYTHONPATH=/workspace/lam3d:/workspace/LAM \\
      /usr/local/miniconda3/envs/lam/bin/python scripts/make_rot_compare.py \\
      > /tmp/rot_compare.log 2>&1 &
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, rot_sync, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/rot_compare_job")
OUT = Path("/tmp/rot_compare.png")
SIZE = 512


def frame(sess, arkit, yaw):
    return arkit_live.render_frame(sess, arkit, yaw=yaw, size=SIZE)


def base_modulator(renderer):
    mod = tga_covariance._base_module().DynamicCovarianceModulator(
        renderer.flame_model)
    mod.to("cuda")
    return mod


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="rotcmp")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    ones = np.ones(52, dtype=np.float32)
    zeros = np.zeros(52, dtype=np.float32)

    rows = {k: [] for k in range(6)}
    for yaw in (0.0, 0.610865):          # 0 deg / 35 deg
        base = frame(sess, zeros, yaw)

        renderer._cov_modulator = base_modulator(renderer)
        lam = frame(sess, ones, yaw)

        renderer._cov_modulator = rot_sync.install(renderer, force=True)
        rot = frame(sess, ones, yaw)

        renderer._cov_modulator = tga_covariance.install(renderer, force=True)
        tga = frame(sess, ones, yaw)

        d_rl = np.clip(np.abs(rot.astype(np.float32) - lam.astype(np.float32)) * 8.0, 0, 255)
        d_tr = np.clip(np.abs(tga.astype(np.float32) - rot.astype(np.float32)) * 8.0, 0, 255)
        for k, v in enumerate((base, lam, rot, tga,
                               d_rl.astype(np.uint8), d_tr.astype(np.uint8))):
            rows[k].append(v)

    grid = np.concatenate([np.concatenate(rows[k], axis=1) for k in range(6)], axis=0)
    from PIL import Image as PILImage
    PILImage.fromarray(grid).save(OUT)
    print("saved", OUT, grid.shape, flush=True)

    b = rows[0][0].astype(np.float32)
    for k, name in ((1, "LAM"), (2, "ROT"), (3, "TGA")):
        d = np.abs(rows[k][0].astype(np.float32) - b)
        print("%s vs B=0: mean=%.5f changed=%.2f%%"
              % (name, d.mean(), 100.0 * (d.sum(-1) > 1e-6).mean()), flush=True)
    for k, a, bb in ((4, 2, 1), (5, 3, 2)):
        if k == 4:
            d = np.abs(rows[2][0].astype(np.float32) - rows[1][0].astype(np.float32))
            print("ROT vs LAM: mean=%.5f changed=%.2f%%"
                  % (d.mean(), 100.0 * (d.sum(-1) > 1e-6).mean()), flush=True)
        else:
            d = np.abs(rows[3][0].astype(np.float32) - rows[2][0].astype(np.float32))
            print("TGA vs ROT: mean=%.5f changed=%.2f%%"
                  % (d.mean(), 100.0 * (d.sum(-1) > 1e-6).mean()), flush=True)


if __name__ == "__main__":
    main()
