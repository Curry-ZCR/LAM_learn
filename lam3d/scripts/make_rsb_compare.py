#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RSB vs RotSync 渲染视觉对照(fix 生效区:嘴唇/下巴/鼻翼)。

输出 /tmp/rsb_compare.png:
  行 = {B0, preset-RotSync, preset-RSB, |RSB-ROT|x8}
  列 = {正面, yaw35} x {ones, extreme} 两个 preset 块纵向拼接

运行(后台,约 300s):
    cd /workspace/lam3d && nohup env PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/make_rsb_compare.py \
      > /tmp/rsb_compare.log 2>&1 &
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, rot_sync, rsb  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/rsb_compare_job")
OUT = Path("/workspace/lam3d/docs/images/rsb_compare.png")
SIZE = 512
PRESETS = {
    "ones": np.ones(52, dtype=np.float32),
    "extreme": None,  # 复用 p99 实验的 extreme 向量构造
}

def extreme_vec():
    v = np.zeros(52, dtype=np.float32)
    # 与 s-b/s-c 实验同源的逐通道常值压力向量(0.8 幅度全通道)
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
           26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
    for i in idx: v[i] = 0.8
    return v

def frame(sess, arkit, yaw):
    return arkit_live.render_frame(sess, arkit, yaw=yaw, size=SIZE)

def main():
    PRESETS["extreme"] = extreme_vec()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="rsbcmp")
    sess = pipe.arkit_session
    renderer = sess["renderer"]

    blocks = []
    stats = []
    for pname, vec in PRESETS.items():
        rows = {k: [] for k in range(4)}
        for yaw in (0.0, 0.610865):
            renderer._cov_modulator = rot_sync.install(renderer, force=True)
            rot = frame(sess, vec, yaw)
            renderer._cov_modulator = rsb.install(renderer, force=True,
                                                  normal_cap_deg=25.0, coherence_gate=0.3)
            rsb_img = frame(sess, vec, yaw)
            renderer._cov_modulator = rot_sync.install(renderer, force=True)
            base = frame(sess, np.zeros(52, dtype=np.float32), yaw)
            d = np.abs(rsb_img.astype(np.float32) - rot.astype(np.float32))
            diff8 = np.clip(d * 8.0, 0, 255).astype(np.uint8)
            for k, v in enumerate((base, rot, rsb_img, diff8)):
                rows[k].append(v)
            dd = d
            stats.append("%s yaw=%.0f: |RSB-ROT| mean=%.5f changed=%.2f%% p99=%.1f"
                         % (pname, yaw * 57.2958, dd.mean(),
                            100.0 * (dd.sum(-1) > 1e-6).mean(),
                            float(np.percentile(dd, 99))))
        blocks.append(np.concatenate([np.concatenate(rows[k], axis=1) for k in range(4)], axis=0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate(blocks, axis=0)
    from PIL import Image as PILImage
    PILImage.fromarray(grid).save(OUT)
    print("saved", OUT, grid.shape, flush=True)
    for s in stats: print(s, flush=True)

if __name__ == "__main__":
    main()
