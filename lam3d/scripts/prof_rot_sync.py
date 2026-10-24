#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐段剖析 RotSyncModulator 在真实会话上的开销（定位 R13 超预算的部分）。

做法：不重建 animation_forward 的调用链（容易与 renderer 的实现漂移），
而是给 rot_sync 模块里的纯函数和 Modulator 的方法打点，然后跑真实的
arkit_live.compute_splat(sess, ones)，每段用 cuda 同步计时累加。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, rot_sync  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/prof_rot_job")
WARM = 4
REP = 12

STATS = {}


class timed:
    """记录累计耗时的包装器；作为类属性时按描述符绑定 self。"""

    def __init__(self, label, fn):
        self.label = label
        self.fn = fn

    def _record(self, t0):
        torch.cuda.synchronize()
        stat = STATS.get(self.label)
        if stat is None:
            stat = STATS[self.label] = [0.0, 0]
        stat[0] += time.perf_counter() - t0
        stat[1] += 1

    def __call__(self, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.fn(*a, **kw)
        self._record(t0)
        return out

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        bound = self.fn.__get__(obj, objtype)

        class _Bound:
            def __call__(inner, *a, **kw):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = bound(*a, **kw)
                self._record(t0)
                return out

        return _Bound()


def main():
    print("== building pipeline ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="profrot")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    mod = rot_sync.install(renderer, force=True)
    N = int(sess["gs_model"].xyz.shape[0])
    F = int(mod.faces.shape[0])
    print("N=%d F=%d installed=%s" % (N, F, sess.get("cov_modulator")), flush=True)

    ones = np.ones(52, dtype=np.float32)

    orig = {}
    for name in ("polar_rotation", "matrix_to_quat", "quat_multiply",
                 "quat_angle", "slerp_identity", "identity_quat", "install"):
        if hasattr(rot_sync, name):
            orig[name] = getattr(rot_sync, name)
            if name != "install":
                setattr(rot_sync, name, timed(name, orig[name]))
    for name in ("face_areas", "face_jacobians", "vertex_jacobians",
                 "rotation_field", "apply", "delta_scale"):
        key = "." + name
        orig[key] = getattr(rot_sync.RotSyncModulator, name)
        setattr(rot_sync.RotSyncModulator, name, timed(key, orig[key]))

    with torch.no_grad():
        for _ in range(WARM):
            arkit_live.compute_splat(sess, ones)
    STATS.clear()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(REP):
            arkit_live.compute_splat(sess, ones)
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / REP * 1e3

    print("\n== compute_splat 端到端 = %.2f ms/次 (rep=%d) ==" % (total, REP), flush=True)
    rows = []
    for k, (t, n) in STATS.items():
        rows.append((k, t / max(n, 1) * 1e3, n / float(REP)))
    rows.sort(key=lambda r: -r[1])
    covered = 0.0
    for name, ms, per in rows:
        covered += ms * per
        print("  %-22s %8.3f ms/call   x%.1f/iter" % (name, ms, per), flush=True)
    print("  打点覆盖合计 = %.2f ms；未覆盖 = %.2f ms" % (covered, total - covered), flush=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(REP):
            arkit_live.compute_splat(sess, None)
    print("  B=0 静态路径 = %.2f ms/次" % ((time.perf_counter() - t0) / REP * 1e3), flush=True)

    for name, fn in orig.items():
        if name.startswith("."):
            setattr(rot_sync.RotSyncModulator, name[1:], fn)
        elif name != "install":
            setattr(rot_sync, name, fn)


if __name__ == "__main__":
    main()
