# -*- coding: utf-8 -*-
"""逐面极分解 + Markley 聚合 vs 对照方案的直接 A/B（真实捕获张量）。

回答报告里的两件事在代码里是否真的生效，以及代价/收益：
  1) "必须先逐面极分解再聚合，不能先把 J 平均再分解"
  2) "聚合必须对 q/-q 的符号双覆盖免疫"

四组对照（全部在 /tmp/rot_capture.pt 的真实 x0/xB 上）：
  A  face_markley   逐面极分解 -> Markley 主特征向量      （当前默认，rot_sync.py face 分支）
  B  face_linear    逐面极分解 -> 半球对齐后线性平均      （上一版实现）
  C  face_w0        逐面极分解 -> 全翻到 w>=0 后线性平均  （最朴素的免符号做法）
  D  vertex_then    先把 J 平均到顶点 -> 再极分解         （报告说的错误做法）

判据：以 A 为基线看 B/C/D 的角度偏差；D 是否真的会在下颌/嘴角跳变。

运行:
  PYTHONPATH=/workspace/lam3d:/workspace/LAM /usr/local/miniconda3/envs/lam/bin/python scripts/offline_ab_polar_markley.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import rot_sync as RS  # noqa: E402
from lam3d.core.tga_covariance import matrix_to_quat  # noqa: E402

DEG = 180.0 / 3.141592653589793
CAP = "/tmp/rot_capture.pt"


class _Shim:
    pass


def build_mod():
    cap = torch.load(CAP, weights_only=False)
    m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
    torch.nn.Module.__init__(m)
    sh = _Shim()
    sh.faces = cap["faces"].cuda()
    sh.faces_flat = sh.faces.reshape(-1)
    m.base = sh
    m.eps = 1e-12
    return m, cap


def q_of(J):
    return F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)


def ang_deg(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEG


def scatter_avg(qf, area, flat, n_vert):
    w = qf.repeat_interleave(3, 0) * (area / 3.0).repeat_interleave(3, 0)[:, None]
    acc = torch.zeros(n_vert, 4, device=qf.device, dtype=w.dtype)
    acc.index_add_(0, flat, w)
    return acc


def path_A(m, x0, xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    qf = q_of(Jf)[0]
    q, conf, wsum = m._markley_mean(qf[None], area[None], x0.shape[0])
    return F.normalize(q[0], dim=-1), conf[0], qf, area


def path_B(m, x0, xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    qf = q_of(Jf)[0]
    q_ref = q_of(m.vertex_jacobians(x0[None], xB, Jf=Jf[None]))[0]
    flat = m.base.faces_flat
    pref = q_ref[flat].reshape(-1, 3, 4).sum(1).repeat_interleave(3, 0)
    goto = qf.repeat_interleave(3, 0)
    sign = torch.where((goto * pref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    q = F.normalize(scatter_avg(sign * goto, area, flat, x0.shape[0]), dim=-1)
    return q, q_ref, qf, area


def path_C(m, x0, xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    qf = q_of(Jf)[0]
    sign = torch.where(qf[:, :1] < 0, -1.0, 1.0)
    flat = m.base.faces_flat
    q = F.normalize(scatter_avg(sign * qf, area, flat, x0.shape[0]), dim=-1)
    return q, None, qf, area


def path_D(m, x0, xB):
    Jf = m.face_jacobians(x0[None], xB)
    Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    return q_of(Jv)[0], None, None, None


def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def pct(t):
    t = t.flatten()
    return [float(torch.quantile(t, q)) for q in (0.5, 0.9, 0.99, 1.0)]


def main():
    m, cap = build_mod()
    names = cap["names"]
    print("capture N=%d F=%d variants=%s" % (cap["x0"].shape[0], cap["faces"].shape[0], names), flush=True)
    x0 = cap["x0"].cuda()
    n_face = cap["faces"].shape[0]

    with torch.no_grad():
        for i, nm in enumerate(names):
            xB = cap["xB"][i].cuda()
            qA, confA, qf, area = path_A(m, x0, xB)
            qB, q_ref, _, _ = path_B(m, x0, xB)
            qC, _, _, _ = path_C(m, x0, xB)
            qD, _, _, _ = path_D(m, x0, xB)
            print()
            print("=== %s ===" % nm, flush=True)
            for tag, q in (("B linear+ref", qB), ("C w0-only", qC), ("D vertex-then-polar", qD)):
                d = ang_deg(q, qA)
                p50, p90, p99, mx = pct(d)
                print("  %-20s mean=%8.3f p50=%7.3f p90=%8.3f p99=%9.3f max=%9.3f  >30deg=%d/%d"
                      % (tag, float(d.mean()), p50, p90, p99, mx, int((d > 30).sum()), d.numel()), flush=True)
            print("  conf(A) p1=%.3f p50=%.3f  <0.9: %d"
                  % (float(torch.quantile(confA, 0.01)), float(torch.quantile(confA, 0.5)), int((confA < 0.9).sum())), flush=True)
            dD = ang_deg(qD, qA)
            bad = dD > 30
            if int(bad.sum()) > 0:
                cb = confA[bad]
                print("  D 偏差>30 度的 %d 个顶点: conf p50=%.3f min=%.3f"
                      % (int(bad.sum()), float(torch.quantile(cb, 0.5)), float(cb.min())), flush=True)

        print()
        print("== 聚合段耗时（GPU Event, ms）==")
        xB = cap["xB"][0].cuda()
        area = m.face_areas(x0[None])[0]
        Jf = m.face_jacobians(x0[None], xB)[0]
        qf = q_of(Jf)
        flat = m.base.faces_flat
        N = x0.shape[0]
        q_ref = q_of(m.vertex_jacobians(x0[None], xB, Jf=Jf[None]))[0]
        pref = q_ref[flat].reshape(-1, 3, 4).sum(1).repeat_interleave(3, 0)
        goto = qf.repeat_interleave(3, 0)
        signB = torch.where((goto * pref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        signC = torch.where(qf[:, :1] < 0, -1.0, 1.0)
        qw = signC * qf
        tJv = bench(lambda: q_of(m.vertex_jacobians(x0[None], xB, Jf=Jf[None])))
        premult = signB * goto * (area / 3.0).repeat_interleave(3, 0)[:, None]
        premultC = qw * (area / 3.0).repeat_interleave(3, 0)[:, None]
        print("  A markley_mean               %7.3f" % bench(lambda: m._markley_mean(qf[None], area[None], N)))
        print("  B 参考 polar(Jv)+m2q         %7.3f" % tJv)
        print("  B 参考后再线性平均            %7.3f" % bench(lambda: F.normalize(scatter_avg(signB * goto, area, flat, N), dim=-1)))
        print("  C w>=0 线性平均               %7.3f" % bench(lambda: F.normalize(scatter_avg(qw, area, flat, N), dim=-1)))
        print("  D vertex_jacobians+polar      %7.3f" % tJv)
        print("  (qf 本身: %7.3f ms)" % bench(lambda: q_of(Jf)))


if __name__ == "__main__":
    main()
