# -*- coding: utf-8 -*-
"""半球参考的三种取法：精度 + 速度。

A: 现方案 —— 参考 = polar(Jv) 的顶点四元数（要额外一次 Jv 极分解，~2.5 ms）
B: 廉价 —— 参考 = 每个面自己，先翻到 w>=0 再面积加权（零额外 polar）
C: 混合 —— 参考 = 该面三个顶点**未做极分解**的退化占位（不实现，仅记录）
"""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
from lam3d.core import rot_sync as RS
from lam3d.core.tga_covariance import matrix_to_quat

cap = torch.load("/tmp/rot_capture.pt", weights_only=False)
x0 = cap["x0"].cuda(); faces = cap["faces"].cuda(); names = cap["names"]
DEGI = 180.0 / 3.14159265

class Shim: pass
def make_mod():
    m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
    torch.nn.Module.__init__(m)
    m.base = Shim(); m.base.faces = faces; m.base.faces_flat = faces.reshape(-1)
    m.rotation_gain = 1.0; m.max_angle_deg = 45.0; m.deadzone_deg = 0.0
    m.average = "face"; m.temporal_smoothing = 0.0; m.eps = 1e-12
    m._prev_delta = None; m._last_diag = None
    return m
m = make_mod()

def quats_of(J):
    return F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)

def avg_to_vertex(qf, pref_face):
    """qf [F,4], pref_face [F,3,4] 参考（每个面对应三个顶点的参考四元数）。"""
    fa = m.face_areas(x0[None])[0]                       # [F]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)             # [F,3,4]
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    nrm = acc.norm(dim=-1, keepdim=True)
    return F.normalize(acc, dim=-1), nrm.squeeze(-1)

def angle_between(p, q):
    d = (p * q).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

def run(xB):
    Jf = m.face_jacobians(x0[None], xB)
    qf = quats_of(Jf)[0]          # 注意取 [0]：否则 qv_ref[faces] 会广播成 [F,3,N,4]（38 GB）
    # A: 参考 = polar(Jv)
    Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    qv_ref = quats_of(Jv)[0]
    qA, nA = avg_to_vertex(qf, qv_ref[faces])
    # B: 参考 = 自身 w>=0
    qf_pos = torch.where(qf[:, 0:1] < 0, -qf, qf)
    qB, nB = avg_to_vertex(qf_pos, qf_pos.unsqueeze(1).expand(-1, 3, -1))
    # 未平均的粗糙基线（只取第一个相邻面）作参照
    return (qA, nA), (qB, nB), qf

with torch.no_grad():
    for i, nm in enumerate(names):
        xB = cap["xB"][i:i+1].cuda()
        (qA, nA), (qB, nB), qf = run(xB)
        valid = nA > 1e-8
        d = angle_between(qA, qB)[valid]
        # 参考方案自身的差异：用 qf 直接取任一相邻面 vs 面积平均，衡量平均本身的影响
        print("%-12s A vs B: mean=%.5f p99=%.5f max=%.5f deg | valid=%d"
              % (nm, d.mean(), d.quantile(.99), d.max(), int(valid.sum())))

    xB = cap["xB"][0:1].cuda()
    def bench(fn, name, n=15, warm=4):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort(); med = ts[len(ts)//2]
        print("  %-34s %.3f ms" % (name, med)); return med
    print("\n计时（F=39904 / N=20018）:")
    Jf = m.face_jacobians(x0[None], xB); Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    bench(lambda: RS.polar_rotation(Jf), "polar(Jf) it=8  [必需]")
    bench(lambda: RS.polar_rotation(Jv), "polar(Jv) it=8  [仅作参考]")
    bench(lambda: (quats_of(Jf)), "quats_of(Jf) 含 matrix_to_quat")
    bench(lambda: m.rotation_field(x0[None], xB), "rotation_field(face) 当前实现")
