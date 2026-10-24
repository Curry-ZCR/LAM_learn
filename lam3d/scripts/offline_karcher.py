# -*- coding: utf-8 -*-
"""以 Karcher（Riemannian/测地线）均值为真值，裁决 Markley vs 半球对齐 chordal。

Karcher 均值 = argmin sum w_i d(q_i,q)^2，正是 SurFhead 建议的
exp(sum w log U) 平均。用 3 步不动点迭代求。
"""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
from lam3d.core import rot_sync as RS
from lam3d.core.tga_covariance import matrix_to_quat

cap = torch.load("/tmp/rot_capture.pt", weights_only=False)
x0 = cap["x0"].cuda(); faces = cap["faces"].cuda()
DEGI = 180.0 / 3.14159265

class Shim: pass
m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
torch.nn.Module.__init__(m)
m.base = Shim(); m.base.faces = faces; m.base.faces_flat = faces.reshape(-1)
m.eps = 1e-12

def quats_of(J):
    return F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)

def qconj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)

def qlog(q):
    """单位四元数 -> 旋转向量 [..,3]"""
    w = q[..., 0].clamp(-1.0, 1.0)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(n, w)
    factor = torch.where(n > 1e-9, ang / n.clamp(min=1e-12), torch.full_like(n, 2.0))
    return v * factor

def qexp(v):
    ang = v.norm(dim=-1, keepdim=True)
    half = 0.5 * ang
    s = torch.where(ang > 1e-9, torch.sin(half) / ang.clamp(min=1e-12), torch.full_like(ang, 0.5))
    return F.normalize(torch.cat((torch.cos(half), v * s), dim=-1), dim=-1)

def qmul(a, b):
    return RS.quat_multiply(a, b)

def avg_linear(qf, pref_face):
    fa = m.face_areas(x0[None])[0]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    return F.normalize(acc, dim=-1)

def avg_markley(qf):
    fa = m.face_areas(x0[None])[0]
    M = (qf[:, :, None] * qf[:, None, :]) * fa[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    acc = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), wire)
    ev, evec = torch.linalg.eigh(acc.view(x0.shape[0], 4, 4))
    return F.normalize(evec[..., -1], dim=-1)

def avg_karcher(qf, init, iters=3):
    fa = m.face_areas(x0[None])[0]
    wface = fa[:, None, None].expand(-1, 3, 1).reshape(-1, 1)   # [3F,1]
    cur = init
    for _ in range(iters):
        rel = qmul(qconj(cur)[faces.reshape(-1)], qf.repeat_interleave(3, dim=0))  # [3F,4]
        # 半球：rel 的 w 取正（等价于走最短测地线）
        rel = torch.where(rel[:, 0:1] < 0, -rel, rel)
        v = qlog(rel) * wface
        num = torch.zeros(x0.shape[0], 3, device=x0.device, dtype=qf.dtype)
        num.index_add_(0, faces.reshape(-1), v)
        den = torch.zeros(x0.shape[0], 1, device=x0.device, dtype=qf.dtype)
        den.index_add_(0, faces.reshape(-1), wface)
        step = qexp(num / den.clamp(min=1e-20))
        cur = F.normalize(qmul(cur, step), dim=-1)
    return cur

def angle_between(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

with torch.no_grad():
    for i, nm in enumerate(cap["names"]):
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        qf = quats_of(Jf)[0]
        Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
        qv = quats_of(Jv)[0]
        qL = avg_linear(qf, qv[faces])
        qM = avg_markley(qf)
        qK = avg_karcher(qf, qL, iters=4)
        dL, dM = angle_between(qL, qK), angle_between(qM, qK)
        print("%-12s vs Karcher(4it):  linear mean=%.4f p99=%.4f max=%.3f | markley mean=%.4f p99=%.4f max=%.3f"
              % (nm, dL.mean(), dL.quantile(.99), dL.max(), dM.mean(), dM.quantile(.99), dM.max()))
        # 迭代收敛性：不同 iters 的 Karcher 是否稳定
        qK2 = avg_karcher(qf, qL, iters=2)
        print("             Karcher 2it vs 4it: mean=%.5f max=%.4f deg"
              % (angle_between(qK2, qK).mean(), angle_between(qK2, qK).max()))

    xB = cap["xB"][0:1].cuda()
    def bench(fn, name, n=15, warm=4):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort(); print("  %-32s %.3f ms" % (name, ts[len(ts)//2]))
    Jf = m.face_jacobians(x0[None], xB); qf = quats_of(Jf)[0]
    Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf); qv = quats_of(Jv)[0]
    qL = avg_linear(qf, qv[faces]); qM = avg_markley(qf)
    print("\n计时:")
    bench(lambda: avg_markley(qf), "avg_markley")
    bench(lambda: avg_karcher(qf, qM, iters=1), "avg_karcher init=markley it=1")
    bench(lambda: avg_karcher(qf, qM, iters=3), "avg_karcher init=markley it=3")
