# -*- coding: utf-8 -*-
"""修正上一版的 bug：qA = F.normalize(acc.clamp(min=0)) 里的 clamp(min=0)
是笔误（四元数分量有负值，clamp 会毁掉方向）。这里去掉后重测。"""
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
m = RS.RotSyncModulator.__new__(RS.RotSyncModulator); torch.nn.Module.__init__(m)
m.base = Shim(); m.base.faces = faces; m.base.faces_flat = faces.reshape(-1); m.eps = 1e-12
flat = m.base.faces_flat

q_of = lambda J: F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)
def ang(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

def path_A(xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    q_ref = q_of(m.vertex_jacobians(x0[None], xB, Jf=Jf))[0]
    qf = q_of(Jf)[0]
    pair = q_ref[flat].reshape(-1, 3, 4).sum(1)
    goto = qf.repeat_interleave(3, 0); pref = pair.repeat_interleave(3, 0)
    sign = torch.where((goto * pref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * area.repeat_interleave(3, 0)[:, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device)
    acc.index_add_(0, flat, w)
    return F.normalize(acc, dim=-1), q_ref, qf, area

def path_B(xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    qf = q_of(Jf)[0]
    M = (qf[:, :, None] * qf[:, None, :]) * area[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    acc = torch.zeros(x0.shape[0], 16, device=x0.device)
    acc.index_add_(0, flat, wire)
    ev, evec = torch.linalg.eigh(acc.view(-1, 4, 4))
    return F.normalize(evec[:, :, -1], dim=-1), ev, qf, area

def bench(fn, arg, n=12):
    for _ in range(4): fn(arg)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(n): fn(arg)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

with torch.no_grad():
    print("== v 与参考向量校验（前 3 个顶点, A vs B, 用无符号夹角）==")
    for i in range(3):
        nm = cap["names"][i]; xB = cap["xB"][i:i+1].cuda()
        qA, q_ref, qf, area = path_A(xB)
        qB, ev, _, _ = path_B(xB)
        d = ang(qA, qB)
        lam1 = ev[:, -1]; lam2 = ev[:, -2]
        conf = (1 - lam2 / lam1.clamp(min=1e-20)).clamp(0, 1)
        print("\n-- %s -- A(半球+线性) vs B(Markley):" % nm)
        print("   mean=%.4f p50=%.4f p90=%.4f p99=%.4f max=%.3f 度"
              % (d.mean(), d.median(), d.quantile(.9), d.quantile(.99), d.max()))
        for th in (1.0, 5.0, 12.0, 30.0, 90.0):
            print("      > %5.1f 度: %5d (%.2f%%)" % (th, int((d > th).sum()), 100.0*int((d>th).sum())/d.numel()))
        print("   conf 分位: p1=%.3f p5=%.3f p50=%.3f" % (conf.quantile(.01), conf.quantile(.05), conf.median()))
        for c in (0.5, 0.7, 0.9, 0.95):
            keep = conf >= c; dk = d[keep]
            print("   conf>=%.2f 保留 %.1f%% | 差 mean=%.4f p99=%.3f max=%.3f | >5度 %d"
                  % (c, 100.0*keep.float().mean(), dk.mean(), dk.quantile(.99), dk.max(), int((dk>5).sum())))
    print("\n== 耗时 (ms, 12 次均值) ==")
    xB = cap["xB"][0:1].cuda()
    tA = bench(path_A, xB); tB = bench(path_B, xB)
    print("  path_A(当前 full: Jv+polar(Jv)+m2q+半球+线性) = %.3f ms" % tA)
    print("  path_B(新 full: Jf+polar(Jf)+m2q+外积+eigh)   = %.3f ms" % tB)
    print("  => 省 %.3f ms (%.0f%%)" % (tA - tB, 100.0 * (tA - tB) / tA))
