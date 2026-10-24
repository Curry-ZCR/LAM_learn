# -*- coding: utf-8 -*-
"""离线全链路：polar_rotation 新内核 + rotation_field 端到端，跑在真实捕获张量上。"""
import sys, time
import torch
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
from lam3d.core import rot_sync as RS

cap = torch.load("/tmp/rot_capture.pt", weights_only=False)
x0 = cap["x0"].cuda()
faces = cap["faces"].cuda()
names = cap["names"]

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

def bench(fn, name, n=15, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts)//2]
    print("  %-30s median %.3f ms" % (name, med))
    return med

with torch.no_grad():
    for i, nm in enumerate(names):
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        # 精度：新内核 vs float64 SVD 真值
        Rf = RS.polar_rotation(Jf)
        Rs = RS.polar_rotation(Jf, exact=True)
        d = (Rf - Rs).abs().amax(dim=(-1,-2))[0]
        orth = ((Rf.transpose(-1,-2) @ Rf) - torch.eye(3, device=Rf.device)).abs().amax(dim=(-1,-2))[0]
        print("%-12s vs SVD: p50=%.2e p99=%.2e max=%.2e | orth p99=%.2e max=%.2e"
              % (nm, d.median(), d.quantile(.99), d.max(), orth.quantile(.99), orth.max()))
        dq, diag = m.rotation_field(x0[None], xB)
        ang = RS.quat_angle(dq)[0]
        print("             dq finite=%s |q|=1 maxerr=%.2e angle deg mean=%.3f p95=%.3f max=%.3f clipped=%d"
              % (bool(torch.isfinite(dq).all()), (dq.norm(dim=-1)-1).abs().max(),
                 ang.mean()*180/3.14159265, ang.quantile(.95)*180/3.14159265,
                 ang.max()*180/3.14159265, int(diag["clipped"].sum())))

    xB = cap["xB"][0:1].cuda()
    Jf = m.face_jacobians(x0[None], xB)
    print("\n离线计时（F=39904 faces / N=20018）:")
    t_pf = bench(lambda: RS.polar_rotation(Jf), "polar_rotation(Jf) it=8")
    Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    t_pv = bench(lambda: RS.polar_rotation(Jv), "polar_rotation(Jv)")
    t_rf = bench(lambda: m.rotation_field(x0[None], xB), "rotation_field(face) TOTAL")
    print("  polar 合计 %.3f ms = %.1f%% of rotation_field" % (t_pf + t_pv, 100*(t_pf+t_pv)/t_rf))
