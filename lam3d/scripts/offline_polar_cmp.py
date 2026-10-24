# -*- coding: utf-8 -*-
"""离线内核对比：inv3 vs linalg.inv_ex，以及两种半球参考的等价性。

跑在 /tmp/rot_capture.pt 的真实张量上，秒级出结果。
"""
import sys, time
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
import torch.nn.functional as F
from lam3d.core import rot_sync as RS
from lam3d.core.tga_covariance import inv3

cap = torch.load("/tmp/rot_capture.pt", weights_only=False)
print("capture:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in cap.items() if k != "faces"})
x0 = cap["x0"].cuda()
names = cap["names"]

class Shim:
    pass

def make_mod():
    m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
    torch.nn.Module.__init__(m)
    m.base = Shim()
    m.base.faces = cap["faces"].cuda()
    m.base.faces_flat = m.base.faces.reshape(-1)
    m.rotation_gain = 1.0
    m.max_angle_deg = 45.0
    m.deadzone_deg = 0.0
    m.average = "face"
    m.temporal_smoothing = 0.0
    m.eps = 1e-12
    m._prev_delta = None
    m._last_diag = None
    return m

m = make_mod()

def inv_ex_safe(x):
    r, info = torch.linalg.inv_ex(x)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

def polar_with(invfn, j, iters=8):
    x = j if j.dtype == torch.float32 else j.float()
    neg_in = torch.linalg.det(j) < 0
    nrm = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1).clamp(min=1e-12)
    x = x / nrm[..., None, None]
    for _ in range(iters):
        x = 0.5 * (x + invfn(x).transpose(-1, -2))
    orth = (x.transpose(-1, -2) @ x - torch.eye(3, device=x.device)).abs().amax(dim=(-1, -2))
    det = torch.linalg.det(x)
    finite = torch.isfinite(x).all(-1).all(-1)
    scale = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1)
    bad = (~finite) | (scale <= 1e-6) | (orth > 1e-6) | (det <= 0) | neg_in
    return x, bad, orth

def bench(fn, name, n=12, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    print("  %-26s median %.3f ms" % (name, ts[len(ts)//2]))
    return ts[len(ts)//2]

with torch.no_grad():
    Jf = m.face_jacobians(x0[None], cap["xB"][0:1].cuda())
    print("Jf shape", tuple(Jf.shape), "cond p50/p99/max:",
          torch.linalg.cond(Jf)[0].quantile(torch.tensor([.5, .99, 1.0], device='cuda')).tolist())

    a, bada, ortha = polar_with(lambda t: inv3(t, 1e-9), Jf)
    b, badb, orthb = polar_with(inv_ex_safe, Jf)
    print("inv3 it=8 : bad=%d orth max %.3e" % (int(bada.sum()), ortha.max().item()))
    print("inv_ex it=8: bad=%d orth max %.3e" % (int(badb.sum()), orthb.max().item()))
    print("max|Ra - Rb| = %.3e" % (a - b).abs().max().item())
    ta = bench(lambda: polar_with(lambda t: inv3(t, 1e-9), Jf), "polar inv3 it=8")
    tb = bench(lambda: polar_with(inv_ex_safe, Jf), "polar inv_ex it=8")
    print("  speedup %.2fx" % (ta / tb))
