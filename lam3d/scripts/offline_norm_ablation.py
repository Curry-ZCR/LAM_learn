# -*- coding: utf-8 -*-
"""验证 eng-practice 的两条主张：(1) Frobenius 预归一化是否反而有害；(2) det 投影时机。"""
import sys
import torch
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
from lam3d.core import rot_sync as RS
from lam3d.core.rot_sync import _safe_inv, _polar_rotation_svd

cap = torch.load("/tmp/rot_capture.pt", weights_only=False)
x0 = cap["x0"].cuda(); names = cap["names"]

class Shim: pass
m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
torch.nn.Module.__init__(m)
m.base = Shim(); m.base.faces = cap["faces"].cuda(); m.base.faces_flat = m.base.faces.reshape(-1)
m.eps = 1e-12

def polar(j, iters=8, normalize=True):
    x = j.float()
    if normalize:
        nrm = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1).clamp(min=1e-12)
        x = x / nrm[..., None, None]
    for _ in range(iters):
        x = 0.5 * (x + _safe_inv(x).transpose(-1, -2))
    return x

def orth_of(x):
    o = (x.transpose(-1, -2) @ x - torch.eye(3, device=x.device)).abs().amax(dim=(-1,-2))
    return o

def report(tag, J, ref):
    for norm in (True, False):
        for it in (6, 8, 10):
            R = polar(J, it, norm)
            o = orth_of(R)
            fin = torch.isfinite(R).all(-1).all(-1)
            d = (R - ref).abs().amax(dim=(-1,-2))
            d = torch.where(fin, d, torch.full_like(d, float('nan')))
            print("  %-9s norm=%-5s it=%-2d orth p50=%.2e p99=%.2e max=%.2e | nonfinite=%d | vs SVD p50=%.2e p99=%.2e"
                  % (tag, norm, it, o.median(), o.quantile(.99), o.max(), int((~fin).sum()),
                     d.nanmedian(), d.nanquantile(.99)))

with torch.no_grad():
    for i, nm in enumerate(["ones", "extreme"]):
        Jf = m.face_jacobians(x0[None], cap["xB"][i:i+1].cuda())
        ref = _polar_rotation_svd(Jf)
        report(nm, Jf, ref)
    # 合成 cond=30 的 SPD（eng-practice 的场景）
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(20000, 3, 3, device='cuda'))
    S = torch.tensor([1.0, 0.2, 1.0/30.0], device='cuda')
    Js = (U * S) @ U.transpose(-1, -2)
    print("\n合成 cond=30:")
    report("synth", Js, _polar_rotation_svd(Js))
