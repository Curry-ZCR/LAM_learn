# -*- coding: utf-8 -*-
"""Markley 四元数平均（主特征向量）替代"参考半球对齐 + 线性加权"。

动机：variants 实测里 '自身 w>=0 作参考' 在近 180 度转角的顶点上与 polar(Jv) 参考
差到 177 度（符号不一致）。Markley 法对 q 与 -q 完全不变（q q^T = (-q)(-q)^T），
因此不需要任何参考，也就不再需要 polar(Jv) 那 2.6 ms。
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
m = RS.RotSyncModulator.__new__(RS.RotSyncModulator)
torch.nn.Module.__init__(m)
m.base = Shim(); m.base.faces = faces; m.base.faces_flat = faces.reshape(-1)
m.rotation_gain = 1.0; m.max_angle_deg = 45.0; m.deadzone_deg = 0.0
m.average = "face"; m.temporal_smoothing = 0.0; m.eps = 1e-12
m._prev_delta = None; m._last_diag = None

def quats_of(J):
    return F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)

def angle_between(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

def ref_linear(qf, pref_face):
    fa = m.face_areas(x0[None])[0]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    nrm = acc.norm(dim=-1, keepdim=True)
    return F.normalize(acc, dim=-1), nrm.squeeze(-1)

def markley(qf):
    fa = m.face_areas(x0[None])[0]
    M = (qf[:, :, None] * qf[:, None, :]) * fa[:, None, None]      # [F,4,4]
    acc = torch.zeros(x0.shape[0], 4, 4, device=x0.device, dtype=qf.dtype)
    # index_add 只支持 float 张量最后一维按 index 累加：展平到 [F,16]
    acc.view(x0.shape[0], 16).index_add_(0, faces[:, 0].repeat_interleave(16), M.reshape(-1, 16)) if False else None
    # 用 scatter_add 三维更直接
    idx = faces.reshape(-1, 1, 1).expand(-1, 3, 16).reshape(-1)
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) * (1.0/3.0)).reshape(-1, 16)
    acc2 = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    acc2.index_add_(0, faces.reshape(-1), wire)
    acc = acc2.view(x0.shape[0], 4, 4)
    evals, evecs = torch.linalg.eigh(acc)
    q = evecs[..., -1]
    return F.normalize(q, dim=-1), evals[..., -1]

with torch.no_grad():
    for i, nm in enumerate(names):
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        qf = quats_of(Jf)[0]
        Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
        qv = quats_of(Jv)[0]
        qA, nA = ref_linear(qf, qv[faces])
        qM, lam = markley(qf)
        d = angle_between(qA, qM)
        print("%-12s Markley vs polar(Jv)-ref: mean=%.5f p99=%.5f max=%.5f deg | lam>0 %d/%d"
              % (nm, d.mean(), d.quantile(.99), d.max(), int((lam > 0).sum()), lam.numel()))

    xB = cap["xB"][0:1].cuda()
    def bench(fn, name, n=15, warm=4):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort(); print("  %-30s %.3f ms" % (name, ts[len(ts)//2]))
    Jf = m.face_jacobians(x0[None], xB); Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    qf = quats_of(Jf)[0]; qv = quats_of(Jv)[0]
    print("\n计时:")
    bench(lambda: ref_linear(qf, qv[faces]), "ref_linear (需要 qv)")
    bench(lambda: markley(qf), "markley eigh")
    M = (qf[:, :, None] * qf[:, None, :])
    wire = M.unsqueeze(1).expand(-1, 3, 4, 4).reshape(-1, 16)
    acc2 = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    bench(lambda: acc2.index_add(0, faces.reshape(-1), wire), "  markley gather/index_add")
    A = acc2.view(x0.shape[0], 4, 4).clone()
    bench(lambda: torch.linalg.eigh(A), "  markley eigh only")
