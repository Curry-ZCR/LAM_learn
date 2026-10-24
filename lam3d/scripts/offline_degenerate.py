# -*- coding: utf-8 -*-
"""关键假设：>30 度分歧的顶点 == 退化/近奇异面的顶点。

真实管线会"清理孤立顶点 20018 -> 19494"，而我们的捕获张量是清理前的 20018。
若分歧顶点集中在退化面上，则在真实 19494 上这些分歧会自然消失。
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
def angle_between(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI
def avg_linear(qf, pref_face):
    fa = m.face_areas(x0[None])[0]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    return F.normalize(acc, dim=-1), acc.norm(dim=-1)
def avg_markley(qf):
    fa = m.face_areas(x0[None])[0]
    M = (qf[:, :, None] * qf[:, None, :]) * fa[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    acc = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), wire)
    ev, evec = torch.linalg.eigh(acc.view(x0.shape[0], 4, 4))
    return F.normalize(evec[..., -1], dim=-1), ev

with torch.no_grad():
    area = m.face_areas(x0[None])[0]
    used = torch.zeros(x0.shape[0], dtype=torch.bool, device=x0.device)
    used[faces.reshape(-1)] = True
    print("网格: F=%d N=%d | 未被任何面引用的顶点 %d 个 | 面积 == 0 的面 %d 个 | 面积 < 1e-10 的面 %d 个"
          % (faces.shape[0], x0.shape[0], int((~used).sum()), int((area == 0).sum()), int((area < 1e-10).sum())))
    q = area.quantile(torch.tensor([0.0, 0.01, 0.05, 0.5], device='cuda', dtype=torch.float32))
    print("面积分位: min=%.3e p1=%.3e p5=%.3e p50=%.3e" % (area.min(), area.quantile(.01), area.quantile(.05), area.median()))

    Jf = m.face_jacobians(x0[None], cap["xB"][0:1].cuda())
    cond = torch.linalg.cond(Jf)[0]
    badface = (cond > 1e3) | (area < 1e-10)
    print("cond>1e3 或面积~0 的面: %d 个" % int(badface.sum()))

    xB = cap["xB"][0:1].cuda()
    qf = quats_of(Jf)[0]
    Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
    qv = quats_of(Jv)[0]
    qL, nL = avg_linear(qf, qv[faces])
    qM, ev = avg_markley(qf)
    d = angle_between(qL, qM)
    for thr in (30.0, 90.0):
        idx = (d > thr).nonzero().squeeze(-1)
        hits = 0
        if idx.numel():
            # 这些顶点的一环里有多少个坏面
            vset = set(idx.tolist())
            fhit = [f for f in range(faces.shape[0]) if set(faces[f].tolist()) & vset]
            nb = sum(1 for f in fhit if bool(badface[f]))
            print("  分歧>%.0f 度: %d 顶点；它们的一环共 %d 个面，其中坏面 %d 个"
                  % (thr, idx.numel(), len(fhit), nb))
        else:
            print("  分歧>%.0f 度: 0 顶点" % thr)
    # 只看"好顶点"（一环全好面）的分歧
    goodv = torch.ones(x0.shape[0], dtype=torch.bool, device=x0.device)
    # 坏面标记传播到顶点
    bv = torch.zeros(x0.shape[0], dtype=torch.bool, device=x0.device)
    bv[faces[badface].reshape(-1)] = True
    dg = d[~bv]
    print("  排除坏面顶点后(%d 个顶点): linear vs markley mean=%.5f p99=%.5f max=%.4f deg"
          % (int((~bv).sum()), dg.mean(), dg.quantile(.99), dg.max()))
