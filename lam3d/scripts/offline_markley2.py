# -*- coding: utf-8 -*-
"""谁对？线性半球平均 vs Markley：检查大差异顶点上的归一化残差 nA。"""
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

def ref_linear(qf, pref_face):
    fa = m.face_areas(x0[None])[0]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    nrm = acc.norm(dim=-1, keepdim=True)
    return F.normalize(acc, dim=-1), nrm.squeeze(-1), fa

def markley(qf):
    fa = m.face_areas(x0[None])[0]
    M = (qf[:, :, None] * qf[:, None, :]) * fa[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    acc = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), wire)
    evals, evecs = torch.linalg.eigh(acc.view(x0.shape[0], 4, 4))
    return F.normalize(evecs[..., -1], dim=-1), evals[..., -1]

def angle_between(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

with torch.no_grad():
    for i, nm in enumerate(["ones", "extreme"]):
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        qf = quats_of(Jf)[0]
        Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
        qv = quats_of(Jv)[0]
        qA, nA, fa = ref_linear(qf, qv[faces])
        qM, lam = markley(qf)
        d = angle_between(qA, qM)
        big = d > 30.0
        print("%-8s 顶点数=%d | >30deg: %d 个 | >90deg: %d 个"
              % (nm, d.numel(), int(big.sum()), int((d > 90).sum())))
        # 每个顶点的一环面数
        deg = torch.zeros(x0.shape[0], device=x0.device)
        deg.index_add_(0, faces.reshape(-1), torch.ones(faces.numel(), device=x0.device))
        for label, mask in [("全部", torch.ones_like(d, dtype=torch.bool)), (">30deg", big), ("<=30deg", ~big)]:
            if mask.sum() == 0: continue
            print("    %-8s nA: p50=%.4f min=%.4f | 一环面数 p50=%.0f max=%.0f"
                  % (label, nA[mask].median(), nA[mask].min(), deg[mask].median(), deg[mask].max()))
        # 一环面数 == 1 或 2 的顶点（线性平均等价于单点，本该完全一致）
        few = deg <= 2
        print("    一环<=2 面的顶点 %d 个，其中 max 差 %.3f deg" % (int(few.sum()), d[few].max() if few.sum() else float('nan')))
