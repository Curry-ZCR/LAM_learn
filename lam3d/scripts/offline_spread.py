# -*- coding: utf-8 -*-
"""只看那几十个"两法分歧 >30 度"的顶点：一环面四元数的真实散布有多大？
散布小 -> 半球对齐有 bug；散布 ~180 度 -> 两种平均都无定义，必须靠置信度门控。
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
    for i in range(3):
        nm = cap["names"][i]
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        qf = quats_of(Jf)[0]
        Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
        qv = quats_of(Jv)[0]
        qL, nL = avg_linear(qf, qv[faces])
        qM, ev = avg_markley(qf)
        d = angle_between(qL, qM)
        bad = d > 30.0
        idx = bad.nonzero().squeeze(-1)
        print("\n== %s == 分歧>30deg 的顶点 %d 个" % (nm, idx.numel()))
        # 逐个顶点取一环面
        vf = {}
        for f in range(faces.shape[0]):
            for v in faces[f].tolist():
                vf.setdefault(v, []).append(f)
        spreads, nfaces = [], []
        for v in idx.tolist()[:60]:
            fl = vf.get(v, [])
            nfaces.append(len(fl))
            if len(fl) < 2: continue
            sub = qf[fl]                                     # [k,4]
            # 两两夹角（取 min(a,180-a) 的半球最短）
            A = angle_between(sub[:, None, :], sub[None, :, :])
            spreads.append(A.max().item())
        spreads = torch.tensor(spreads)
        print("  一环面数: %s" % sorted(set(nfaces)))
        if spreads.numel():
            print("  一环内最大两两夹角: p50=%.1f p90=%.1f max=%.1f 度 | >90 度的顶点 %d/%d"
                  % (spreads.median(), spreads.quantile(.9), spreads.max(),
                     int((spreads > 90).sum()), spreads.numel()))
        lam = ev[..., -1]; lam2 = ev[..., -2]
        conf = (1 - lam2 / lam.clamp(min=1e-20)).clamp(0, 1)
        print("  conf(全体 p1=%.3f) 分歧组: min=%.3f median=%.3f max=%.3f"
              % (conf.quantile(.01), conf[idx].min(), conf[idx].median(), conf[idx].max()))
        print("  分歧组 |acc| (半球加权范数, <0.05 说明几乎相消): min=%.4f p50=%.4f max=%.4f"
              % (nL[idx].min(), nL[idx].median(), nL[idx].max()))
