# -*- coding: utf-8 -*-
"""一环一致性置信度：Markley 的 lambda_max - lambda_2nd 能否识别"相邻面旋转打架"的顶点？

若能，则用它乘/门控旋转，就能自动消掉线性平均在 7 个顶点上限出的 >90 度假旋转。
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

def linear_avg(qf, pref_face):
    fa = m.face_areas(x0[None])[0]
    goto = qf.unsqueeze(1).expand(-1, 3, -1)
    sign = torch.where((goto * pref_face).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = (sign * goto) * fa[:, None, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), w.reshape(-1, 4))
    nrm = acc.norm(dim=-1, keepdim=True)
    return F.normalize(acc, dim=-1), nrm.squeeze(-1)

def markley_spec(qf):
    fa = m.face_areas(x0[None])[0]
    M = (qf[:, :, None] * qf[:, None, :]) * fa[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    acc = torch.zeros(x0.shape[0], 16, device=x0.device, dtype=qf.dtype)
    acc.index_add_(0, faces.reshape(-1), wire)
    evals, evecs = torch.linalg.eigh(acc.view(x0.shape[0], 4, 4))
    return F.normalize(evecs[..., -1], dim=-1), evals

def angle_between(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

def local_axis(x0, faces, N):
    """每个顶点一环的几何法向（面积加权），用于定位"""
    a, b, c = faces.unbind(-1)
    e1 = x0[b] - x0[a]; e2 = x0[c] - x0[a]
    n = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    acc = torch.zeros(N, 3, device=x0.device); acc.index_add_(0, a, n)
    return F.normalize(acc, dim=-1)

with torch.no_grad():
    for i in range(2):
        nm = cap["names"][i]
        xB = cap["xB"][i:i+1].cuda()
        Jf = m.face_jacobians(x0[None], xB)
        qf = quats_of(Jf)[0]
        Jv = m.vertex_jacobians(x0[None], xB, Jf=Jf)
        qv = quats_of(Jv)[0]
        qL, nL = linear_avg(qf, qv[faces])
        qM, ev = markley_spec(qf)
        lam1, lam2 = ev[..., -1], ev[..., -2]
        conf = (1.0 - lam2 / lam1.clamp(min=1e-20)).clamp(0, 1)   # 0 = 完全打架, 1 = 完全一致
        d = angle_between(qL, qM)
        axis = local_axis(x0, faces, x0.shape[0])
        bad = d > 30.0
        print("\n== %s ==  >30deg 顶点 %d 个" % (nm, int(bad.sum())))
        print("  conf 全体: p1=%.3f p5=%.3f p50=%.3f | bad 组: %s"
              % (conf.quantile(.01), conf.quantile(.05), conf.median(),
                 ("%.3f~%.3f median %.3f" % (conf[bad].min(), conf[bad].max(), conf[bad].median())) if bad.any() else "n/a"))
        # 用 conf 门控：conf < tau 时旋转回缩到恒等
        for tau in (0.1, 0.3, 0.5):
            kept = conf >= tau
            print("  tau=%.1f: 保留 %d 顶点(%.1f%%), bad 组里被门控掉 %d/%d"
                  % (tau, int(kept.sum()), 100*kept.float().mean(),
                     int((~kept)[bad].sum()), int(bad.sum())))
        # 这些 bad 顶点在哪？看它们的法向绝对分量（z 分量大 = 面向摄像机/前方）
        if bad.any():
            ax = axis[bad].abs()
            print("  bad 组一环法向 |x|,|y|,|z| 中位数: %.2f %.2f %.2f ; x0 z 范围 %.4f~%.4f"
                  % (ax[:,0].median(), ax[:,1].median(), ax[:,2].median(),
                     x0[bad][:,2].min(), x0[bad][:,2].max()))
