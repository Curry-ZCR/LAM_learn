# -*- coding: utf-8 -*-
"""coherence = |sum(w*q)| / sum(w) —— 这是当前实现里**已经算出**的中间量，
不需要 eigh、不需要 polar(Jv)。检验它能否当免费置信度：
  1) 全体分布
  2) 与"线性平均 vs Markley"分歧的相关性
  3) 用它门控能否自动消掉那几十个 180 度无定义顶点
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
m = RS.RotSyncModulator.__new__(RS.RotSyncModulator); torch.nn.Module.__init__(m)
m.base = Shim(); m.base.faces = faces; m.base.faces_flat = faces.reshape(-1); m.eps = 1e-12
flat = m.base.faces_flat
q_of = lambda J: F.normalize(matrix_to_quat(RS.polar_rotation(J)), dim=-1)
def ang(p, q):
    d = (p * q).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d) * DEGI

def run(xB):
    area = m.face_areas(x0[None])[0]
    Jf = m.face_jacobians(x0[None], xB)
    q_ref = q_of(m.vertex_jacobians(x0[None], xB, Jf=Jf))[0]
    qf = q_of(Jf)[0]
    pair = q_ref[flat].reshape(-1, 3, 4).sum(1)
    goto = qf.repeat_interleave(3, 0); pref = pair.repeat_interleave(3, 0)
    sign = torch.where((goto * pref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = area.repeat_interleave(3, 0)[:, None]
    acc = torch.zeros(x0.shape[0], 4, device=x0.device)
    acc.index_add_(0, flat, sign * goto * w)
    den = torch.zeros(x0.shape[0], 1, device=x0.device)
    den.index_add_(0, flat, w)
    coh = acc.norm(dim=-1) / den.squeeze(-1).clamp(min=1e-20)
    qA = F.normalize(acc, dim=-1)
    # Markley
    M = (qf[:, :, None] * qf[:, None, :]) * area[:, None, None]
    wire = (M.unsqueeze(1).expand(-1, 3, 4, 4) / 3.0).reshape(-1, 16)
    accM = torch.zeros(x0.shape[0], 16, device=x0.device)
    accM.index_add_(0, flat, wire)
    qM = F.normalize(torch.linalg.eigh(accM.view(-1, 4, 4))[1][:, :, -1], dim=-1)
    return coh, qA, qM

with torch.no_grad():
    print("%-12s %8s %8s %8s %8s %8s %8s" % ("vector", "p1", "p5", "p50", "p95", "min", "<0.9 个数"))
    store = {}
    for i in range(5):
        nm = cap["names"][i]; xB = cap["xB"][i:i+1].cuda()
        coh, qA, qM = run(xB)
        d = ang(qA, qM)
        store[nm] = (coh, d)
        print("%-12s %8.4f %8.4f %8.4f %8.4f %8.4f %8d" %
              (nm, coh.quantile(.01), coh.quantile(.05), coh.median(), coh.quantile(.95),
               coh.min(), int((coh < 0.9).sum())))
        print("            与 Markley 分歧: %-10s  coh<0.9 组的分歧 mean=%.3f p99=%.2f max=%.2f"
              % ("", d[coh < 0.9].mean() if (coh<0.9).any() else 0.0,
                 d[coh < 0.9].quantile(.99) if (coh<0.9).any() else 0.0,
                 d[coh < 0.9].max() if (coh<0.9).any() else 0.0))
        print("            coh>=0.9 组: mean=%.4f p99=%.3f max=%.2f | >5度 %d/%d" %
              (d[coh >= 0.9].mean(), d[coh >= 0.9].quantile(.99), d[coh >= 0.9].max(),
               int((d[coh >= 0.9] > 5).sum()), int((coh >= 0.9).sum())))

    print("\n== 门控方案：angle_eff = angle * g(coh)，g = clamp((coh-tau)/(1-tau),0,1) ==")
    coh, qA, qM = store["ones"]
    for tau in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        g = ((coh - tau) / max(1.0 - tau, 1e-9)).clamp(0, 1)
        # 被压到 0 的就是被门掉的
        killed = (g < 1e-6)
        print("  tau=%.2f: 完全门掉 %5d (%.2f%%) | 部分衰减 %5d (%.2f%%) | 全保留 %5d"
              % (tau, int(killed.sum()), 100.0*killed.float().mean(),
                 int(((g>1e-6)&(g<1-1e-6)).sum()), int((g>1-1e-6).sum())))
