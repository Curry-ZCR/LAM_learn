
# -*- coding: utf-8 -*-
"""方法四（候选设计）的离线原型：在已捕获的真实会话张量上复算各方法的几何指标。

不改 LAM，不改 rot_sync/tga；只 import 它们的数值工具，用来：
  1) 复现 docs/cov_methods_eval.md 里 rot_sync / tga 的 shape_err、rigid_err 等指标；
  2) 在同一口径下评测方法四的各个候选（法向轴拉伸 / 法向协变对齐 / SPD 测地插值）。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
os.environ.setdefault("LAM_ROOT", "/workspace/LAM")

from lam3d.core.rot_sync import polar_rotation, quat_multiply
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat, inv3

torch.set_grad_enabled(False)
DEV = "cuda"
CAP = "/tmp/rot_capture.pt"


def load():
    d = torch.load(CAP, map_location="cpu")
    x0 = d["x0"].to(DEV).float()                       # [N,3]
    xB = d["xB"].to(DEV).float()                       # [M,N,3]
    faces = d["faces"].to(DEV).long()                  # [F,3]
    q0 = F.normalize(d["gs_quat"].to(DEV).float(), dim=-1)
    s0 = d["gs_scale"].to(DEV).float().clamp(min=1e-9)
    return x0, xB, faces, q0, s0, d["names"]


def face_jacobians(x0, xB, faces):
    # x0 [N,3] -> [1,N,3]; xB [M,N,3]
    x0b = x0.unsqueeze(0)
    a, b, c = faces.unbind(-1)
    e1 = x0b[..., b, :] - x0b[..., a, :]
    e2 = x0b[..., c, :] - x0b[..., a, :]
    d1 = xB[..., b, :] - xB[..., a, :]
    d2 = xB[..., c, :] - xB[..., a, :]
    n0 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-12)
    n1 = F.normalize(torch.cross(d1, d2, dim=-1), dim=-1, eps=1e-12)
    J = torch.stack((d1, d2, n1), -1) @ inv3(torch.stack((e1, e2, n0), -1))
    return torch.nan_to_num(J, 0., 0., 0.).clamp(-64, 64), n0


def face_areas(x0, faces, M=None):
    x0b = x0.unsqueeze(0)
    a, b, c = faces.unbind(-1)
    ar = 0.5 * torch.cross(x0b[..., b, :] - x0b[..., a, :], x0b[..., c, :] - x0b[..., a, :], dim=-1).norm(dim=-1)
    ar = torch.nan_to_num(ar, 0.).clamp(min=0.)
    if ar.dim() == 3 and ar.shape[0] == 1:
        ar = ar[0]
    if M is not None and ar.shape[0] != M:
        ar = ar.expand(M, -1)
    return ar


def scatter_vertex(per_face, area, faces, n_vert):
    """[M,F,9] -> [M,N,9] area-weighted; returns also den [M,N]."""
    M, Fn = area.shape
    flat = faces.reshape(-1)
    idx = flat.view(1, -1, 1).expand(M, -1, 9)
    vals = (per_face * area[..., None, None]).reshape(M, Fn, 9).repeat_interleave(3, 1)
    num = torch.zeros(M, n_vert, 9, device=per_face.device, dtype=per_face.dtype)
    num.scatter_add_(1, idx, vals)
    den = torch.zeros(M, n_vert, device=area.device, dtype=area.dtype)
    den.scatter_add_(1, flat.view(1, -1).expand(M, -1), area.repeat_interleave(3, 1))
    out = num / den.clamp(min=1e-12).unsqueeze(-1)
    eye = torch.eye(3, device=per_face.device, dtype=per_face.dtype).reshape(1, 1, 9)
    out = torch.where((den <= 1e-12).unsqueeze(-1), eye.expand_as(out), out)
    return out, den


def vertex_jacobians(Jf, area, faces, n_vert):
    Jv, den = scatter_vertex(Jf, area, faces, n_vert)
    return Jv.view(*Jv.shape[:-1], 3, 3)


def markley(q_face, area, faces, n_vert):
    M, Fn = area.shape
    flat = faces.reshape(-1)
    pw = (q_face.unsqueeze(-1) * q_face.unsqueeze(-2)) * area[:, :, None, None]
    wire = (pw / 3.0).reshape(M, Fn, 16).repeat_interleave(3, 1)
    idx = flat.view(1, -1, 1).expand(M, -1, 16)
    acc = torch.zeros(M, n_vert, 16, device=q_face.device, dtype=torch.float32)
    acc.scatter_add_(1, idx, wire.to(torch.float32))
    ev, evec = torch.linalg.eigh(acc.view(M, n_vert, 4, 4))
    q = evec[..., -1]
    lam1 = ev[..., -1].clamp(min=1e-20)
    conf = (1.0 - ev[..., -2] / lam1).clamp(0., 1.)
    return F.normalize(q, dim=-1), conf


def quat_angle(q):
    w = q[..., 0].abs().clamp(max=1.)
    return 2.0 * torch.atan2(q[..., 1:].norm(dim=-1), w)


def rot_angle(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(((tr - 1) * 0.5).clamp(-1., 1.))


def kabsch_rings(x0, xB, faces, n_vert):
    """逐顶点一环 Kabsch：返回 (R_kabsch [M,N,3,3], motion_deg [M,N])。"""
    M = xB.shape[0]
    N = n_vert
    src = (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    ei = torch.cat([e[0] for e in src]); ej = torch.cat([e[1] for e in src])
    ei2 = torch.cat([ei, ej]); ej2 = torch.cat([ej, ei])
    # 邻接（双向、去重）
    key = ei2 * N + ej2
    key = torch.unique(key)
    ei2 = torch.div(key, N, rounding_mode="floor"); ej2 = key % N
    cnt = torch.zeros(N, device=x0.device).index_add_(0, ei2, torch.ones_like(ei2, dtype=torch.float32))
    w = 1.0 / cnt.clamp(min=1.).index_select(0, ei2)        # 行归一化权重
    P = x0.unsqueeze(0).expand(M, N, 3)                      # [M,N,3]
    Q = xB
    pbar = torch.zeros(M, N, 3, device=x0.device).index_add_(1, ei2.view(1, -1).expand(M, -1), P.index_select(1, ej2) * w.view(1, -1, 1))
    qbar = torch.zeros(M, N, 3, device=x0.device).index_add_(1, ei2.view(1, -1).expand(M, -1), Q.index_select(1, ej2) * w.view(1, -1, 1))
    Pj = P.index_select(1, ej2); Qj = Q.index_select(1, ej2)
    op = (Qj.unsqueeze(-1) * Pj.unsqueeze(-2)).reshape(M, -1, 9) * w.view(1, -1, 1)
    Mqp = torch.zeros(M, N, 9, device=x0.device).index_add_(1, ei2.view(1, -1, 1).expand(M, -1, 9), op)
    H = Mqp.view(M, N, 3, 3) - qbar.unsqueeze(-1) * pbar.unsqueeze(-2)
    U, S, Vh = torch.linalg.svd(H.double())
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.ones(M, N, 3, device=x0.device, dtype=torch.float64)
    D[..., 2] = d
    R = U @ torch.diag_embed(D) @ Vh
    return R.float()


def ring_jitter(qdelta, faces, n_vert):
    """一环相邻顶点旋转夹角均值（度）。"""
    ei = faces[:, [0, 1]]; ej = faces[:, [1, 2]]
    a = torch.cat([ei[:, 0], ej[:, 0], faces[:, 2]]); b = torch.cat([ei[:, 1], ej[:, 1], faces[:, 0]])
    key = torch.unique(torch.cat([a * n_vert + b, b * n_vert + a]))
    ai = torch.div(key, n_vert, rounding_mode="floor"); bi = key % n_vert
    qa = qdelta.index_select(1, ai); qb = qdelta.index_select(1, bi)
    sgn = torch.where((qa * qb).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    rel = F.normalize(qa * sgn - qb, dim=-1)     # 相对旋转的向量部
    # angle of qa^{-1} qb: 用 |dot| 更稳
    dot = (qa * qb).sum(-1).abs().clamp(max=1.)
    ang = 2 * torch.acos(dot) * 180 / np.pi
    return ang.mean(dim=1)      # [M]
