#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TGA 雅可比协方差形变的纯数学自检（不需要 checkpoint/FLAME 资产）。

只验 apply() 的代数：给定 (R0, s0) 与形变梯度 J，回写出来的 (q', s') 是否
让光栅器的 Sigma = R' diag(s'^2) R'^T 精确等于 J Sigma0 J^T。

运行:
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_covariance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAM3D_ROOT = Path("/workspace/lam3d")
for p in (str(LAM3D_ROOT), "/workspace/LAM"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from lam3d.core import tga_covariance as tga  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             ("  " + detail) if detail else ""))


def make_mod(min_ratio=0.25, max_ratio=1.35):
    """不经过 __init__（那会加载 FLAME 资产），只装配 apply() 需要的标量。"""
    m = tga.TgaCovarianceModulator.__new__(tga.TgaCovarianceModulator)
    torch.nn.Module.__init__(m)
    m.eps = 1e-12
    m.max_scale_ratio = float(max_ratio)
    m.min_scale_ratio = float(min_ratio)
    m.jacobian_strength = 1.0
    return m


def random_rotation(n, dtype=torch.float64):
    q = torch.randn(n, 4, dtype=dtype, device=DEV)
    q = q / q.norm(dim=-1, keepdim=True)
    return q.to(torch.float32)


def sigma_of(q, s):
    R = tga.quat_to_matrix(q.float().to(DEV))
    return R @ torch.diag_embed(s.float().to(DEV) ** 2) @ R.transpose(-1, -2)


def main():
    torch.manual_seed(0)
    n = 4096
    mod = make_mod()

    # ---------------------------------------------------------------- M1 往返
    q0 = random_rotation(n)
    s0 = torch.rand(n, 3, dtype=torch.float32, device=DEV) * 0.02 + 1e-3
    # 温和的 J（接近刚体，不会触发任何截断）
    axis = torch.randn(n, 3, dtype=torch.float64, device=DEV)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = (torch.rand(n, dtype=torch.float64, device=DEV) - 0.5) * 1.2
    J = _rodrigues(axis, ang) + 0.05 * torch.randn(n, 3, 3, dtype=torch.float64, device=DEV)
    with torch.no_grad():
        q1, s1 = mod.apply(None, J, q0, s0)
    S_in = sigma_of(q0, s0).double()
    S_want = J @ S_in @ J.transpose(-1, -2)
    S_got = sigma_of(q1, s1).double()
    rel = ((S_got - S_want).norm(dim=(-1, -2))
           / S_want.norm(dim=(-1, -2)).clamp(min=1e-20))
    report("M1 回写保真: ||R' diag(s'^2) R'^T - J Sigma J^T|| / ||.|| < 1e-3",
           bool(rel.max() < 1e-3),
           "mean={:.3e} p99={:.3e} max={:.3e}".format(
               rel.mean().item(), rel.quantile(0.99).item(), rel.max().item()))

    # ------------------------------------------------------------ M2 中性 J=I
    # 用 **特征值分离良好** 的 s0: 当两个轴的特征值几乎相等时, 该特征子空间
    # 内的基是任意的, 重建出的 R' 可以与 R0 差一个子空间内旋转, 而 Sigma 仍
    # 完全一致 (M1 已证实 1e-7)。那种情况在 M2c 单独验证。
    s_sep = torch.stack((0.020 + torch.rand(n, device=DEV) * 0.01,
                         0.008 + torch.rand(n, device=DEV) * 0.004,
                         0.002 + torch.rand(n, device=DEV) * 0.001), dim=-1)
    eye = torch.eye(3, dtype=torch.float64, device=DEV).expand(n, 3, 3)
    with torch.no_grad():
        qn, sn = mod.apply(None, eye, q0, s_sep)
    # 注意用 **矩阵** 比较而不是 2*acos(|<q,q>|): 后者在单位阵附近把 1e-7 的
    # 浮点残差放大成 0.05 度量级的假误差 (sqrt 型病态), 不是真实偏差。
    dR = (tga.quat_to_matrix(qn) - tga.quat_to_matrix(q0)).abs().amax(dim=(-1, -2))
    ds = (sn.float() - s_sep.float()).abs() / s_sep.float().clamp(min=1e-12)
    report("M2 J=I -> R' == R0（特征值分离良好时, 1e-5）",
           bool(dR.max() < 1e-5),
           "max|dR|={:.3e}".format(dR.max().item()))
    report("M2b J=I -> s' == s0", bool(ds.max() < 1e-2),
           "max rel={:.3e}".format(ds.max().item()))
    # 近退化轴: R' 可以漂移 (子空间基任意), 但 Sigma 必须仍然精确。
    with torch.no_grad():
        qd, sd = mod.apply(None, eye, q0, s0)
    S_d_want = sigma_of(q0, s0).double()
    S_d_got = sigma_of(qd, sd).double()
    rel_d = ((S_d_got - S_d_want).norm(dim=(-1, -2))
             / S_d_want.norm(dim=(-1, -2)).clamp(min=1e-20))
    report("M2c J=I 且尺度近退化 -> Sigma 仍精确（R 可漂移）",
           bool(rel_d.max() < 1e-4),
           "max rel={:.3e}".format(rel_d.max().item()))

    # --------------------------------------------------------- M3 上限贴合
    # 极端拉伸：沿 x 拉伸 8x, 沿 y 压缩 0.1x
    big = torch.zeros(n, 3, 3, dtype=torch.float64, device=DEV)
    big[:, 0, 0] = 8.0
    big[:, 1, 1] = 0.1
    big[:, 2, 2] = 1.0
    with torch.no_grad():
        qb, sb = mod.apply(None, big, q0, s0)
    ratio = (sb.float() / s0.float().clamp(min=1e-30))
    report("M3 极端 J 仍无 NaN/Inf", bool(torch.isfinite(sb).all() and torch.isfinite(qb).all()))
    report("M3b 逐点 s'/s0 <= 1.35 (契约 A6)",
           bool(np_max(ratio) <= 1.35 + 1e-4),
           "max={:.6f}".format(np_max(ratio)))
    report("M3c 逐点 s'/s0 >= min_scale_ratio",
           bool(np_min(ratio) >= 0.25 - 1e-4),
           "min={:.6f}".format(np_min(ratio)))
    qnorm = qb.float().norm(dim=-1)
    report("M3d |q'| == 1", bool(((qnorm - 1).abs() <= 1e-5).all()),
           "max|1-|q||={:.3e}".format((qnorm - 1).abs().max().item()))

    # --------------------------------------------- M4 各向异性确实被引入
    # 单轴拉伸 2x, 检查 Sigma' 的主轴相对 Sigma0 是否真的变扁/变长
    stretch = torch.eye(3, dtype=torch.float64, device=DEV).expand(n, 3, 3).clone()
    stretch[:, 0, 0] = 2.0
    with torch.no_grad():
        qs, ss = mod.apply(None, stretch, q0, s0)
    aniso0 = _aniso(sigma_of(q0, s0).double())
    aniso1 = _aniso(sigma_of(qs, ss).double())
    report("M4 单轴拉伸改变了主轴各向异性（不是各向同性缩放）",
           bool((aniso1 - aniso0).abs().mean() > 1e-3),
           "mean sqrt(l1/l3): {:.4f} -> {:.4f}".format(
               aniso0.mean().item(), aniso1.mean().item()))

    # -------------------------------------------------- M5 四元数往返
    q_rt = tga.matrix_to_quat(tga.quat_to_matrix(q0))
    dR_rt = (tga.quat_to_matrix(q_rt) - tga.quat_to_matrix(q0)).abs().amax(dim=(-1, -2))
    dM_rt = (tga.quat_to_matrix(q_rt) - tga.quat_to_matrix(q0)).abs().amax(dim=(-1, -2))
    report("M5 matrix_to_quat(quat_to_matrix(q)) == R（矩阵级, 1e-5）",
           bool(dM_rt.max() < 1e-5),
           "max|dR|={:.3e}".format(dR_rt.max().item()))

    # ------------------------------------------- M6 与 CUDA computeCov3D 一致
    # 由 diag_cov_path.py 的 (B) 实验确认 sigma_from_sr 与光栅器等价的约定，
    # 这里只复核 quat_to_matrix 的列填充顺序与 computeCov3D 相同。
    q_ref = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32, device=DEV)
    R_ref = tga.quat_to_matrix(q_ref)[0].cpu()
    expect = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    report("M6 quat_to_matrix 与 computeCov3D 的 R_std 约定一致",
           bool((R_ref - expect).abs().max() < 1e-6),
           "R(0.5,0.5,0.5,0.5)={}".format(R_ref.tolist()))

    ok = all(RESULTS)
    print("")
    print("{}/{} PASS".format(sum(RESULTS), len(RESULTS)))
    print("RESULT: {}".format("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


def _rodrigues(axis, ang):
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1.0 - c
    out = torch.stack((
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ), dim=-1)
    return out.reshape(*axis.shape[:-1], 3, 3)


def _aniso(S):
    ev = torch.linalg.eigvalsh(S).clamp(min=1e-30)
    return (ev[..., -1] / ev[..., 0]).sqrt()


def np_max(t):
    return float(t.max().item())


def np_min(t):
    return float(t.min().item())


if __name__ == "__main__":
    sys.exit(main())
