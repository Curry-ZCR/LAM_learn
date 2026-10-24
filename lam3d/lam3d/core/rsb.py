# -*- coding: utf-8 -*-
"""RSB（Rotation + Surface-Bundle）协方差调制器 —— 方法四。

在 RotSync（方法一：逐面极分解 + Markley 平均 → 逐高斯刚体旋转）之上，加一层
「法向协变修正」：
  1) 首帧从 x0（ARKit 中性、FLAME posed 的规范姿态网格）缓存面积加权顶点法向 n0v；
  2) 每帧用父类透出的逐面 Jf 算协变法向 n_v = normalize(J_f^{-T} n_0) 的面积加权
     聚合（严格协变，剪切下与 R·n0 不同）；
  3) 对 delta_q 旋转后的规范法向 u = R(delta_q)·n0v 与 n_v 之间的夹角做
     tanh 软限幅的最小旋转修正 fix，并以门控系数 a（限幅×coherence 门）加权；
  4) apply: q' = normalize(fix ⊗ (delta_q ⊗ q0))，尺度逐位透传（s'≡s0）。

设计依据：lam3d/docs/method4_rsb_design.md §9（伪码/衔接点/陷阱表/验收 W1–W8）。
实测（原型，5 preset capture）：rig50 8.501→5.507（−35.2%），rig90 −17.7%，
jitter −18.2%；推荐 τ=0.3, p=1.5, θ_cap=45°, n_cap=25°，尺度支路 OFF。
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .rot_sync import RotSyncModulator, quat_multiply, install as _rot_install

_DEG = math.pi / 180.0


def _rotate_by_quat(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用四元数 q=[w,x,y,z] 旋转矢量 v（两次叉积形式，[...,4]×[...,3]→[...,3]）。"""
    w = q[..., :1]
    u = q[..., 1:]
    return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


class RsbModulator(RotSyncModulator):
    """RotSync + 法向协变修正。参数与父类一致，另加 4 个（默认=实测推荐值）。"""

    def __init__(
        self,
        flame_model,
        arkit_bs_path: Optional[str] = None,
        *,
        normal_cap_deg: float = 25.0,
        enable_normal_fix: bool = True,
        use_stretch: bool = False,
        gate_power: float = 1.5,
        **kwargs,
    ):
        super().__init__(flame_model, arkit_bs_path, **kwargs)
        self.normal_cap_deg = float(normal_cap_deg)
        self.enable_normal_fix = bool(enable_normal_fix)
        self.use_stretch = bool(use_stretch)
        self.gate_power = float(gate_power)
        self._n0v: Optional[torch.Tensor] = None   # [N,3] 首帧缓存（规范姿态法向）
        self._fix_diag: Optional[dict] = None

    # ------------------------------------------------ 法向核（规范/协变共用同一路径）
    def _vertex_normals(self, x: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """面积加权顶点法向 [B,N,3]，权重用静止面积 area [B,F]。

        face 法向 n_f = normalize(cross(d1, d2))（循环边 d1=v0-v2, d2=v1-v0）。
        变形网格上这就是**严格协变**方向：cross(J d1, J d2) = det(J)·J^{-T} cross(d1,d2)，
        归一化后即 normalize(J^{-T} n_0)——比从 Jf 取列更准（J 的第三列是 J·e3，
        不是 J·n0）且省一次 3×3 逆。det<0 的翻折面法向自动翻号（协变本义）。
        """
        B = x.shape[0]
        f = self.faces_flat                                  # [3F] int64
        v0 = x[:, f[0::3]]; v1 = x[:, f[1::3]]; v2 = x[:, f[2::3]]
        d1 = v0 - v2
        d2 = v1 - v0
        nf = F.normalize(torch.cross(d1, d2, dim=-1), dim=-1, eps=self.eps)  # [B,F,3]
        nv = x.new_zeros(B, x.shape[1], 3)
        nv.index_add_(1, f, (nf * area[..., None]).repeat_interleave(3, dim=1))
        den = x.new_zeros(B, x.shape[1])
        den.index_add_(1, f, area.repeat_interleave(3, dim=1))
        return F.normalize(nv / den.clamp_min(self.eps)[..., None], dim=-1, eps=self.eps)

    # ------------------------------------------------ W1: 规范法向（首帧惰性缓存）
    def _canonical_normals(self, x0: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """n0v [N,3]，首帧缓存（规范姿态 = ARKit 中性下的 FLAME posed 网格）。"""
        if self._n0v is not None and self._n0v.device == x0.device:
            return self._n0v
        n0v = self._vertex_normals(x0, area)[0]              # 与协变同一路径 → 恒等帧逐位一致
        self._n0v = n0v
        return n0v

    # ------------------------------------------------ W2: 协变法向（每帧，直接从 xB 网格算）
    def _covariant_normals(self, xB: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """n_v = normalize(Σ_f a_f · n_f(xB)) [B,N,3]，a_f 为静止面积。"""
        return self._vertex_normals(xB, area)

    # ------------------------------------------------ W3/W4: 法向修正四元数
    def _normal_fix(
        self, delta_q: torch.Tensor, n0v: torch.Tensor, nv: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """u=R(delta_q)·n0v → nv 的最小旋转，tanh 软限幅 + 门控加权 → fix [B,N,4]。"""
        u = _rotate_by_quat(delta_q, n0v.unsqueeze(0).to(delta_q.dtype))   # [B,N,3]
        cr = torch.cross(u, nv, dim=-1)
        s = cr.norm(dim=-1)                                  # sin θ
        c = (u * nv).sum(-1)                                 # cos θ
        theta = torch.atan2(s, c)                            # [B,N]
        cap = max(self.normal_cap_deg * _DEG, 1e-6)
        theta_t = cap * torch.tanh(theta / cap)              # 软限幅
        a = gate.clamp(0.0, 1.0) ** self.gate_power          # 门控（限幅×coh 门）^p
        half = a * theta_t * 0.5
        axis = cr / s.clamp_min(1e-9).unsqueeze(-1)
        fix = torch.cat(
            [torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * axis], dim=-1
        )
        # a==0 或 θ→0 → fix 严格 = [1,0,0,0]（浮点吸附，保证恒等帧逐位透传）：
        # θ=0 时 axis=cr/‖cr‖ 是 0/0，clamp 后是尘级方向，sin(half)≈0 但不严格为 0，
        # 会让 W8-2b 类逐位断言看到 1e-8 级残差。
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=fix.device, dtype=fix.dtype)
        # 1e-6 rad ≈ 5.7e-5 deg：恒等帧 n0v/nv 同路径浮点尘 ~1e-8 rad 在此之下，
        # 物理转角（死区之上）不会触达。
        pass_through = (a <= 0.0) | (theta.abs() < 1e-6)
        return torch.where(pass_through.unsqueeze(-1), ident, fix)

    # ------------------------------------------------ 与 LAM 冻结调用点对接
    def delta_rotation_scale(self, x0: torch.Tensor, xB: torch.Tensor):
        # SimpleNamespace 自检范式下 super() 不可用，显式走父类实现。
        delta_q, payload = RotSyncModulator.delta_rotation_scale(self, x0, xB)
        if not self.enable_normal_fix:
            self._fix_diag = {"disabled": True}
            return delta_q, payload
        diag = dict(payload)
        area, gate = payload["area"], payload["gate"]
        n0v = self._canonical_normals(x0, area)
        nv = self._covariant_normals(xB, area)
        fix = self._normal_fix(delta_q, n0v, nv, gate)
        diag["fix"] = fix
        theta_fix = 2.0 * torch.acos(fix[..., 0].clamp(-1.0, 1.0).abs())
        a = gate.clamp(0.0, 1.0) ** self.gate_power
        normal_diag = {
            "normal_angle_deg": (theta_fix / _DEG).detach(),
            "normal_applied_deg": ((theta_fix / _DEG) * a).detach(),
        }
        diag.update(normal_diag)
        self._fix_diag = normal_diag
        self._last_diag = diag          # 与父类同口径，scripts 里 _last_diag 消费者可用
        return delta_q, diag

    def delta_scale(self, payload, scaling: torch.Tensor) -> torch.Tensor:
        return payload                    # 尺度逐位透传（use_stretch=False 冻结）

    def apply(self, delta_q, payload, rotation, scaling):
        """q' = normalize(fix ⊗ (delta_q ⊗ q0))，s' = s0（逐位透传）。

        与父类同约定：rotation [N,4] / scaling [N,3]（规范单帧），返回
        (q' [B,N,4], s' [B,N,3])。严格恒等（fix=I 且 delta_q=I）处直接透传 q0，
        保持与 RotSync / LAM 基线的逐位一致语义。
        """
        fix = payload.get("fix") if isinstance(payload, dict) else None
        q0 = rotation.unsqueeze(0).to(dtype=delta_q.dtype)
        if fix is None:
            q = quat_multiply(delta_q, q0)
            is_identity = (delta_q[..., 0] == 1.0) & (delta_q[..., 1:] == 0.0).all(-1)
        else:
            q = quat_multiply(quat_multiply(fix.to(delta_q.dtype), delta_q), q0)
            is_identity = (
                (delta_q[..., 0] == 1.0) & (delta_q[..., 1:] == 0.0).all(-1)
                & (fix[..., 0] == 1.0) & (fix[..., 1:] == 0.0).all(-1)
            )
        q = torch.where(
            is_identity.unsqueeze(-1), q0, F.normalize(q, dim=-1, eps=self.eps)
        )
        s = scaling.unsqueeze(0).to(dtype=delta_q.dtype).expand(
            delta_q.shape[0], -1, -1
        ).contiguous()
        return q, s


# ------------------------------------------------------------------- 注入
def enabled() -> bool:
    return os.environ.get("LAM3D_RSB", "0") not in ("0", "false", "False", "")


def install(renderer, *, force: bool = False, **kwargs):
    """把 RSB 装进 renderer._cov_modulator（占位接管，不改 LAM 一行）。"""
    if not enabled():
        return None
    existing = getattr(renderer, "_cov_modulator", None)
    if existing is not None and not force:
        return existing
    params = {
        "normal_cap_deg": 25.0,
        "enable_normal_fix": True,
        "use_stretch": False,
        "gate_power": 1.5,
    }
    for key, env_key in (
        ("normal_cap_deg", "LAM3D_RSB_NORMAL_CAP"),
        ("enable_normal_fix", "LAM3D_RSB_FIX"),
        ("use_stretch", "LAM3D_RSB_STRETCH"),
        ("gate_power", "LAM3D_RSB_GATE_P"),
    ):
        raw = os.environ.get(env_key)
        if raw is not None and raw != "":
            params[key] = raw if key == "enable_normal_fix" else float(raw)
    params.update(kwargs)
    mod = RsbModulator(renderer.flame_model, **params)
    device = getattr(renderer, "device", None) or torch.device("cuda")
    mod.to(device)
    mod.eval()
    renderer._cov_modulator = mod
    return mod
