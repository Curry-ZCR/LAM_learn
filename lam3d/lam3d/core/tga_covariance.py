"""TGA 的雅可比协方差形变 —— 沿 LAM 冻结的 (q, s) 通道回写。

论文
----
TGA: True-to-Geometry Avatar Dynamic Reconstruction (NeurIPS 2025 Spotlight),
Perspective-Aware Gaussian Transformation (PGT)::

    J = E_hat * E^-1                    # 三角形静止/形变边矩阵（含单位法向第三列）
    mu' = J * mu + t
    Sigma' = J * R * S * S^T * R^T * J^T    # S 是线性尺度, S S^T = diag(s^2)

它替代了"按父三角形面积做各向同性缩放"的做法 —— 后者在拉伸/剪切方向上
over-blend、在压缩方向上 under-blend, 表现为脸动起来像橡皮。

为什么本模块必须回写 (q, s) 而不是把 Sigma 传进光栅器
----------------------------------------------------
LAM 官方渲染路径 **丢弃一切外部协方差**:

* /workspace/LAM/lam/models/rendering/gs_renderer.py:534 里
  cov3D_precomp = None 是硬编码的（第 530-531 行注释仍写着
  "If precomputed 3d covariance is provided, use it", 但代码并没有这么做）;
* 第 535-536 行无条件用 gs.scaling / gs.rotation 覆盖 scales/rotations;
* 第 556-558 行把 scales/rotations 传给 rasterizer, cov3D_precomp 只传 None。

于是 diff-gaussian-rasterization 的 computeCov3D() 永远走
"由 (scale, rot) 现算 Sigma" 那条分支, Python 侧挂上去的任何 Sigma
(GaussianModel 属性、外部张量) 都不参与渲染。scripts/diag_cov_path.py
用真实 checkpoint 做了同 xyz 的干净 A/B/C 实验证明这一点:

    (A) 按调制后 (s, R) 渲染                                  = 基准
    (B) Python 重建 Sigma -> raw rasterizer cov3D_precomp     ~= 基准 (1.7e-6, 仅 float32 舍入)
    (C) 同一 Sigma 挂成属性、走官方 forward_single_batch       与基准 **逐位相同** (被丢弃)
    (C') 同一 Sigma 走 raw rasterizer                          mean|diff| 0.067 (钩子是好的)

结论: 数学没问题, 是渲染路径丢协方差。所以本模块 **不碰 LAM、不碰 CUDA**,
把 TGA 的 Sigma' 沿同一条 (q, s) 通道精确回写:

    Sigma' --对称特征分解--> R' (特征向量阵), s' (sqrt 特征值)
    q' = quat(R'),  s' = sqrt(lambda)

因为 Sigma' 实对称正定,  R' diag(s'^2) R'^T 与 Sigma' **逐元素相等**
(只有触发 1.35*s0 上限截断的极端轴会有偏差), 所以光栅器从 (q', s') 现算
出来的就是 TGA 的协方差 —— 渲染结果里真的带上了雅可比形变。

与 LAM 原调制器的关系
--------------------
原 DynamicCovarianceModulator 只做面内 (2x2) 旋转 + 各向同性面积缩放
(ds = s0 * 0.35 * tanh(2 * log det A)), 永远无法表达方向性拉伸。本类用
**组合** 复用它的 ARKit basis / 参数校验 / Loop 上采样 (arkit_bs_up 由
FlameHeadSubdivided.upsample_mesh_cpu 得到), 只把 delta_rotation_scale +
apply 换成 3-D 雅可比版本。

调用点是 LAM 里 **冻结** 的三行 (gs_renderer.py:668-674)::

    delta_q, stretch = modulator.delta_rotation_scale(ret["animated"], mean_3d)
    gs_rotation, gs_scaling = modulator.apply(
        delta_q, modulator.delta_scale(stretch, gs_attr.scaling),
        gs_attr.rotation, gs_attr.scaling)

本类把"每高斯 3x3 形变梯度 J"当作那个 payload 传下去
(delta_rotation_scale 的第二个返回值 -> delta_scale 原样透传 -> apply 消费),
delta_q 位置返回 None。这样既不用改 LAM, 也保持 apply(delta_q, delta_scale,
rotation, scaling) 的签名。

契约
----
* ARKit 全 0: 渲染器走静态短路, 本模块不被调用; 即使被调用 J = I 逐位成立。
* |J| 有界: 面退化 (sliver face, det(E) 可小到 1e-8) 时按面积加权并对 J 逐元素
  clamp, 绝不产生 NaN/Inf。
* s' <= max_scale_ratio * s0 (默认 1.35, CONTRACT_v2.md 第 8 节 A6)。
* 无沿着 N 个高斯的 Python for: 只有 einsum / matmul / scatter_add / eigh。
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

import torch
import torch.nn as nn

from . import paths

__all__ = ["TgaCovarianceModulator", "install", "enabled", "DEFAULTS",
           "quat_to_matrix", "matrix_to_quat"]

# ARKit -> FLAME 的基与常量都在 LAM 的这个文件里; 按文件路径加载可以绕开
# lam/__init__.py (它会拖进整个 ModelLAM 与 CUDA 扩展)。
_LAM_MODULE_PATH = (
    paths.LAM_ROOT / "lam" / "models" / "rendering" / "covariance_modulation.py"
)
_BASE_MOD = None

DEFAULTS = {
    "jacobian_strength": 1.0,   # J_eff = I + strength * (J - I); 1.0 = 论文原式
    "max_scale_ratio": 1.35,    # s' / s0 的硬上限 (CONTRACT_v2.md 第 8 节 A6)
    # 每轴压缩下限。契约只约束上限 (1.35), 但不设下限时一个被强剪切的轴可以
    # 缩到 ~0, 高斯在该方向塌成一条线、深度排序里直接消失。取 0.25 = 允许 4x
    # 压缩 (远宽于原调制器的 0.65 各向同性下限), 同时保住数值安全。
    "min_scale_ratio": 0.25,
}


def _base_module():
    """按文件路径惰性加载 LAM 的 covariance_modulation (只取基类与元数据)。"""
    global _BASE_MOD
    if _BASE_MOD is None:
        if not _LAM_MODULE_PATH.is_file():
            raise FileNotFoundError(
                "找不到 LAM 协方差调制模块: {} (请确认 LAM_ROOT 指向 LAM 检出)".format(
                    _LAM_MODULE_PATH
                )
            )
        spec = importlib.util.spec_from_file_location(
            "lam_covariance_modulation_tga", _LAM_MODULE_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _BASE_MOD = mod
    return _BASE_MOD


# ---------------------------------------------------------------- 数值小工具
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """wxyz 四元数 -> 3x3 旋转矩阵 (与 CUDA computeCov3D 同一约定)。

    computeCov3D 内部把 R 按列填充 (1-2(y^2+z^2), 2(xy+wz), 2(xz-wy)),
    即它的 R = R_std^T、Sigma = R^T S^2 R = R_std S^2 R_std^T。这里直接给出
    R_std, 因此 R_std diag(s^2) R_std^T 与光栅器逐位 (到 float32 舍入) 一致
    —— 已由 diag_cov_path.py 的 (B) 实验证实 (1.7e-6)。
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(dim=-1)
    r = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """3x3 旋转矩阵 -> wxyz 四元数 (Shepperd 四分支, 用 where 合成, 无 Python 分支)。

    输入必须是 det=+1 的正交阵 (调用方负责修 det)。返回 w >= 0 的规范形式。
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    eps = 1e-12

    s0 = (trace + 1.0).clamp(min=eps).sqrt() * 2.0
    q0 = torch.stack(
        (0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0), dim=-1
    )
    s1 = (1.0 + m00 - m11 - m22).clamp(min=eps).sqrt() * 2.0
    q1 = torch.stack(
        ((m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1), dim=-1
    )
    s2 = (1.0 + m11 - m00 - m22).clamp(min=eps).sqrt() * 2.0
    q2 = torch.stack(
        ((m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2), dim=-1
    )
    s3 = (1.0 + m22 - m00 - m11).clamp(min=eps).sqrt() * 2.0
    q3 = torch.stack(
        ((m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3), dim=-1
    )

    best = torch.stack((trace, m00, m11, m22), dim=-1).argmax(dim=-1, keepdim=True)
    cand = torch.stack((q0, q1, q2, q3), dim=-2)            # [..., 4(候选), 4(wxyz)]
    q = cand.gather(-2, best.unsqueeze(-1).expand(*best.shape[:-1], 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def inv3(m: torch.Tensor, det_floor: float = 1e-12) -> torch.Tensor:
    """批量 3x3 求逆 (伴随矩阵闭式, det 带下限)。m: [..., 3, 3]。"""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    c00 = e * i - f * h
    c01 = f * g - d * i
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    sign = torch.where(det < 0, -1.0, 1.0)
    safe = torch.where(det.abs() < det_floor, sign * det_floor, det)
    adj = torch.stack(
        (
            c00, c * h - b * i, b * f - c * e,
            c01, a * i - c * g, c * d - a * f,
            c02, b * g - a * h, a * e - b * d,
        ),
        dim=-1,
    ).reshape(m.shape)
    return adj / safe.unsqueeze(-1).unsqueeze(-1)


# ------------------------------------------------------------------ 调制器
class TgaCovarianceModulator(nn.Module):
    """ARKit 驱动的 3-D 雅可比动态协方差形变 (TGA / PGT)。

    Args:
        flame_model: renderer 持有的 FlameHeadSubdivided (取 faces_up 与其自带的
            Loop 上采样器)。
        arkit_bs_path: 可选, ARKit 基文件; 默认沿用 LAM 的内置 flame_arkit_bs.npy。
        jacobian_strength: J_eff = I + s * (J - I); 1.0 = 论文原式, 小于 1 是保守
            退让 (调试/性能对比用)。
        max_scale_ratio: s' <= max_scale_ratio * s0 的硬上限, 默认 1.35
            (CONTRACT_v2.md 第 8 节 A6 / scripts/arkit_splat_test.py:524)。
    """

    def __init__(
        self,
        flame_model: nn.Module,
        arkit_bs_path: Optional[str] = None,
        *,
        jacobian_strength: float = DEFAULTS["jacobian_strength"],
        max_scale_ratio: float = DEFAULTS["max_scale_ratio"],
        min_scale_ratio: float = DEFAULTS["min_scale_ratio"],
        eps: float = 1e-12,
    ):
        super().__init__()
        if not 0.0 < jacobian_strength <= 1.0:
            raise ValueError(
                "jacobian_strength must be in (0, 1]; got {!r}".format(jacobian_strength)
            )
        if not max_scale_ratio > 1.0:
            raise ValueError(
                "max_scale_ratio must be > 1; got {!r}".format(max_scale_ratio)
            )
        if not 0.0 < min_scale_ratio < 1.0:
            raise ValueError(
                "min_scale_ratio must be in (0, 1); got {!r}".format(min_scale_ratio)
            )
        base_mod = _base_module()
        # 组合而不是继承: LAM 的基类必须按文件路径加载, 因此这里持有一个实例。
        self.base = base_mod.DynamicCovarianceModulator(
            flame_model, arkit_bs_path, eps=eps
        )
        self.jacobian_strength = float(jacobian_strength)
        self.max_scale_ratio = float(max_scale_ratio)
        self.min_scale_ratio = float(min_scale_ratio)
        self.eps = float(eps)

    # ---------------------------------------------------------------- 透传
    @property
    def num_gaussians(self) -> int:
        return self.base.num_gaussians

    @property
    def n_arkit(self) -> int:
        return self.base.n_arkit

    @property
    def faces(self) -> torch.Tensor:
        return self.base.faces

    @property
    def faces_flat(self) -> torch.Tensor:
        return self.base.faces_flat

    @property
    def arkit_bs_up(self) -> torch.Tensor:
        return self.base.arkit_bs_up

    @staticmethod
    def is_neutral(arkit_params) -> bool:
        return _base_module().DynamicCovarianceModulator.is_neutral(arkit_params)

    def prepare_params(self, arkit_params, num_view: int, device, dtype):
        return self.base.prepare_params(arkit_params, num_view, device, dtype)

    def arkit_offsets(self, arkit_params: torch.Tensor) -> torch.Tensor:
        return self.base.arkit_offsets(arkit_params)

    # ------------------------------------------------------- 每面雅可比
    def face_jacobians(self, x0: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """逐面 3x3 形变梯度 J_f = E_hat * E^-1 (TGA 的 PGT)。

        E  = [e1, e2, n0]: 静止边对 + 单位法向 (列向量);
        E^ = [d1, d2, n1]: 形变后的同一构造。第三列取 **单位** 法向, 与论文
        一致 (只有前两列承载尺度), 保证 J 在法向上不注入虚假膨胀。

        Returns:
            [B, F, 3, 3] 的 J。
        """
        faces = self.faces.to(x0.device)
        a, b, c = faces.unbind(dim=-1)
        e1 = x0[..., b, :] - x0[..., a, :]
        e2 = x0[..., c, :] - x0[..., a, :]
        d1 = xB[..., b, :] - xB[..., a, :]
        d2 = xB[..., c, :] - xB[..., a, :]

        n0 = torch.cross(e1, e2, dim=-1)
        n0 = n0 / n0.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        n1 = torch.cross(d1, d2, dim=-1)
        n1 = n1 / n1.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        E = torch.stack((e1, e2, n0), dim=-1)      # [B,F,3,3], 列向量
        Eb = torch.stack((d1, d2, n1), dim=-1)
        J = torch.matmul(Eb, inv3(E))
        return torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).clamp(-64.0, 64.0)

    def vertex_jacobians(self, x0: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """把逐面 J_f 按 **面积加权** 平均到顶点 (one-ring), 得到 J_v。

        面积加权是必要的: sliver 面的 E^-1 病态 (det 可到 1e-8), 等权平均会被
        少数退化面主导。权重为 0 的面贡献 0, 配合 nan_to_num 不会污染。
        没有任何入射面的孤立顶点回落到恒等阵, 绝不产生 NaN。
        """
        Jf = self.face_jacobians(x0, xB)                        # [B,F,3,3]
        faces = self.faces.to(x0.device)
        a, b, c = faces.unbind(dim=-1)
        e1 = x0[..., b, :] - x0[..., a, :]
        e2 = x0[..., c, :] - x0[..., a, :]
        area = 0.5 * torch.cross(e1, e2, dim=-1).norm(dim=-1)   # [B,F]
        area = torch.nan_to_num(area, nan=0.0).clamp(min=0.0)

        batch, n_face = area.shape
        n_vert = x0.shape[-2]
        idx = self.faces_flat.to(x0.device).view(1, -1, 1).expand(batch, -1, 9)
        jac3 = (Jf * area[..., None, None]).reshape(batch, n_face, 9)
        num = torch.zeros(batch, n_vert, 9, device=x0.device, dtype=x0.dtype)
        num.scatter_add_(1, idx, jac3.repeat_interleave(3, dim=1))
        idx1 = self.faces_flat.to(x0.device).view(1, -1).expand(batch, -1)
        area3 = area.repeat_interleave(3, dim=1)                # [B,3F]
        den = torch.zeros(batch, n_vert, device=x0.device, dtype=x0.dtype)
        den.scatter_add_(1, idx1, area3)

        # num/den 都是 [B, N, 9]（J 摊平成 9 个通道，便于一次 scatter_add_）
        Jv = num / den.clamp(min=self.eps).unsqueeze(-1)
        eye_flat = torch.eye(3, device=x0.device, dtype=x0.dtype).reshape(1, 1, 9)
        Jv = torch.where((den <= self.eps).unsqueeze(-1), eye_flat.expand_as(Jv), Jv)
        return Jv.view(batch, n_vert, 3, 3)

    # ------------------------------------------- 与 LAM 冻结调用点对接
    def delta_rotation_scale(self, x0: torch.Tensor, xB: torch.Tensor):
        """返回 (None, J_v): [B,N,3,3] 的每高斯形变梯度作为 payload。

        渲染器 (gs_renderer.py:668-674) 把第二个返回值交给本类的 delta_scale
        再交给 apply, 因此 payload 原样透传。第一个返回值 (LAM 语义里的
        delta_q) 在本类里没有意义, 返回 None —— apply 不读它。
        """
        Jv = self.vertex_jacobians(x0, xB)
        strength = self.jacobian_strength
        if strength != 1.0:
            eye = torch.eye(3, device=Jv.device, dtype=Jv.dtype)
            Jv = eye + strength * (Jv - eye)
        return None, Jv

    def delta_scale(self, payload: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
        """透传: 本实现把 s' 的求解放在 apply (需要同时看 rotation)。"""
        return payload

    def apply(self, delta_q, jacobian: torch.Tensor,
              rotation: torch.Tensor, scaling: torch.Tensor):
        """Sigma' = J Sigma J^T -> 对称特征分解 -> (q', s')。

        Args:
            delta_q: 未使用 (LAM 调用点的占位参数)。
            jacobian: [B, N, 3, 3] 每高斯形变梯度。
            rotation: [N, 4] 规范四元数; scaling: [N, 3] 线性尺度。
        Returns:
            (q', s'), 形状 [B, N, 4] / [B, N, 3]。

        因为 Sigma' 实对称正定, R' diag(s'^2) R'^T == Sigma' 逐元素成立, 所以
        写回去的 (q', s') 让光栅器从 (s, R) 现算出来的就是 TGA 的协方差。
        """
        # 渲染器给的是 [num_view, N, 3, 3]；也接受单批次 [N, 3, 3]。
        single = jacobian.dim() == 3
        if single:
            jacobian = jacobian.unsqueeze(0)
        B, N = jacobian.shape[0], jacobian.shape[1]
        R0 = quat_to_matrix(rotation.float())                       # [N,3,3]
        s0 = scaling.float().reshape(N, 3)                          # [N,3]
        S0 = R0 @ torch.diag_embed(s0 * s0) @ R0.transpose(-1, -2)  # [N,3,3]

        # eigh 在 float64 里做: 尺度动态范围大 (s 跨几个数量级), float32 的近退化
        # 特征分解会带来 1e-3 量级的相对误差。
        J64 = jacobian.double()
        Sig = J64 @ S0.double() @ J64.transpose(-1, -2)
        Sig = 0.5 * (Sig + Sig.transpose(-1, -2))
        lam, vec = torch.linalg.eigh(Sig)                     # 升序: lam[...,0] <= ...
        s_eig = torch.nan_to_num(lam.sqrt(), nan=0.0, posinf=0.0).clamp(min=self.eps)

        # ---- 契约 A6 的尺度上限 (逐轴, 上限 1.35 / 下限 min_scale_ratio)。
        # Sigma0 = R0 diag(s0^2) R0^T 的特征值恰好是 s0^2 这个 **集合**(只是顺序
        # 不同), 所以 Sigma' 的特征值与 **升序** 的 s0 一一对应, 可以逐轴算 ratio。
        # 夹住之后无论 J 多病态, 都不会出现"某轴涨 5 倍 / 缩到 0"的爆炸或吞噬;
        # 形态上等价于原调制器的 tanh 软截断, 只是方向性由 J 决定、不再各向同性。
        rmax = float(self.max_scale_ratio)
        rmin = float(self.min_scale_ratio)
        s0_sorted, s_idx = torch.sort(s0.double(), dim=-1)                  # [N,3] 升序
        ratio = s_eig / s0_sorted.unsqueeze(0).clamp(min=self.eps)          # [B,N,3]
        ratio = ratio.clamp(min=rmin, max=rmax)
        s_capped = s0_sorted.unsqueeze(0) * ratio                           # [B,N,3]

        # ---- 回到 s0 的原始槽位。
        # 升降序对齐只是为了让特征值与 s0 的分量一一对上; 写回时必须回到原下标,
        # 因为 LSPL 消费者按 **下标** 比较 s'/s0 (scripts/arkit_splat_test.py:524
        # 逐点断言 max(s/s0) <= 1.35)。s_idx 是置换, 每个槽位恰好写一次; vec 的
        # 列按同一置换搬动, 所以 Sigma = R diag(s^2) R^T 保持不变。
        s_out = torch.zeros_like(s_capped)
        s_out.scatter_(-1, s_idx.unsqueeze(0).expand_as(s_capped), s_capped)
        vec_out = torch.zeros_like(vec)
        vec_out.scatter_(-1, s_idx.view(1, N, 1, 3).expand(B, N, 3, 3), vec)

        # ---- 特征向量的符号必须逐列对齐到原旋转 R0。
        # eigh 的列符号是完全任意的, 而 Sigma 对"偶数次翻列"是盲的 —— 也就是说
        # 除了 Sigma 之外, 重建出的 R' 可能与 R0 相差一个 180 度翻转, 视觉上等价
        # 但会让 "无形变应该还原原旋转" 这一性质失效。用与 R0 对应列的符号积
        # 逐列翻转即可同时得到: (a) R' 与 R0 最近的旋转, (b) det(R') = +1
        # (因为 R' 与 R0 的列都正交归一且符号对齐)。Sigma 完全不变。
        dots = (vec_out * R0.double()).sum(dim=-2)                          # [B,N,3]
        vec_out = vec_out * torch.where(dots < 0, -1.0, 1.0).unsqueeze(-2)

        # 兜底: 极端病态 (特征值几乎相等, 特征子空间在内部任意旋转) 时对齐不一定
        # 能保证 det = +1; 那就翻转最小特征值那一列 —— 它不改变 Sigma。
        bad = (torch.linalg.det(vec_out) < 0).unsqueeze(-1).unsqueeze(-1)
        last = torch.zeros_like(vec_out)
        last[..., -1:] = -vec_out[..., -1:]
        vec_out = torch.where(bad, torch.cat((vec_out[..., :-1], last[..., -1:]), dim=-1),
                              vec_out)

        # 四元数也在 float64 里解出再降精度: 否则 R 里那 1e-7 级的正交性残差会被
        # 放大成 ~0.06 度的旋转误差。
        q = matrix_to_quat(vec_out).to(torch.float32)
        s_final = s_out.to(torch.float32).clamp(min=self.eps)
        if single:
            return q[0], s_final[0]
        return q, s_final


# ------------------------------------------------------------------- 注入
def enabled() -> bool:
    """环境变量开关: LAM3D_TGA_COV=0 退回 LAM 原调制器 (对照/回滚用)。"""
    return os.environ.get("LAM3D_TGA_COV", "1") not in ("0", "false", "False", "")


def install(renderer, *, force: bool = False, **kwargs) -> Optional[nn.Module]:
    """把 TGA 调制器装进 renderer (替换 _cov_modulator, 无需改 LAM)。

    GSRenderer._get_cov_modulator (gs_renderer.py:571-578) 是惰性的: 只有
    self._cov_modulator 为 None 时才构造 LAM 自带的 DynamicCovarianceModulator。
    因此预先把它设成我们的实例, animate_gs_model 就会走 TGA 路径。会话建立
    时调用一次。

    Returns:
        装好的调制器; 若开关关闭或已装过 (非 force) 则返回已有/None。
    """
    if not enabled():
        return None
    existing = getattr(renderer, "_cov_modulator", None)
    if existing is not None and not force:
        return existing
    params = dict(DEFAULTS)
    for k, env_key in (("jacobian_strength", "LAM3D_TGA_STRENGTH"),
                       ("max_scale_ratio", "LAM3D_TGA_MAX_RATIO"),
                       ("min_scale_ratio", "LAM3D_TGA_MIN_RATIO")):
        raw = os.environ.get(env_key)
        if raw:
            params[k] = float(raw)
    params.update(kwargs)
    mod = TgaCovarianceModulator(
        renderer.flame_model,
        jacobian_strength=params["jacobian_strength"],
        max_scale_ratio=params["max_scale_ratio"],
        min_scale_ratio=params["min_scale_ratio"],
    )
    device = getattr(renderer, "device", None) or torch.device("cuda")
    mod.to(device)
    mod.eval()
    renderer._cov_modulator = mod
    return mod
