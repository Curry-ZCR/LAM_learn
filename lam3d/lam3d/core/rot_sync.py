"""旋转同步（Rotation Sync）—— 让高斯朝向跟着皮肤局部形变转，严格保轴长。

一句话
------
把每帧的局部形变梯度 J 做极分解 J = R S，**只取旋转因子 R**，按
Sigma' = R Sigma0 R^T 更新高斯朝向。特征值（三个轴长）逐位不变，所以这是
"只转方向、不拉伸"的最小改动。

与 TGA 的关系（同一族的两端）
----------------------------
tga_covariance.py 用完整的 Sigma' = J Sigma0 J^T，等于同时应用了 R 与 S。
本模块把它砍成 Sigma' = R Sigma0 R^T（等价于把 S 换成单位阵）。两者共用同一套
逐面/逐顶形变梯度与 (q, s) 回写通道，因此可以互为消融对照：

    mode="rot"    本模块    保轴长，只改朝向     —— 用户报告的方法一
    mode="full"   TGA       改朝向 + 改轴长      —— 报告方法二的上界

为什么不是"重复计算"
--------------------
用户报告 5.1 担心全局头部旋转与局部表情旋转叠加两遍。这条在本管线上**不成立**，
原因有三，都可核验：

1. arkit_live.neutral_flame_data() 把 rotation / neck_pose / jaw_pose /
   eyes_pose / translation / expr 全部 zero_()，只留 betas。ARKit 会话里的头部
   姿态由**相机**（orbit_c2w）转，不写进 FLAME 的 pose。
2. animate_gs_model 的未调制分支把 gs_attr.rotation **原样拷贝**到每个视角
   （gs_renderer.py:683-686），也就是说基线对高斯朝向的变换是恒等 B = I。
3. x0 = animation_forward(dynamic_offset=None) 与 xB = animation_forward(
   dynamic_offset=arkit_offsets) 用的是**同一组** pose 参数，所以逐面差分
   J = E_b E^-1 只包含 ARKit 动态偏移带来的局部形变，LBS 的刚体部分被约掉。

于是报告 8.2 的残差式在这里退化成 C = J B^-1 = J，
A = Exp(cR Log R_C) Exp(cS log S_C) B 取 cR = 1, cS = 0 就是本模块。
两个门都开到 1 才是 TGA。这就是"新增控制只作用在尚未处理的残差上"。

为什么值得做（本模型上方法一不是"转了等于没转"）
------------------------------------------------
用户报告 5.4 担心"接近球形的高斯转了也看不出来"。实测 LAM 的 20018 个规范高斯
（workspace/jobs/1cce50fbc0d0/input_gaussians.ply）极度各向异性：

    lmax/lmin   p50 = 3.37   p75 = 8.29   p90 = 83.0   max = 1.9e6
    各向异性 > 2 的占比 80.4%，> 5 的占比 35.9%

也就是绝大多数高斯是薄片/细条，朝向一错就斜插出皮肤表面；反过来，朝向一改，
屏幕足迹立刻明显变化。方法一在本模型上不是收益小，而是收益最直接。

工程上必须补的六件事（报告没写，但实测会咬人）
----------------------------------------------
1. **逐面极分解后再聚合，而不是把 J 平均后再极分解**。下颌边界一带有相邻面朝
   相反方向转，平均后的 J 可能接近奇异，极分解出来的 R 会跳变。逐面取 R 再
   在 SO(3) 上聚合（ASAP 的旋转场）才稳。
2. **四元数聚合对 q / -q 双覆盖免疫**。q 与 -q 是同一个旋转，直接逐元素求和
   会在符号不一致的顶点互相抵消。本模块用 Markley 平均（每个顶点解
   argmax q^T (sum_f A_f q_f q_f^T) q，即 4x4 外积矩阵的主特征向量）——
   外积里符号自动抵消，所以既不需要参考四元数做半球对齐，也不需要为此多算
   一次极分解。实测：原实现为了定半球额外算了 polar(J_v) + matrix_to_quat，
   共 3.5 ms，占当时旋转场的 37%；改用 Markley 后这两步全部省掉。
   反例（曾踩过）：拿 Markley 结果只当"半球参考"再做线性平均，特征向量在
   近简并时方向不稳，反而把个别面翻错，25 度刚体旋转测出 std=1.66 度。
   **拿一个可能不稳的量当参考系，比不要参考系更危险。**
3. **det(J) < 0 的翻转面**。极分解的 R 在 det<0 时是反射，转成四元数会得到
   垃圾。必须投影到最近的旋转（翻掉最小奇异值对应的那一列）。实测真实 J 上
   det(U) < 0 的计数为 0，这条是兜底而非日常路径。
4. **限幅 + 置信度门控，且两者合成一次 slerp**。J 在 sliver 面上可达 64（见
   face_jacobians 的 clamp），极分解出来的角度可能是 170 度。按 max_angle 沿
   测地线回缩，避免个别高斯被甩飞；同时用"一环内面旋转的合矢量长度"作
   置信度（Markley 的特征值间隔 lambda1 - lambda2 与它等价，免费拿到）。
   实测这类无定义顶点有 46~76 个（占 0.2%~0.4%），一环面夹角中位数 176 度。
   把限幅系数与门控系数**乘成一个 t_shrink**、最后只做一次 slerp；
   实测两次 slerp 会多约 2.3 ms。
5. **恒等吸附**。F.normalize 与极分解有 1e-8 级残差，会让中性帧的 delta_q 与
   [1,0,0,0] 逐位不等，破坏"与改动前像素级一致"的可验证性。所以在算完角度
   后加一道 angle < IDENT_EPS(1e-6 rad) 的吸附，命中处直接置回严格单位四元数。
6. **孤立顶点兜底**。_markley_mean 额外返回 wsum（一环面积和）；wsum <= eps
   时 4x4 矩阵是零阵，特征向量无意义，必须显式置回单位四元数并把置信度置 0。
   真实 LAM 网格没有孤立顶点，但这是"可被任意网格复用"的必要防御。

坐标与契约
----------
* 全部用 wxyz Hamilton 四元数，与 covariance_modulation.quaternion_multiply
  和 tga_covariance.quat_to_matrix 同一约定。
* ARKit 全 0 时渲染器走静态短路，本模块不被调用；即便被调用，J = I ->
  R = I -> q' = q0（误差 < 1e-6 度），s' = s0 **逐位相等**。
* 无沿 N 个高斯的 Python for；只有 matmul / einsum / scatter_add / 迭代求逆。
* 不改 /workspace/LAM/** 下任何文件。

用法
----
    from lam3d.core import rot_sync
    rot_sync.install(renderer)                        # 装进 renderer._cov_modulator
    renderer._cov_modulator = tga_covariance.install(renderer, force=True)   # 换回 TGA

环境变量
--------
    LAM3D_ROT_SYNC=0       关掉（install 返回 None，退回 LAM 原调制器）
    LAM3D_ROT_GAIN=1.0     旋转门 cR，Exp(cR Log R)，[0,1]
    LAM3D_ROT_MAX_ANGLE=45 逐高斯旋转角上限（度），<=0 表示不限
    LAM3D_ROT_DEADZONE=0   低于该角度（度）的旋转视为噪声清零
    LAM3D_ROT_AVG=face     face = 逐面极分解后按面积加权平均；
                           vertex = 先把 J 按面积加权平均再极分解（对照用）
    LAM3D_ROT_SMOOTH=0     时间 EMA 系数 [0,1)
    LAM3D_ROT_COH=0        置信度门控 tau：一环合矢量长度 R < tau 的高斯不转，
                           (R-tau)/(1-tau) 之间线性过渡。0 = 关闭
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tga_covariance import _base_module, inv3, matrix_to_quat, quat_to_matrix

__all__ = [
    "DEFAULTS",
    "RotSyncModulator",
    "polar_rotation",
    "quat_multiply",
    "quat_angle",
    "slerp_identity",
    "enabled",
    "install",
]

DEG = 3.141592653589793 / 180.0

# 小于该角度（弧度）的旋转直接吸附成严格单位四元数。
# 1e-6 rad = 5.7e-5 度，远小于任何可见/可测阈值。
IDENT_EPS = 1e-6

DEFAULTS = {
    "rotation_gain": 1.0,       # cR: Exp(cR * Log R)
    "max_angle_deg": 45.0,      # 逐高斯转角上限；<= 0 关闭
    "deadzone_deg": 0.0,        # 小于该角度直接归零（抑制数值抖动）
    "average": "face",          # "face" | "vertex"
    "temporal_smoothing": 0.0,  # 0 = 关闭；>0 = 对 delta_q 做 EMA
    "coherence_gate": 0.0,      # 0 = 关闭；>0 = tau，一环合矢量长度低于 tau 不转
}


# --------------------------------------------------------------- 数值小工具
def identity_quat(ref: torch.Tensor) -> torch.Tensor:
    """与 ref 同 dtype/device 的单位四元数 [1,0,0,0]。"""
    q = torch.zeros(ref.shape[:-1] + (4,), dtype=ref.dtype, device=ref.device)
    q[..., 0] = 1.0
    return q


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton 积 a (x) b，wxyz 顺序（与 LAM 的 quaternion_multiply 逐位一致）。"""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_angle(q: torch.Tensor) -> torch.Tensor:
    """四元数对应的旋转角（弧度）。对 q 与 -q 对称，值域 [0, pi]。"""
    w = q[..., 0].abs().clamp(max=1.0)
    v = q[..., 1:].norm(dim=-1)
    # 2*acos(|w|) 在 w->1 附近对 float32 有灾难性抵消，改用 atan2 形式。
    return 2.0 * torch.atan2(v, w)


def slerp_identity(q: torch.Tensor, t) -> torch.Tensor:
    """从单位旋转 slerp 到 q，插值系数 t（[...] 或标量）。

    等价于 Exp(t * Log q)：沿测地线把转角乘 t，转轴不变。
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=q.dtype, device=q.device)
    t = t.unsqueeze(-1)
    v = q[..., 1:]
    vnorm = v.norm(dim=-1, keepdim=True)
    # 半球对齐：把 q 翻到 w>=0 再取 axis（v 随 q 一起变号，表示同一个旋转）。
    # 不对齐时 cos(t*half) 恒 >= 0，t=1 重建不出 w<0 的原四元数（恒等性破坏，
    # W8 自检实测 |dot| 低到 0.11），且 w<0 的顶点插值方向整体反号。
    sgn = torch.where(q[..., :1] < 0, -1.0, 1.0).to(q.dtype)
    axis = torch.where(
        vnorm > 1e-12,
        sgn * v / vnorm.clamp(min=1e-12),
        torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device),
    )
    half = 0.5 * quat_angle(q).unsqueeze(-1)          # 已对 |w| 对称 -> [0, pi/2]
    out = torch.cat((torch.cos(t * half), axis * torch.sin(t * half)), dim=-1)
    return F.normalize(out, dim=-1)


# Newton 迭代步数默认值。见 polar_rotation 的说明：真实 J 的条件数 p50 = 2.22 /
# p99 = 7.73 / max = 22.0，8 步已把正交残差压到 1.8e-7，与 10 步的差仅 4.8e-7
# （float32 精度量级）。10 步的收益为 0，代价是多两次 cuBLAS 求逆。
POLAR_ITERS = 8

# 初始尺度归一后，矩阵可直接求逆的 det 下限（相对量纲：此时 ||X||_F = 1）。
_POLAR_DET_FLOOR = 1e-9

# Newton 快路径的质量门限（|R^T R - I| 的逐元素最大绝对值）。实测真实 J 上
# 10 步后约 2.4e-7；门限设 1e-6 使快路径输出与 SVD 同为 float32 精度量级，
# 超限的样本才会被送去 SVD 精修（真实数据上命中率 0）。
_POLAR_TOL = 1e-6


def _safe_inv(m: torch.Tensor) -> torch.Tensor:
    """批量 3x3 求逆，奇异输入不抛异常。

    为什么不用 torch.linalg.inv
    ---------------------------
    它在 batch 里只要有一个奇异元素就整批抛 LinAlgError::

        torch._C._LinAlgError: linalg.inv: (Batch element 1): The diagonal element
        1 is zero, the inversion could not be completed because the input matrix
        is singular.

    而真实数据里确实有 1~6 / 39904 个 det <= 0 的 sliver 面（见 /tmp/cond_all.py），
    所以必须用 inv_ex——它返回 (result, info)，奇异位置给 inf/nan 而不抛。

    为什么不用闭式伴随矩阵 inv3
    --------------------------
    快 3.7 倍的反面：inv3 的 9 个 stack 元素会展开成三条独立的 elementwise kernel
    链（stack 后接 reshape 才合并），实测 39904 个矩阵 inv3 = 0.380 ms，而
    inv_ex = 0.102 ms（cuBLAS batched getrf+getri）。整个 polar 因此从 4.14 ms
    降到 2.06 ms。精度上两者等价（bad 数 47 vs 48，都是同一批 cond ~1e8 的样本）。

    info 标志位不可用
    ----------------
    CUDA 上 inv_ex 的 info 实测**恒为 0**（奇异输入也返回 0），所以不能靠它判坏
    样本，只能 nan_to_num 兜住再让上层的 orth / det / finite 三重判据接手。
    """
    r, _info = torch.linalg.inv_ex(m)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _polar_rotation_svd(j: torch.Tensor) -> torch.Tensor:
    """精确路径：float64 SVD。慢（4 万个面约 46 ms），只给自检做参照。"""
    j64 = j.double() if j.dtype != torch.float64 else j
    u, _s, vh = torch.linalg.svd(j64, full_matrices=False)
    r = u @ vh
    det = torch.linalg.det(r)
    flip = torch.where(det < 0, -1.0, 1.0)
    u = torch.cat(
        (u[..., :-1], u[..., -1:] * flip.unsqueeze(-1).unsqueeze(-1)), dim=-1
    )
    r = u @ vh
    # 极端退化（J 秩 < 2）时零空间任意，再兜一次 det。
    bad = (torch.linalg.det(r) < 0).unsqueeze(-1).unsqueeze(-1)
    r = torch.where(bad, torch.cat((r[..., :-1], -r[..., -1:]), dim=-1), r)
    return r.to(dtype=torch.float32)


def polar_rotation(
    j: torch.Tensor, *, exact: bool = False, iters: Optional[int] = None
) -> torch.Tensor:
    """逐矩阵极分解 J = R S，只返回旋转因子 R（det = +1 的真旋转）。

    Args:
        j: [..., 3, 3] 形变梯度。
        exact: True 走 float64 SVD（自检参照用，慢 10 倍以上）。
        iters: Newton 迭代步数，默认 POLAR_ITERS。
    Returns:
        [..., 3, 3]，正交且 det = +1；退化到无法定义旋转的位置返回单位阵。

    为什么默认走 Newton 而不是 SVD
    ------------------------------
    契约要求单帧调制 < 30 ms，而逐面极分解一次要吃 38890 个 3x3 矩阵。三种
    路径在 RTX 4090 上的实测（见 scripts/check 与 /tmp/bench*.py）：

        float64 SVD   45.8 ms    ← 早期实现，单这一项就吃掉 138 ms 里的 1/3
        float32 SVD    6.66 ms
        float32 Newton 3.20 ms（10 步）  正交残差 2.4e-7

    Newton 迭代 X <- (X + X^-T)/2 对**非奇异**实矩阵必收敛到正交极因子
    （Higham 1986, "Computing the polar decomposition"），且 X 与极因子同号
    det。真实 J 的条件数实测 p50 = 2.22 / p99 = 7.73 / max = 22.0，无病态样本，
    10 步足够。

    退化兜底
    --------
    J 秩不足（面积为 0 的 sliver 面）时极因子不唯一、Newton 会发散。做法是：
    初始按 Frobenius 范数归一（奇异值进 (0,1]），求逆带相对 det 下限，
    迭代后把非有限值和 det < 0 的元素吸附到单位阵，并另外给出 is_valid 掩码
    供上层做可信度加权（见 rotation_field 的 face_weight）。
    """
    if exact:
        return _polar_rotation_svd(j)

    steps = POLAR_ITERS if iters is None else int(iters)
    x = j if j.dtype == torch.float32 else j.to(torch.float32)
    # 在 float64 里判坏样本，float32 里迭代（判据与快路径精度解耦）。
    with torch.no_grad():
        neg_in = torch.linalg.det(j) < 0
    # Frobenius 归一：把奇异值压进 (0, 1]，Newton 的吸引域。
    nrm = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1).clamp(min=1e-12)
    x = x / nrm[..., None, None]

    for _ in range(steps):
        x = 0.5 * (x + _safe_inv(x).transpose(-1, -2))

    # 质量判据：正交残差 + det 符号 + 有限性。真实 J 上 bad 恒为空，
    # 只有对抗性/退化输入（镜像、近奇异、随机大条件数矩阵）才命中。
    orth = (x.transpose(-1, -2) @ x - torch.eye(3, dtype=x.dtype, device=x.device))
    orth = orth.abs().amax(dim=(-1, -2))
    det = torch.linalg.det(x)
    finite = torch.isfinite(x).all(-1).all(-1)
    scale = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1)
    bad = (~finite) | (scale <= 1e-6) | (orth > _POLAR_TOL) | (det <= 0) | neg_in

    if bool(bad.any()):
        # 只对坏样本走精确 SVD：index_select -> SVD -> index_copy，全部是
        # 张量算子（不是沿高斯的 Python for）。good 路径开销 = 一次掩码归约。
        idx = bad.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        flat = x.reshape(-1, 3, 3)
        # 注意：必须对**原始 J** 做 SVD，不能对已经收敛的 x 做 —— 镜像样本的
        # Newton 极限是 det=-1 的正交阵，它自己就是"极因子"，SVD 会原样返回，
        # 翻列也只会得到另一个正交阵而非最近旋转。原始 J 的 SVD 走 Kabsch
        # 惯例（det(UV^T) = -1 时翻 U 的最后一列），才是最近真旋转。
        src = j.reshape(-1, 3, 3).index_select(0, idx)
        fixed = _polar_rotation_svd(src)
        x = flat.index_copy(0, idx, fixed.to(flat.dtype)).reshape(x.shape)
    else:
        # 极少数 det < 0 已在 bad 里覆盖；这里只处理"全 good 但仍有镜像"的
        # 理论分支（正常构造不会进）。
        neg = (det < 0)[..., None, None]
        x = torch.where(neg, torch.cat((x[..., :-1], -x[..., -1:]), dim=-1), x)
        # 发散到非有限 / 塌成零阵的位置：视为"旋转无定义"，回单位阵。
        ok = finite & (scale > 1e-6)
        eye = torch.eye(3, dtype=x.dtype, device=x.device).expand_as(x)
        x = torch.where(ok[..., None, None], torch.nan_to_num(x, 0.0, 0.0, 0.0), eye)
    return x


# ------------------------------------------------------------------- 调制器
class RotSyncModulator(nn.Module):
    """ARKit 驱动的「高斯旋转同步」调制器（只转朝向、严格保轴长）。

    与 TgaCovarianceModulator 接口逐位相同。渲染器里冻结的调用点是

        delta_q, stretch = modulator.delta_rotation_scale(ret["animated"], mean_3d)
        gs_rotation, gs_scaling = modulator.apply(
            delta_q, modulator.delta_scale(stretch, gs_attr.scaling),
            gs_attr.rotation, gs_attr.scaling)

    本类把逐高斯单位四元数 delta_q 放在第一个返回值（LAM 原语义就是"左乘的旋转
    增量"），第二个返回值留 None（本模块不动尺度）。

    Args:
        flame_model: renderer 持有的 FlameHeadSubdivided（取 faces_up 与 ARKit 基）。
        arkit_bs_path: 可选，ARKit 基 npy 路径；默认沿用 LAM 内置资产。
        rotation_gain: cR in [0,1]，Exp(cR Log R)；1 = 完全同步，0 = 完全不动。
        max_angle_deg: 逐高斯转角硬上限（度），<= 0 关闭。
        deadzone_deg: 小于该角度视为 0，抑制浮点抖动。
        average: "face"（逐面极分解 + 面积加权四元数平均，推荐）或
            "vertex"（J 先平均再极分解，仅作对照）。
        temporal_smoothing: [0,1)，对 delta_q 做 EMA；0 = 无状态。
        eps: 数值下限。
    """

    def __init__(
        self,
        flame_model: nn.Module,
        arkit_bs_path: Optional[str] = None,
        *,
        rotation_gain: float = DEFAULTS["rotation_gain"],
        max_angle_deg: float = DEFAULTS["max_angle_deg"],
        deadzone_deg: float = DEFAULTS["deadzone_deg"],
        average: str = DEFAULTS["average"],
        temporal_smoothing: float = DEFAULTS["temporal_smoothing"],
        coherence_gate: float = DEFAULTS["coherence_gate"],
        eps: float = 1e-12,
    ):
        super().__init__()
        if not 0.0 <= rotation_gain <= 1.0:
            raise ValueError("rotation_gain must be in [0, 1]; got {!r}".format(rotation_gain))
        if not 0.0 <= deadzone_deg < 180.0:
            raise ValueError("deadzone_deg must be in [0, 180); got {!r}".format(deadzone_deg))
        if average not in ("face", "vertex"):
            raise ValueError("average must be 'face' or 'vertex'; got {!r}".format(average))
        if not 0.0 <= temporal_smoothing < 1.0:
            raise ValueError(
                "temporal_smoothing must be in [0, 1); got {!r}".format(temporal_smoothing)
            )
        if not 0.0 <= coherence_gate < 1.0:
            raise ValueError(
                "coherence_gate must be in [0, 1); got {!r}".format(coherence_gate)
            )
        # 组合而不是继承：LAM 的基类必须按文件路径加载。
        self.base = _base_module().DynamicCovarianceModulator(flame_model, arkit_bs_path, eps=eps)
        self.rotation_gain = float(rotation_gain)
        self.max_angle_deg = float(max_angle_deg)
        self.deadzone_deg = float(deadzone_deg)
        self.average = str(average)
        self.temporal_smoothing = float(temporal_smoothing)
        self.coherence_gate = float(coherence_gate)
        self.eps = float(eps)
        self._prev_delta = None
        self._last_diag = None

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

    # ---------------------------------------------------------- 形变梯度
    def face_jacobians(self, x0: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """逐面形变梯度 J_f = [d1 d2 n1] [e1 e2 n0]^-1，[B,F,3,3]。

        与 tga_covariance.face_jacobians 同一构造（第三列取**单位**法向，避免把
        法向长度变化当成尺度变化）。本模块单独实现以便独立自检。
        """
        faces = self.faces.to(x0.device)
        a, b, c = faces.unbind(dim=-1)
        e1 = x0[..., b, :] - x0[..., a, :]
        e2 = x0[..., c, :] - x0[..., a, :]
        d1 = xB[..., b, :] - xB[..., a, :]
        d2 = xB[..., c, :] - xB[..., a, :]
        n0 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=self.eps)
        n1 = F.normalize(torch.cross(d1, d2, dim=-1), dim=-1, eps=self.eps)
        J = torch.stack((d1, d2, n1), dim=-1) @ inv3(torch.stack((e1, e2, n0), dim=-1))
        return torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).clamp(-64.0, 64.0)

    def face_areas(self, x0: torch.Tensor) -> torch.Tensor:
        """逐面面积 [B,F]（静止网格），用于面积加权。"""
        faces = self.faces.to(x0.device)
        a, b, c = faces.unbind(dim=-1)
        area = 0.5 * torch.cross(
            x0[..., b, :] - x0[..., a, :], x0[..., c, :] - x0[..., a, :], dim=-1
        ).norm(dim=-1)
        return torch.nan_to_num(area, nan=0.0).clamp(min=0.0)

    def vertex_jacobians(
        self, x0: torch.Tensor, xB: torch.Tensor, Jf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """面积加权把 J_f 摊到顶点，[B,N,3,3]。孤立顶点回退单位阵。

        Args:
            Jf: 可选，已算好的逐面 J（[B,F,3,3]）。rotation_field 里逐面与逐顶点
                两条通路都要用 J_f，传进来可省一次（约 6 ms 的）重复构造。
        """
        if Jf is None:
            Jf = self.face_jacobians(x0, xB)                    # [B,F,3,3]
        area = self.face_areas(x0)                              # [B,F]
        B, n_face = area.shape
        n_vert = x0.shape[-2]
        idx = self.faces_flat.to(x0.device).view(1, -1, 1).expand(B, -1, 9)
        vals = (Jf * area[..., None, None]).reshape(B, n_face, 9).repeat_interleave(3, dim=1)
        num = torch.zeros(B, n_vert, 9, device=x0.device, dtype=x0.dtype)
        num.scatter_add_(1, idx, vals)
        idx1 = self.faces_flat.to(x0.device).view(1, -1).expand(B, -1)
        den = torch.zeros(B, n_vert, device=x0.device, dtype=x0.dtype)
        den.scatter_add_(1, idx1, area.repeat_interleave(3, dim=1))
        eye = torch.eye(3, device=x0.device, dtype=x0.dtype).reshape(1, 1, 9)
        Jv = num / den.clamp(min=self.eps).unsqueeze(-1)
        Jv = torch.where((den <= self.eps).unsqueeze(-1), eye.expand_as(Jv), Jv)
        return Jv.view(B, n_vert, 3, 3)

    # ------------------------------------------------------------ 旋转场
    def _markley_mean(self, q_face: torch.Tensor, area: torch.Tensor, n_vert: int):
        """把逐面四元数按面积加权平均到顶点（Markley / 最大特征向量法）。

        对每个顶点解 argmax_q  q^T (sum_f A_f q_f q_f^T) q，等价于取该 4x4 实对称
        矩阵最大特征值对应的特征向量。相对逐元素线性平均的优势是**对 q 与 -q
        的符号完全不敏感**（qq^T 里符号自动抵消），所以不需要任何参考四元数
        来做半球对齐。

        代价：一次 [3F,16] 的 scatter_add + 一次 [N,4,4] 的 eigh。实测 0.63 ms，
        其中 eigh 只有 0.12 ms；这比它替换掉的那条
        "J_f 摊到顶点 → 极分解 → matrix_to_quat"（约 3.5 ms）便宜得多。

        Args:
            q_face: [B,F,4] 单位四元数。
            area:   [B,F] 静止网格的逐面面积。
            n_vert: 顶点数 N。
        Returns:
            (q_mean [B,N,4] 单位四元数（已归一化）,
             conf   [B,N]   特征值间隔 1 - lam2/lam1，等价于该顶点一环的
                            "合矢量长度"，1 = 完全一致，0 = 一半对一半反向；
             wsum   [B,N]   该顶点三个面面积之和，用于识别无入射面的孤立顶点)
        """
        B, n_face = area.shape
        flat = self.faces_flat.to(q_face.device)
        # 外积 [B,F,4,4]，乘面积后按三个顶点各摊 1/3。
        pw = (q_face.unsqueeze(-1) * q_face.unsqueeze(-2)) * area[:, :, None, None]
        wire = (pw / 3.0).reshape(B, n_face, 16).repeat_interleave(3, dim=1)
        idx = flat.view(1, -1, 1).expand(B, -1, 16)
        acc = torch.zeros(B, n_vert, 16, device=q_face.device, dtype=torch.float32)
        acc.scatter_add_(1, idx, wire.to(torch.float32))
        # 面积权重和：与 16 维累加同一次 scatter 的降维版，单独算一次更省内存。
        wsq = (area / 3.0).repeat_interleave(3, dim=1)                      # [B,3F]
        wsum = torch.zeros(B, n_vert, device=q_face.device, dtype=torch.float32)
        wsum.scatter_add_(1, flat.view(1, -1).expand(B, -1), wsq)
        ev, evec = torch.linalg.eigh(acc.view(B, n_vert, 4, 4))
        q = evec[..., -1]
        # eigh 输出的特征向量已单位化，但 float32 下仍有 1e-7 级残差；
        # 这里不做 normalize，交给 rotation_field 统一处理（避免重复归一化）。
        lam1 = ev[..., -1].clamp(min=1e-20)
        conf = (1.0 - ev[..., -2] / lam1).clamp(0.0, 1.0)
        return q, conf, wsum

    def rotation_field(self, x0: torch.Tensor, xB: torch.Tensor):
        """返回 (delta_q [B,N,4], 诊断 dict)。

        诊断里的量只给脚本/调参用，不参与渲染：
            angle_deg  [B,N]  逐高斯最终转角（度，限幅前）
            clipped    [B,N]  被 max_angle 截断标记
            isolated   [B,N]  无入射面的顶点
        """
        B = x0.shape[0]
        N = x0.shape[-2]
        device = x0.device

        # J_f 在两条通路里都要用（顶点通路摊到顶点、面通路极分解），算一次共用。
        area = self.face_areas(x0)                                           # [B,F]
        Jf = self.face_jacobians(x0, xB)                                     # [B,F,3,3]

        if self.average == "vertex":
            Jv = self.vertex_jacobians(x0, xB, Jf=Jf)                        # [B,N,3,3]
            delta = F.normalize(matrix_to_quat(polar_rotation(Jv)), dim=-1)  # [B,N,4]
            coh = torch.ones(B, N, 1, device=device, dtype=delta.dtype)
        else:
            q_face = F.normalize(matrix_to_quat(polar_rotation(Jf)), dim=-1)  # [B,F,4]
            # 直接用 Markley 平均：对 q/-q 翻号免疫，所以既不需要参考四元数做
            # 半球对齐，也不需要"把 J_f 摊到顶点再做一次极分解"（省约 3.5 ms）。
            delta, coh, wsum = self._markley_mean(q_face, area, N)           # [B,N,4], [B,N]
            # 无入射面的孤立顶点：Markley 的 4x4 矩阵是零阵，特征向量无意义。
            # coh 与 iso 同形 [B,N]，不要先 unsqueeze 再 where——[B,N,1] 与 [B,N]
            # 广播会得到 [B,N,N]（N=20018 时是 1.6 GB 的假张量）。
            iso = wsum <= self.eps                                            # [B,N]
            delta = torch.where(iso.unsqueeze(-1), identity_quat(delta), delta)
            coh = torch.where(iso, torch.zeros_like(coh), coh).unsqueeze(-1)  # [B,N,1]
            delta = F.normalize(delta, dim=-1)

        angle = quat_angle(delta)                                            # [B,N] 弧度

        # ---- 恒等吸附：中性/数值噪声一律回到严格的 [1,0,0,0]。
        # ARKit 全 0 时渲染器走静态短路、本模块根本不被调用；但"不动"的语义
        # 要在浮点上也可验证，否则 F.normalize 与极分解的 1e-8 残差会让
        # delta_q 与 [1,0,0,0] 逐位不等，破坏"与改动前像素级一致"的自检。
        snap = (angle < IDENT_EPS).unsqueeze(-1)
        delta = torch.where(snap, identity_quat(delta), delta)
        angle = torch.where(snap.squeeze(-1), torch.zeros_like(angle), angle)

        # ---- 死区：抑制 float 抖动带来的方向噪声。
        if self.deadzone_deg > 0:
            dead = (angle < self.deadzone_deg * DEG).unsqueeze(-1)
            delta = torch.where(dead, identity_quat(delta), delta)

        # ---- 旋转门 cR = Exp(cR Log R)，沿测地线回缩。
        if self.rotation_gain != 1.0:
            delta = slerp_identity(delta, self.rotation_gain)

        # ---- 统一回缩：逐高斯角上限 与 置信度门控 合成一个插值系数，
        #      只做一次 slerp（两次 slerp 会被测出约 2.3 ms 的额外开销）。
        clipped = torch.zeros_like(angle, dtype=torch.bool)
        t_shrink = torch.ones_like(angle)
        gated = torch.zeros_like(angle, dtype=torch.bool)
        if self.max_angle_deg > 0:
            cap = self.max_angle_deg * DEG               # 弧度
            clipped = angle > cap
            t_shrink = torch.where(
                clipped, cap / angle.clamp(min=1e-9), torch.ones_like(angle)
            )
        if self.coherence_gate > 0:
            # g = clamp((R - tau) / (1 - tau), 0, 1)：R 低于 tau 完全不动，
            # R = 1（一环内所有面旋转一致）时完全同步。
            tau = self.coherence_gate
            g = ((coh.squeeze(-1) - tau) / max(1.0 - tau, 1e-9)).clamp(0.0, 1.0)
            gated = g < 1.0
            t_shrink = t_shrink * g
        if bool((t_shrink < 1.0).any()):
            delta = slerp_identity(delta, t_shrink.clamp(max=1.0))

        # ---- 可选时间平滑（逐帧 EMA，需会话内顺序调用）。
        if self.temporal_smoothing > 0:
            prev = self._prev_delta
            if prev is not None and prev.shape == delta.shape:
                s = torch.where((delta * prev).sum(-1, keepdim=True) < 0, -1.0, 1.0)
                a = self.temporal_smoothing
                delta = F.normalize(a * prev + (1.0 - a) * delta * s, dim=-1)
            self._prev_delta = delta.detach()

        delta = F.normalize(delta, dim=-1).to(x0.dtype)
        return delta, {
            "angle_deg": (angle / DEG).detach(),   # 对外以度为单位（限幅/门控前）
            "angle_rad": angle.detach(),
            "clipped": clipped.detach(),
            "coherence": coh.squeeze(-1).detach(),  # 一环合矢量长度 R ∈ [0,1]
            "gated": gated.detach(),
            "shrink": t_shrink.detach(),
            # ---- RSB 衔接键（方法四子类用；零开销，detach 只断梯度）
            "Jf": Jf.detach(),
            "area": area.detach(),
            "gate": t_shrink.detach(),
        }

    # ------------------------------------------- 与 LAM 冻结调用点对接
    def delta_rotation_scale(self, x0: torch.Tensor, xB: torch.Tensor):
        """返回 (delta_q [B,N,4], diag)。diag 作为 payload 传给 delta_scale/apply；
        本模块的 delta_scale/apply 忽略其中的键（基线行为逐位不变）。"""
        delta_q, diag = self.rotation_field(x0, xB)
        self._last_diag = diag
        return delta_q, diag

    def delta_scale(self, payload, scaling: torch.Tensor) -> torch.Tensor:
        """本模块不改尺度，payload 原样透传（apply 里忽略）。"""
        return payload

    def apply(self, delta_q: torch.Tensor, payload, rotation: torch.Tensor,
              scaling: torch.Tensor):
        """q' = normalize(delta_q (x) q0)，s' = s0（逐位相等）。

        Args:
            delta_q: [B,N,4] 逐高斯旋转增量。
            payload: 未使用。
            rotation: [N,4] 规范四元数。
            scaling: [N,3] 规范尺度。
        Returns:
            (q' [B,N,4], s' [B,N,3])
        """
        q0 = rotation.unsqueeze(0).to(dtype=delta_q.dtype)
        q = quat_multiply(delta_q, q0)
        # 严格恒等增量（含 rotation_gain=0 与恒等吸附两类）直接透传 q0：
        # quat_multiply(I, q0) 在浮点上逐位等于 q0，再 normalize 反而会引入
        # 1e-8 级重归一化残差，破坏 "gain=0 与不启用逐位一致" 的可验证性。
        is_identity = (delta_q[..., 0] == 1.0) & (delta_q[..., 1:] == 0.0).all(-1)
        q = torch.where(
            is_identity.unsqueeze(-1), q0, F.normalize(q, dim=-1, eps=self.eps)
        )
        s = scaling.unsqueeze(0).to(dtype=delta_q.dtype).expand(
            delta_q.shape[0], -1, -1
        ).contiguous()
        return q, s

    # ------------------------------------------------------------- 诊断
    @staticmethod
    def sigma(rotation: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
        """Sigma = R diag(s^2) R^T，[...,3,3]。自检脚本用。"""
        r = quat_to_matrix(rotation.float())
        return r @ torch.diag_embed(scaling.float().pow(2)) @ r.transpose(-1, -2)


# ------------------------------------------------------------------- 注入
def enabled() -> bool:
    return os.environ.get("LAM3D_ROT_SYNC", "1") not in ("0", "false", "False", "")


def install(renderer, *, force: bool = False, **kwargs) -> Optional[nn.Module]:
    """把旋转同步调制器装进 renderer._cov_modulator（不改 LAM 一行）。

    GSRenderer._get_cov_modulator 只在 _cov_modulator is None 时惰性构造自带的
    DynamicCovarianceModulator，所以预先占位即可接管。
    """
    if not enabled():
        return None
    existing = getattr(renderer, "_cov_modulator", None)
    if existing is not None and not force:
        return existing
    params = dict(DEFAULTS)
    for key, env_key in (
        ("rotation_gain", "LAM3D_ROT_GAIN"),
        ("max_angle_deg", "LAM3D_ROT_MAX_ANGLE"),
        ("deadzone_deg", "LAM3D_ROT_DEADZONE"),
        ("average", "LAM3D_ROT_AVG"),
        ("temporal_smoothing", "LAM3D_ROT_SMOOTH"),
        ("coherence_gate", "LAM3D_ROT_COH"),
    ):
        raw = os.environ.get(env_key)
        if raw is not None and raw != "":
            params[key] = raw if key == "average" else float(raw)
    params.update(kwargs)
    mod = RotSyncModulator(
        renderer.flame_model,
        rotation_gain=params["rotation_gain"],
        max_angle_deg=params["max_angle_deg"],
        deadzone_deg=params["deadzone_deg"],
        average=params["average"],
        temporal_smoothing=params["temporal_smoothing"],
        coherence_gate=params["coherence_gate"],
    )
    device = getattr(renderer, "device", None) or torch.device("cuda")
    mod.to(device)
    mod.eval()
    renderer._cov_modulator = mod
    return mod
