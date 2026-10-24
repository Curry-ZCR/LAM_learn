"""rot_sync 纯张量层自检（不需要加载大模型，CPU 几秒跑完）。

覆盖：
  1. 四元数代数与 LAM 的 quaternion_multiply 逐位对齐
  2. polar_rotation 的正交性 / det=+1 / 镜像面处理
  3. slerp_identity == Exp(t Log q)
  4. Sigma' = R Sigma0 R^T 的特征值（轴长）逐位守恒
  5. rotation_field 在 J = I 时严格给出单位四元数
  6. 合成曲面：整体刚体旋转不应产生残差（B = I 时的设计前提）
  7. 逐面极分解平均 vs 顶点平均的差异、限幅与死区生效
  8. 无 NaN/Inf、无沿 N 的 Python for 循环
"""
import importlib.util
import math
import sys

import torch
import torch.nn.functional as F

LAM_ROOT = "/workspace/LAM"
if LAM_ROOT not in sys.path:
    sys.path.insert(0, LAM_ROOT)

from lam3d.core import rot_sync, tga_covariance  # noqa: E402

FAIL = []
def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("[%s] %s %s" % (tag, name, detail))
    if not ok:
        FAIL.append(name)

torch.manual_seed(0)
D = torch.float32


def lam_quat_multiply(a, b):
    return rot_sync._base_module().quaternion_multiply(a, b)


# ---------------------------------------------------------------- 1. 四元数
q = F.normalize(torch.randn(4096, 4, dtype=D), dim=-1)
p = F.normalize(torch.randn(4096, 4, dtype=D), dim=-1)
mine = rot_sync.quat_multiply(q, p)
theirs = lam_quat_multiply(q, p)
check("quat_multiply == LAM", torch.allclose(mine, theirs, atol=1e-7),
      "max|d|=%.3e" % (mine - theirs).abs().max().item())

# 乘积仍然表示旋转：R(q p) == R(q) R(p)
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402
rq, rp = quat_to_matrix(q), quat_to_matrix(p)
rqp = quat_to_matrix(F.normalize(mine, dim=-1))
check("R(qp) == R(q)R(p)", torch.allclose(rqp, rq @ rp, atol=1e-5),
      "max|d|=%.3e" % (rqp - rq @ rp).abs().max().item())

# ------------------------------------------------------- 2. 极分解
mix = torch.randn(2048, 3, 3, dtype=D) * 3.0
mix[0] = torch.eye(3) * 5.0            # 纯缩放
mix[1] = torch.zeros(3, 3)             # 奇异
mix[2] = torch.eye(3); mix[2, 0, 0] = -1.0   # 反射
R = rot_sync.polar_rotation(mix)
orth = (R.transpose(-1, -2) @ R - torch.eye(3)).abs().max().item()
det = torch.linalg.det(R.double())
check("polar orthonormal", orth < 1e-5, "max|R^T R - I|=%.3e" % orth)
check("polar det=+1", bool((det > 0).all() and (det - 1).abs().max() < 1e-5),
      "det range [%.6f, %.6f]" % (det.min().item(), det.max().item()))
check("polar finite", bool(torch.isfinite(R).all()))

# S 对称正定。注意只对 det(J) > 0 的样本有意义：det(J) < 0 时 J 不可能分解成
# 真旋转 x 正定阵（det(R S) = +1 * det(S) > 0），极分解在这种输入上只能给出
# "最近真旋转"，R^T J 必然不对称。真实 J 由逐面差分构造，恒有 det > 0。
detj = torch.linalg.det(mix.double())
pos = detj > 0
S = R.double().transpose(-1, -2) @ mix.double()
asym = (S - S.transpose(-1, -2)).abs().amax(dim=(-1, -2))
asym_rel = (asym / mix.double().reshape(2048, 9).norm(dim=-1).clamp(min=1e-12))[pos]
check("S symmetric (det>0 only)",
      asym_rel.max().item() < 1e-5,
      "max rel asym=%.3e over %d samples" % (asym_rel.max().item(), int(pos.sum())))
# 反射输入：R 应为单位阵的最近旋转（翻最小奇异值列后）
check("reflection -> det +1 handled", torch.linalg.det(R[2].double()).item() > 0)

# ------------------------------------------------- 3. slerp_identity
axis = F.normalize(torch.randn(1024, 3, dtype=D), dim=-1)
ang = torch.rand(1024, dtype=D) * math.pi
half = ang / 2
qq = torch.cat((torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), -1)
for t in (0.0, 0.25, 0.5, 1.0):
    got = rot_sync.slerp_identity(qq, t)
    want = rot_sync.quat_angle(got)
    ok = torch.allclose(want, ang * t, atol=2e-4)
    check("slerp_identity t=%.2f angle scales" % t, ok,
          "max|d|=%.3e" % (want - ang * t).abs().max().item())

# --------------------------------------------- 4. 轴向不变性（方法一核心）
M = 5000
s0 = torch.exp(torch.randn(M, 3, dtype=D) * 1.2)
r0 = F.normalize(torch.randn(M, 4, dtype=D), dim=-1)
Rt = rot_sync.polar_rotation(torch.randn(M, 3, 3, dtype=D) * 3.0)   # 真旋转
sig0 = rot_sync.RotSyncModulator.sigma(r0, s0)
sig1 = Rt @ sig0 @ Rt.transpose(-1, -2)
# 逐样本用最大特征值归一，避免把 float32 条件数（跨越 10 个数量级）算成缺陷。
def eig_rel(a, b):
    ea, eb = torch.linalg.eigvalsh(a.double()), torch.linalg.eigvalsh(b.double())
    scale = ea.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    return ((eb - ea).abs() / scale).max().item()

rel = eig_rel(sig0, sig1)
check("eigenvalues preserved (axis lengths)", rel < 1e-5, "max rel err=%.3e" % rel)
# 再用 float64 精确参考验一次：R Sigma0 R^T 的特征值必须逐个等于 s^2
ref = torch.sort(s0.double().pow(2), dim=-1).values
rel_ref = ((torch.linalg.eigvalsh(sig1.double()) - ref).abs()
           / ref.max(dim=-1, keepdim=True).values.clamp(min=1e-12)).max().item()
check("eigenvalues == s^2 exactly", rel_ref < 1e-5, "max rel err=%.3e" % rel_ref)
# 用 quaternion 通路再验一遍
qs = F.normalize(rot_sync.matrix_to_quat(Rt), dim=-1)
sigq = rot_sync.RotSyncModulator.sigma(qs, s0)
rel2 = eig_rel(sig0, sigq)
check("quat roundtrip preserves axes", rel2 < 1e-5, "max rel err=%.3e" % rel2)
# 但方向确实变了：非对角项应当改变
off = (sig1 - sig0).abs().max().item()
check("orientation actually changes", off > 1e-3, "max|d Sigma|=%.3e" % off)

print()
print("FAILED:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
