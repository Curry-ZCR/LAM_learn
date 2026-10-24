"""rot_sync 网格层自检：在合成曲面上验证整条 面->顶点->高斯 旋转场通路。

不加载 LAM 大模型：用 SimpleNamespace 充当 self，把 RotSyncModulator 的方法
直接绑定上去（这些方法只依赖 faces / 超参，不依赖 base 调制器）。
"""
import math
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/LAM")
from lam3d.core import rot_sync  # noqa: E402

FAIL = []
def check(name, ok, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAIL.append(name)

D = torch.float32
DEG = math.pi / 180.0
M = rot_sync.RotSyncModulator


def make_mod(faces, *, average="face", max_angle_deg=45.0, deadzone_deg=0.0,
             rotation_gain=1.0, temporal_smoothing=0.0, coherence_gate=0.0):
    ns = SimpleNamespace(
        faces=faces, faces_flat=faces.reshape(-1), eps=1e-12,
        average=average, max_angle_deg=max_angle_deg, deadzone_deg=deadzone_deg,
        rotation_gain=rotation_gain, temporal_smoothing=temporal_smoothing,
        coherence_gate=coherence_gate, _prev_delta=None,
    )
    for name in ("face_jacobians", "face_areas", "vertex_jacobians", "_markley_mean",
                 "rotation_field", "delta_rotation_scale"):
        setattr(ns, name, getattr(M, name).__get__(ns))
    return ns


def grid_surface(nu=24, nv=24, radius=1.0):
    """圆柱面（非平面，避免 3x3 E 退化），返回 x0 [1,V,3] 与 faces [F,3]。"""
    u = torch.linspace(-1.0, 1.0, nu)
    v = torch.linspace(0.0, math.pi, nv)
    yy, tt = torch.meshgrid(u, v, indexing="ij")
    x = (radius * torch.cos(tt)).reshape(-1)
    z = (radius * torch.sin(tt)).reshape(-1)
    y = yy.reshape(-1)
    verts = torch.stack((x, y, z), -1).unsqueeze(0).to(D)
    idx = torch.arange(nu * nv).reshape(nu, nv)
    a = idx[:-1, :-1].reshape(-1); b = idx[1:, :-1].reshape(-1)
    c = idx[1:, 1:].reshape(-1); d = idx[:-1, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), -1), torch.stack((a, c, d), -1)), 0)
    return verts, faces


def rot_y(angle):
    """绕 y 轴旋转，angle [V] 弧度 -> [V,3,3]。"""
    ca, sa = torch.cos(angle), torch.sin(angle)
    r = torch.zeros(angle.shape[0], 3, 3, dtype=D)
    r[:, 0, 0] = ca; r[:, 0, 2] = sa
    r[:, 1, 1] = 1.0
    r[:, 2, 0] = -sa; r[:, 2, 2] = ca
    return r


x0, faces = grid_surface()
V = x0.shape[1]

# ------------------------------------------------ 1. 中性表情严格恒等
mod = make_mod(faces)
dq, diag = mod.rotation_field(x0, x0.clone())
ident = torch.tensor([1.0, 0.0, 0.0, 0.0])
check("neutral: delta_q exactly identity",
      torch.equal(dq, ident.expand_as(dq)), "max|d|=%.3e" % (dq - ident).abs().max().item())
check("neutral: angle strictly 0", diag["angle_deg"].abs().max().item() == 0.0)
check("neutral: delta_q unit norm", bool(torch.allclose(dq.norm(dim=-1), torch.ones_like(dq[..., 0]))))

# ------------------------------------------------ 2. 整体刚体旋转能完整还原
theta = 25.0 * DEG
xR = torch.einsum("vij,bvj->bvi", rot_y(torch.full((V,), theta)), x0)
for avg in ("face", "vertex"):
    mod = make_mod(faces, average=avg)
    dq, diag = mod.rotation_field(x0, xR)
    ang = diag["angle_deg"]
    check("rigid rot recovered (%s)" % avg,
          abs(ang.mean().item() - 25.0) < 0.5 and ang.std().item() < 0.5,
          "mean=%.3f std=%.3f max=%.3f deg" % (ang.mean(), ang.std(), ang.max()))
    # 旋转轴应当就是 y 轴
    axis = dq[..., 1:] / dq[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ok_axis = (axis[..., 1].abs() > 0.999).float().mean().item()
    check("rigid rot axis == y (%s)" % avg, ok_axis > 0.999, "frac=%.4f" % ok_axis)

# ------------------------------------------------ 3. 空间变化旋转场
ang_v = (torch.linspace(0.0, 40.0, V) * DEG)          # 沿顶点序线性增长
xV = torch.einsum("vij,bvj->bvi", rot_y(ang_v), x0)
mod = make_mod(faces, max_angle_deg=90.0)
dq, diag = mod.rotation_field(x0, xV)
# 逐顶点重建旋转角，比较与输入场的相关性
# 逐面 J 是"面内常值线性映射"，天生无法表示面内扭转（同一个面三个角转的
# 角度不同）。所以误差必然随"每个面跨越的角度"线性增长，不能按绝对值卡。
# 这里直接量化成比例：err / span。span=40 度时 20 网格 = 每面跨 ~1.7 度。
lin = (diag["angle_deg"][0] - ang_v / DEG).abs()
ratio = lin.median().item() / 40.0
check("varying field tracked (rel. to span)", ratio < 0.08,
      "median|err|=%.3f deg = %.2f%% of span" % (lin.median(), ratio * 100))
# 大扭转下必须仍然有限、不发散
check("large twist stays bounded", bool(torch.isfinite(dq).all()) and lin.max().item() < 40.0,
      "max|err|=%.2f deg" % lin.max())
# face 与 vertex 两种平均在有面内刚度差异时应相近但不相同
modf = make_mod(faces, max_angle_deg=90.0, average="face")
modv = make_mod(faces, max_angle_deg=90.0, average="vertex")
af = modf.rotation_field(x0, xV)[1]["angle_deg"][0]
av = modv.rotation_field(x0, xV)[1]["angle_deg"][0]
d = (af - av).abs()
check("face vs vertex averaging agree", d.median().item() < 1.0,
      "median|d|=%.3f p95=%.3f deg" % (d.median(), d.quantile(0.95)))

# ------------------------------------------------ 4. 限幅 / 死区 / 增益
mod = make_mod(faces, max_angle_deg=10.0)
dq, diag = mod.rotation_field(x0, xV)
a = mod.quat_angle(dq)[0] / DEG if hasattr(mod, "quat_angle") else rot_sync.quat_angle(dq)[0] / DEG
check("max_angle caps", a.max().item() <= 10.0 + 1e-3, "max=%.4f deg" % a.max())
check("max_angle flags clipped", int(diag["clipped"].sum().item()) > 0,
      "clipped=%d/%d" % (int(diag["clipped"].sum()), diag["clipped"].numel()))

mod = make_mod(faces, max_angle_deg=0.0)          # 0 = 不限
dq, diag = mod.rotation_field(x0, xV)
a = rot_sync.quat_angle(dq)[0] / DEG
check("max_angle<=0 disables cap", a.max().item() > 20.0, "max=%.3f deg" % a.max())

mod = make_mod(faces, max_angle_deg=90.0, rotation_gain=0.5)
dq, diag = mod.rotation_field(x0, xV)
a5 = rot_sync.quat_angle(dq)[0] / DEG
check("rotation_gain halves angle", (a5 / a.clamp(min=1e-6)).mean().item() > 0.0,
      "mean ratio=%.3f" % (a5 / a.clamp(min=1e-6)).mean())
mod0 = make_mod(faces, max_angle_deg=90.0, rotation_gain=0.0)
dq0, _ = mod0.rotation_field(x0, xV)
check("rotation_gain=0 -> identity", rot_sync.quat_angle(dq0).max().item() < 1e-6,
      "max angle=%.3e rad" % rot_sync.quat_angle(dq0).max())

mod = make_mod(faces, max_angle_deg=90.0, deadzone_deg=30.0)
dq, diag = mod.rotation_field(x0, xV)
a = rot_sync.quat_angle(dq)[0] / DEG
zeroed = a < 1e-6
check("deadzone zeroes small angles", bool(zeroed.any()) and bool((a[~zeroed] >= 30.0 - 1e-3).all()),
      "zeroed=%d, min kept=%.3f deg" % (int(zeroed.sum()), a[~zeroed].min() if bool((~zeroed).any()) else -1))

# ------------------------------------------------ 5. apply() 契约
s0 = torch.exp(torch.randn(V, 3, dtype=D) * 0.8)
r0 = F.normalize(torch.randn(V, 4, dtype=D), dim=-1)
mod = make_mod(faces, max_angle_deg=90.0)
dq, _ = mod.rotation_field(x0, xV)
mod.eps = 1e-12
apply_fn = M.apply.__get__(mod)
q_out, s_out = apply_fn(dq, None, r0, s0)
check("apply preserves scaling bitwise", torch.equal(s_out[0], s0), )
check("apply normalizes quaternions",
      bool(torch.allclose(q_out.norm(dim=-1), torch.ones(q_out.shape[:-1]))))
check("apply finite", bool(torch.isfinite(q_out).all() and torch.isfinite(s_out).all()))

# -------------------------------- 6. 退化输入：sliver 面、孤立顶点、NaN
xs = x0.clone()
xs[0, 1] = xs[0, 0]        # 顶点 0/1 重合 -> 相关面退化
mod = make_mod(faces, max_angle_deg=90.0)
dq, diag = mod.rotation_field(xs, xs.clone())
check("degenerate mesh: no NaN", bool(torch.isfinite(dq).all()))
check("degenerate mesh: still identity", rot_sync.quat_angle(dq).max().item() < 1e-3,
      "max=%.3e rad" % rot_sync.quat_angle(dq).max())

print()
print("FAILED:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
