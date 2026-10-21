"""对比 3D 网格的几种重建方案，挑出最适合前端展示的。

LAM 的规范高斯带 offset（中位 0.011，头半径仅 0.067），
用 FLAME 拓扑把它们连成三角面时，相邻点深度差过大会让表面自交/起刺。
渲染时被高斯混合掩盖，但导出成网格后在浏览器里就暴露成暗斑。

候选：
  A 原始 gs.xyz（当前实现）
  B 拉普拉斯平滑 1 次
  C 拉普拉斯平滑 3 次
  D 纯 FLAME 模板 + shape（上游 save_shaped_mesh 的做法，最干净但丢细节）
"""
import sys

sys.path.insert(0, "/workspace/LAM")
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix

from lam.models.rendering.flame_model.flame import FlameHeadSubdivided

flame = FlameHeadSubdivided(
    300, 100, add_teeth=False, add_shoulder=False,
    flame_model_path="model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
    flame_lmk_embedding_path="model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy",
    flame_template_mesh_path="model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj",
    flame_parts_path="model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl",
    subdivide_num=1,
)
F = flame.faces_up.cpu().numpy()
neutral = flame.v_template_up.cpu().numpy()

JID = open("/tmp/jobid.txt").read().strip()
D = f"/workspace/lam3d/workspace/jobs/{JID}"
job = __import__("json").loads(open(f"{D}/result.json").read())
# 取渲染前的规范几何：从 result.json 找不到，直接用 glb + 还原
# 这里重跑一次不方便，改用已保存的 gaussians.ply（与 glb 同源）
from plyfile import PlyData

d = PlyData.read(f"{D}/input_gaussians.ply")
v = d["vertex"]
V_raw = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
C_glb = trimesh.load(f"{D}/input_head.glb", process=False)
g0 = list(C_glb.geometry.values())[0]
C = np.asarray(g0.visual.vertex_colors)[:, :3].astype(float) / 255.0
# GLB 已做颈部裁剪 + 孤立顶点清理，顶点数少于 20018。
# 用最近邻把 GLB 顶点映射回原始 20018 空间，保证 V/C/F 三者对齐。
from scipy.spatial import cKDTree as _KDTree
GLB_F = np.asarray(g0.faces)
GLB_V = np.asarray(g0.vertices)
_idx = _KDTree(V_raw).query(GLB_V)[1]
_aligned = np.zeros(len(V_raw), dtype=bool); _aligned[_idx] = True
V_raw = V_raw[_idx]           # 只保留 GLB 里实际存在的顶点
F_keep = GLB_F                 # 面索引已针对 GLB 顶点编号
F = F_keep

# 用 shape 还原干净模板
shape = np.load("/workspace/LAM/output/tracking/export/raw/canonical_flame_param.npz")["shape"]
with torch.no_grad():
    V_flame = flame.get_cano_verts(torch.FloatTensor(shape).unsqueeze(0)).cpu().numpy()[0]

# --- 邻接矩阵（用于拉普拉斯平滑与法线） ---
def build_adj(faces, n):
    r = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    c = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    r = np.concatenate([r, c]); c = np.concatenate([c, r[: len(c)]])
    data = np.ones(len(r))
    A = coo_matrix((data, (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A

A = build_adj(F, len(V_raw))   # 注意：F 与 V_raw 已按 GLB 顶点顺序对齐
deg = np.asarray(A.sum(1)).ravel()
deg[deg == 0] = 1
W = coo_matrix((1.0 / deg, (np.arange(len(V_raw)), np.arange(len(V_raw)))), shape=A.shape).tocsr() @ A

def laplacian(V, iters, lam=0.5):
    out = V.copy()
    for _ in range(iters):
        out = out + lam * (W @ out - out)
    return out

def roughness(V):
    """每个顶点偏离邻居均值的程度 —— 就是网格的"刺"有多尖。"""
    return np.linalg.norm(V - W @ V, axis=1)

V_flame = V_flame[_idx]   # 同样对齐到 GLB 顶点顺序
variants = [
    ("A 原始 offset", V_raw),
    ("B 平滑 x1", laplacian(V_raw, 1)),
    ("C 平滑 x3", laplacian(V_raw, 3)),
    ("D 纯 FLAME 模板", V_flame),
]
print("粗糙度（中位 / p99），越小越平滑：")
for name, V in variants:
    r = roughness(V)
    print(f"  {name:16s} median {np.median(r):.5f}  p99 {np.percentile(r,99):.5f}  max {r.max():.5f}")
print()
print("与原始 offset 几何的偏离量（保留了多少细节）：")
for name, V in variants:
    d = np.linalg.norm(V - V_raw, axis=1)
    print(f"  {name:16s} mean {d.mean():.5f}  max {d.max():.5f}")

# --- 渲染对比 ---
W_, H_ = 340, 340
def render(V, C, cam):
    cam = np.array(cam, float); cam /= np.linalg.norm(cam)
    up = np.array([0, 1, 0.0]); right = np.cross(up, cam); right /= np.linalg.norm(right)
    up2 = np.cross(cam, right)
    P = V @ np.stack([right, up2, cam], 1)
    # 用平滑后的法线做 Lambert 光照，模拟 MeshStandardMaterial
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    light = np.array([0.45, 0.55, 0.70]); light /= np.linalg.norm(light)
    shade = np.clip(0.35 + 0.65 * np.clip(vn @ light, 0, 1), 0, 1.4)

    uv, dd = P[:, :2], P[:, 2]
    lo, hi = neutral[:, :2].min(0), neutral[:, :2].max(0)
    c = (lo + hi) / 2; s = max(hi - lo) * 1.35
    px = ((uv[:, 0] - c[0]) / s + 0.5) * (W_ - 1)
    py = (0.5 - (uv[:, 1] - c[1]) / s) * (H_ - 1)
    img = np.full((H_, W_, 3), 0.06); zb = np.full((H_, W_), -1e9)
    for t in F:
        x, y, z = px[t], py[t], dd[t]
        x0, x1 = max(int(x.min()), 0), min(int(x.max()) + 1, W_)
        y0, y1 = max(int(y.min()), 0), min(int(y.max()) + 1, H_)
        if x1 <= x0 or y1 <= y0: continue
        X, Y = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        den = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(den) < 1e-12: continue
        w0 = ((x[1] - X) * (y[2] - Y) - (x[2] - X) * (y[1] - Y)) / den
        w1 = ((x[2] - X) * (y[0] - Y) - (x[0] - X) * (y[2] - Y)) / den
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any(): continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zb[y0:y1, x0:x1]; u = m & (zz > sub); sub[u] = zz[u]
        col = (w0[..., None] * C[t[0]] + w1[..., None] * C[t[1]] + w2[..., None] * C[t[2]])
        sh = (w0 * shade[t[0]] + w1 * shade[t[1]] + w2 * shade[t[2]])[..., None]
        col = col * sh
        img[y0:y1, x0:x1][u] = np.clip(col, 0, 1)[u]
    return (img * 255).astype(np.uint8)

tiles = [render(V, C, (0.45, 0.12, 0.88)) for _, V in variants]
Image.fromarray(np.concatenate(tiles, 1)).save("/tmp/smooth_cmp.png")
print("\n已保存 /tmp/smooth_cmp.png （左→右：A 原始 / B 平滑x1 / C 平滑x3 / D 纯模板）")
