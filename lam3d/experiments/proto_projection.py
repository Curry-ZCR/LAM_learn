"""验证顶点→像素投影约定：能不能把网格顶点正确投到 LAM 渲染帧的脸上。

正确性判据：投出来的 2D 点应该落在帧的**人像区域**内，且分布与帧里的人脸轮廓吻合。
"""
import sys

sys.path.insert(0, "/workspace/LAM")
import numpy as np
import subprocess
import imageio_ffmpeg
from PIL import Image

from lam.runners.infer.head_utils import _load_pose

jid = open("/tmp/jobid.txt").read().strip()
D = f"/workspace/lam3d/workspace/jobs/{jid}"

# 读回我们上一次跑用的动作序列帧参数（与渲染时同源）
import json

motion = "Anti_Drugs"
mj = json.load(open(f"/workspace/LAM/assets/sample_motion/export/{motion}/transforms.json"))

# 取网格顶点
from plyfile import PlyData
d = PlyData.read(f"{D}/input_gaussians.ply")
v = d["vertex"]
V = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)

# 读一帧渲染图
ff = imageio_ffmpeg.get_ffmpeg_exe()
r = subprocess.run([ff, "-ss", "0.3", "-i", f"{D}/video.mp4", "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
frame = np.frombuffer(r.stdout, np.uint8).reshape(512, 512, 3)

# 对应第 9 帧左右（0.3s @30fps）
fi = mj["frames"][9]
c2w, intr = _load_pose(fi)
c2w = np.asarray(c2w, dtype=np.float64)
intr = np.asarray(intr, dtype=np.float64)
intr = intr * 0.5           # prepare_motion_seqs 里做了 scale_intrs(intr, .5, .5)
intr[2, 2] = 1.0
print("intr fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]))

def project(V, c2w, intr):
    w2c = np.linalg.inv(c2w)
    cam = V @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    ok = z > 1e-6
    inv = 1.0 / np.where(ok, z, 1)
    u = intr[0, 0] * cam[:, 0] * inv + intr[0, 2]
    vv = intr[1, 1] * cam[:, 1] * inv + intr[1, 2]
    return u, vv, ok

u, vv, ok = project(V, c2w, intr)
inside = ok & (u >= 0) & (u < 512) & (vv >= 0) & (vv < 512)
print("落在画面内: %d / %d (%.1f%%)" % (inside.sum(), len(V), inside.mean() * 100))

# 采样这些像素的颜色，看是否集中在肤色/石色（非背景白）
px = frame[np.clip(np.rint(vv).astype(int), 0, 511), np.clip(np.rint(u).astype(int), 0, 511)]
fg = (px.reshape(-1, 3).astype(int).sum(1) < 720)  # 排除接近纯白背景
print("投影点中落在前景(非纯白)的比例: %.1f%%" % (fg.mean() * 100))
print("投影点平均 RGB: %s" % px.mean(0).round(1))

# 做个可视化：把投影点画在帧上
vis = frame.copy()
sel = np.flatnonzero(inside)
for i in sel[::7]:
    x, y = int(u[i]), int(vv[i])
    if 0 <= x < 512 and 0 <= y < 512:
        vis[max(0, y-1):y+2, max(0, x-1):x+2] = [255, 0, 0]
Image.fromarray(np.concatenate([frame, vis], 1)).save("/tmp/proj_check.png")
print("saved /tmp/proj_check.png (左=原始帧, 右=投影点叠加)")
