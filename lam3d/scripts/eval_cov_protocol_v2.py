#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测协议 v2 的可执行实现（协议文档 docs/eval_protocol_v2.md 的配套代码）。

设计动机见文档 §1：现有指标族有两类结构性盲区
  (a) 像素族对方法间差异没有信息量（E_var/E_sys = 1e-4）；
  (b) 几何族（eig/rigid/false_move/roughness）是对方法一「定义」的自证。
本脚本给出 v2 的**可观测**端点，全部不依赖任何被测方法：

  SCE  silhouette coverage error —— 用形变 FLAME 网格的轮廓当独立锚，
       完全去掉造成参考系 A 失败的那个 11 dB 着色/纹理工序误差。
  MAD  multi-view artifact dispersion —— 同一帧在多视角下的 SCE 离散度。
  TJA  temporal jerk amplification —— 输出轨迹 jerk / 驱动信号 jerk。
  RTT  rotation transport target —— 切平面极分解（2x2）独立估计的旋转参考。
  STAT 效应量 / MDE / 嵌套自由度审计。

阶段：
  sce     渲染 + 轮廓覆盖误差 + alpha/RGB 方差分解（GPU，约 3 min）
  rtt     旋转参考与切平面漂移（GPU，约 2 min）
  traj    轨迹库生成（CPU，秒级）
  stats   把上面的 JSON 合成统计表（CPU）

用法（后台，必须，前台 20 s 会被杀）：
  cd /workspace/lam3d && nohup env \
    PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_protocol_v2.py \
    --stage sce,rtt,traj --assets messi,barbara,musk --views front,yaw35,yaw60 \
    > /tmp/v2_pilot.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cov_methods as E  # noqa: E402  reuse render_full / MeshRaster / psnr / ssim
from lam3d.core import arkit_live, cov_switch, tga_covariance  # noqa: E402

METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
SHORT = {"lam": "M-LAM", "tga": "M-TGA", "rot_sync": "M-RS", "vertex_then_polar": "M-VTP"}
SIZE = 512
SCE_RIM = 2          # 轮廓带半宽（像素），与 coverage_metrics(tol=2) 对齐
ROI_THR_MM = 3.0     # 表情 ROI：形变位移超过 3 mm 的顶点才纳入锚
ROI_DILATE = 12      # ROI 外扩像素（覆盖唇/眼周过渡带）
VIEWS = {"front": 0.0, "yaw35": 0.610865, "yaw60": 1.047198,
         "yawm35": -0.610865, "yawm60": -1.047198}
OUT = Path(os.environ.get("V2_OUT", "/tmp/cov_v2"))   # 可分片运行，最后 merge 成一份
DEG = 180.0 / math.pi


def log(msg):
    print("[v2] " + str(msg), flush=True)


# --------------------------------------------------------------------------
# 轮廓算子（不引入新依赖：scipy.ndimage 已在环境里）
# --------------------------------------------------------------------------
def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def dilate(m, r=2):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, structure=_disk(r)) if r > 0 else m


def erode(m, r=2):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, structure=_disk(r)) if r > 0 else m


def project_points(x, c2w, intr):
    """世界点 (N,3) -> 像素 (N,2) + 相机系深度 z。与 MeshRaster 同一约定。"""
    w2c = np.linalg.inv(c2w)
    pc = x @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    u = intr[0, 0] * pc[:, 0] / np.maximum(np.abs(z), 1e-9) + intr[0, 2]
    v = intr[1, 1] * pc[:, 1] / np.maximum(np.abs(z), 1e-9) + intr[1, 2]
    return np.stack([u, v], 1), z


def roi_from_displacement(x0, xb, c2w, intr, hw, thr_mm=3.0, dil=12):
    """表情 ROI：形变位移 > thr_mm 的顶点投影后取稠密区域。

    ★ 这个 ROI 只依赖 FLAME 网格顶点（x0/xb 来自 mesh_vertices），
      与四个协方差方法完全无关，因此可以作为 SCE/DEP 的合法锚。
      动机：整头掩膜里 60% 是头发（FLAME 网格不含头发），
      直接算 halo 会把「头发」误记成「溢出面」（pilot 实测 halo=0.62）。
    """
    from scipy.ndimage import binary_fill_holes
    disp = np.linalg.norm(xb - x0, axis=-1) * 1000.0     # mm
    sel = disp > thr_mm
    h, w = hw
    m = np.zeros((h, w), bool)
    if sel.sum() < 8:
        return m, float(sel.mean())
    uv, z = project_points(xb[sel], c2w, intr)
    ok = np.isfinite(uv).all(1) & (z > 0)
    uv = uv[ok]
    if len(uv) < 8:
        return m, float(sel.mean())
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    m[vi, ui] = True
    m = binary_fill_holes(dilate(m, 3))
    m = dilate(m, dil)
    return m, float(sel.mean())


def depth_protrusion(gs_depth, mesh_z, roi, tau_mm=1.5):
    """DEP：3DGS 深度比网格更靠近相机（薄片斜插出皮肤）的像素比例与幅度。

    ★ 这是「artifact 侧」指标：不比较谁更接近某个参考，只问
      「有没有高斯穿出了可见表面」。四方法的差别主要在这里。
    符号约定先自动对齐（两种 z 方向都试，取与网格深度相关性更高者）。
    """
    ok = roi & np.isfinite(mesh_z) & np.isfinite(gs_depth) & (mesh_z > 0)
    if int(ok.sum()) < 100:
        return {"frac": float("nan"), "p95_mm": float("nan"), "n": int(ok.sum())}
    a = gs_depth[ok].astype(np.float64)
    b = mesh_z[ok].astype(np.float64)
    if np.corrcoef(a, b)[0, 1] < 0:
        a = -a
    d = (b - a) * 1000.0                    # >0 = 高斯在网格前面（穿出）
    scale = np.median(np.abs(b)) * 1000.0
    return {"frac": float((d > tau_mm).mean()),
            "p95_mm": float(np.percentile(d, 95)),
            "median_mm": float(np.median(d)),
            "frac_gt": {str(t): float((d > t).mean()) for t in (1, 2, 4, 8)},
            "n": int(ok.sum())}


def coverage_metrics(alpha, mesh_mask, thr=0.5, tol=2, roi=None):
    """SCE 家族。全部以形变网格轮廓 M 为锚，与任何协方差方法无关。

    halo  溢出面 : 3DGS 覆盖到 (M 外扩 tol 像素) 之外的比例 —— 薄片斜插出皮肤
    holes 空洞率 : M 内缩 tol 像素之后仍不透明的面积比例 —— 高斯塌陷/覆盖不足
    iou        : 与既有 mesh_iou 同定义，保留可比性
    edge_mass  : 边界带（M 与 erode(M) 之间）的平均 alpha —— 边缘糊/硬
    """
    G = alpha > thr
    M = mesh_mask.astype(bool)
    Mo = dilate(M, tol)
    Mi = erode(M, tol)
    R = np.ones_like(M) if roi is None else roi.astype(bool)
    inter = float((G & M & R).sum())
    union = float(((G | M) & R).sum())
    # halo/holes 的分母用 ROI 内的网格面积，而不是整头掩膜：
    # 整头掩膜 60% 是头发（FLAME 网格无头发），否则 halo 会被头发支配。
    halo = float((G & ~Mo & R).sum()) / max(float((M & R).sum()), 1.0)
    hole = float((Mi & ~(alpha > 0.3) & R).sum()) / max(float((Mi & R).sum()), 1.0)
    band = M & ~Mi & R
    edge = float(alpha[band].mean()) if band.any() else float("nan")
    return {"iou": inter / max(union, 1.0),
            "halo": halo,
            "holes": hole,
            "edge_mass": edge,
            "roi_frac": float(R.mean()),
            "area_g": float((G & R).mean()), "area_m": float((M & R).mean())}


# --------------------------------------------------------------------------
# 几何参考：切平面极分解（2x2）——与四方法代码路径无关的独立构造
# --------------------------------------------------------------------------
def tangent_basis(n):
    """对每个法向给出一组正交切基 [N,3,2]，用固定辅助轴避免 branch cut。"""
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    aux = np.where(np.abs(n[..., :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]),
                   np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(n, np.broadcast_to(aux, n.shape))
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2], axis=-1)


def tangential_rotation(x0, xB, faces, n0, nB):
    """每个顶点的切平面 2x2 形变梯度 -> 极分解 -> 旋转角（度）。

    这是「表面映射的切向旋转」的无偏估计：表面是 2 维流形，其切向线性映射
    唯一确定一个 SO(2) 极因子，且这个因子不由四种方法中的任何一种定义。
    Kabsch-on-edges（既有 rigid_err 的参考）在 3D 里做同一件事，会把法向弯曲
    也算进旋转；本函数把它投影掉，两者之差本身就是一个诊断量。
    """
    nv = x0.shape[0]
    f = faces.astype(np.int64)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    # 边方向在 rest 切基里的分量 -> 2x2 映射 E2 = E1 @ F^T
    T0 = tangent_basis(n0)                      # [N,3,2]
    T1 = tangent_basis(nB)
    p = x0[dst] - x0[src]
    q = xB[dst] - xB[src]
    # 局部坐标：把边投影到 *源点* 的切基上
    p2 = np.einsum("mi,mij->mj", p, T0[src])    # [E,2]
    q2 = np.einsum("mi,mij->mj", q, T1[src])    # [E,2]
    H = np.zeros((nv, 2, 2))
    for i in range(2):
        for j in range(2):
            H[:, i, j] = np.bincount(src, weights=p2[:, i] * q2[:, j], minlength=nv)
    # 2x2 极分解：M = U S V^T 的 U，用 SVD（nv=20k，2x2，10 ms 量级）
    u, _, vt = np.linalg.svd(H.transpose(0, 2, 1))
    R = np.einsum("nij,njk->nik", u, vt)
    ang = np.degrees(np.arctan2(R[:, 1, 0] - R[:, 0, 1], R[:, 0, 0] + R[:, 1, 1]))
    return ang, np.linalg.norm(H.transpose(0, 2, 1), axis=(1, 2))


def rotation_angle_deg_3d(a, b=None):
    """两组 3x3 旋转之间的测地角（度）。b=None 时等价于与单位阵比较。"""
    a = np.asarray(a, np.float64)
    if b is None:
        b = np.broadcast_to(np.eye(3), a.shape)
    b = np.asarray(b, np.float64)
    if b.shape != a.shape:
        b = np.broadcast_to(b, a.shape)
    tr = np.einsum("nij,nij->n", np.swapaxes(a, 1, 2), b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))


def quat_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def sigma_of(q, s):
    R = quat_to_R(np.asarray(q, np.float64))
    d = np.asarray(s, np.float64) ** 2
    return (R * d[:, None, :]) @ np.swapaxes(R, 1, 2)


def pctl(x, qs=(50, 90, 99)):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    out = {"mean": float(x.mean()), "max": float(x.max())}
    for q in qs:
        out["p%d" % q] = float(np.percentile(x, q))
    return out


# --------------------------------------------------------------------------
# 轨迹库（时域协议 §5）——纯 CPU，输出 npz 供渲染阶段使用
# --------------------------------------------------------------------------
def raised_cosine(n, a0=0.0, a1=1.0):
    t = np.linspace(0.0, 1.0, n)
    return a0 + (a1 - a0) * 0.5 * (1.0 - np.cos(np.pi * t))


def trajectory_library(n_channel=52, names=None, fps=30):
    """返回 {name: (T x 52) float32, meta}。全部是 ARKit52 驱动信号。"""
    idx = {n: i for i, n in enumerate(names or [])}
    lib = {}

    def place(track):
        Tn = max(len(np.atleast_1d(v)) for v in track.values())
        out = np.zeros((Tn, n_channel), np.float32)
        for ch, amp in track.items():
            if ch in idx:
                out[:, idx[ch]] = np.asarray(amp, np.float32).reshape(-1)[:Tn]
        return out

    T = 60
    # T1 单通道 raised-cosine 往返：无端点 jerk，检迟滞与过冲
    for ch in ("jawOpen", "mouthSmileLeft", "browInnerUp", "eyeBlinkLeft"):
        lib["ramp_%s" % ch] = place({ch: np.concatenate([raised_cosine(T // 2),
                                                         raised_cosine(T // 2)[::-1]])})
    # T2 阶跃：检振铃 / 单帧跳变
    step = np.zeros(T)
    step[T // 2:] = 1.0
    lib["step_jawOpen"] = place({"jawOpen": step})
    lib["step_smile"] = place({"mouthSmileLeft": step, "mouthSmileRight": step})
    # T3 带限抖动：检噪声放大（jerk 放大率的直接压力测试）
    rng = np.random.default_rng(20261024)
    for amp in (0.25, 0.5):
        k = 3
        white = rng.standard_normal((T, n_channel))
        ker = np.ones(k) / k
        sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, white)
        sm = np.clip(0.5 + amp * sm / max(np.abs(sm).std(), 1e-9) * 0.25, 0.0, 1.0)
        lib["jitter_a%02d" % int(amp * 100)] = sm.astype(np.float32)
    # T4 共发音：两组通道同相/反相，检聚合器在冲突旋转下的行为
    lib["co_smile_jaw"] = place({"mouthSmileLeft": raised_cosine(T),
                                 "mouthSmileRight": raised_cosine(T),
                                 "jawOpen": raised_cosine(T)[::-1]})
    # T5 对抗：嘴内/眼内翻折区（无解区顶点）的驱动
    lib["adv_mouth_funnel"] = place({"mouthFunnel": raised_cosine(T),
                                     "mouthPucker": raised_cosine(T)[::-1]})
    lib["adv_wink"] = place({"eyeBlinkLeft": raised_cosine(T)})
    meta = {k: {"fps": fps, "frames": int(v.shape[0]),
                "channels": [int(i) for i in np.nonzero(np.abs(v).sum(0) > 1e-6)[0]]}
            for k, v in lib.items()}
    return lib, meta


# --------------------------------------------------------------------------
# 统计核心（§4）
# --------------------------------------------------------------------------
def cohen_dz(diff):
    d = np.asarray(diff, np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-15 else float("inf") * np.sign(d.mean())


def mde_two_sided(dz, n, alpha=0.05, power=0.80):
    """给定配对效应量与样本量，反解可检出的最小效应（近似正态）。"""
    from scipy.stats import norm
    if not np.isfinite(dz) or n < 2:
        return float("nan")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) / math.sqrt(n))


def cluster_bootstrap(diffs, n_boot=2000, seed=7):
    """对「资产」层做簇 bootstrap（不是对高斯做——那是伪重复）。"""
    d = np.asarray(diffs, np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    if n < 2:
        return {"mean": float(d.mean()) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(n)}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(1)
    return {"mean": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(n), "n_boot": n_boot}


# --------------------------------------------------------------------------
# 阶段 SCE
# --------------------------------------------------------------------------
def build_session(asset):
    from lam3d.core.pipeline import SingleImage3DPipeline
    img = Path("/workspace/LAM/assets/sample_input/%s" % asset)
    if not img.exists():                      # 允许只给 stem（dufu -> dufu.jpg）
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = img.with_name(img.name + ext)
            if cand.exists():
                img = cand
                break
    if not img.exists():
        raise FileNotFoundError("asset not found: %s" % img)
    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    # ★ job_dir 里绝不能出现 '.png'：LAM 的 preprocess 用
    #   os.path.join(alpha_dir, name).replace('.png', '.jpg') 生成 alpha 路径，
    #   replace 作用在整条路径上，会把 /tmp/v2_job_messi.png/... 改写成
    #   /tmp/v2_job_messi.jpg/... 从而写入一个不存在的镜像目录（静默丢帧）。
    stem = Path(asset).stem
    pipe.run(image_path=img, job_dir="/tmp/v2_job_%s" % stem,
             job_id="v2_%s" % stem)
    log("session %s ready %.1fs" % (asset, time.time() - t0))
    return pipe.arkit_session


def stage_sce(assets, view_names, vectors):
    from lam3d.core import arkit_live, cov_switch, tga_covariance
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    rep = {"assets": assets, "views": view_names, "vectors": vectors,
           "per": {}, "variance": {}}
    for asset in assets:
        sess = build_session(asset)
        renderer = sess["renderer"]
        meta = arkit_live.arkit_meta()
        faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
        base = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
        base.to("cuda").eval()
        raster = E.MeshRaster(faces, SIZE, SIZE)
        rep["per"][asset] = {}

        x_rest = E.mesh_vertices(renderer, sess["flame_data"],
                                 E.vector_for(meta, "zeros"), base)
        for vec_name in vectors:
            vec = E.vector_for(meta, vec_name)
            mv = E.mesh_vertices(renderer, sess["flame_data"], vec, base)
            frames = {}
            for name in METHODS:
                cov_switch.set_method(name, renderer=renderer)
                for view in view_names:
                    frames[(name, view)] = E.render_full(
                        sess, vec, yaw=VIEWS[view], size=SIZE)
            for view in view_names:
                fr0 = frames[("lam", view)]
                cols = np.clip(fr0["col"], 0.0, 1.0)
                best = None
                for sign_y in (-1.0, 1.0):
                    raster.set_view(fr0["c2w"], fr0["intr"], sign_y)
                    img, mask, _ = raster.render(mv, cols)
                    gm = fr0["alpha"] > 0.5
                    iou = float((mask & gm).sum()) / max(float((mask | gm).sum()), 1.0)
                    if best is None or iou > best[0]:
                        best = (iou, sign_y, img, mask)
                iou, sign_y, gt_img, gt_mask = best
                # 重取网格深度（best 循环里最后一轮不一定是选中那轮）
                raster.set_view(fr0["c2w"], fr0["intr"], sign_y)
                _, gt_mask, gt_z = raster.render(mv, cols)
                hh, ww = gt_mask.shape
                roi, roi_frac = roi_from_displacement(
                    x_rest, mv, fr0["c2w"], fr0["intr"], (hh, ww),
                    thr_mm=ROI_THR_MM, dil=ROI_DILATE)
                entry = {"mesh_iou_raw": iou, "roi_frac": roi_frac,
                         "roi_npix": int(roi.sum())}
                alphas, rgbs = [], []
                for name in METHODS:
                    fr = frames[(name, view)]
                    cm = coverage_metrics(fr["alpha"], gt_mask, roi=roi)
                    cm["psnr_vs_mesh"] = E.psnr(fr["rgb"], gt_img)
                    cm["ssim_vs_mesh"] = E.ssim(fr["rgb"], gt_img)
                    cm["dep"] = depth_protrusion(fr["depth"], gt_z, roi)
                    entry[name] = cm
                    alphas.append(fr["alpha"].astype(np.float64))
                    rgbs.append(fr["rgb"].astype(np.float64))
                # 方差分解：同一个公式，分别作用在 alpha 与 RGB 上。
                # ★ 关键修正（pilot 实测驱动）：整幅图的 E_var/E_sys 被大片
                #   饱和区域（头皮 alpha≡1、纯背景 alpha≡0）稀释 2~3 个数量级，
                #   比值 3e-06 是「没信息」，不是「方法一致」。信息只在轮廓带，
                #   所以必须同时给受限区域版本，否则该比值不能区分方法与噪声。
                Mm = gt_mask.astype(bool)
                regions = {
                    "full": np.ones_like(Mm),
                    "ring8": dilate(Mm, 8) & ~erode(Mm, 8),   # 轮廓两侧 8px
                    "rim2": dilate(Mm, SCE_RIM) & ~erode(Mm, SCE_RIM),
                    "inside8": erode(Mm, 8),                  # 内部（RGB 用）
                    "outside8": ~dilate(Mm, 8),               # 纯背景（漏光用）
                }
                A = np.stack(alphas, 0)
                C = np.stack(rgbs, 0)
                var = {}
                for rname, rmask in regions.items():
                    if int(rmask.sum()) < 200:
                        continue
                    for sig, stack, ref in (
                            ("alpha", A, Mm.astype(np.float64)),
                            ("rgb", C, gt_img.astype(np.float64))):
                        mean_r = stack.mean(0)
                        e_sys = float(np.mean((mean_r[rmask] - ref[rmask]) ** 2))
                        e_var = float(np.mean((stack[:, rmask] - mean_r[rmask]) ** 2))
                        var["%s|%s" % (sig, rname)] = {
                            "E_sys": e_sys, "E_var": e_var,
                            "ratio": e_var / max(e_sys, 1e-20),
                            "npix": int(rmask.sum())}
                # 落盘以便离线复算（alpha fp16、rgb uint8，体积可控）
                np.savez_compressed(
                    OUT / "frames" / ("%s_%s_%s.npz" % (Path(asset).stem, vec_name, view)),
                    alpha=A.astype(np.float16), rgb=C.astype(np.uint8),
                    mesh_mask=Mm, methods=np.array(METHODS))
                alphas, rgbs = None, None
                rep["per"][asset]["%s|%s" % (vec_name, view)] = entry
                rep["variance"]["%s|%s|%s" % (asset, vec_name, view)] = var
                log("%-8s %-10s %-7s  mesh_iou %.4f | halo %s | "
                    "Evar/Esys alpha rim2 %.2e | rgb rim2 %.2e | alpha full %.2e"
                    % (asset, vec_name, view, iou,
                       " ".join("%s:%.4f" % (SHORT[m], entry[m]["halo"]) for m in METHODS),
                       var.get("alpha|rim2", {}).get("ratio", float("nan")),
                       var.get("rgb|rim2", {}).get("ratio", float("nan")),
                       var.get("alpha|full", {}).get("ratio", float("nan"))))
        del sess
        import torch
        torch.cuda.empty_cache()
        # 每个资产一落盘：资产之间 session 要 75 s，崩在最后一个资产不该
        # 丢掉前面已经算完的帧（第一轮就踩过这个坑）。
        (OUT / "sce.json").write_text(json.dumps(rep, indent=1))
    (OUT / "sce.json").write_text(json.dumps(rep, indent=1))
    log("wrote %s" % (OUT / "sce.json"))
    return rep


# --------------------------------------------------------------------------
# 阶段 RTT
# --------------------------------------------------------------------------
def stage_rtt(assets, vectors):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"assets": assets, "vectors": vectors, "per": {}}
    for asset in assets:
        sess = build_session(asset)
        renderer = sess["renderer"]
        meta = arkit_live.arkit_meta()
        faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
        gs = sess["gs_model"]
        q0 = gs.rotation.detach().float().cpu().numpy().astype(np.float64)
        s0 = gs.scaling.detach().float().cpu().numpy().astype(np.float64)
        x0 = gs.xyz.detach().float().cpu().numpy().astype(np.float64)
        base = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
        base.to("cuda").eval()
        rep["per"][asset] = {}

        n0, _ = E.vertex_normals(x0, faces)
        thin0 = np.argmin(s0, -1)
        R0 = quat_to_R(q0)
        thin0v = R0[np.arange(len(q0)), :, thin0]
        cos0 = np.abs((thin0v * n0).sum(-1))
        ang0 = np.degrees(np.arccos(np.clip(cos0, 0.0, 1.0)))
        aniso0 = s0.max(-1) / np.maximum(s0.min(-1), 1e-12)

        for vec_name in vectors:
            vec = E.vector_for(meta, vec_name)
            xb = E.mesh_vertices(renderer, sess["flame_data"], vec, base)
            nB, _ = E.vertex_normals(xb, faces)
            t_ang, t_weight = tangential_rotation(x0, xb, faces, n0, nB)
            entry = {"tangential_rotation_deg": pctl(np.abs(t_ang), (50, 90, 99)),
                     "tangential_weight": pctl(t_weight, (50, 90)),
                     "rest_normal_angle": pctl(ang0, (50, 90)),
                     "aniso_rest": pctl(aniso0, (50, 90))}
            for name in METHODS:
                cov_switch.set_method(name, renderer=renderer)
                mod = renderer._cov_modulator
                q1, s1, _ = E.modulate(mod, renderer, sess["flame_data"], vec,
                                       gs.xyz.detach().unsqueeze(0).float(),
                                       gs.rotation.detach().float(),
                                       gs.scaling.detach().float())
                q1 = q1.astype(np.float64); s1 = s1.astype(np.float64)
                R1 = quat_to_R(q1)
                thin1 = np.argmin(s1, -1)
                thin1v = R1[np.arange(len(q1)), :, thin1]
                cos1 = np.abs((thin1v * nB).sum(-1))
                ang1 = np.degrees(np.arccos(np.clip(cos1, 0.0, 1.0)))
                # NTT：薄轴相对 *真实* 曲面法向的漂移（薄片 -> 0 是唯一正解）
                drift = np.abs(ang1 - ang0)
                planar = aniso0 > 2.0            # 只有平面型高斯薄轴才有意义
                # 与切平面独立参考比旋转：3D 极分解角 vs 2D 切向角
                r_delta = R1 @ np.swapaxes(R0, 1, 2)
                d3 = rotation_angle_deg_3d(r_delta)
                entry[name] = {
                    "ntt_drift_deg": pctl(drift[planar], (50, 90, 99)),
                    "ntt_frac_lt5": float((drift[planar] < 5).mean()),
                    "ntt_frac_lt10": float((drift[planar] < 10).mean()),
                    "ntt_frac_gt30": float((drift[planar] > 30).mean()),
                    "applied_deg": pctl(d3[planar], (50, 90)),
                    "frac_planar": float(planar.mean()),
                }
            rep["per"][asset][vec_name] = entry
            log("%-8s %-10s tangential p50 %.3f p99 %.3f | NTT p50 %s | <10deg %s"
                % (asset, vec_name, entry["tangential_rotation_deg"]["p50"],
                   entry["tangential_rotation_deg"]["p99"],
                   " ".join("%s:%.2f" % (SHORT[m], entry[m]["ntt_drift_deg"]["p50"])
                            for m in METHODS),
                   " ".join("%s:%.3f" % (SHORT[m], entry[m]["ntt_frac_lt10"])
                            for m in METHODS)))
        del sess
        import torch
        torch.cuda.empty_cache()
    (OUT / "rtt.json").write_text(json.dumps(rep, indent=1))
    log("wrote %s" % (OUT / "rtt.json"))
    return rep


# --------------------------------------------------------------------------
# 阶段 STAT
# --------------------------------------------------------------------------
def stage_stats():
    rep = {}
    p = OUT / "sce.json"
    if p.is_file():
        sce = json.loads(p.read_text())
        rows = sce["per"]
        out = {}
        for metric in ("iou", "halo", "holes", "psnr_vs_mesh", "ssim_vs_mesh"):
            per_asset = {}
            for asset, conds in rows.items():
                vals = {m: [conds[c][m][metric] for c in conds] for m in METHODS}
                per_asset[asset] = vals
            # 方法间配对差（以方法一为基准），以「条件」为单位，再在资产上簇 bootstrap
            for base_m in ("rot_sync", "lam", "tga"):
                for m in METHODS:
                    if m == base_m:
                        continue
                    diffs_all = []
                    for asset, vals in per_asset.items():
                        d = np.array(vals[m], np.float64) - np.array(vals[base_m], np.float64)
                        diffs_all.append(float(np.nanmean(d)))
                    bs = cluster_bootstrap(np.array(diffs_all))
                    out.setdefault("%s|%s_vs_%s" % (metric, m, base_m), bs)
            # 条件内配对的效应量（跨所有资产/条件池化，仅作参考）
            dz = {}
            for base_m in ("rot_sync", "lam", "tga"):
                if metric in ("iou", "psnr_vs_mesh", "ssim_vs_mesh"):
                    pass
                d = []
                for asset, vals in per_asset.items():
                    d.append(np.array(vals["rot_sync"], np.float64)
                             - np.array(vals[base_m], np.float64))
                dz["rot_sync_vs_%s" % base_m] = cohen_dz(np.concatenate(d))
            out.setdefault("%s|dz" % metric, dz)
        rep["sce"] = out
        # 方差分解汇总
        v = sce["variance"]
        # 键形如 "alpha|rim2" / "rgb|inside8"：按 (通道, 分区) 分别聚合。
        # 分区很关键——整幅图的 E_var/E_sys 被饱和区稀释，分区的比值才是判别量。
        keys = sorted({k for c in v.values() for k in c})
        agg = {}
        for k in keys:
            rows = [v[c][k] for c in v if k in v[c]]
            if not rows:
                continue
            sys_e = float(np.mean([r["E_sys"] for r in rows]))
            var_e = float(np.mean([r["E_var"] for r in rows]))
            agg[k] = {"E_sys": sys_e, "E_var": var_e,
                      "ratio": var_e / max(sys_e, 1e-20),
                      "n_cond": len(rows)}
        rep["variance_summary"] = agg
        for k in ("alpha|rim2", "rgb|rim2", "alpha|inside8", "rgb|inside8"):
            if k in agg:
                log("variance %-14s E_var/E_sys %.2e" % (k, agg[k]["ratio"]))

    p = OUT / "rtt.json"
    if p.is_file():
        rtt = json.loads(p.read_text())
        out = {}
        for metric in ("ntt_drift_deg", "ntt_frac_lt10", "ntt_frac_gt30"):
            per_asset = {}
            for asset, conds in rtt["per"].items():
                vals = {}
                for m in METHODS:
                    if metric in ("ntt_drift_deg",):
                        vals[m] = [conds[c][m][metric]["p50"] for c in conds]
                    else:
                        vals[m] = [conds[c][m][metric] for c in conds]
                per_asset[asset] = vals
            for base_m in ("rot_sync", "lam", "tga"):
                for m in METHODS:
                    if m == base_m:
                        continue
                    d = np.array([np.nanmean(np.array(per_asset[a][m], np.float64)
                                             - np.array(per_asset[a][base_m], np.float64))
                                  for a in per_asset])
                    out["%s|%s_vs_%s" % (metric, m, base_m)] = cluster_bootstrap(d)
        rep["rtt"] = out
    (OUT / "stats.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps(rep, indent=1)[:4000])
    return rep


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sce,rtt,traj,stats")
    ap.add_argument("--assets", default="messi")
    ap.add_argument("--views", default="front,yaw35")
    ap.add_argument("--vectors", default="ones,half,rand,extreme,bigSmile")
    args = ap.parse_args()
    # LAM 的 pipeline 在构造时会自己 parse_args(sys.argv)，我们的 --stage/--assets
    # 会把它打挂（see eval_cov_methods.py:331 'never use argparse here'）。
    # 解析完立刻把 argv 清干净，让下游 LAM 看到一个空的参数表。
    sys.argv = [sys.argv[0]]
    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    vectors = [v.strip() for v in args.vectors.split(",") if v.strip()]
    # 资产名保留原样（不带扩展名），由 build_session 逐个后缀探测，
    # 因为 sample_input 里混着 .png/.jpg（dufu.jpg、james.png ...）。
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "traj" in stages:
        from lam3d.core import arkit_live
        meta = arkit_live.arkit_meta()
        lib, tmeta = trajectory_library(meta["count"], meta["names"])
        np.savez_compressed(OUT / "trajectories.npz", **lib)
        (OUT / "trajectories_meta.json").write_text(json.dumps(tmeta, indent=1))
        log("trajectories: %d 条 -> %s" % (len(lib), OUT / "trajectories.npz"))
    if "sce" in stages:
        stage_sce(assets, views, vectors)
    if "rtt" in stages:
        stage_rtt(assets, vectors)
    if "stats" in stages:
        stage_stats()
    log("done %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
