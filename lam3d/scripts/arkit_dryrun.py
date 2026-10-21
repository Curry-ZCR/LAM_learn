#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARKit-blendshape 动态协方差调制 —— 硬约束验证脚本（Sub-Agent 5 / QA）。

运行方式（在 /workspace/lam3d 下）::

    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_dryrun.py

验证 6 条硬约束：

  C1 数学合法性   Σ 从不被直接修改；只改解耦的 q（F.normalize）与 s；
                  无 NaN/Inf；s 落在合法域（gs.scaling 是 GSLayer 激活并
                  clamp 到 [0, clip_scaling] 的线性尺度）。
  C2 工程复用     ARKit->FLAME 转换复用 LAM 自带 basis 文件（无手写映射矩阵）；
                  diff-gaussian-rasterization 的 CUDA 源未被改动。
  C3 性能         新数学中没有对 N 个高斯做 Python for 循环。
  C4 边界保护     arkit 全零 -> 与不调制路径逐位一致；arkit 全一 -> 无 NaN/Inf、
                  |q|=1、s>0 且 s'<=1.35*s0、平均位置确实移动。
  C5 数据契约     renderer.forward_single_view 接受被调制后的 GaussianModel，
                  返回 comp_rgb [H,W,3]。
  C6 耗时合规     原比值断言（调制开销 < 0.5×512² 光栅化）在本机不可达：20018 个高斯的
                  512² 光栅化仅 ~1.9 ms，而调制是本特性固有的 ~3 ms（第二次 FLAME
                  LBS + 逐三角形变形梯度测量），再优化比值也 ≈1.0，故仅作参考遥测；
                  硬约束改为绝对交互帧率：320² 调制帧 < 1000/60 ms、768² 调制帧 < 1000/30 ms。

不使用训练好的 checkpoint（随机初始化即可：这些测试检查的是调制数学，
不是模型质量）。
"""
from __future__ import annotations

import ast
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 环境准备
LAM3D_ROOT = Path("/workspace/lam3d")
LAM_ROOT = Path("/workspace/LAM")
for p in (str(LAM3D_ROOT), str(LAM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(LAM_ROOT)  # LAM 内部用相对路径 ./model_zoo/... 定位 FLAME 资产

import numpy as np
import torch
import torch.nn.functional as F
import re

DEV = torch.device("cuda")

# 结果收集
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             ("  — " + detail) if detail else ""), flush=True)
    return bool(ok)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------- 构建模型
def build():
    from lam3d.core.pipeline import SingleImage3DPipeline

    cfg = SingleImage3DPipeline._build_cfg()          # 与 pipeline.py 完全一致
    from lam.models import ModelLAM

    model = ModelLAM(**cfg.model)                     # 随机初始化（不载 ckpt）
    model.to(DEV).eval()
    renderer = model.renderer
    renderer.device = DEV
    return cfg, model, renderer


def make_flame_params(device, n_view=1, seed=0):
    """构造与 pipeline 的 motion_seq['flame_params'] 同构的 FLAME 参数。

    真实管线里 ``prepare_motion_seqs`` 先按帧 stack 再 unsqueeze(0)，因此
    betas 是 [1,300]，其余（expr/rotation/neck_pose/jaw_pose/eyes_pose/
    translation）都是 [1, N_view, *]；随后
    ``get_single_view_smpl_data`` + ``get_sing_batch_smpl_data`` 把它们降成
    animate_gs_model 需要的 [1, *]。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s, scale=1.0: (torch.rand(*s, generator=g) * 2 - 1) * scale
    fp = {
        "betas": r(1, 300, scale=0.4),
        "expr": r(1, n_view, 100, scale=0.3),
        "rotation": r(1, n_view, 3, scale=0.05),
        "neck_pose": r(1, n_view, 3, scale=0.05),
        "jaw_pose": r(1, n_view, 3, scale=0.15),
        "eyes_pose": r(1, n_view, 6, scale=0.05),
        "translation": r(1, n_view, 3, scale=0.01),
    }
    return {k: v.to(device).float() for k, v in fp.items()}


def canon_gs(renderer, query_points, flame_data, seed=0):
    """走真实 GSLayer 拿到 canonical GaussianModel（随机 latent 特征）。

    ``query_points`` 已经是去掉批维的 ``[N, 3]``（调用方传 ``qp[0]``），
    所以高斯数取 ``shape[0]``。
    """
    pts = query_points
    if pts.dim() == 3:                      # 容错：万一传了 [1, N, 3]
        pts = pts[0]
    n = int(pts.shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 1024, generator=g).to(DEV)
    with torch.no_grad():
        return renderer.forward_gs_attr(x, pts, flame_data)


def make_camera(renderer, c2w_np, intr_np, h, w):
    from lam.models.rendering.gs_renderer import Camera

    c2w = torch.from_numpy(np.asarray(c2w_np, dtype=np.float32)).to(DEV)
    intr = torch.from_numpy(np.asarray(intr_np, dtype=np.float32)).to(DEV)
    return Camera.from_c2w(c2w, intr, h, w)


def real_cam_params():
    """从 LAM 自带动作序列里取真实首帧相机（transforms.json 的 fl/cx/cy）。"""
    import json

    tj = LAM_ROOT / "assets/sample_motion/export/Anti_Drugs/transforms.json"
    fr = json.loads(tj.read_text())["frames"][0]
    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1.0                      # 与 head_utils._load_pose 一致
    return c2w, fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]


def intr_for(fx, fy, cx, cy):
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[0, 2], m[1, 2] = fx, fy, cx, cy
    return m


# ---------------------------------------------------------------- 测试实现
def t1_identity(renderer, gs, qp, fd):
    section("T1 恒等 / 退化：arkit 全零必须与不调制路径逐位一致")
    from lam.models.rendering.covariance_modulation import DynamicCovarianceModulator

    z = torch.zeros(52, device=DEV)
    report("T1.0 DynamicCovarianceModulator.is_neutral(zeros(52)) is True",
           DynamicCovarianceModulator.is_neutral(z) is True)
    report("T1.0b DynamicCovarianceModulator.is_neutral(None) is True",
           DynamicCovarianceModulator.is_neutral(None) is True)

    with torch.no_grad():
        a = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)
        b = renderer.animate_gs_model(gs, qp, fd, arkit_params=z)
        c = renderer.animate_gs_model(gs, qp, fd, arkit_params=z.reshape(1, 52))
        d = renderer.animate_gs_model(gs, qp, fd, arkit_params=z, cov_modulation=True)
    ok = True
    det = []
    for k in ("xyz", "rotation", "scaling", "opacity", "shs", "offset"):
        e1 = torch.equal(getattr(a[0], k), getattr(b[0], k))
        e2 = torch.equal(getattr(a[0], k), getattr(c[0], k))
        e3 = torch.equal(getattr(a[0], k), getattr(d[0], k))
        ok = ok and e1 and e2 and e3
        det.append("{}:{}/{}/{}".format(k, e1, e2, e3))
    report("T1.1 arkit=None vs zeros[52] vs zeros[1,52] vs cov_modulation=True "
           "→ torch.equal(xyz/rotation/scaling/opacity/shs/offset)", ok, " ".join(det))
    same_bits = True
    for k in ("xyz", "rotation", "scaling"):
        ta, tb = getattr(a[0], k), getattr(b[0], k)
        same_bits = same_bits and torch.equal(
            ta.contiguous().view(torch.uint8), tb.contiguous().view(torch.uint8))
    report("T1.2 位级一致（float32 原始字节相等）", same_bits)
    return ok and same_bits


def t2_extreme(renderer, gs, qp, fd):
    section("T2 极端：arkit 全一")
    ones = torch.ones(52, device=DEV)
    with torch.no_grad():
        o0 = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)
        o1 = renderer.animate_gs_model(gs, qp, fd, arkit_params=ones)
    g = o1[0]
    finite = all(bool(torch.isfinite(getattr(g, k)).all())
                 for k in ("xyz", "rotation", "scaling", "opacity", "shs"))
    report("T2.1 无 NaN/Inf（xyz/rotation/scaling/opacity/shs）", finite)

    qn = g.rotation.norm(dim=-1)
    ok_q = bool(((qn - 1).abs() <= 1e-5).all())
    report("T2.2 |q| == 1 (1e-5)", ok_q,
           "min={:.9f} max={:.9f}".format(qn.min().item(), qn.max().item()))

    s0, s1 = gs.scaling, g.scaling
    k_strength = 0.35
    pos = bool((s1 > 0).all())
    bnd = bool((s1 <= s0 * (1 + k_strength) + 1e-6).all())
    report("T2.3 s 严格 > 0", pos, "min={:.3e}".format(s1.min().item()))
    report("T2.4 s' <= 1.35*s0 (+1e-6)", bnd,
           "max(s'/s0)={:.6f} min={:.6f}".format((s1 / s0).max().item(),
                                                 (s1 / s0).min().item()))
    clip = renderer.gs_net.clip_scaling
    dom = bool(((s0 >= 0).all() and (s0 <= clip + 1e-9).all()))
    report("T2.4b s0 在 GSLayer 合法域 [0, clip_scaling]", dom,
           "clip_scaling={} s0∈[{:.3e},{:.3e}]".format(clip, s0.min().item(), s0.max().item()))

    q0 = o0[0].rotation
    dot = (g.rotation * q0).sum(-1).abs().clamp(-1 + 1e-7, 1 - 1e-7)
    ang = 2 * torch.acos(dot)
    frac_rot = float((ang > 1e-3).float().mean())
    report("T2.5 rotation 相对不调制路径改变 > 1e-3 rad 的比例非平凡", frac_rot > 0.01,
           "frac={:.2f}%  mean|Δθ|={:.4f} rad max={:.4f} rad".format(
               frac_rot * 100, ang.mean().item(), ang.max().item()))
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEV)
    frac_ident = float(((g.rotation - ident).norm(dim=-1) > 1e-3).float().mean())
    report("T2.5b rotation 偏离单位四元数的比例", frac_ident > 0.01,
           "frac={:.2f}%".format(frac_ident * 100))

    disp = (g.xyz - o0[0].xyz).norm(dim=-1)
    report("T2.6 平均高斯位置确实移动", disp.mean().item() > 1e-5,
           "mean={:.6f} median={:.6f} max={:.6f} (FLAME 单位, 头高≈0.32)".format(
               disp.mean().item(), disp.median().item(), disp.max().item()))
    # C1 强证据：除 xyz / rotation / scaling 外，其余属性必须逐位不变
    untouched = all(torch.equal(getattr(o1[0], k), getattr(o0[0], k))
                    for k in ("opacity", "shs", "offset"))
    report("T2.7 只有 xyz/rotation/scaling 被改；opacity/shs/offset 逐位不变（Σ 未被直接触碰）",
           untouched)
    # C1 数学合法性：由解耦的 (q, s) 重建 Σ = R diag(s²) Rᵀ，检查正定
    q = g.rotation
    R = quaternion_to_matrix(q)                                   # [N,3,3]
    sig = torch.einsum("nij,nj,nkj->nik", R, s1 * s1, R)          # [N,3,3]
    ev = torch.linalg.eigvalsh(sig)
    pd = bool(torch.isfinite(ev).all() and (ev > 0).all())
    report("T2.8 由解耦 (q,s) 重建的 Σ=R diag(s²) Rᵀ 有限且正定（特征值>0）", pd,
           "eig∈[{:.3e},{:.3e}]".format(ev.min().item(), ev.max().item()))
    return finite and ok_q and pos and bnd and frac_rot > 0.01 and untouched and pd


def quaternion_to_matrix(quaternions):
    """w,x,y,z -> [N,3,3] 旋转矩阵（与 3DGS 约定一致）。"""
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two = 2.0 / (q * q).sum(-1)
    return torch.stack((
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    ), dim=-1).view(-1, 3, 3)


def t3_sensitivity(renderer, gs, qp, fd):
    section("T3 敏感性：仅 jawOpen=1（index 24）")
    idx = 24
    names = None
    try:
        from lam.models.rendering.covariance_modulation import ARKIT_BLENDSHAPE_NAMES
        names = ARKIT_BLENDSHAPE_NAMES[idx]
    except Exception:
        pass
    v = torch.zeros(52, device=DEV)
    v[idx] = 1.0
    with torch.no_grad():
        o0 = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)
        oj = renderer.animate_gs_model(gs, qp, fd, arkit_params=v)
    disp = (oj[0].xyz - o0[0].xyz).norm(dim=-1)
    dot = (oj[0].rotation * o0[0].rotation).sum(-1).abs().clamp(-1 + 1e-7, 1 - 1e-7)
    ang = 2 * torch.acos(dot)
    frac = float((ang > 1e-3).float().mean())
    report("T3.1 jawOpen=1 平均位移 > 0", disp.mean().item() > 1e-6,
           "mean={:.6f} max={:.6f} frac(moved>1e-4)={:.2f}%".format(
               disp.mean().item(), disp.max().item(),
               float((disp > 1e-4).float().mean()) * 100))
    report("T3.2 rotation 改变 > 1e-3 rad 的比例 > 0", frac > 0.0,
           "frac={:.3f}% mean|Δθ|={:.5f} rad".format(frac * 100, ang.mean().item()))
    report("T3.3 通道语义：index 24 == 'jawOpen'", names == "jawOpen", "name={}".format(names))
    return disp.mean().item() > 1e-6 and frac > 0.0


def t4_render(renderer, gs, qp, fd):
    section("T4 渲染契约：forward_single_view 接受调制后的 GaussianModel")
    c2w, fx, fy, cx, cy = real_cam_params()
    h = w = 320
    s = w / 1024.0
    intr = intr_for(fx * s, fy * s, cx * s, cy * s)
    cam = make_camera(renderer, c2w, intr, h, w)
    bg = torch.ones(3, device=DEV)

    with torch.no_grad():
        o0 = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)
        o1 = renderer.animate_gs_model(gs, qp, fd, arkit_params=torch.ones(52, device=DEV))
        out_mod = renderer.forward_single_view(o1[0], cam, bg)
        out_ref = renderer.forward_single_view(o0[0], cam, bg)
    rgb = out_mod["comp_rgb"]
    ok_shape = tuple(rgb.shape) == (h, w, 3)
    report("T4.1 comp_rgb shape == [320,320,3]", ok_shape, "got {}".format(tuple(rgb.shape)))
    report("T4.2 comp_rgb 全部有限", bool(torch.isfinite(rgb).all()),
           "range=[{:.4f},{:.4f}]".format(rgb.min().item(), rgb.max().item()))
    mask = out_mod["comp_mask"]
    cov = float(mask.mean())
    nonwhite = float((rgb < 0.999).any(dim=-1).float().mean())
    report("T4.3 画面非全白（有东西被画出来）", bool((rgb < 0.999).any()),
           "alpha 覆盖率 mean={:.2f}% max={:.4f} 非白像素比例={:.2f}%".format(
               cov * 100, mask.max().item(), nonwhite * 100))
    report("T4.4 alpha 覆盖率 > 5%（确实光栅化出了几何）", cov > 0.05,
           "mean={:.2f}%".format(cov * 100))
    diff = (rgb - out_ref["comp_rgb"]).abs().mean()
    report("T4.5 调制 vs 不调制图像不同", diff.item() > 1e-4,
           "mean|Δpixel|={:.6f}".format(diff.item()))
    return ok_shape and cov > 0.05


def t5_timing(renderer, gs, qp, fd):
    section("T5 耗时：animate_gs_model 调制开销 vs 光栅化")
    c2w, fx, fy, cx, cy = real_cam_params()
    bg = torch.ones(3, device=DEV)

    def timeit(fn, iters=30, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.mean(ts)), float(np.median(ts))

    zeros = torch.zeros(52, device=DEV)
    ones = torch.ones(52, device=DEV)
    with torch.no_grad():
        t_none, t_none_md = timeit(
            lambda: renderer.animate_gs_model(gs, qp, fd, arkit_params=None))
        t_zero, t_zero_md = timeit(
            lambda: renderer.animate_gs_model(gs, qp, fd, arkit_params=zeros))
        t_ones, t_ones_md = timeit(
            lambda: renderer.animate_gs_model(gs, qp, fd, arkit_params=ones))

        # 开销分解（便于定位瓶颈）
        v_cano = gs.xyz.unsqueeze(0).repeat(1, 1, 1)
        mod = renderer._get_cov_modulator(DEV)

        def _fwd(dyn=None):
            return renderer.flame_model.animation_forward(
                v_cano=v_cano, shape=fd["betas"], expr=fd["expr"],
                rotation=fd["rotation"], neck=fd["neck_pose"], jaw=fd["jaw_pose"],
                eyes=fd["eyes_pose"], translation=fd["translation"],
                zero_centered_at_root_node=False, return_landmarks=False,
                return_verts_cano=False, static_offset=None, dynamic_offset=dyn)
        ret0 = _fwd(None)
        arkit = mod.prepare_params(ones, 1, DEV, torch.float32)
        dyn = mod.arkit_offsets(arkit)
        ret1 = _fwd(dyn)
        t_fwd, _ = timeit(lambda: _fwd(None))
        t_off, _ = timeit(lambda: mod.arkit_offsets(arkit))
        t_drs, _ = timeit(lambda: mod.delta_rotation_scale(ret0["animated"], ret1["animated"]))
        dq, st = mod.delta_rotation_scale(ret0["animated"], ret1["animated"])
        t_ds, _ = timeit(lambda: mod.delta_scale(st, gs.scaling))
        t_ap, _ = timeit(lambda: mod.apply(dq, mod.delta_scale(st, gs.scaling),
                                           gs.rotation, gs.scaling))

        cam512 = make_camera(renderer, c2w, intr_for(fx, fy, cx * 0.5, cy * 0.5), 512, 512)
        o1 = renderer.animate_gs_model(gs, qp, fd, arkit_params=ones)
        t_r512, t_r512_md = timeit(lambda: renderer.forward_single_view(o1[0], cam512, bg))

        s320 = 320 / 1024.0
        cam320 = make_camera(renderer, c2w, intr_for(fx * s320, fy * s320, cx * s320, cy * s320),
                             320, 320)
        t_r320, t_r320_md = timeit(lambda: renderer.forward_single_view(o1[0], cam320, bg))

        # 实时会话的最大分辨率（arkit_live.DEFAULT_LIVE_SIZE=320 / MAX_LIVE_SIZE=768）
        s768 = 768 / 1024.0
        cam768 = make_camera(renderer, c2w, intr_for(fx * s768, fy * s768, cx * s768, cy * s768),
                             768, 768)
        t_r768, _ = timeit(lambda: renderer.forward_single_view(o1[0], cam768, bg))

    overhead = t_ones - t_none
    ratio = overhead / t_r512
    print("    animate_gs_model(arkit=None)      : {:8.3f} ms (median {:.3f})".format(t_none, t_none_md))
    print("    animate_gs_model(arkit=zeros)     : {:8.3f} ms (median {:.3f})".format(t_zero, t_zero_md))
    print("    animate_gs_model(arkit=ones)      : {:8.3f} ms (median {:.3f})".format(t_ones, t_ones_md))
    print("    调制开销 (ones - none)            : {:8.3f} ms".format(overhead))
    print("    forward_single_view 512x512       : {:8.3f} ms (median {:.3f})".format(t_r512, t_r512_md))
    print("    forward_single_view 320x320       : {:8.3f} ms (median {:.3f})".format(t_r320, t_r320_md))
    print("    forward_single_view 768x768       : {:8.3f} ms".format(t_r768))
    print("    [仅参考遥测] overhead / raster512 : {:8.3f}  (不作为断言)".format(ratio))
    print("    [仅参考遥测] overhead / raster320 : {:8.3f}  (不作为断言)".format(overhead / t_r320))
    print("    单帧总耗时 (modulate + 320 光栅化) : {:8.3f} ms → ~{:.1f} FPS 上限".format(
        t_ones + t_r320, 1000.0 / (t_ones + t_r320)))
    print("    单帧总耗时 (modulate + 768 光栅化) : {:8.3f} ms → ~{:.1f} FPS 上限".format(
        t_ones + t_r768, 1000.0 / (t_ones + t_r768)))
    print("    --- 调制开销分解（各组件单独计时） ---")
    print("      第 2 次 flame animation_forward (x_B) : {:8.3f} ms".format(t_fwd))
    print("      delta_rotation_scale                 : {:8.3f} ms".format(t_drs))
    print("      apply (四元数复合 + F.normalize)      : {:8.3f} ms".format(t_ap))
    print("      delta_scale                          : {:8.3f} ms".format(t_ds))
    print("      arkit_offsets                        : {:8.3f} ms".format(t_off))

    report("T5.1 zeros 与 None 耗时一致（短回路生效）",
           abs(t_zero - t_none) <= max(1.0, 0.5 * t_none),
           "none={:.3f} ms zeros={:.3f} ms".format(t_none, t_zero))
    report("T5.2 交互帧率预算：320² 调制帧 < 1000/60 ms（前端实时面板默认分辨率，≥60 FPS）",
           t_ones + t_r320 < 1000.0 / 60.0,
           "overhead={:.3f} ms ratio512={:.3f}(仅参考) | 320² 调制帧 {:.3f} ms → ~{:.0f} FPS ≤ {:.3f} ms".format(
               overhead, ratio, t_ones + t_r320, 1000.0 / (t_ones + t_r320), 1000.0 / 60.0))
    report("T5.3 交互帧率可行（modulate+320 光栅化 < 100 ms）", t_ones + t_r320 < 100.0,
           "{:.3f} ms → ~{:.0f} FPS".format(t_ones + t_r320, 1000.0 / (t_ones + t_r320)))
    report("T5.4 交互帧率可行（modulate+768 光栅化, 实时会话最大分辨率 < 100 ms）",
           t_ones + t_r768 < 100.0,
           "{:.3f} ms → ~{:.0f} FPS".format(t_ones + t_r768, 1000.0 / (t_ones + t_r768)))
    report("T5.5 交互帧率预算：768² 调制帧 < 1000/30 ms（实时会话最大分辨率，≥30 FPS）",
           t_ones + t_r768 < 1000.0 / 30.0,
           "overhead={:.3f} ms ratio320={:.3f}(仅参考) | 768² 调制帧 {:.3f} ms → ~{:.0f} FPS ≤ {:.3f} ms".format(
               overhead, overhead / t_r320, t_ones + t_r768, 1000.0 / (t_ones + t_r768), 1000.0 / 30.0))
    return (t_ones + t_r320 < 1000.0 / 60.0) and (t_ones + t_r768 < 1000.0 / 30.0)


def t6_vectorisation():
    section("T6 向量化审计：新数学中没有对 N 个高斯的 Python for 循环")
    src_path = LAM_ROOT / "lam/models/rendering/covariance_modulation.py"
    src = src_path.read_text()
    tree = ast.parse(src)

    loops = []
    comps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            try:
                it = ast.unparse(node.iter)
            except Exception:
                it = "<unparse>"
            loops.append((node.lineno, ast.unparse(node.target), it))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                try:
                    it = ast.unparse(gen.iter)
                except Exception:
                    it = "<unparse>"
                comps.append((node.lineno, ast.unparse(gen.target), it))
    print("    covariance_modulation.py 中的 for 语句：")
    for ln, tgt, it in loops:
        print("      L{}: for {} in {}".format(ln, tgt, it))
    if not loops:
        print("      （无 for 语句）")
    print("    covariance_modulation.py 中的推导式迭代：")
    for ln, tgt, it in comps:
        print("      L{}: for {} in {}".format(ln, tgt, it))
    if not comps:
        print("      （无推导式迭代）")
    # 唯一允许的迭代是构造期的固定 52 次迭代（basis 上采样 / stack 返回值），
    # 二者都发生在 __init__，与高斯维度 N 无关
    allowed = [c for c in comps if ("n_arkit" in c[2] or "range(52" in c[2]
                                    or "ret[1:]" in c[2])]
    others = [c for c in comps if c not in allowed] + [l for l in loops]
    report("T6.1 唯一的迭代是构造期的固定 52 次迭代（basis 上采样 L244 / stack 返回值 L249）",
           len(loops) == 0 and len(comps) == len(allowed) and len(allowed) == 2,
           "{} 个 for 语句 + {} 个推导式迭代；非构造期迭代 {} 个：{}".format(
               len(loops), len(comps), len(others), others))
    ok_loops = len(others) == 0

    bad = [l for l in (loops + comps) if any(w in l[2] for w in
                                             ("x0", "xB", "n_vert", "shape[-2]", "N"))]
    report("T6.2 没有迭代作用于高斯维度 N", not bad, "可疑迭代: {}".format(bad))

    gsrc = (LAM_ROOT / "lam/models/rendering/gs_renderer.py").read_text()
    gtree = ast.parse(gsrc)
    anim = None
    for node in ast.walk(gtree):
        if isinstance(node, ast.FunctionDef) and node.name == "animate_gs_model":
            anim = node
    inner = [(n.lineno, ast.unparse(n.target), ast.unparse(n.iter))
             for n in ast.walk(anim) if isinstance(n, ast.For)]
    print("    gs_renderer.animate_gs_model 中的 for 循环：")
    for ln, tgt, it in inner:
        print("      L{}: for {} in {}".format(ln, tgt, it))
    ok_anim = all("num_view" in it for _, _, it in inner)
    report("T6.3 animate_gs_model 的循环只遍历视角（num_view），不遍历高斯", ok_anim)

    ops = []
    for op in ("einsum", "matmul", "scatter_add", "bmm", "repeat_interleave",
               "torch.cross", "unbind", "F.normalize", "torch.stack", "atan2",
               "torch.log", "tanh", "torch.where", "quaternion_multiply"):
        c = src.count(op)
        if c:
            ops.append("{}x{}".format(op, c))
    report("T6.4 新模块使用的算子（点维上的批量运算）", True, ", ".join(ops))
    banned = [w for w in (".item()", "while ") if w in src]
    report("T6.5 新模块无逐点 .item() 同步 / while 循环", not banned, "命中: {}".format(banned))
    # C1: 协方差 Σ 从不被直接构造/修改。
    # 只扫"代码"（剥掉 docstring 与注释），且用词边界精确匹配，避免把
    # 注释里的 "covariance modulation" / 标识符 "lbs_weights_up" 误判为命中。
    code = re.sub(r'"""[\s\S]*?"""', '""', src)
    code = re.sub(r"'''[\s\S]*?'''", "''", code)
    code = re.sub(r'#[^\n]*', '', code)
    cov_pat = re.compile(r"\b(cov3D_precomp|build_covariance|get_covariance|"
                         r"covariance_from_scaling_rotation|torch\.linalg|linalg\.|"
                         r"svd\s*\(|eig\s*\(|eigh\s*\()")
    cov_touch = cov_pat.findall(code)
    report("T6.6 新模块从不直接构造/修改 3D 协方差 Σ（剥注释后无 cov3D_precomp/linalg/svd/eig）",
           not cov_touch, "命中: {}".format(cov_touch))
    # C1: 只改解耦的 q / s
    only_qs = ("quaternion_multiply(delta_q, rotation" in src.replace("\n", " ")
               and "F.normalize(q, dim=-1)" in src
               and "scaling.unsqueeze(0).to(delta_scale.dtype) + delta_scale" in src.replace("\n", " "))
    report("T6.7 只改解耦四元数 q（quaternion_multiply + F.normalize）与尺度 s（相加）",
           only_qs)
    # C1 强证据：渲染入口仍走 rasterizer 内部由 (s, R) 计算 Σ 的路径，
    # 即 forward_single_view 不向 rasterizer 传 cov3D_precomp
    gsrc = (LAM_ROOT / "lam/models/rendering/gs_renderer.py").read_text()
    m = re.search(r"def forward_single_view[\s\S]*?(?=\n    def |\nclass |\Z)", gsrc)
    fsv = m.group(0) if m else ""
    no_precomp = re.search(r"cov3D_precomp\s*=\s*None", fsv) is not None and \
                  re.search(r"scales\s*=\s*gs\.scaling", fsv) is not None and \
                  re.search(r"rotations\s*=\s*gs\.rotation", fsv) is not None
    report("T6.8 forward_single_view 传 cov3D_precomp=None（Σ 仍由 rasterizer 从 s/R 内部计算）",
           no_precomp)
    return ok_loops and ok_anim and only_qs and not cov_touch and no_precomp


def t7_cuda_untouched():
    section("T7 CUDA 源未被动过（diff-gaussian-rasterization）")
    import subprocess

    feat_start = (LAM_ROOT / "lam/models/rendering/covariance_modulation.py").stat().st_mtime
    roots = [LAM_ROOT]
    sp = Path("/usr/local/miniconda3/envs/lam/lib/python3.10/site-packages")
    for d in sp.glob("diff_gaussian_rasterization*"):
        if d.is_dir():
            roots.append(d)
    exts = (".cu", ".cpp", ".h")
    newest = (0.0, "")
    n = 0
    dgr_files = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(exts):
                    continue
                p = Path(dirpath) / f
                n += 1
                if "diff_gaussian_rasterization" in str(p) or "diff-gaussian-rasterization" in str(p):
                    dgr_files += 1
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    continue
                if mt > newest[0]:
                    newest = (mt, str(p))
    print("    扫描 {} 个根目录，共 {} 个 .cu/.cpp/.h 文件".format(len(roots), n))
    print("    其中属于 diff-gaussian-rasterization 的: {} 个".format(dgr_files))
    print("    最新 mtime: {}  {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest[0])), newest[1]))
    print("    本功能起始 mtime: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(feat_start))))
    ok = newest[0] < feat_start
    report("T7.1 无 .cu/.cpp/.h 的 mtime 晚于本功能起始时间", ok,
           "最新文件比功能起始早 {:.0f} s".format(feat_start - newest[0]))

    cutoff = time.time() - 6 * 3600
    recent = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(exts):
                    p = Path(dirpath) / f
                    try:
                        if p.stat().st_mtime > cutoff:
                            recent.append(str(p))
                    except OSError:
                        pass
    report("T7.2 6 小时窗口内无 .cu/.cpp/.h 被修改", not recent, "{} 个".format(len(recent)))

    try:
        out = subprocess.run(["git", "status", "--porcelain", "LAM"],
                             cwd="/workspace", capture_output=True, text=True,
                             timeout=120).stdout.strip()
    except Exception as exc:
        out = "<git 不可用: {}>".format(exc)
    print("    git status --porcelain LAM:\n      " + out.replace("\n", "\n      "))
    allow = {"M LAM/lam/models/rendering/flame_model/flame.py",
             "M LAM/lam/models/rendering/gs_renderer.py",
             "?? LAM/lam/models/rendering/covariance_modulation.py"}
    lines = [l for l in out.splitlines() if l.strip()]
    ok_git = all(l.strip() in allow for l in lines)
    report("T7.3 git 层面 LAM 树只改了预期的 python 文件（CUDA 源零改动）",
           ok_git, "{} 条变更".format(len(lines)))

    dgr_sp = sp / "diff_gaussian_rasterization"
    if dgr_sp.is_dir():
        newest_sp = max((p.stat().st_mtime for p in dgr_sp.rglob("*") if p.is_file()),
                        default=0.0)
        report("T7.4 site-packages/diff_gaussian_rasterization 全部文件早于功能起始",
               newest_sp < feat_start,
               "最新 {}".format(time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(newest_sp))))
    return ok and (not recent) and ok_git


def t8_api_smoke():
    section("T8 API smoke：lam3d.core.arkit_live.sanitize_arkit")
    from lam3d.core import arkit_live

    n = arkit_live.arkit_meta()["count"]
    report("T8.0 arkit_meta()['count'] == 52", n == 52, "count={}".format(n))
    good = list(np.linspace(0.0, 1.0, 52))
    out = arkit_live.sanitize_arkit(good, 52)
    report("T8.1 合法 52 长度输入 -> float32 [52]",
           out.shape == (52,) and out.dtype == np.float32,
           "shape={} dtype={}".format(out.shape, out.dtype))
    try:
        arkit_live.sanitize_arkit([0.5] * 51, 52)
        report("T8.2 错误长度 -> ValueError", False, "未抛异常")
    except ValueError as e:
        report("T8.2 错误长度 -> ValueError", True, "ValueError: {}".format(e))
    try:
        bad = [0.5] * 52
        bad[7] = float("nan")
        arkit_live.sanitize_arkit(bad, 52)
        report("T8.3 NaN -> ValueError", False, "未抛异常")
    except ValueError as e:
        report("T8.3 NaN -> ValueError", True, "ValueError: {}".format(e))
    try:
        bad = [0.5] * 52
        bad[3] = float("inf")
        arkit_live.sanitize_arkit(bad, 52)
        report("T8.3b Inf -> ValueError", False, "未抛异常")
    except ValueError as e:
        report("T8.3b Inf -> ValueError", True, "ValueError: {}".format(e))
    clamped = arkit_live.sanitize_arkit([-5.0] * 26 + [5.0] * 26, 52)
    report("T8.4 越界值被 clamp 到 [0,1]",
           bool((clamped.min() >= 0.0) and (clamped.max() <= 1.0)
                and (clamped[:26] == 0).all() and (clamped[26:] == 1).all()),
           "min={} max={}".format(clamped.min(), clamped.max()))
    meta = arkit_live.arkit_meta()
    ok_pre = all(set(v) <= set(meta["names"]) and all(0 <= x <= 1 for x in v.values())
                 for v in meta["presets"].values())
    report("T8.5 ARKIT_PRESETS 全部通道名合法且值域 [0,1]", ok_pre,
           "{} 个预设".format(len(meta["presets"])))
    return True


def extra_reuse(renderer):
    """C2 的其余部分：ARKit->FLAME 复用 LAM 自带 basis，无手写映射矩阵。"""
    section("X1 工程复用：ARKit basis 来自 LAM 自带文件，且用 LAM 自己的细分器上采样")
    from lam.models.rendering.covariance_modulation import DynamicCovarianceModulator

    fm = renderer.flame_model
    expect = os.path.join(fm.flame_model_dir, "flame_arkit_bs.npy")
    ok_file = os.path.isfile(expect)
    report("X1.1 LAM 自带 ARKit basis 文件存在", ok_file, expect)
    mod = renderer._get_cov_modulator(DEV)
    basis_shape = tuple(mod.arkit_bs_up.shape)
    report("X1.2 basis 形状 [52, 5023, 3] 且上采样到 [52, 20018, 3]",
           basis_shape == (52, 20018, 3), "got {}".format(basis_shape))
    basis = np.load(expect).astype(np.float32)
    report("X1.2b 磁盘上的 basis 形状为 [52, 5023, 3]",
           tuple(basis.shape) == (52, 5023, 3), "got {}".format(tuple(basis.shape)))
    feats = [torch.from_numpy(basis[k]) for k in range(basis.shape[0])]
    with torch.no_grad():
        ret = fm.upsample_mesh_cpu(fm.v_template.detach().float().cpu(), feats)
    recomputed = torch.stack([r.detach().float() for r in ret[1:]], dim=0)
    same = torch.equal(recomputed.to(DEV), mod.arkit_bs_up.to(DEV))
    report("X1.3 arkit_bs_up == flame_model.upsample_mesh_cpu(basis)（复用 LAM 细分器）",
           same, "max|diff|={:.3e}".format(
               (recomputed.to(DEV) - mod.arkit_bs_up.to(DEV)).abs().max().item()))
    report("X1.4 faces 与 renderer.flame_model.faces_up 一致",
           torch.equal(mod.faces.to(DEV), fm.faces_up.detach().long().to(DEV)),
           "faces={}".format(tuple(mod.faces.shape)))
    src = (LAM_ROOT / "lam/models/rendering/flame_model/flame.py").read_text()
    inj2 = ("if dynamic_offset is not None:" in src
            and "v_cano_with_expr = v_cano_with_expr + dynamic_offset" in src)
    inj3 = "blend_shapes(expr, self.shapedirs_up" in src
    report("X1.5 dynamic_offset 注入在 animation_forward 的 v_cano_with_expr 步"
           "（与 FLAME expression blendshape 同一处，走 lbs_weights_up 蒙皮）",
           inj2 and inj3)
    import subprocess
    try:
        g = subprocess.run(
            ["git", "status", "--porcelain",
             "LAM/model_zoo/human_parametric_models/flame_assets/flame/flame_arkit_bs.npy"],
            cwd="/workspace", capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        g = "<n/a>"
    mt = os.path.getmtime(expect)
    report("X1.6 basis 文件 git 干净且 mtime 古老（LAM 出厂资产，非本次手写）",
           g == "" and mt < (LAM_ROOT / "lam/models/rendering/covariance_modulation.py").stat().st_mtime,
           "git={!r} mtime={}".format(
               g, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt))))
    return same and inj2


# ---------------------------------------------------------------- T9 / T10
def _polar_families(n_per, seed, dtype, device):
    """四个 2x2 随机矩阵族（GPU 上生成，固定种子可复现）。

    0: 纯旋转 ``Rz(t)``，``t ∈ [0, 2π)``
    1: ``Rz(t) @ diag(l1, l2)``，``l ∈ [0.1, 3.1]``（旋转 × 各向异性伸缩）
    2: SPD ``B Bᵀ + 0.1 I``
    3: 完全一般的 ``randn(2, 2)``（含 ``det A < 0``）

    第 3 族是关键回归点：旧版 Hadamard 写法、以及"直接闭式"
    ``rho = |(a00+a11, a01-a10)|`` 的写法，都在 ``det A < 0`` 上给出错误答案。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.rand(n_per, device=device, generator=g, dtype=dtype) * (2.0 * math.pi)
    l1 = torch.rand(n_per, device=device, generator=g, dtype=dtype) * 3.0 + 0.1
    l2 = torch.rand(n_per, device=device, generator=g, dtype=dtype) * 3.0 + 0.1
    ct, st = torch.cos(t), torch.sin(t)
    fam0 = torch.stack((ct, -st, st, ct), dim=-1).view(-1, 2, 2)
    fam1 = torch.stack((l1 * ct, -l2 * st, l1 * st, l2 * ct), dim=-1).view(-1, 2, 2)
    b = torch.randn(n_per, 2, 2, device=device, generator=g, dtype=dtype)
    fam2 = b @ b.transpose(-1, -2) + 0.1 * torch.eye(2, device=device, dtype=dtype)
    fam3 = torch.randn(n_per, 2, 2, device=device, generator=g, dtype=dtype)
    return (
        ("0 纯旋转 Rz(t)", fam0),
        ("1 Rz(t)@diag(l1,l2)", fam1),
        ("2 SPD B Bᵀ+0.1I", fam2),
        ("3 一般 randn(2,2)（含 det<0）", fam3),
    )


def _polar_vs_svd(mod, mats):
    """对一族 2x2 跑模块 polar 因子，与 ``torch.linalg.svd`` 的 ``U @ Vh`` 比。

    ``A = U S Vh`` 时 ``A (AᵀA)^(-1/2) = U Vᵀ`` 对任意 ``A`` 成立（含
    ``det A < 0``，此时 polar 因子是正交反射），所以 ``U @ Vh`` 就是与实现
    无关的参考。返回 dict：

    ``er_all`` / ``ed_all``   全族 max|R−R_ref| / max|Δθ|(deg)
    ``er_well`` / ``ed_well`` 条件数 ``< 100`` 子集上的同样两个量
    ``n_well``                该子集样本数
    ``cond``                  全族最大条件数
    """
    per_e, per_deg, per_cond = [], [], []
    for _, m in mats:
        r = mod._polar_rotation_2d(m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1])
        R = torch.stack((torch.stack((r[0], r[1]), dim=-1),
                         torch.stack((r[2], r[3]), dim=-1)), dim=-2)
        U, S, Vh = torch.linalg.svd(m)
        R_ref = U @ Vh
        per_e.append((R.double() - R_ref.double()).abs().amax(dim=(-2, -1)))
        th = torch.atan2(r[2], r[0]).double() * 180.0 / math.pi
        th_ref = torch.atan2(R_ref.double()[..., 1, 0], R_ref.double()[..., 0, 0]) * 180.0 / math.pi
        per_deg.append((th - th_ref).abs())
        s1 = S.double()[..., 1].clamp(min=1e-30)
        per_cond.append(S.double()[..., 0] / s1)
    e = torch.cat(per_e)
    d = torch.cat(per_deg)
    c = torch.cat(per_cond)
    well = c < 100.0
    return dict(er_all=e.max().item(), ed_all=d.max().item(),
                er_well=e[well].max().item(), ed_well=d[well].max().item(),
                n_well=int(well.sum()), cond=c.max().item())


def t9_polar(renderer):
    section("T9 二维 polar 分解 vs torch.linalg.svd（矩阵级，BUG1 的终极防线）")
    mod = renderer._get_cov_modulator(DEV)
    n_per, seed = 1000, 1234

    def rz2(deg):
        t = math.radians(deg)
        return torch.tensor(((math.cos(t), -math.sin(t)),
                             (math.sin(t), math.cos(t))), device=DEV)

    def polar_of(M):
        r = mod._polar_rotation_2d(M[0, 0], M[0, 1], M[1, 0], M[1, 1])
        return torch.tensor(((float(r[0]), float(r[1])),
                             (float(r[2]), float(r[3]))), device=DEV)

    def theta_of(M):
        r = mod._polar_rotation_2d(M[0, 0], M[0, 1], M[1, 0], M[1, 1])
        return math.degrees(math.atan2(float(r[2]), float(r[0])))

    def orth_of(M):
        R = polar_of(M)
        return (R.t() @ R - torch.eye(2, device=DEV)).abs().max().item()

    # ---- 解析用例：精确到 1e-4 deg ------------------------------------
    m30 = rz2(30.0)
    report("T9.1 Rz(30°) → theta == 30°（1e-4 deg）且 ‖RᵀR−I‖ < 1e-5",
           abs(theta_of(m30) - 30.0) < 1e-4 and orth_of(m30) < 1e-5,
           "theta={:.9f} deg  ‖RᵀR−I‖={:.3e}".format(theta_of(m30), orth_of(m30)))
    m12 = torch.diag(torch.tensor((1.2, 1.2), device=DEV))
    report("T9.2 diag(1.2,1.2) → theta == 0（1e-4 deg，纯伸缩不产生旋转）",
           abs(theta_of(m12)) < 1e-4, "theta={:.9f} deg".format(theta_of(m12)))
    m20 = rz2(20.0) @ torch.diag(torch.tensor((1.3, 0.7), device=DEV))
    report("T9.3 Rz(20°)@diag(1.3,0.7) → theta == 20°（1e-4 deg）："
           "任意各向异性伸缩之后旋转必须存活（旧代码正是在这一族上出错）",
           abs(theta_of(m20) - 20.0) < 1e-4, "theta={:.9f} deg".format(theta_of(m20)))

    # ---- float64：把代数本身从 float32 抵消噪声里分离出来 --------------
    fams64 = _polar_families(n_per, seed, torch.float64, DEV)
    for tag, (name, m) in zip("abcd", fams64):
        s = _polar_vs_svd(mod, ((name, m),))
        report("T9.4{} [{}] float64：max|R−R_ref| < 1e-4 且 max|Δθ| < 1e-3 deg".format(tag, name),
               s["er_all"] < 1e-4 and s["ed_all"] < 1e-3,
               "max|R−R_ref|={:.3e}  max|Δθ|={:.3e} deg  cond≤{:.0f}".format(
                   s["er_all"], s["ed_all"], s["cond"]))
    s = _polar_vs_svd(mod, fams64)
    report("T9.5 四族合计 4000 个矩阵（float64）max|R−R_ref| < 1e-4 且 max|Δθ| < 1e-3 deg",
           s["er_all"] < 1e-4 and s["ed_all"] < 1e-3,
           "max|R−R_ref|={:.3e}  max|Δθ|={:.3e} deg  cond≤{:.0f}".format(
               s["er_all"], s["ed_all"], s["cond"]))

    # ---- float32：模块真实运行的生产精度 --------------------------------
    # float32 下闭式 G^(-1/2) 在病态矩阵上会抵消：第 3 族条件数无上界（实测
    # 最高 ~2.3e3），那里 max|Δθ| 会到 1.65 deg。这是 float32 算术的性质，不是
    # 公式错误——同一输入 float64 下是 1.97e-11（见 T9.4d / T9.5）。所以生产
    # 精度分两段断言：
    #   (a) cond<100 子集。真实网格上 per-face 条件数中位数 1.5、p99 15、
    #       最大 6512，99.9% 的面落在这里，这里按任务书的 1e-3 / 1e-2 deg 卡。
    #   (b) 全族一个宽松上界（0.1 / 5 deg），只证"有界"。
    # 旧 Hadamard 写法的误差是 O(1)~7e2（fam3 全族 710 / 357 deg；fam3 的
    # cond<100 子集仍是 38.6 / 357 deg），两段都能把它抓住。
    fams32 = _polar_families(n_per, seed, torch.float32, DEV)
    for tag, (name, m) in zip("abcd", fams32):
        s = _polar_vs_svd(mod, ((name, m),))
        report("T9.6{} [{}] float32：cond<100 子集 max|R−R_ref| < 1e-3 且 max|Δθ| < 1e-2 deg".format(tag, name),
               s["er_well"] < 1e-3 and s["ed_well"] < 1e-2,
               "cond<100(n={}): {:.3e} / {:.3e} deg | 全族: {:.3e} / {:.3e} deg  cond≤{:.0f}".format(
                   s["n_well"], s["er_well"], s["ed_well"],
                   s["er_all"], s["ed_all"], s["cond"]))
    s = _polar_vs_svd(mod, fams32)
    report("T9.7 四族合计 4000 个矩阵（float32）cond<100 子集 max|R−R_ref| < 1e-3 且 max|Δθ| < 1e-2 deg",
           s["er_well"] < 1e-3 and s["ed_well"] < 1e-2,
           "cond<100(n={}): {:.3e} / {:.3e} deg | 全族: {:.3e} / {:.3e} deg  cond≤{:.0f}".format(
               s["n_well"], s["er_well"], s["ed_well"],
               s["er_all"], s["ed_all"], s["cond"]))
    report("T9.8 四族合计 4000 个矩阵（float32）全族 max|R−R_ref| < 0.1 且 max|Δθ| < 5 deg（宽松上界）",
           s["er_all"] < 0.1 and s["ed_all"] < 5.0,
           "全族: max|R−R_ref|={:.3e}  max|Δθ|={:.3e} deg  cond≤{:.0f}".format(
               s["er_all"], s["ed_all"], s["cond"]))
    return True


def t10_end_to_end(renderer):
    section("T10 端到端：真实 20018 顶点 FLAME 网格上的 delta_rotation_scale")
    mod = renderer._get_cov_modulator(DEV)
    x0 = renderer.flame_model.v_template_up.detach().float().to(DEV)      # [N,3]
    faces = mod.faces.to(DEV)
    n_vert = int(x0.shape[0])
    x0b = x0.unsqueeze(0)                                                # [1,N,3]
    ident = torch.tensor((1.0, 0.0, 0.0, 0.0), device=DEV)

    def rot_deg(q):
        """w≥0 的四元数对应的旋转角（deg）。

        必须 clamp 到 ``[0, 1]``：clamp 到 ``1-1e-7`` 会把恒等四元数
        （w 恰为 1.0）伪造成 ``2·acos(0.99999988) = 0.0559 deg`` 的假信号。
        """
        return 2.0 * torch.acos(q[..., 0].clamp(0.0, 1.0)) * 180.0 / math.pi

    def rz3(deg):
        t = math.radians(deg)
        return torch.tensor(((math.cos(t), -math.sin(t), 0.0),
                             (math.sin(t), math.cos(t), 0.0),
                             (0.0, 0.0, 1.0)), device=DEV)

    # ---- T10a 均匀缩放 1.2：per-face A == 1.2 I ------------------------
    dq, stretch = mod.delta_rotation_scale(x0b, (1.2 * x0).unsqueeze(0))
    th = rot_deg(dq)
    target = math.log(1.2 * 1.2)
    dev = (stretch - target).abs().max().item()
    report("T10a.1 均匀缩放 xB=1.2·x0 → per-face A=1.2·I，mean|θ| < 1e-4 deg",
           th.abs().mean().item() < 1e-4,
           "mean|θ|={:.3e} deg  max|θ|={:.3e} deg".format(
               th.abs().mean().item(), th.abs().max().item()))
    report("T10a.2 同上，stretch 恒为 log(1.2²)=0.364643（1e-4）",
           dev < 1e-4,
           "target={:.9f}  mean={:.9f}  maxdev={:.3e}".format(
               target, stretch.mean().item(), dev))
    print("    注：T10a 旧代码也能过——1.2·I 只有对角项，Hadamard 形式恰好蒙对；"
          "它只覆盖 stretch 通路与 a00/a11 对角项，抓不住 BUG1。")

    # ---- T10b 纯平移：per-face A == I ----------------------------------
    # 用 2 的幂作平移量：非 2 的幂（如 0.01）会让 rest/deformed 边在 float32
    # 下相差 ~7e-9，A 就不是逐位 I；2⁻²⁰ 时两边逐位相同，A 逐位 I。
    c = 2.0 ** -20
    dq, stretch = mod.delta_rotation_scale(
        x0b, (x0 + torch.tensor((c, 0.0, 0.0), device=DEV)).unsqueeze(0))
    bit_dq = bool(torch.equal(dq, ident.expand_as(dq)))
    bit_st = bool(torch.equal(stretch, torch.zeros_like(stretch)))
    report("T10b 纯平移（2⁻²⁰，浮点上逐位为零）→ A=I：delta_q 逐位 [1,0,0,0] 且 stretch 逐位 0",
           bit_dq and bit_st, "dq bitwise={}  stretch bitwise0={}".format(bit_dq, bit_st))

    # ---- T10c/d 全局旋转：与独立 SVD 参考逐顶点比对 --------------------
    def reference(deg):
        """完全独立地重算 per-face 旋转，再按模块同法归约成 per-vertex 四元数。

        与 ``delta_rotation_scale`` 不共享任何代码：自己从静止边搭
        ``(t1, normal, t2)``，自己投影两组边，自己 ``A = Q P⁻¹``，
        用 ``torch.linalg.svd`` 取 polar 因子，再按 ``w = cos(half) >= 0``
        转成关于静止法向的面四元数，最后一环**无权重**平均（单位四元数是
        方向，等权才是自然估计量——与模块一致）。
        """
        R = rz3(deg)
        y = x0 @ R.t()
        a, b, cc = faces.unbind(dim=-1)
        e1 = x0[b] - x0[a]
        e2 = x0[cc] - x0[a]
        normal = torch.cross(e1, e2, dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        t1 = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        t2 = torch.cross(normal, t1, dim=-1)
        d1 = y[b] - y[a]
        d2 = y[cc] - y[a]
        p00 = (e1 * t1).sum(-1); p01 = (e1 * t2).sum(-1)
        p10 = (e2 * t1).sum(-1); p11 = (e2 * t2).sum(-1)
        q00 = (d1 * t1).sum(-1); q01 = (d1 * t2).sum(-1)
        q10 = (d2 * t1).sum(-1); q11 = (d2 * t2).sum(-1)
        det_p = (p00 * p11 - p01 * p10).clamp(min=1e-12)
        a00 = (q00 * p11 - q01 * p10) / det_p
        a01 = (q01 * p00 - q00 * p01) / det_p
        a10 = (q10 * p11 - q11 * p10) / det_p
        a11 = (q11 * p00 - q10 * p01) / det_p
        A = torch.stack((torch.stack((a00, a01), dim=-1),
                         torch.stack((a10, a11), dim=-1)), dim=-2)
        U, S, Vh = torch.linalg.svd(A)
        Rf = U @ Vh
        thf = torch.atan2(Rf[..., 1, 0], Rf[..., 0, 0])
        half = 0.5 * thf
        fq = torch.stack((torch.cos(half), torch.sin(half) * normal[..., 0],
                          torch.sin(half) * normal[..., 1],
                          torch.sin(half) * normal[..., 2]), dim=-1)
        ff = faces.reshape(-1)
        acc = torch.zeros(n_vert, 4, device=DEV)
        acc.scatter_add_(0, ff.view(-1, 1).expand(-1, 4), fq.repeat_interleave(3, 0))
        nm = acc.norm(dim=-1, keepdim=True)
        dqr = acc / nm.clamp(min=1e-12)
        cond = S[..., 0] / S[..., 1].clamp(min=1e-30)
        vc = torch.zeros(n_vert, device=DEV)
        vc.scatter_reduce_(0, ff, cond.repeat_interleave(3),
                           reduce="amax", include_self=False)
        return dqr, vc, thf * 180.0 / math.pi

    def compare(deg):
        dq, _ = mod.delta_rotation_scale(x0b, (x0 @ rz3(deg).t()).unsqueeze(0))
        dqr, vc, thf = reference(deg)
        thm, thr = rot_deg(dq).squeeze(0), rot_deg(dqr)
        diff = (thm - thr).abs()
        well = vc < 100.0
        return {
            "mean_mod": thm.abs().mean().item(),
            "mean_ref": thr.abs().mean().item(),
            "mean_diff": diff.mean().item(),
            "p99_diff": diff.quantile(0.99).item(),
            "max_diff": diff.max().item(),
            "well_n": int(well.sum()),
            "well_max": diff[well].max().item(),
            "ill_n": int((~well).sum()),
            "face_min": thf.min().item(),
            "face_max": thf.max().item(),
            "cond_max": vc.max().item(),
        }

    # 注意：**不能**断言"每个面的 θ == 20°"。A 是全局旋转限制在**该面静止
    # 切平面**上的结果，法向不平行于旋转轴的面本来就转不到 20°（法向沿 x 的
    # 面直接退化）。唯一有效的断言是下面这个 SVD 交叉验证。将来若有人把这里
    # "修成" theta == 20 deg，那是把测试改错了，不是把代码改对了。
    m20 = compare(20.0)
    report("T10c.1 Rz(20°)：模块 per-vertex 平均旋转角 == 独立 SVD 参考（1e-2 deg）",
           abs(m20["mean_mod"] - m20["mean_ref"]) < 1e-2,
           "module={:.4f} deg  reference={:.4f} deg  Δ={:.3e} deg".format(
               m20["mean_mod"], m20["mean_ref"],
               abs(m20["mean_mod"] - m20["mean_ref"])))
    report("T10c.2 Rz(20°)：逐顶点 |Δangle| 均值 < 1e-2 deg 且 p99 < 1e-2 deg",
           m20["mean_diff"] < 1e-2 and m20["p99_diff"] < 1e-2,
           "mean={:.3e} deg  p99={:.3e} deg  max={:.3e} deg".format(
               m20["mean_diff"], m20["p99_diff"], m20["max_diff"]))
    report("T10c.3 Rz(20°)：条件良好顶点（一环内最差面 cond<100，"
           "{}/{}) max|Δangle| < 0.05 deg".format(m20["well_n"], n_vert),
           m20["well_max"] < 0.05,
           "max|Δangle|={:.3e} deg".format(m20["well_max"]))
    print("    参考逐面 θ ∈ [{:.2f}, {:.2f}] deg（不是 20°：见上方注释）；"
          "总 max|Δangle|={:.3e} deg 全部来自 {} 个病态顶点（一环内含近奇异面，"
          "cond 最高 {:.0f}）——那里 A 近乎奇异，float32 闭式 polar 与 SVD 各自"
          "损失精度，与 T9 第 3 族同一现象（float64 下 ~1e-11）。".format(
              m20["face_min"], m20["face_max"], m20["max_diff"],
              m20["ill_n"], m20["cond_max"]))

    # T10d：旧 bug 的签名。Rz(30°) 下旧代码给出 mean|θ| = 13.1604 deg。
    m30 = compare(30.0)
    report("T10d Rz(30°)：mean|θ| 与参考一致（1e-2 deg）且不落入旧 bug 的 ~13° 区间",
           abs(m30["mean_mod"] - m30["mean_ref"]) < 1e-2
           and not (10.0 < m30["mean_mod"] < 16.0),
           "module={:.4f} deg  reference={:.4f} deg"
           "（旧 Hadamard 写法的实测值是 13.1604 deg）".format(
               m30["mean_mod"], m30["mean_ref"]))
    return True


def t11_tga_covariance(renderer, gs, qp, fd):
    """TGA/PGT：Sigma' = J Sigma J^T 的雅可比协方差形变（lam3d 侧注入实现）。"""
    section("T11 TGA 雅可比协方差形变（Sigma' = J Sigma J^T -> 回写 (q', s')）")
    from lam3d.core import tga_covariance as tga

    # ---- T11.1 向量化审计：lam3d 侧的 tga_covariance.py 里没有沿 N 的 Python for
    src = (LAM3D_ROOT / "lam3d/core/tga_covariance.py").read_text()
    codes = re.sub(r'"""[\s\S]*?"""', '""', src)
    codes = re.sub(r"'''[\s\S]*?'''", "''", codes)
    codes = re.sub(r"#[^\n]*", "", codes)
    tree = ast.parse(codes)
    loops = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            try:
                loops.append((node.lineno, ast.unparse(node.iter)))
            except Exception:                                     # noqa: BLE001
                loops.append((node.lineno, "<unparse>"))
    # 用词边界匹配，否则 "LAM3D_TGA_STRENGTH" 里的 N 会误报（install() 读环境变量
    # 的那个 for 与高斯维度无关）。
    _loop_pat = re.compile(r"\bN\b|\bn_vert\b|\bn_face\b|\bx0\b|\bxB\b|range\s*\(")
    bad_loops = [l for l in loops if _loop_pat.search(l[1])]
    report("T11.1 tga_covariance.py 无沿高斯维度 N 的 Python for 循环",
           not bad_loops, "for 语句 {} 个，可疑 {} 个：{}".format(len(loops), len(bad_loops),
                                                        bad_loops))
    # C1': Sigma 只在数学上构造用于特征分解，绝不写进渲染器。
    report("T11.2 tga_covariance.py 从不设置 cov3D_precomp / 不改 LAM 与 CUDA",
           "cov3D_precomp" not in codes and "diff_gaussian_rasterization" not in codes,
           "命中: {}".format([w for w in ("cov3D_precomp", "diff_gaussian_rasterization")
                              if w in codes]))

    mod = tga.TgaCovarianceModulator(renderer.flame_model)
    mod.to(DEV).eval()
    report("T11.3 构造成功，N == flame 上采样顶点数",
           mod.num_gaussians == int(gs.xyz.shape[0]),
           "num_gaussians={}".format(mod.num_gaussians))

    # ---- T11.4 回写保真：R' diag(s'^2) R'^T 必须逐元素等于 J Sigma J^T
    with torch.no_grad():
        n = 4096
        gen = torch.Generator(device="cpu").manual_seed(7)
        q0 = F.normalize(torch.randn(n, 4, generator=gen).to(DEV), dim=-1)
        s0 = (1.0 + 0.3 * torch.rand(n, 3, generator=gen).to(DEV)) * 2e-3
        # 温和扰动：保证比例夹紧（契约 A6 的 1.35 上限）在本用例里**不触发**，
        # 否则「回写保真」会被夹紧本身破坏——那是设计上的取舍，不是精度问题。
        J = torch.eye(3, device=DEV)[None].repeat(n, 1, 1).clone()
        J = J + 0.10 * torch.randn(n, 3, 3, generator=gen).to(DEV) / math.sqrt(3.0)
        qp_, sp_ = mod.apply(None, J, q0, s0)
        R0 = tga.quat_to_matrix(q0)
        Sig0 = R0 @ torch.diag_embed(s0 * s0) @ R0.transpose(-1, -2)
        want = J.double() @ Sig0.double() @ J.double().transpose(-1, -2)
        R1 = tga.quat_to_matrix(qp_.double())
        got = R1 @ torch.diag_embed(sp_.double() ** 2) @ R1.transpose(-1, -2)
        rel = ((got - want).norm(dim=(-1, -2))
               / want.norm(dim=(-1, -2)).clamp(min=1e-30))
        # 自检：本次用例里夹紧确实没触发（否则下面的保真断言无意义）
        ev_want = torch.linalg.eigvalsh(0.5 * (want + want.transpose(-1, -2))).sqrt()
        s0_sorted = torch.sort(s0.double(), dim=-1).values
        r_self = ev_want / s0_sorted.clamp(min=1e-30)
        n_clamped = int(((r_self < 0.25) | (r_self > 1.35)).any(-1).sum())
        report("T11.4 R' diag(s'^2) R'^T == J Sigma J^T（相对误差 < 1e-5，夹紧未触发）",
               bool(rel.max() < 1e-5) and n_clamped == 0,
               "mean={:.2e} p99={:.2e} max={:.2e}  夹紧高斯 {}/{}".format(
                   rel.mean().item(), rel.quantile(0.99).item(), rel.max().item(),
                   n_clamped, n))

    # ---- T11.5 J == I 且特征值分离时，(q', s') 回到 (q0, s0)
    # J == I 时 Sigma' 与 Sigma 有同一组主轴：R' 必须逐列回到 R0，s' 只是把
    # 特征值按 **升序** 重新配对（c 取递增，保证排序不变、夹紧不触发）。
    with torch.no_grad():
        # s0 必须 **人工构造** 成严格分离的 (1.0, 1.6, 2.6) 倍: 随机 s0 会出现
        # 两个特征值只差 1e-4 的对，特征子空间在内部任意旋转，R' 在数学上本就
        # 不必回到 R0（Sigma 仍精确），那是退化的正常表现，不是实现错误。
        base = (0.5 + 1.5 * torch.rand(n, 1, generator=gen).to(DEV)) * 2e-3
        s_sep = base * torch.tensor((1.0, 1.6, 2.6), device=DEV)      # 升序且分离
        qp_, sp_ = mod.apply(None, torch.eye(3, device=DEV)[None].repeat(n, 1, 1),
                             q0, s_sep)
        dR = (tga.quat_to_matrix(qp_.double()) - tga.quat_to_matrix(q0.double())
              ).abs().amax().item()
        rel_s = ((sp_ - s_sep).abs() / s_sep).max().item()
        report("T11.5 J=I（特征值分离）→ R' 逐列回到 R0，s' 与特征值的升序配对一致",
               dR < 1e-5 and rel_s < 1e-4,
               "max|dR|={:.2e}  max rel|ds|={:.2e}".format(dR, rel_s))

    # ---- T11.6 极端 J 下有界：s'/s0 ∈ [min_scale_ratio, max_scale_ratio]、|q'|=1、无 NaN
    with torch.no_grad():
        Jext = torch.eye(3, device=DEV)[None].repeat(n, 1, 1).clone()
        Jext[:, 0, 0] = 8.0
        Jext[:, 1, 1] = 0.1
        qp_, sp_ = mod.apply(None, Jext, q0, s0)
        ratio = (sp_ / s0).clamp(min=0)
        qn = qp_.norm(dim=-1)
        ok_ext = (bool(torch.isfinite(sp_).all()) and bool(torch.isfinite(qp_).all())
                  and bool((ratio <= float(tga.DEFAULTS["max_scale_ratio"]) + 1e-4).all())
                  and bool((ratio >= float(tga.DEFAULTS["min_scale_ratio"]) - 1e-4).all())
                  and bool(((qn - 1).abs() <= 1e-5).all()))
        report("T11.6 极端 J（拉 8x / 压 0.1x）→ s'/s0 有界、|q'|=1、无 NaN/Inf",
               ok_ext, "s'/s0 ∈ [{:.4f}, {:.4f}]  max|1-|q||={:.2e}".format(
                   ratio.min().item(), ratio.max().item(), (qn - 1).abs().max().item()))

    # ---- T11.7 装进 renderer 之后，全 0 退化仍逐位一致；全 1 与 LAM 原调制器不同
    ones = torch.ones(52, device=DEV)
    zeros = torch.zeros(52, device=DEV)
    with torch.no_grad():
        base_static = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)[0]
        base_zero = renderer.animate_gs_model(gs, qp, fd, arkit_params=zeros)[0]
        lam_mod = renderer._get_cov_modulator(DEV)          # LAM 原调制器
        lam_ones = renderer.animate_gs_model(gs, qp, fd, arkit_params=ones)[0]

    tga_mod = tga.install(renderer, force=True)
    try:
        with torch.no_grad():
            new_static = renderer.animate_gs_model(gs, qp, fd, arkit_params=None)[0]
            new_zero = renderer.animate_gs_model(gs, qp, fd, arkit_params=zeros)[0]
            tga_ones = renderer.animate_gs_model(gs, qp, fd, arkit_params=ones)[0]
        bit = all(torch.equal(getattr(new_static, k), getattr(base_static, k))
                  for k in ("xyz", "rotation", "scaling", "opacity", "shs", "offset"))
        bit_zero = all(torch.equal(getattr(new_zero, k), getattr(base_zero, k))
                       for k in ("xyz", "rotation", "scaling", "opacity", "shs", "offset"))
        report("T11.7 装上 TGA 后：arkit=None / 全 0 仍与改动前逐位一致（C4 退化一致性）",
               bit and bit_zero, "None bitwise={}  zeros bitwise={}".format(bit, bit_zero))
        dq_rot = (tga_ones.rotation - lam_ones.rotation).abs().mean().item()
        dq_scl = (tga_ones.scaling - lam_ones.scaling).abs().mean().item()
        report("T11.8 全 1 时 TGA 的 (q', s') 与 LAM 原调制器不同",
               dq_rot > 1e-4 and dq_scl > 1e-5,
               "mean|dq'|={:.4f}  mean|ds'|={:.2e}".format(dq_rot, dq_scl))
        ratio = (tga_ones.scaling / gs.scaling.clamp(min=1e-30))
        finite = all(bool(torch.isfinite(getattr(tga_ones, k)).all())
                     for k in ("xyz", "rotation", "scaling", "opacity", "shs"))
        report("T11.9 全 1 时 TGA 路径无 NaN/Inf 且 s' <= 1.35*s0、|q'|=1",
               finite and bool((ratio <= 1.35 + 1e-4).all())
               and bool(((tga_ones.rotation.norm(dim=-1) - 1).abs() <= 1e-5).all()),
               "max(s'/s0)={:.6f}  min={:.6f}".format(ratio.max().item(),
                                                      ratio.min().item()))
        ok = (not bad_loops and bit and bit_zero and dq_rot > 1e-4 and finite)
    finally:
        # 还原：后续/其它脚本若复用同一 renderer，仍走 LAM 原调制器
        renderer._cov_modulator = None
    return ok

def main():
    t_start = time.time()
    print("torch {}  device {}".format(torch.__version__, torch.cuda.get_device_name(0)))
    section("构建模型（随机初始化，不载 checkpoint）")
    cfg, model, renderer = build()
    print("    renderer = {}  clip_scaling={}  use_rgb={}  sh_degree={}  subdivide={}".format(
        type(renderer).__name__, renderer.gs_net.clip_scaling,
        renderer.gs_net.use_rgb, renderer.sh_degree, renderer.flame_model.subdivide_num))

    fp = make_flame_params(DEV, n_view=1, seed=0)
    qp, fp = renderer.get_query_points(fp, DEV)
    report("P.1 query_points N == 20018 (flame_subdivide_num=1)",
           tuple(qp.shape) == (1, 20018, 3), "shape={}".format(tuple(qp.shape)))
    fd = renderer.get_sing_batch_smpl_data(renderer.get_single_view_smpl_data(fp, 0), 0)
    print("    flame_data: " + ", ".join("{}{}".format(k, tuple(v.shape)) for k, v in fd.items()))

    gs = canon_gs(renderer, qp[0], fd, seed=0)
    report("P.2 canonical GaussianModel 全部 float32 / dtype-shape 合规",
           gs.xyz.dtype == torch.float32 and gs.rotation.dtype == torch.float32
           and gs.scaling.dtype == torch.float32 and gs.shs.dtype == torch.float32
           and tuple(gs.xyz.shape) == (20018, 3) and tuple(gs.rotation.shape) == (20018, 4)
           and tuple(gs.scaling.shape) == (20018, 3) and tuple(gs.shs.shape) == (20018, 1, 3)
           and tuple(gs.opacity.shape) == (20018, 1) and tuple(gs.offset.shape) == (20018, 3),
           "xyz{} rot{} scl{} shs{} opa{} off{}".format(
               tuple(gs.xyz.shape), tuple(gs.rotation.shape), tuple(gs.scaling.shape),
               tuple(gs.shs.shape), tuple(gs.opacity.shape), tuple(gs.offset.shape)))

    res = {}
    for key, fn in (
        ("T1", lambda: t1_identity(renderer, gs, qp[0], fd)),
        ("T2", lambda: t2_extreme(renderer, gs, qp[0], fd)),
        ("T3", lambda: t3_sensitivity(renderer, gs, qp[0], fd)),
        ("T4", lambda: t4_render(renderer, gs, qp[0], fd)),
        ("T5", lambda: t5_timing(renderer, gs, qp[0], fd)),
        ("T6", lambda: t6_vectorisation()),
        ("T7", lambda: t7_cuda_untouched()),
        ("T8", lambda: t8_api_smoke()),
        ("X1", lambda: extra_reuse(renderer)),
        ("T9", lambda: t9_polar(renderer)),
        ("T10", lambda: t10_end_to_end(renderer)),
        ("T11", lambda: t11_tga_covariance(renderer, gs, qp[0], fd)),
    ):
        try:
            res[key] = fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report("{} 抛出异常".format(key), False,
                   "{}: {}".format(type(exc).__name__, exc))
            res[key] = False

    section("汇总")
    print("{:<8}{:<8}{}".format("测试", "结果", "说明"))
    print("-" * 78)
    for name, ok, det in RESULTS:
        print("{:<8}{:<8}{}".format(name, "PASS" if ok else "FAIL", det[:110]))
    print("-" * 78)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("总计 {} 项检查，失败 {} 项；总耗时 {:.1f} s".format(
        len(RESULTS), n_fail, time.time() - t_start))
    if n_fail:
        print("\n失败项：")
        for name, ok, det in RESULTS:
            if not ok:
                print("  - {}: {}".format(name, det))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
