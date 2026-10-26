#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A 口径 v2:对齐后的中性重构保真(2026-10-26)

v1 缺陷:渲染是头部特写,输入原图是半身照,PSNR/SSIM 被取景与背景差异主导(7.19 dB 无意义)。
v2 修复:
  1) 双图 haar 人脸检测 → 以人脸为中心、k×脸宽 的正方形裁剪(两张图脸占比一致)→ 512×512;
  2) 渲染 alpha>0.5 掩膜 → 只在头部区域算 PSNR(背景完全排除);
  3) LPIPS/SSIM 在对齐裁剪上算。
中性态五方法仍应逐位一致(顺带再校验一次)。

运行:
    cd /workspace/lam3d && nohup env PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/eval_a2_aligned.py \
      > /tmp/eval_a2.log 2>&1 &
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core import arkit_live, cov_switch  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

import torch  # noqa: E402
import lpips as lpips_lib  # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402
import cv2  # noqa: E402

METHODS = ["lam", "tga", "rot_sync", "rsb", "vertex_then_polar"]
IMAGE = "/workspace/LAM/assets/sample_input/messi.png"
SIZE = 512


def render_rgba(sess, arkit, yaw=0.0, size=SIZE):
    """与 arkit_live.render_frame 同路径,额外返回 alpha(0-255 uint8)。"""
    renderer = sess["renderer"]
    h0, w0 = sess["render_hw"]
    scale = size / float(w0)
    h, w = int(round(h0 * scale)), size
    with arkit_live._RENDER_LOCK:
        with torch.no_grad():
            gs_list = renderer.animate_gs_model(
                sess["gs_model"], sess["query_points"], sess["flame_data"],
                arkit_params=torch.from_numpy(np.asarray(arkit, dtype=np.float32)))
            c2w = arkit_live.orbit_c2w(sess["c2w"], sess["pivot"], yaw, 0.0, 1.0)
            intr = np.array(sess["intr"], dtype=np.float64, copy=True)
            intr[0, 0] *= scale; intr[1, 1] *= scale
            intr[0, 2] *= scale; intr[1, 2] *= scale
            device = gs_list[0].xyz.device
            out = renderer.forward_single_batch(
                [gs_list[0]],
                torch.from_numpy(c2w).float().to(device)[None],
                torch.from_numpy(intr).float().to(device)[None], h, w,
                sess["bg"].to(device)[None])
            rgb = out["comp_rgb"][0].clamp(0.0, 1.0)
            alpha = out["comp_mask"][0]
            if alpha.dim() == 3 and alpha.shape[-1] in (1, 3):
                alpha = alpha[..., 0]  # [H,W,1]/[H,W,3] → [H,W]
            elif alpha.dim() == 3 and alpha.shape[0] in (1, 3):
                alpha = alpha[0]
            return ((rgb * 255 + 0.5).to(torch.uint8).cpu().numpy(),
                    (alpha.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))


_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_box(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fs = _CAS.detectMultiScale(g, 1.1, 5, minSize=(max(24, min(g.shape) // 8),) * 2)
    if len(fs) == 0:
        return None
    return max(map(tuple, fs), key=lambda f: f[2] * f[3])


def aligned_crop(img, k=1.9):
    fb = face_box(img)
    H, W = img.shape[:2]
    if fb is None:
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        crop = img[y0:y0 + s, x0:x0 + s]
        fbinfo = None
    else:
        x, y, w, h = fb
        cx, cy = x + w / 2.0, y + h * 0.55
        s = int(round(k * max(w, h)))
        x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        crop = img[max(0, y0):min(H, y0 + s), max(0, x0):min(W, x0 + s)]
        pl, pt = max(0, -x0), max(0, -y0)
        pr, pb = max(0, x0 + s - W), max(0, y0 + s - H)
        if pl or pt or pr or pb:
            crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        fbinfo = {"box": [int(v) for v in fb], "src_hw": [H, W]}
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA), fbinfo


def psnr_masked(a, b, m):
    d = a[m].astype(np.float64) - b[m].astype(np.float64)
    mse = float((d * d).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def lpips_pair(lp, a, b, dev):
    with torch.no_grad():
        t1 = torch.from_numpy(a.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
        t2 = torch.from_numpy(b.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
        return float(lp(t1, t2))


def main():
    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=Path(IMAGE), job_dir=Path("/tmp/psl_job"), job_id="psl")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    print(f"session ready {time.time()-t0:.1f}s", flush=True)

    lp = lpips_lib.LPIPS(net="alex", verbose=False).to("cuda").eval()
    neutral = np.zeros(52, dtype=np.float32)

    photo = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)
    photo_c, photo_fb = aligned_crop(photo)
    print("photo face box:", photo_fb, flush=True)

    res = {"meta": {"size": SIZE, "k": 1.9, "lpips_net": "alex",
                    "note": "A_v2=人脸对齐裁剪(k=1.9x脸宽)+alpha掩膜PSNR(仅头部);中性态,五方法逐位一致"}}
    ref = None
    for m in METHODS:
        cov_switch.set_method(m, renderer=renderer)
        rgb, alpha = render_rgba(sess, neutral, yaw=0.0)
        rend_c, rend_fb = aligned_crop(rgb)
        mask = alpha > 127  # 渲染坐标即裁剪前坐标一致(裁剪作用于同一张 rgb)
        # mask 需与 rend_c 同尺寸:裁剪时 mask 也要走同一几何变换——简化:重建 crop 后的 mask
        # 做法:把 mask 当单通道图走同样的 aligned_crop 几何(face_box 检测在 rgb 上做,复用 rend_fb)
        # 这里直接在渲染图上检测(与 rgb 同步),对 mask 用相同裁剪:
        H, W = mask.shape
        if rend_fb is None:
            s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2
            mc = mask[y0:y0 + s, x0:x0 + s]
        else:
            x, y, w, h = rend_fb["box"]
            cx, cy = x + w / 2.0, y + h * 0.55
            s = int(round(1.9 * max(w, h)))
            x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
            mc = mask[max(0, y0):min(H, y0 + s), max(0, x0):min(W, x0 + s)]
            pl, pt = max(0, -x0), max(0, -y0)
            pr, pb = max(0, x0 + s - W), max(0, y0 + s - H)
            if pl or pt or pr or pb:
                mc = cv2.copyMakeBorder(mc.astype(np.uint8), pt, pb, pl, pr, cv2.BORDER_REPLICATE) > 0
        mc = cv2.resize(mc.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0

        if ref is None:
            ref = rgb
            bitwise = True
        else:
            bitwise = bool((rgb == ref).all())
        res[m] = {
            "bitwise_identical_neutral": bitwise,
            "psnr_head_masked": round(psnr_masked(photo_c, rend_c, mc), 3),
            "ssim_aligned": round(float(ssim_fn(photo_c, rend_c, channel_axis=2, data_range=255)), 4),
            "lpips_aligned": round(lpips_pair(lp, photo_c, rend_c, "cuda"), 4),
            "head_coverage": round(float(mc.mean()), 4),
            "render_face_box": rend_fb,
        }
        print(m, res[m], flush=True)
        # 保存对照图供目视检查
        if m == "lam":
            vis = np.concatenate([photo_c, rend_c], axis=1)
            cv2.imwrite("/tmp/a2_align_check.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    out = "/workspace/lam3d/experiments/eval_a2_aligned.json"
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
