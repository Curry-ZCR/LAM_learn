#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PSNR / SSIM / LPIPS 评测(2026-10-26)

口径(单图资产无逐帧照片 GT,如实报告):
  A) 中性重构保真:输入照片(=LAM 拟合目标,合法 GT)vs 中性态(arkit=0)正面渲染。
     同时校验:五方法中性帧应逐位一致(表情调制不污染静态)。
  B) 方法间偏离(参考=LAM 原版渲染):226 帧驱动序列抽帧 × yaw{0,35},
     量化各协方差方法相对基线的渲染差异(感知口径的改动可见度)。

运行:
    cd /workspace/lam3d && nohup env PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/eval_psnr_ssim_lpips.py \
      > /tmp/eval_psl.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, sys, time
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


def to_uint8(arr) -> np.ndarray:
    # render_frame 已返回 uint8 [H,W,3];防御性处理 float 输入
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1)
        a = (a * 255 + 0.5).astype(np.uint8)
    return a


def psnr(a, b) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def center_crop(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[top:top + s, left:left + s], (size, size), interpolation=cv2.INTER_AREA)


def lpips_pair(lp, a, b, dev):
    with torch.no_grad():
        t1 = torch.from_numpy(a.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
        t2 = torch.from_numpy(b.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
        return float(lp(t1, t2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="/workspace/lam3d/experiments/psnr_ssim_lpips.json")
    args = ap.parse_args()

    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=Path(IMAGE), job_dir=Path("/tmp/psl_job"), job_id="psl")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    print(f"session ready {time.time()-t0:.1f}s", flush=True)

    dev = "cuda"
    lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev).eval()
    size = args.size

    photo = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)
    photo = center_crop(photo, size)

    results = {"meta": {"frames": args.frames, "size": size, "lpips_net": "alex",
                        "input": IMAGE,
                        "note": "A=重构保真(GT=输入原图;渲染为头部特写取景,PSNR 受取景/背景差异主导,仅作参考);"
                                "B=方法间偏离(参考=LAM原版渲染,量化协方差方法的感知级改动量)"}}

    # ===== A) 中性重构 + 逐位一致校验 =====
    results["A_input_recon"] = {}
    ref = None
    for m in METHODS:
        cov_switch.set_method(m, renderer=renderer)
        img = to_uint8(arkit_live.render_frame(sess, np.zeros(52), yaw=0.0, size=size))
        if ref is None:
            ref = img
            results["A_neutral_bitwise_identical"] = {m: True}
        else:
            results["A_neutral_bitwise_identical"][m] = bool((img == ref).all())
        results["A_input_recon"][m] = {
            "psnr": round(psnr(photo, img), 3),
            "ssim": round(float(ssim_fn(photo, img, channel_axis=2, data_range=255)), 4),
            "lpips": round(lpips_pair(lp, photo, img, dev), 4),
        }
        print("A", m, results["A_input_recon"][m], "bitwise=", results["A_neutral_bitwise_identical"][m], flush=True)

    # ===== B) 驱动序列偏离 =====
    npz_dir = Path("/workspace/my_motion/flame_param")
    frames = sorted(npz_dir.glob("*.npz"))
    idx = np.linspace(0, len(frames) - 1, args.frames).astype(int)

    def drive(p):
        d = np.load(p)
        v = np.full(52, 0.8, dtype=np.float32)
        v[0] = float(np.clip(d["jaw_pose"].reshape(-1)[0], -1, 1))
        return v

    results["B_vs_lam_baseline"] = {}
    for yaw in (0.0, 35.0):
        cov_switch.set_method("lam", renderer=renderer)
        base = {i: to_uint8(arkit_live.render_frame(sess, drive(frames[i]), yaw=yaw, size=size)) for i in idx}
        for m in METHODS + ["_selftest"]:  # _selftest = 同方法重渲染,应逐位相等
            if m == "_selftest":
                results[f"B_selftest_yaw{int(yaw)}_bitwise_equal"] = True
                for i in idx:
                    img = to_uint8(arkit_live.render_frame(sess, drive(frames[i]), yaw=yaw, size=size))
                    if not (img == base[i]).all():
                        results[f"B_selftest_yaw{int(yaw)}_bitwise_equal"] = False
                continue
            cov_switch.set_method(m, renderer=renderer)
            acc = {"psnr": [], "ssim": [], "lpips": []}
            for i in idx:
                img = to_uint8(arkit_live.render_frame(sess, drive(frames[i]), yaw=yaw, size=size))
                acc["psnr"].append(psnr(base[i], img))
                acc["ssim"].append(float(ssim_fn(base[i], img, channel_axis=2, data_range=255)))
                acc["lpips"].append(lpips_pair(lp, base[i], img, dev))
            key = f"{m}|yaw{int(yaw)}"
            results["B_vs_lam_baseline"][key] = {
                "psnr_mean": round(float(np.mean(acc["psnr"])), 3),
                "ssim_mean": round(float(np.mean(acc["ssim"])), 5),
                "lpips_mean": round(float(np.mean(acc["lpips"])), 5),
                "lpips_p95": round(float(np.percentile(acc["lpips"], 95)), 5),
            }
            print("B", key, results["B_vs_lam_baseline"][key], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
