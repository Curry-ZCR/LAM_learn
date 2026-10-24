#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dynamic (temporal) metrics for the four covariance modulators.

Static frames cannot separate the methods: on a single expression the four
images differ by less than 0.005 LPIPS.  What actually separates them is how
the deformation behaves ALONG a motion.  This script drives a smooth ramp of
one ARKit coefficient (0 -> 1 in N steps) and measures:

  T1 rotation roughness : per gaussian, mean |a(t+1)-a(t)| / mean a(t) where
                          a(t) = angle(q_t, q_{t-1}).  A method that only
                          rotates produces a smooth rotation curve; one that
                          adds shear/stretch can produce stepwise jumps.
  T2 pixel flicker      : mean |I(t+1) - 2 I(t) + I(t-1)| over the pixels
                          that move.  Second order temporal difference of
                          the rendered image.
  T3 monotonicity       : Spearman correlation between the driving
                          coefficient and the mean absolute displacement of
                          the gaussians (or of the image).
  T4 depth definition   : affine-fit of the 3DGS depth against the mesh
                          depth; a high R^2 with a non-unit slope proves the
                          two depth buffers use different normalisation.

Run (background, build ~70 s):
  cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_dynamics.py \
    > /tmp/eval_cov4.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from lam3d.core import arkit_live, cov_switch  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/eval_cov4_job")
OUT_JSON = Path("/tmp/cov_dynamics.json")
METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
STEPS = 24
SIZE = 384
DEG = 180.0 / math.pi
RAMPS = ["jawOpen"
         , "mouthSmileLeft"
         , "browInnerUp"
         , "eyeBlinkLeft"
        ]


def log(msg):
    print(msg, flush=True)


def quat_angle_deg(a, b):
    """Per-row unsigned angle between unit quaternions, degrees."""
    d = np.abs((a * b).sum(-1))
    return 2.0 * np.degrees(np.arccos(np.clip(d, 0.0, 1.0)))


def render_full(sess, vec, yaw=0.0, size=SIZE):
    renderer = sess["renderer"]
    h0, w0 = sess["render_hw"]
    scale = size / float(w0)
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    with arkit_live._RENDER_LOCK:
        with torch.no_grad():
            gs_list = renderer.animate_gs_model(
                sess["gs_model"], sess["query_points"], sess["flame_data"],
                arkit_params=torch.from_numpy(np.asarray(vec, np.float32)),
            )
            c2w = arkit_live.orbit_c2w(sess["c2w"], sess["pivot"], yaw, 0.0, 1.0)
            intr = np.array(sess["intr"], np.float64, copy=True)
            intr[0, 0] *= scale
            intr[1, 1] *= scale
            intr[0, 2] *= scale
            intr[1, 2] *= scale
            dev = gs_list[0].xyz.device
            out = renderer.forward_single_batch(
                [gs_list[0]],
                torch.from_numpy(c2w).float().to(dev)[None],
                torch.from_numpy(intr).float().to(dev)[None],
                h, w, sess["bg"].to(dev)[None])
            rgb = (out["comp_rgb"][0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            alpha = out["comp_mask"][0].cpu().numpy()[..., 0]
            depth = out["comp_depth"][0].cpu().numpy()[..., 0]
            gs = gs_list[0]
            xyz = gs.xyz.detach().cpu().numpy()
            rot = gs.rotation.detach().cpu().numpy()
            scl = gs.scaling.detach().cpu().numpy()
    return dict(rgb=rgb, alpha=alpha, depth=depth, c2w=c2w, intr=intr,
                hw=(h, w), xyz=xyz, rot=rot, scl=scl)


def proj_matrix(intr, h, w, sign_y=1.0, znear=0.01, zfar=100.0):
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    fovx = 2.0 * math.atan2(w, 2.0 * fx)
    fovy = 2.0 * math.atan2(h, 2.0 * fy)
    top = math.tan(fovy / 2) * znear
    right = math.tan(fovx / 2) * znear
    p = np.zeros((4, 4), np.float64)
    p[0, 0] = znear / right
    p[1, 1] = sign_y * znear / top
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def mesh_depth(faces, verts, c2w, intr, h, w):
    ctx = dr.RasterizeCudaContext()
    Ft = torch.from_numpy(faces.astype(np.int32)).cuda()
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    mvp = proj_matrix(intr, h, w, 1.0) @ w2c
    vh = np.concatenate([verts, np.ones((verts.shape[0], 1))], 1)
    pos = torch.from_numpy((vh @ mvp.T).astype(np.float32))[None].cuda()
    rast, _ = dr.rasterize(ctx, pos, Ft, (h, w))
    attr = (vh @ w2c.T)[:, 2:3].astype(np.float32)
    out, _ = dr.interpolate(torch.from_numpy(attr)[None].cuda(), rast, Ft)
    mask = (rast[0, :, :, 3] > 0).cpu().numpy()
    z = out[0, :, :, 0].cpu().numpy().astype(np.float32)
    z[~mask] = np.nan
    return z, mask


def mesh_vertices(renderer, fd, vec, lam, device="cuda"):
    v_cano = renderer.flame_model.v_template_up.unsqueeze(0).float()
    p = lam.prepare_params(torch.from_numpy(np.asarray(vec, np.float32)).to(device),
                           1, device, torch.float32)
    dyn = lam.arkit_offsets(p)
    with torch.no_grad():
        out = renderer.flame_model.animation_forward(
            v_cano=v_cano, shape=fd["betas"].repeat(1, 1), expr=fd["expr"],
            rotation=fd["rotation"], neck=fd["neck_pose"], jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"], translation=fd["translation"],
            zero_centered_at_root_node=False, return_landmarks=False,
            return_verts_cano=False, static_offset=None, dynamic_offset=dyn)
    return out["animated"][0].detach().cpu().numpy()


def main():
    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="evalcov4")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    fd = sess["flame_data"]
    meta = arkit_live.arkit_meta()
    names = meta["names"]
    n = int(meta["count"])
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    log("ready %.1fs  N=%d" % (time.time() - t0, n))

    from lam3d.core import tga_covariance
    lam = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    lam.to("cuda").eval()

    report = {"ramps": {}, "methods": METHODS, "steps": STEPS, "size": SIZE}

    # ---------------- T4 depth definition check (one frame)
    vec_ones = np.ones(n, np.float32)
    mv = mesh_vertices(renderer, fd, vec_ones, lam)
    cov_switch.set_method("rot_sync", renderer=renderer)
    fr = render_full(sess, vec_ones)
    mz, mmask = mesh_depth(faces, mv, fr["c2w"], fr["intr"], *fr["hw"])
    gm = fr["alpha"] > 0.5
    both = mmask & gm & np.isfinite(mz) & np.isfinite(fr["depth"])
    a = fr["depth"][both].astype(np.float64)
    b = mz[both].astype(np.float64)
    A = np.stack([b, np.ones_like(b)], 1)
    sol, *_ = np.linalg.lstsq(A, a, rcond=None)
    pred = A @ sol
    r2 = 1.0 - float(((a - pred) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12))
    report["depth_fit"] = {"slope": float(sol[0]), "offset": float(sol[1]),
                           "r2": r2, "n_px": int(both.sum()),
                           "gs_depth_p50": float(np.nanmedian(a)),
                           "mesh_depth_p50": float(np.nanmedian(b)),
                           "residual_p50": float(np.median(np.abs(a - b)))}
    log("T4 depth affine fit: gs = %.4f * mesh + %.4f   R2=%.4f  n=%d"
        % (sol[0], sol[1], r2, both.sum()))
    log("   raw medians gs=%.4f mesh=%.4f  median |diff|=%.4f"
        % (np.nanmedian(a), np.nanmedian(b), np.median(np.abs(a - b))))

    # ---------------- temporal ramps
    for ramp_name in RAMPS:
        if ramp_name not in names:
            log("skip %s (not an ARKit channel)" % ramp_name)
            continue
        report["ramps"][ramp_name] = {}
        log("== ramp %s" % ramp_name)
        for method in METHODS:
            cov_switch.set_method(method, renderer=renderer)
            qs, xs, imgs, dofs = [], [], [], []
            for k in range(STEPS + 1):
                a = k / float(STEPS)
                vec = np.zeros(n, np.float32)
                vec[names.index(ramp_name)] = a
                fr = render_full(sess, vec)
                qs.append(fr["rot"].astype(np.float64))
                xs.append(fr["xyz"].astype(np.float64))
                imgs.append(fr["rgb"].astype(np.float32))
                dofs.append(float(np.abs(xs[-1] - xs[0]).mean()))
            qs = np.stack(qs)
            xs = np.stack(xs)
            imgs = np.stack(imgs)
            step_ang = np.stack([quat_angle_deg(qs[k], qs[k - 1])
                                 for k in range(1, len(qs))])
            d2 = np.abs(np.diff(step_ang, axis=0))
            mean_step = float(step_ang.mean())
            rough = float(d2.mean() / max(mean_step, 1e-9))
            rough_max = float(d2.mean(1).max() / max(mean_step, 1e-9))
            base = imgs[0]
            moved = (np.abs(imgs[-1] - base).sum(-1) > 3)
            if moved.sum() < 100:
                moved = np.ones_like(moved)
            i2 = np.abs(imgs[2:] - 2.0 * imgs[1:-1] + imgs[:-2]).mean(-1)
            flick = float(i2[:, moved].mean())
            disp = np.array([np.abs(im - base).mean() for im in imgs])
            rho = float(spearmanr(np.linspace(0, 1, len(disp)), disp).statistic)
            report["ramps"][ramp_name][method] = {
                "rot_step_mean_deg": mean_step,
                "rot_roughness": rough,
                "rot_roughness_worst_step": rough_max,
                "rot_d2_mean_deg": float(d2.mean()),
                "pixel_flicker": flick,
                "pixel_moved_frac": float(moved.mean()),
                "pixel_disp_final": float(disp[-1]),
                "dispatch_monotonic_spearman": rho,
                "gauss_disp_final": float(dofs[-1]),
            }
            log("  %-18s rot step %6.3f deg  roughness %.4f  worst %.4f  "
                "flicker %.4f  moved %.1f%%  rho %.5f"
                % (method, mean_step, rough, rough_max, flick,
                   100.0 * moved.mean(), rho))

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("wrote %s in %.1f s" % (OUT_JSON, time.time() - t0))


if __name__ == "__main__":
    main()
