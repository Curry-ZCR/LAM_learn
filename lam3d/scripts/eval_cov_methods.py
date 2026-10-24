#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantitative comparison of the four covariance modulators.

Methods: lam (LAM original), tga (Jacobian cov), rot_sync (rotation only),
vertex_then_polar (average J to vertices first, then polar-decompose).

Because all four act on the SAME single-image reconstruction there is no
pixel ground truth, so three independent reference frames are reported:
  R1 geometry pseudo-GT : FLAME subdivided mesh deformed by the SAME ARKit
                          offset field, rasterised with vertex colours and
                          the exact same camera / projection as the 3DGS
                          renderer.  Independent of the covariance method.
  R2 change magnitude   : the same metric computed against the LAM baseline
                          (measures how far a method moves the image, NOT
                          whether it is better).
  R3 geometry agreement : per-gaussian thin axis vs deformed mesh normal
                          (arccos of the absolute cosine), anisotropy and
                          eigenvalue preservation.  Requires no rendering.

Run (must be backgrounded, build takes ~70 s):
  cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_methods.py \
    > /tmp/eval_cov.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

from lam3d.core import arkit_live, cov_switch, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/eval_cov_job")
OUT_JSON = Path("/tmp/cov_metrics.json")
OUT_DIR = Path("/tmp/eval_cov_frames")

METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
PRESETS = ["smile", "bigSmile", "laugh", "surprise", "angry", "sad", "kiss",
           "mouthOpen", "blink"]
ALL_VECTORS = ["zeros", "ones", "half", "rand"] + PRESETS
RENDER_VECTORS = ["ones", "surprise", "laugh", "bigSmile", "kiss"]
YAWS = [("front", 0.0), ("yaw35", 0.610865)]
SIZE = 512
DEG = 180.0 / math.pi

_LOG = []


def log(msg):
    print(msg, flush=True)
    _LOG.append(msg)


def vector_for(meta, name):
    n = int(meta["count"])
    if name == "zeros":
        return np.zeros(n, np.float32)
    if name == "ones":
        return np.ones(n, np.float32)
    if name == "half":
        return np.full(n, 0.5, np.float32)
    if name == "rand":
        return np.random.default_rng(20261022).uniform(0.0, 1.0, n).astype(np.float32)
    vec = np.zeros(n, np.float32)
    preset = meta["presets"][name]
    for k, v in preset.items():
        vec[meta["names"].index(k)] = float(v)
    return vec


# ------------------------------------------------------------------ geometry
def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    area = 0.5 * np.linalg.norm(fn, axis=-1)
    norm = np.linalg.norm(vn, axis=-1, keepdims=True)
    return vn / np.maximum(norm, 1e-12), area


def one_ring(faces, n_vert):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    e = np.unique(e, axis=0)
    order = np.argsort(e[:, 0], kind="stable")
    e = e[order]
    deg = np.bincount(e[:, 0], minlength=n_vert).astype(np.float32)
    return e[:, 0], e[:, 1], deg


def ring_jitter(q, src, dst, deg):
    """Mean angle (deg) between each vertex rotation and its 1-ring mean."""
    acc = np.zeros((q.shape[0], 4), np.float32)
    np.add.at(acc, src, q[dst])
    mean = acc / np.maximum(deg, 1.0)[:, None]
    dot = np.abs((q * mean).sum(-1))
    dot = np.clip(dot / np.maximum(np.linalg.norm(mean, axis=-1), 1e-12), 0.0, 1.0)
    return 2.0 * np.arccos(dot) * DEG


def eig_spectrum(q, s):
    r = quat_to_matrix(torch.from_numpy(q).float())
    d = torch.diag_embed(torch.from_numpy(s).float() ** 2)
    sig = r @ d @ r.transpose(-1, -2)
    lam = torch.linalg.eigvalsh(sig)
    return lam.numpy()


# ------------------------------------------------------------------ raster
def proj_matrix(intr, h, w, sign_y, znear=0.01, zfar=100.0):
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


class MeshRaster:
    """nvdiffrast mesh rasteriser reproducing the 3DGS camera convention."""

    def __init__(self, faces, h, w):
        self.F = torch.from_numpy(faces.astype(np.int32)).cuda()
        self.ctx = dr.RasterizeCudaContext()
        self.h, self.w = int(h), int(w)
        self.mvp = None
        self.w2c = None

    def set_view(self, c2w, intr, sign_y=-1.0):
        self.w2c = np.linalg.inv(np.asarray(c2w, np.float64))
        self.mvp = proj_matrix(intr, self.h, self.w, sign_y) @ self.w2c

    def render(self, verts, colors):
        vh = np.concatenate([verts, np.ones((verts.shape[0], 1))], 1)
        pos = torch.from_numpy((vh @ self.mvp.T).astype(np.float32))[None].cuda()
        rast, _ = dr.rasterize(self.ctx, pos, self.F, (self.h, self.w))
        attr = np.concatenate(
            [colors.astype(np.float32), (vh @ self.w2c.T)[:, 2:3].astype(np.float32)], 1)
        out, _ = dr.interpolate(
            torch.from_numpy(attr)[None].cuda(), rast, self.F)
        mask = (rast[0, :, :, 3] > 0).cpu().numpy()
        vals = out[0].cpu().numpy()
        rgb = np.ones((self.h, self.w, 3), np.float32)
        rgb[mask] = np.clip(vals[mask][:, :3], 0.0, 1.0)
        z = vals[..., 3].copy()
        z[~mask] = np.nan
        return (rgb * 255.0).astype(np.uint8), mask, z


# ------------------------------------------------------------------ metrics
def psnr(a, b, mask=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        if int(mask.sum()) < 50:
            return float("nan")
        a = a[mask]
        b = b[mask]
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))


def ssim(a, b, mask=None):
    ga = a.astype(np.float64).mean(-1)
    gb = b.astype(np.float64).mean(-1)
    if mask is not None:
        ys, xs = np.where(mask)
        if ys.size < 50:
            return float("nan")
        y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, ga.shape[0])
        x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, ga.shape[1])
        ga = ga[y0:y1, x0:x1]
        gb = gb[y0:y1, x0:x1]
    if min(ga.shape) < 7:
        return float("nan")
    win = min(7, min(ga.shape) // 2 * 2 + 1)
    return float(structural_similarity(ga, gb, data_range=255.0, win_size=win))


class Lpips:
    def __init__(self, device="cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    def __call__(self, a, b):
        ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)
        tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)
        ta = ta.permute(2, 0, 1)[None].to(self.device)
        tb = tb.permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            return float(self.net(ta, tb).item())


def pct(x, qs):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max()),
            **{"p%d" % q: float(np.percentile(x, q)) for q in qs}}


# ------------------------------------------------------------------ render
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
            device = gs_list[0].xyz.device
            out = renderer.forward_single_batch(
                [gs_list[0]],
                torch.from_numpy(c2w).float().to(device)[None],
                torch.from_numpy(intr).float().to(device)[None],
                h, w, sess["bg"].to(device)[None])
            rgb = (out["comp_rgb"][0].clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            rgb = rgb.cpu().numpy()
            alpha = out["comp_mask"][0].cpu().numpy()[..., 0]
            depth = out["comp_depth"][0].cpu().numpy()[..., 0]
            gs = gs_list[0]
            n = int(gs.xyz.shape[0])
            col = gs.shs
            if col is None:
                col = torch.zeros(n, 3, device=device)
            col = col.reshape(-1)
            if col.numel() != n * 3:
                col = col.reshape(n, -1)[:, :3]
            arr = dict(
                xyz=gs.xyz.detach().cpu().numpy(),
                rot=gs.rotation.detach().cpu().numpy(),
                scl=gs.scaling.detach().cpu().numpy(),
                col=col.reshape(n, 3).detach().cpu().numpy(),
            )
    return dict(rgb=rgb, alpha=alpha, depth=depth, c2w=c2w, intr=intr,
                hw=(h, w), **arr)


def mesh_vertices(renderer, fd, vec, lam, device="cuda"):
    v_cano = renderer.flame_model.v_template_up.unsqueeze(0).float()
    p = lam.prepare_params(torch.from_numpy(np.asarray(vec, np.float32)).to(device),
                           1, device, torch.float32)
    dyn = lam.arkit_offsets(p)
    with torch.no_grad():
        out = renderer.flame_model.animation_forward(
            v_cano=v_cano,
            shape=fd["betas"].repeat(1, 1),
            expr=fd["expr"],
            rotation=fd["rotation"],
            neck=fd["neck_pose"],
            jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"],
            translation=fd["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=False,
            static_offset=None,
            dynamic_offset=dyn)
    return out["animated"][0].detach().cpu().numpy()


def modulate(mod, renderer, fd, vec, x0, q0, s0):
    """Run the exact frozen call sequence from gs_renderer.animate_gs_model."""
    device = "cuda"
    p = mod.prepare_params(torch.from_numpy(np.asarray(vec, np.float32)).to(device),
                           1, device, torch.float32)
    dyn = mod.arkit_offsets(p)
    with torch.no_grad():
        xb = renderer.flame_model.animation_forward(
            v_cano=x0,
            shape=fd["betas"].repeat(1, 1),
            expr=fd["expr"],
            rotation=fd["rotation"],
            neck=fd["neck_pose"],
            jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"],
            translation=fd["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=False,
            static_offset=None,
            dynamic_offset=dyn)["animated"]
        dq, payload = mod.delta_rotation_scale(x0, xb)
        s2 = mod.delta_scale(payload, s0)
        qo, so = mod.apply(dq, s2, q0, s0)
    return (qo[0].detach().cpu().numpy().astype(np.float32),
            so[0].detach().cpu().numpy().astype(np.float32), xb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-lpips", action="store_true")
    args = ap.parse_args()
    vectors = [v.strip() for v in args.only.split(",") if v.strip()] or ALL_VECTORS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="evalcov")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    fd = sess["flame_data"]
    meta = arkit_live.arkit_meta()
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    n_vert = int(renderer.flame_model.v_template_up.shape[0])
    log("ready %.1fs  N_vert=%d  F=%d  arkit=%d"
        % (time.time() - t0, n_vert, faces.shape[0], meta["count"]))

    base = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    gs = sess["gs_model"]
    x0 = gs.xyz.detach().unsqueeze(0).float()
    q0 = gs.rotation.detach().float()
    s0 = gs.scaling.detach().float()
    q0_np = q0.cpu().numpy().astype(np.float32)
    s0_np = s0.cpu().numpy().astype(np.float32)
    src, dst, deg = one_ring(faces, n_vert)
    lam0 = eig_spectrum(q0_np, s0_np)

    # ---- R0: all-zero must stay bit-identical across methods
    zeros = vector_for(meta, "zeros")
    ref0 = {}
    for name in METHODS:
        cov_switch.set_method(name, renderer=renderer)
        mod = renderer._cov_modulator
        qo, so, _ = modulate(mod, renderer, fd, zeros, x0, q0, s0)
        ref0[name] = (qo, so)
    ident = all(np.array_equal(ref0["lam"][0], ref0[m][0]) for m in METHODS)
    ident_s = all(np.array_equal(ref0["lam"][1], ref0[m][1]) for m in METHODS)
    log("R0 neutral identity: q bitwise=%s  s bitwise=%s" % (ident, ident_s))
    log("R0 lam == q0 bitwise: %s ; s == s0 bitwise: %s"
        % (np.array_equal(ref0["lam"][0], q0_np),
           np.array_equal(ref0["lam"][1], s0_np)))

    lpips_net = None if args.skip_lpips else Lpips()
    report = {"vectors": vectors, "methods": METHODS,
              "neutral_identity": {"q": bool(ident), "s": bool(ident_s)},
              "geometry": {}, "pixel": {}, "timing": {}}

    # ---- efficiency: median wall time of a full modulated frame
    timing = {}
    for name in METHODS:
        cov_switch.set_method(name, renderer=renderer)
        for _ in range(4):
            render_full(sess, np.ones(meta["count"], np.float32))
        reps = []
        for _ in range(15):
            t1 = time.time()
            render_full(sess, np.ones(meta["count"], np.float32))
            reps.append((time.time() - t1) * 1000.0)
        timing[name] = {"median_ms": float(np.median(reps)),
                        "p90_ms": float(np.percentile(reps, 90))}
        log("timing %-18s median %.2f ms  p90 %.2f ms"
            % (name, timing[name]["median_ms"], timing[name]["p90_ms"]))
    # neutral frame cost (modulator bypassed) as the baseline
    for _ in range(4):
        render_full(sess, zeros)
    reps = []
    for _ in range(15):
        t1 = time.time()
        render_full(sess, zeros)
        reps.append((time.time() - t1) * 1000.0)
    timing["neutral(bypass)"] = {"median_ms": float(np.median(reps)),
                                 "p90_ms": float(np.percentile(reps, 90))}
    log("timing %-18s median %.2f ms" % ("neutral(bypass)", np.median(reps)))
    report["timing"] = timing

    raster = MeshRaster(faces, SIZE, SIZE)

    for vec_name in vectors:
        vec = vector_for(meta, vec_name)
        mv = mesh_vertices(renderer, fd, vec, base)
        vn, area = vertex_normals(mv, faces)
        views = [(n, y) for (n, y) in YAWS if vec_name in RENDER_VECTORS] or [YAWS[0]]

        frames = {}
        for name in METHODS:
            cov_switch.set_method(name, renderer=renderer)
            mod = renderer._cov_modulator
            qo, so, xb = modulate(mod, renderer, fd, vec, x0, q0, s0)

            # ---- R3 geometry agreement (rendering independent)
            thin = np.argmin(so, axis=-1)
            rot = quat_to_matrix(torch.from_numpy(qo)).numpy()
            axis = rot[np.arange(n_vert), :, thin]
            cos = np.abs((axis * vn).sum(-1))
            ang = np.arccos(np.clip(cos, 0.0, 1.0)) * DEG
            aniso = so.max(-1) / np.maximum(so.min(-1), 1e-12)
            lam1 = eig_spectrum(qo, so)
            rel = np.abs(lam1 - lam0) / np.maximum(lam0.max(-1, keepdims=True), 1e-20)
            jit = ring_jitter(qo, src, dst, deg)
            report["geometry"].setdefault(vec_name, {})[name] = {
                "normal_angle_deg": pct(ang, [10, 20, 30]),
                "frac_lt5": float((ang < 5).mean()),
                "frac_lt10": float((ang < 10).mean()),
                "frac_lt20": float((ang < 20).mean()),
                "aniso": pct(aniso, [25, 75]),
                "aniso_p99_over_p50": float(
                    np.percentile(aniso, 99) / max(np.percentile(aniso, 50), 1e-9)),
                "eig_rel_err_max": float(rel.max()),
                "eig_rel_err_p99": float(np.percentile(rel, 99)),
                "ring_jitter_deg": pct(jit, [50, 90]),
            }
            log("  %-18s %-10s normal mean %6.2f p90 %6.2f frac<10 %.3f | "
                "aniso p50 %6.2f | eig err p99 %.2e | jitter mean %.3f"
                % (name, vec_name,
                   report["geometry"][vec_name][name]["normal_angle_deg"]["mean"],
                   report["geometry"][vec_name][name]["normal_angle_deg"]["p90"],
                   report["geometry"][vec_name][name]["frac_lt10"],
                   report["geometry"][vec_name][name]["aniso"]["p50"],
                   report["geometry"][vec_name][name]["eig_rel_err_p99"],
                   report["geometry"][vec_name][name]["ring_jitter_deg"]["mean"]))

            for view_name, yaw in views:
                fr = render_full(sess, vec, yaw=yaw)
                frames[(name, view_name)] = fr

        # ---- R1/R2 pixel metrics
        for view_name, yaw in views:
            raster.set_view(frames[(METHODS[0], view_name)]["c2w"],
                            frames[(METHODS[0], view_name)]["intr"])
            colors = np.clip(frames[(METHODS[0], view_name)]["col"], 0.0, 1.0)
            best = None
            for sign_y in (-1.0, 1.0):
                raster.set_view(frames[(METHODS[0], view_name)]["c2w"],
                                frames[(METHODS[0], view_name)]["intr"], sign_y)
                img, mask, z = raster.render(mv, colors)
                gm = frames[(METHODS[0], view_name)]["alpha"] > 0.5
                iou = float((mask & gm).sum()) / max(float((mask | gm).sum()), 1.0)
                if best is None or iou > best[0]:
                    best = (iou, sign_y, img, mask, z)
            iou, sign_y, gt_img, gt_mask, gt_z = best
            if vec_name == vectors[0] and view_name == "front":
                log("R1 mesh raster: sign_y=%.0f silhouette IoU=%.4f mask=%.3f"
                    % (sign_y, iou, gt_mask.mean()))
                raster.set_view(frames[(METHODS[0], view_name)]["c2w"],
                                frames[(METHODS[0], view_name)]["intr"], sign_y)
                gt_img, gt_mask, gt_z = raster.render(mv, colors)
            report["pixel"].setdefault(vec_name, {}).setdefault(view_name, {})
            report["pixel"][vec_name][view_name]["mesh_iou"] = iou
            if view_name == "front":
                from PIL import Image as PILImage
                PILImage.fromarray(gt_img).save(OUT_DIR / ("%s_gt.png" % vec_name))
            lam_rgb = frames[("lam", view_name)]["rgb"]
            for name in METHODS:
                rgb = frames[(name, view_name)]["rgb"]
                entry = {
                    "psnr_vs_mesh": psnr(rgb, gt_img),
                    "ssim_vs_mesh": ssim(rgb, gt_img),
                    "psnr_vs_mesh_mask": psnr(rgb, gt_img, gt_mask),
                    "ssim_vs_mesh_mask": ssim(rgb, gt_img, gt_mask),
                    "psnr_vs_lam": psnr(rgb, lam_rgb),
                    "ssim_vs_lam": ssim(rgb, lam_rgb),
                    "lpips_vs_mesh": (None if lpips_net is None
                                      else lpips_net(rgb, gt_img)),
                    "lpips_vs_lam": (None if lpips_net is None
                                     else lpips_net(rgb, lam_rgb)),
                    "changed_pct": float(
                        (np.abs(rgb.astype(np.int16) - lam_rgb.astype(np.int16))
                         .sum(-1) > 0).mean() * 100.0),
                }
                report["pixel"][vec_name][view_name][name] = entry
                log("  pixel %-18s %-10s %-6s PSNR#gt %6.2f SSIM#gt %.4f "
                    "LPIPS#gt %.4f | PSNR#lam %6.2f changed %5.2f%%"
                    % (name, vec_name, view_name, entry["psnr_vs_mesh"],
                       entry["ssim_vs_mesh"],
                       -1.0 if entry["lpips_vs_mesh"] is None
                       else entry["lpips_vs_mesh"],
                       entry["psnr_vs_lam"], entry["changed_pct"]))
                if view_name == "front":
                    from PIL import Image as PILImage
                    PILImage.fromarray(rgb).save(
                        OUT_DIR / ("%s_%s.png" % (vec_name, name)))

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("wrote %s (%.1f KB)" % (OUT_JSON, OUT_JSON.stat().st_size / 1024.0))
    log("frames in %s" % OUT_DIR)
    log("total %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
