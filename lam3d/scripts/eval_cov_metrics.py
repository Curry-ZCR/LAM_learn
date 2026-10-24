#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pixel-level metrics for the four covariance modulators (v3).

The first evaluation pass showed that PSNR/SSIM/LPIPS computed against a
mesh pseudo-GT image are dominated by a ~11.3 dB appearance/occlusion
mismatch and therefore CANNOT rank methods (spread < 0.02 dB).  This script
adds four pixel-level references that do discriminate:

  D  depth map       : 3DGS expected depth vs rasterised mesh depth.  The
                       mesh is deformed by the SAME ARKit offset field, so
                       it is method independent.  PSNR / SSIM on depth.
  N  normal map      : 3DGS thin-axis normal (z-buffered point splat) vs
                       interpolated mesh normal.  PSNR / SSIM / LPIPS on
                       the visualised normal map, plus per-pixel angular
                       error in degrees.
  C  changed region  : PSNR / SSIM / LPIPS of each method against the LAM
                       anchor, restricted to pixels that actually moved.
                       This measures the SIZE of the change, not quality.
  X  cross-method    : 4x4 LPIPS / SSIM matrix between methods, plus the
                       mean distance to the other three (consensus outlier).

Run (background, build ~70 s):
  cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_metrics.py \
    > /tmp/eval_cov3.log 2>&1 &
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
from skimage.metrics import structural_similarity  # noqa: E402

from lam3d.core import arkit_live, cov_switch, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/eval_cov3_job")
OUT_JSON = Path("/tmp/cov_pixel_metrics.json")
OUT_DIR = Path("/tmp/eval_cov3_frames")

METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
PRESETS = ["smile", "bigSmile", "laugh", "surprise", "angry", "sad", "kiss",
           "mouthOpen", "blink"]
VECTORS = ["zeros", "ones", "half", "rand"] + PRESETS
SIZE = 512
DEG = 180.0 / math.pi
OFFSET_ON_SURFACE = 0.010


def log(msg):
    print(msg, flush=True)


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
    for k, v in meta["presets"][name].items():
        vec[meta["names"].index(k)] = float(v)
    return vec


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-12)


def psnr(a, b, mask=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask is not None:
        if int(mask.sum()) < 50:
            return float("nan")
        a, b = a[mask], b[mask]
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))


def ssim(a, b, mask=None):
    ga = np.asarray(a, np.float64).mean(-1) if a.ndim == 3 else np.asarray(a, np.float64)
    gb = np.asarray(b, np.float64).mean(-1) if b.ndim == 3 else np.asarray(b, np.float64)
    if mask is not None:
        ys, xs = np.where(mask)
        if ys.size < 50:
            return float("nan")
        y0, y1 = max(int(ys.min()) - 8, 0), min(int(ys.max()) + 9, ga.shape[0])
        x0, x1 = max(int(xs.min()) - 8, 0), min(int(xs.max()) + 9, ga.shape[1])
        ga, gb = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
    if min(ga.shape) < 7:
        return float("nan")
    win = min(7, min(ga.shape) // 2 * 2 + 1)
    return float(structural_similarity(ga, gb, data_range=255.0, win_size=win))


class Lpips:
    def __init__(self, device="cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @staticmethod
    def _prep(x):
        x = np.asarray(x, np.float32)
        if x.ndim == 2:            # depth / single-channel maps: replicate to RGB
            x = np.repeat(x[..., None], 3, axis=-1)
        return torch.from_numpy(x / 127.5 - 1.0).permute(2, 0, 1)[None]

    def __call__(self, a, b):
        ta = self._prep(a)
        tb = self._prep(b)
        with torch.no_grad():
            return float(self.net(ta.to(self.device), tb.to(self.device)).item())


def pct(x, qs=()):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    out = {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
           "p90": float(np.percentile(x, 90)), "max": float(x.max())}
    for q in qs:
        out["p%d" % q] = float(np.percentile(x, q))
    return out


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


class MeshRaster:
    """nvdiffrast mesh rasteriser using the 3DGS camera convention."""

    def __init__(self, faces, h, w):
        self.F = torch.from_numpy(faces.astype(np.int32)).cuda()
        self.ctx = dr.RasterizeCudaContext()
        self.h, self.w = int(h), int(w)
        self.mvp = None
        self.w2c = None

    def set_view(self, c2w, intr, sign_y=1.0):
        self.w2c = np.linalg.inv(np.asarray(c2w, np.float64))
        self.mvp = proj_matrix(intr, self.h, self.w, sign_y) @ self.w2c

    def render(self, verts, colors, normals=None):
        vh = np.concatenate([verts, np.ones((verts.shape[0], 1))], 1)
        pos = torch.from_numpy((vh @ self.mvp.T).astype(np.float32))[None].cuda()
        rast, _ = dr.rasterize(self.ctx, pos, self.F, (self.h, self.w))
        zc = (vh @ self.w2c.T)[:, 2:3]
        attr = np.concatenate([colors.astype(np.float32), zc.astype(np.float32)], 1)
        out, _ = dr.interpolate(torch.from_numpy(attr)[None].cuda(), rast, self.F)
        mask = (rast[0, :, :, 3] > 0).cpu().numpy()
        vals = out[0].cpu().numpy()
        rgb = np.ones((self.h, self.w, 3), np.float32)
        rgb[mask] = np.clip(vals[mask][:, :3], 0.0, 1.0)
        z = vals[..., 3].astype(np.float32).copy()
        z[~mask] = np.nan
        nmap = None
        if normals is not None:
            attr2 = np.concatenate([normals.astype(np.float32), zc.astype(np.float32)], 1)
            out2, _ = dr.interpolate(torch.from_numpy(attr2)[None].cuda(), rast, self.F)
            v2 = out2[0].cpu().numpy()
            nmap = np.zeros((self.h, self.w, 3), np.float32)
            nv = v2[..., :3][mask]
            nmap[mask] = nv / np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-12)
        return (rgb * 255.0).astype(np.uint8), mask, z, nmap


def splat_normal_map(xyz, rot, scl, c2w, intr, h, w, rad=2):
    """Z-buffered point splat of the thin axis of every gaussian."""
    n = int(xyz.shape[0])
    thin = np.argmin(scl, -1)
    r = quat_to_matrix(torch.from_numpy(rot).float()).numpy()
    nrm = r[np.arange(n), :, thin]
    cam = xyz @ np.asarray(c2w, np.float64)[:3, :3].T + np.asarray(c2w, np.float64)[:3, 3]
    z = cam[:, 2]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + w * 0.5
    v = h * 0.5 - fy * cam[:, 1] / np.maximum(z, 1e-6)
    ok = (z > 0.02) & (u >= rad) & (u < w - rad) & (v >= rad) & (v < h - rad)
    idx = np.where(ok)[0]
    uu = u[idx].astype(np.int64)
    vv = v[idx].astype(np.int64)
    zz = z[idx]
    nn = nrm[idx].astype(np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)
    nmap = np.zeros((h, w, 3), np.float32)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            yy = vv + dy
            xx = uu + dx
            m = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            ys, xs, zs, ns = yy[m], xx[m], zz[m], nn[m]
            flat = ys * w + xs
            order = np.argsort(zs, kind="stable")
            flat_s, zs_s, ns_s = flat[order], zs[order], ns[order]
            first = np.ones(flat_s.shape[0], bool)
            first[1:] = flat_s[1:] != flat_s[:-1]
            flat_u, zs_u, ns_u = flat_s[first], zs_s[first], ns_s[first]
            cur = zbuf.reshape(-1)[flat_u]
            better = zs_u < cur
            sel = flat_u[better]
            zbuf.reshape(-1)[sel] = zs_u[better]
            nmap.reshape(-1, 3)[sel] = ns_u[better]
    cmask = np.isfinite(zbuf)
    zmask = zbuf.copy()
    zmask[~cmask] = np.nan
    return nmap, cmask, zmask


def nrm_rgb(nmap, mask):
    out = np.full(nmap.shape, 128, np.uint8)
    out[mask] = np.clip((nmap[mask] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return out


def depth_rgb(d, mask, lo, hi):
    out = np.full(d.shape, 255, np.uint8)
    dv = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    out[mask] = (dv[mask] * 255.0).astype(np.uint8)
    return out


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


def modulate(mod, renderer, fd, vec, x0, q0, s0):
    dev = "cuda"
    p = mod.prepare_params(torch.from_numpy(np.asarray(vec, np.float32)).to(dev),
                           1, dev, torch.float32)
    dyn = mod.arkit_offsets(p)
    with torch.no_grad():
        xb = renderer.flame_model.animation_forward(
            v_cano=x0, shape=fd["betas"].repeat(1, 1), expr=fd["expr"],
            rotation=fd["rotation"], neck=fd["neck_pose"], jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"], translation=fd["translation"],
            zero_centered_at_root_node=False, return_landmarks=False,
            return_verts_cano=False, static_offset=None,
            dynamic_offset=dyn)["animated"]
        dq, payload = mod.delta_rotation_scale(x0, xb)
        s2 = mod.delta_scale(payload, s0)
        qo, so = mod.apply(dq, s2, q0, s0)
    return (qo[0].detach().cpu().numpy().astype(np.float32),
            so[0].detach().cpu().numpy().astype(np.float32))


def main():
    # NOTE: never use argparse here.  Importing LAM pulls in a global config
    # parser that reads sys.argv at import time and aborts on unknown flags
    # ("unrecognized arguments").  Configure through environment variables.
    sel = os.environ.get("EVAL_VECTORS", "")
    vectors = [v.strip() for v in sel.split(",") if v.strip()] or VECTORS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="evalcov3")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    fd = sess["flame_data"]
    meta = arkit_live.arkit_meta()
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    n_vert = int(renderer.flame_model.v_template_up.shape[0])
    log("ready %.1fs  V=%d F=%d" % (time.time() - t0, n_vert, faces.shape[0]))

    base = tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    gs = sess["gs_model"]
    x0 = gs.xyz.detach().unsqueeze(0).float()
    q0 = gs.rotation.detach().float()
    s0 = gs.scaling.detach().float()
    x0_np = x0[0].cpu().numpy()

    # distance of every gaussian centre to the template mesh surface.
    # trimesh.proximity needs rtree, which is not installed, so use a KD-tree
    # over the subdivided template vertices.  The template is a dense Loop
    # subdivision of FLAME (20018 vertices), so nearest-vertex distance is a
    # good proxy for nearest-surface distance.
    from scipy.spatial import cKDTree
    tmpl_v = renderer.flame_model.v_template_up.detach().cpu().numpy()
    sd = cKDTree(tmpl_v).query(x0_np, k=1)[0]
    on_surf = sd < OFFSET_ON_SURFACE
    log("offset to surface: d p50=%.4f p90=%.4f max=%.4f ; on-surface %.1f%%"
        % (np.percentile(sd, 50), np.percentile(sd, 90),
           sd.max(), 100.0 * on_surf.mean()))

    lpips_net = Lpips()
    raster = MeshRaster(faces, SIZE, SIZE)
    report = {"vectors": vectors, "methods": METHODS,
              "on_surface_frac": float(on_surf.mean()), "pixel": {}}

    for vec_name in vectors:
        vec = vector_for(meta, vec_name)
        mv = mesh_vertices(renderer, fd, vec, base)
        vn = vertex_normals(mv, faces)
        frames = {}
        for name in METHODS:
            cov_switch.set_method(name, renderer=renderer)
            frames[name] = render_full(sess, vec, yaw=0.0)
        ref = frames["lam"]
        raster.set_view(ref["c2w"], ref["intr"], sign_y=1.0)
        _, mmask, mz, mnmap = raster.render(mv, np.full((n_vert, 3), 0.5, np.float32), vn)
        gsmask = ref["alpha"] > 0.5
        both = mmask & gsmask
        iou = float(both.sum()) / max(float((mmask | gsmask).sum()), 1.0)
        dlo = float(np.nanpercentile(mz[both], 1))
        dhi = float(np.nanpercentile(mz[both], 99))
        log("%-10s silhouette IoU=%.4f  depth range [%.4f, %.4f]  both=%d px"
            % (vec_name, iou, dlo, dhi, int(both.sum())));
        entry = {"iou": iou}
        for name in METHODS:
            fr = frames[name]
            gm_d = np.zeros((SIZE, SIZE), np.float32)
            gm_d[:] = np.nan
            gm_d[gsmask] = fr["depth"][gsmask]
            gs_nmap, cvg, _ = splat_normal_map(fr["xyz"], fr["rot"], fr["scl"],
                                               fr["c2w"], fr["intr"], SIZE, SIZE)
            pm = both & cvg
            ang = np.full((SIZE, SIZE), np.nan, np.float32)
            if pm.sum() > 50:
                c = np.abs((gs_nmap[pm] * mnmap[pm]).sum(-1))
                ang[pm] = np.arccos(np.clip(c, 0.0, 1.0)) * DEG
            gm_rgb = nrm_rgb(gs_nmap, cvg)
            gt_rgb = nrm_rgb(mnmap, mmask)
            dr = depth_rgb(gm_d, gsmask, dlo, dhi)
            dt = depth_rgb(mz, mmask, dlo, dhi)
            rgb = fr["rgb"]
            diff = (np.abs(rgb.astype(np.int16) - ref["rgb"].astype(np.int16)).sum(-1) > 0)
            entry[name] = {
                "normal_angle_px": pct(ang, [10, 20, 30]),
                "normal_frac_lt10": float((ang[pm] < 10).mean()) if pm.sum() > 50 else None,
                "normal_frac_lt20": float((ang[pm] < 20).mean()) if pm.sum() > 50 else None,
                "normal_psnr": psnr(gm_rgb, gt_rgb, pm),
                "normal_ssim": ssim(gm_rgb, gt_rgb, pm),
                "normal_lpips": lpips_net(gm_rgb, gt_rgb),
                "depth_psnr": psnr(dr, dt, both),
                "depth_ssim": ssim(dr, dt, both),
                "depth_lpips": lpips_net(dr, dt),
                "psnr_vs_lam": psnr(rgb, ref["rgb"]),
                "ssim_vs_lam": ssim(rgb, ref["rgb"]),
                "lpips_vs_lam": lpips_net(rgb, ref["rgb"]),
                "changed_pct": float(diff.mean() * 100.0),
                "psnr_vs_lam_changed": psnr(rgb, ref["rgb"], diff),
                "ssim_vs_lam_changed": ssim(rgb, ref["rgb"], diff),
            }
            log("  %-18s %-9s normal mean %5.2f p90 %5.2f f<10 %.3f | "
                "nPSNR %5.2f nSSIM %.4f nLPIPS %.4f | dPSNR %5.2f dSSIM %.4f | "
                "dPSNR#lam %5.2f chg %5.2f%%"
                % (name, vec_name,
                   entry[name]["normal_angle_px"].get("mean", float("nan")),
                   entry[name]["normal_angle_px"].get("p90", float("nan")),
                   entry[name]["normal_frac_lt10"] or 0.0,
                   entry[name]["normal_psnr"], entry[name]["normal_ssim"],
                   entry[name]["normal_lpips"], entry[name]["depth_psnr"],
                   entry[name]["depth_ssim"], entry[name]["psnr_vs_lam"],
                   entry[name]["changed_pct"]))

        # per-gaussian normal angle restricted to on-surface thin gaussians
        for name in METHODS:
            cov_switch.set_method(name, renderer=renderer)
            mod = renderer._cov_modulator
            qo, so = modulate(mod, renderer, fd, vec, x0, q0, s0)
            thin = np.argmin(so, -1)
            rr = quat_to_matrix(torch.from_numpy(qo)).numpy()
            ax = rr[np.arange(n_vert), :, thin]
            c = np.abs((ax * vn).sum(-1))
            a_all = np.arccos(np.clip(c, 0.0, 1.0)) * DEG
            entry[name]["gauss_normal_all"] = pct(a_all)
            entry[name]["gauss_normal_on_surface"] = pct(a_all[on_surf])
            entry[name]["gauss_normal_frac_lt10_on_surface"] = float(
                (a_all[on_surf] < 10).mean())
            from PIL import Image as PILImage
            if name == "rot_sync":
                PILImage.fromarray(gm_rgb).save(OUT_DIR / ("%s_gs_normal.png" % vec_name))
                PILImage.fromarray(gt_rgb).save(OUT_DIR / ("%s_mesh_normal.png" % vec_name))
                PILImage.fromarray(dr).save(OUT_DIR / ("%s_gs_depth.png" % vec_name))
                PILImage.fromarray(dt).save(OUT_DIR / ("%s_mesh_depth.png" % vec_name))

        report["pixel"][vec_name] = entry

    # cross-method LPIPS / SSIM matrix on the "surprise" expression
    vec = vector_for(meta, "surprise");
    imgs = {}
    for name in METHODS:
        cov_switch.set_method(name, renderer=renderer)
        imgs[name] = render_full(sess, vec, yaw=0.0)["rgb"]
    mat = {}
    for a in METHODS:
        mat[a] = {}
        for b in METHODS:
            mat[a][b] = ({"lpips": lpips_net(imgs[a], imgs[b]),
                          "ssim": ssim(imgs[a], imgs[b]),
                          "psnr": psnr(imgs[a], imgs[b])});
    report["cross_matrix"] = mat
    for a in METHODS:
        others = [mat[a][b]["lpips"] for b in METHODS if b != a]
        log("consensus LPIPS %-18s %.4f  (mean to other three)" % (a, float(np.mean(others))))

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("wrote %s (%.1f KB) in %.1f s"
        % (OUT_JSON, OUT_JSON.stat().st_size / 1024.0, time.time() - t0))


if __name__ == "__main__":
    main()
