"""Diagnose WHY external / precomputed 3D covariance never shows up in LAM
official render path, and measure how strong the existing ARKit modulation is.

Instrumented reproduction of GaussianRenderer.forward_single_view on the REAL
checkpoint session. No mocks.

  E1  old modulator: how large are |Delta q| (deg), ds/s0, ||Sigma-Sigma||/||Sigma||?
  E2  (a) express a desired Sigma through (scales, rotations) -> image changes;
      (b) attach the same Sigma as cov3D_precomp on the GaussianModel -> the
      official wrapper IGNORES it (line 534 hardcodes None), image bit-identical.
  E3  call the rasterizer directly with cov3D_precomp -> matches (a).
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402

from lam.models.rendering.gaussian_model import GaussianModel  # noqa: E402
from lam.models.rendering.gs_renderer import Camera  # noqa: E402
from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/diag_cov_job")


def quat_to_mat(q):
    """wxyz -> 3x3 matrix, same convention as computeCov3D.

    computeCov3D stores R with columns (1-2(y^2+z^2), 2(xy+wz), 2(xz-wy)),
    i.e. its R = R_std^T and Sigma = R^T S^2 R = R_std S^2 R_std^T.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def sigma_from_sr(scaling, rotation):
    R = quat_to_mat(rotation.float())
    S2 = scaling.float() ** 2
    return R @ torch.diag_embed(S2) @ R.transpose(-1, -2)


def rel_frob(A, B):
    return (A - B).norm(dim=(-1, -2)) / B.norm(dim=(-1, -2)).clamp(min=1e-20)


def quat_angle_deg(q0, q1):
    """Angle of the relative rotation between two wxyz quaternions, degrees."""
    d = (q0 * q1).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


def render_raw(renderer, gs, c2w, intr, h, w, bg, cov3d=None):
    """Reimplementation of forward_single_view with a working cov3D_precomp."""
    cam = Camera.from_c2w(c2w, intr, h, w)
    settings = GaussianRasterizationSettings(
        image_height=int(cam.height),
        image_width=int(cam.width),
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg,
        scale_modifier=renderer.scaling_modifier,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform.float(),
        sh_degree=renderer.sh_degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    rast = GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(gs.xyz, device=gs.xyz.device)
    use_rgb = renderer.gs_net.use_rgb
    kw = dict(
        means3D=gs.xyz.float(),
        means2D=means2D.float(),
        shs=None if use_rgb else gs.shs.float(),
        colors_precomp=gs.shs.squeeze(1).float() if use_rgb else None,
        opacities=gs.opacity.float(),
    )
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        if cov3d is None:
            out = rast(scales=gs.scaling.float(), rotations=gs.rotation.float(),
                       cov3D_precomp=None, **kw)
        else:
            out = rast(scales=None, rotations=None, cov3D_precomp=cov3d.float(), **kw)
    return out[0].permute(1, 2, 0)


def sigma_to_upper(sigma):
    return torch.stack([
        sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2],
        sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2],
    ], dim=-1).contiguous()


def main():
    print("== building pipeline (real checkpoint) ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="diagcov")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    gs0 = sess["gs_model"]
    flame = sess["flame_data"]
    N = int(gs0.xyz.shape[0])
    print("session: N=%d render_hw=%s use_rgb=%s scale_mod=%s"
          % (N, sess["render_hw"], renderer.gs_net.use_rgb,
             renderer.scaling_modifier), flush=True)

    arkit = torch.ones(52, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        t0 = time.time()
        gs_mod = renderer.animate_gs_model(gs0, sess["query_points"], flame,
                                          arkit_params=arkit)[0]
        torch.cuda.synchronize()
        print("", flush=True)
        print("[E1] animate_gs_model(all-ones): %.1f ms" % ((time.time() - t0) * 1e3))

    ang = quat_angle_deg(gs0.rotation.float(), gs_mod.rotation.float())
    ds_rel = (gs_mod.scaling.float() - gs0.scaling.float()).abs() / \
        gs0.scaling.float().clamp(min=1e-12)
    S0 = sigma_from_sr(gs0.scaling, gs0.rotation)
    S1 = sigma_from_sr(gs_mod.scaling, gs_mod.rotation)
    rel = rel_frob(S1, S0)
    ev0 = torch.linalg.eigvalsh(S0.double()).flip(-1).clamp(min=1e-30)
    ev1 = torch.linalg.eigvalsh(S1.double()).flip(-1).clamp(min=1e-30)
    aniso0 = (ev0[..., 0] / ev0[..., -1]).sqrt()
    aniso1 = (ev1[..., 0] / ev1[..., -1]).sqrt()
    shift = (gs_mod.xyz.float() - gs0.xyz.float()).norm(dim=-1)
    print("  |xyz| delta      : mean %.3e  max %.3e (model units)"
          % (shift.mean().item(), shift.max().item()))
    print("  |Delta q| angle  : mean %.4f deg  p95 %.4f deg  max %.4f deg"
          % (ang.mean().item(), ang.quantile(0.95).item(), ang.max().item()))
    print("  |ds|/s0          : mean %.4f  max %.4f  (bound = 0.35)"
          % (ds_rel.mean().item(), ds_rel.max().item()))
    print("  ||Sigma2-Sigma1||/||Sigma1|| : mean %.4f  p95 %.4f  max %.4f"
          % (rel.mean().item(), rel.quantile(0.95).item(), rel.max().item()))
    print("  Sigma anisotropy sqrt(l1/l3): canon %.3f -> mod %.3f (max %.3f)"
          % (aniso0.mean().item(), aniso1.mean().item(), aniso1.max().item()))

    dev = gs0.xyz.device
    c2w = torch.from_numpy(np.asarray(sess["c2w"], dtype=np.float32))[None].to(dev)
    intr = torch.from_numpy(np.asarray(sess["intr"], dtype=np.float32))[None].to(dev)
    H, W = int(sess["render_hw"][0]), int(sess["render_hw"][1])
    bgv = sess["bg"]
    bgv = bgv[None] if bgv.dim() == 1 else bgv
    bgv = bgv.to(dev)
    bg0 = bgv[0]
    with torch.no_grad():
        base = render_raw(renderer, gs0, c2w[0], intr[0], H, W, bg0)
    print("", flush=True)
    print("[E2] base render %s" % (tuple(base.shape),))

    # ---- clean A/B/C: SAME xyz everywhere, so the only variable is how the
    # covariance reaches the rasterizer.
    with torch.no_grad():
        A = render_raw(renderer, gs_mod, c2w[0], intr[0], H, W, bg0)   # (s,R)
    print("", flush=True)
    print("[E2] identical xyz (= modulated xyz); modulated (s,R) render = reference")
    print("      (A) (s,R) path            : %s" % (tuple(A.shape),))

    # (B) rebuild Sigma in python with the rasterizer formula, feed cov3D_precomp
    # through the SAME raw path: must be bit-identical to (A).
    S_mod = sigma_from_sr(gs_mod.scaling, gs_mod.rotation)
    with torch.no_grad():
        B = render_raw(renderer, gs_mod, c2w[0], intr[0], H, W, bg0,
                       cov3d=sigma_to_upper(S_mod))
    print("      (B) python Sigma -> cov3D_precomp via raw rasterizer:")
    print("          vs (A): mean %.3e  max %.3e  bit-identical: %s"
          % ((B - A).abs().mean().item(), (B - A).abs().max().item(),
             bool(torch.equal(B, A))))

    # (C) attach a DELIBERATELY WRONG covariance (4x) as an attribute on the
    # GaussianModel and go through the OFFICIAL wrapper. If the attribute were
    # honoured the image would explode; if line 534 drops it the image is
    # bit-identical to (A).
    gs_cov = GaussianModel(xyz=gs_mod.xyz.clone(), opacity=gs_mod.opacity,
                           rotation=gs_mod.rotation.clone(),
                           scaling=gs_mod.scaling.clone(),
                           shs=gs_mod.shs, offset=gs_mod.offset)
    gs_cov.cov3D_precomp = sigma_to_upper(S_mod * 4.0)
    with torch.no_grad():
        C = renderer.forward_single_batch(
            [gs_cov], c2w, intr, H, W, bgv)["comp_rgb"][0]
    same = bool(torch.equal(C, A))
    print("      (C) cov3D_precomp=4*Sigma as attribute, OFFICIAL wrapper:")
    print("          vs (A): mean %.3e  max %.3e  bit-identical: %s"
          % ((C - A).abs().mean().item(), (C - A).abs().max().item(), same))
    print("          attribute honoured? %s" % ("NO (dropped)" if same else "yes"))

    # negative control: the very same 4x covariance delivered the working way
    with torch.no_grad():
        C2 = render_raw(renderer, gs_mod, c2w[0], intr[0], H, W, bg0,
                        cov3d=sigma_to_upper(S_mod * 4.0))
    print("      (C') same 4*Sigma through raw rasterizer (control):")
    print("          vs (A): mean %.3f  max %.3f  bit-identical: %s"
          % ((C2 - A).abs().mean().item(), (C2 - A).abs().max().item(),
             bool(torch.equal(C2, A))))
    print("      => path is the problem, not the covariance math: the CUDA hook")
    print("         works (B/C'), the official wrapper discards it (C).")

    with torch.no_grad():
        gs_neu = renderer.animate_gs_model(gs0, sess["query_points"], flame,
                                          arkit_params=None)[0]
    print("", flush=True)
    print("[neutral] arkit=None reproduces canonical: xyz %s rot %s scale %s"
          % (torch.equal(gs_neu.xyz, gs0.xyz),
             torch.equal(gs_neu.rotation, gs0.rotation),
             torch.equal(gs_neu.scaling, gs0.scaling)))
    print("", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
