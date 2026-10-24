#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discriminating offline metrics for the four covariance methods.

The online pixel metrics (PSNR / SSIM / LPIPS against a mesh pseudo-GT)
turned out to have no resolving power: the mesh and the gaussian shell differ
far more from each other than the four methods differ from each other.  This
script measures properties that do not need a photograph as ground truth.

Reference frame.  The first-order, exactly-correct answer for a surface-bound
gaussian is the linearised deformation of its neighbourhood,

    Sigma_exact = J_v Sigma_0 J_v^T

(SurFhead, arXiv:2410.11682, builds Sigma^{1/2} = J R_c S_c).  Every method is
compared against that.  This is fidelity to the linearised deformation, not
photometric quality -- a method can be exactly faithful and still flicker.

Metrics, per method and per ARKit vector:

  V1  eig_rel_err     max_i |lambda_i(Sigma) / lambda_i(Sigma_0) - 1|
                      "does the method preserve axis lengths?"  Method one
                      is exact here by construction.
  V2  shape_err       ||Sigma - Sigma_exact||_F / ||Sigma_exact||_F
                      "how much of the surface deformation does it keep?"
  V3  rigid_err       angle between the rotation actually applied and the
                      rotation of the one-ring vertex displacement field
                      (Kabsch fit).  Independent of the face-Jacobian code.
  V4  spurious        fraction of gaussians turned by > 20 deg while the
                      local surface barely moved (< 5 deg): false motion.
  V5  stretch_avail   ||S_v - I||_F and cond(S_v) for the polar factor of
                      J_v: an upper bound on the shape method one discards.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402

from lam3d.core import cov_switch, rot_sync  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402

CAPTURE = "/tmp/rot_capture.pt"
OUT = "/tmp/cov_offline.json"
IMAGE = "/workspace/LAM/assets/sample_input/messi.png"
METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
DEV = "cuda"


def pct(x, q):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan");


def brief(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    return {"mean": float(x.mean()), "p50": pct(x, 50), "p90": pct(x, 90),
            "p99": pct(x, 99), "max": float(x.max())}


def rotation_angle_deg(a, b):
    tr = np.einsum("nij,nij->n", np.swapaxes(a, 1, 2), b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)));


def kabsch_one_ring(x0, xb, faces):
    """[N,3,3] rotations best fitting the one-ring edge displacements."""
    n = x0.shape[0]
    f = faces.astype(np.int64)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    p = x0[dst] - x0[src]
    q = xb[dst] - xb[src]
    h = np.zeros((n, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            h[:, i, j] = np.bincount(src, weights=p[:, i] * q[:, j], minlength=n)
    # R = argmax tr(R^T H) over rotations = V U^T where H = U S V^T
    u, _, vt = np.linalg.svd(h.transpose(0, 2, 1))
    r = np.einsum("nji,nkj->nik", vt, u)
    bad = np.linalg.det(r) < 0
    if bad.any():
        vt2 = vt.copy()
        vt2[bad, -1, :] *= -1.0
        r = np.einsum("nji,nkj->nik", vt2, u)
    return r


def call_method(mod, name, x0t, xbt, q0t, s0t):
    # every modulator is written for a BATCH: x0/xB are [B,N,3]
    with torch.no_grad():
        delta_q, payload = mod.delta_rotation_scale(
            x0t.unsqueeze(0), xbt.unsqueeze(0))
        scaled = mod.delta_scale(payload, s0t)
        q1, s1 = mod.apply(delta_q, scaled, q0t, s0t)
    q1 = q1.detach().float().cpu().numpy()
    s1 = s1.detach().float().cpu().numpy()
    if q1.ndim == 3:
        q1 = q1[0]
    if s1.ndim == 3:
        s1 = s1[0]
    return q1.astype(np.float64), s1.astype(np.float64)


def main():
    cap = torch.load(CAPTURE)
    x0 = cap["x0"].numpy().astype(np.float64)
    BX = cap["xB"].numpy().astype(np.float64)
    names = [str(v) for v in cap["names"]]
    faces = cap["faces"].numpy()
    q0 = cap["gs_quat"].numpy().astype(np.float64)
    s0 = cap["gs_scale"].numpy().astype(np.float64)
    n = x0.shape[0]
    print("capture: N=%d  vectors=%s" % (n, names))

    pipe = SingleImage3DPipeline(device=DEV, lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/offline_job", job_id="offline")
    renderer = pipe.arkit_session["renderer"]
    mods = {m: cov_switch._make(m, renderer, {}) for m in METHODS}
    print("modulators:", {k: type(v).__name__ for k, v in mods.items()})

    R0 = quat_to_matrix(torch.from_numpy(q0).float()).numpy()
    D0 = np.square(s0)
    lam0 = np.sort(D0, axis=-1)
    eye = np.eye(3)

    x0t = torch.from_numpy(x0).float().to(DEV)
    q0t = torch.from_numpy(q0).float().to(DEV)
    s0t = torch.from_numpy(s0).float().to(DEV)

    base_ang = rotation_angle_deg(np.broadcast_to(eye, (n, 3, 3)), R0)
    print("rest orientation vs identity: p50=%.2f deg" % pct(base_ang, 50))

    out = {"vectors": names, "methods": METHODS, "per": {}, "stretch": {}}
    for k, nm in enumerate(names):
        xbt = torch.from_numpy(BX[k]).float().to(DEV)
        r_ref = kabsch_one_ring(x0, BX[k], faces)
        rig_ang = rotation_angle_deg(np.broadcast_to(eye, (n, 3, 3)), r_ref)

        # per-vertex linear transform and its polar factor: reference + bound
        jv = mods["tga"].vertex_jacobians(x0t.unsqueeze(0), xbt).detach().double()
        jv = jv.cpu().numpy()[0]
        jv = np.nan_to_num(jv, nan=0.0, posinf=0.0, neginf=0.0)
        jv = np.clip(jv, -64.0, 64.0)
        rv = rot_sync.polar_rotation(
            torch.from_numpy(jv).float().to(DEV), exact=False)
        rv = rv.double().cpu().numpy()
        sv = np.einsum("nji,njk->nik", rv, jv)
        sig_exact = (jv * D0[:, None, :]) @ jv.transpose(0, 2, 1)
        s_sv = np.linalg.svd(sv, compute_uv=False)
        out["stretch"][nm] = {
            "shear_norm": brief(np.linalg.norm(sv - eye, axis=(1, 2))),
            "cond": brief(s_sv[:, 0] / np.maximum(s_sv[:, -1], 1e-9)),
            "det": brief(np.linalg.det(sv)),
        }
        print("");
        print("== %-10s  rigid motion p50=%.3f p90=%.3f deg   "
              % (nm, pct(rig_ang, 50), pct(rig_ang, 90))
              + "shear p50=%.4f  cond p50=%.3f"
              % (out["stretch"][nm]["shear_norm"]["p50"],
                 out["stretch"][nm]["cond"]["p50"]))
        print("   %-18s %10s %10s %10s %10s %10s"
              % ("method", "eig p99", "shape p50", "shape p90",
                 "rigid p50", "false%"))
        for m in METHODS:
            q1, s1 = call_method(mods[m], m, x0t, xbt, q0t, s0t)
            r1 = quat_to_matrix(torch.from_numpy(q1).float()).numpy()
            d1 = np.square(s1)
            lam1 = np.sort(d1, axis=-1)
            eig = np.abs(lam1 / np.maximum(lam0, 1e-30) - 1.0)
            sig1 = (r1 * d1[:, None, :]) @ np.swapaxes(r1, 1, 2)
            num = np.linalg.norm(sig1 - sig_exact, axis=(1, 2))
            den = np.maximum(np.linalg.norm(sig_exact, axis=(1, 2)), 1e-30)
            shape_err = num / den
            # the rotation the method actually APPLIED is R1 R0^T, not R1:
            # R1 still carries the gaussian's arbitrary rest orientation
            # (median 96.9 deg away from identity for this asset), which would
            # swamp the few degrees of motion we are trying to measure.
            r_delta = r1 @ np.swapaxes(R0, 1, 2)
            ang = rotation_angle_deg(r_ref, r_delta)
            false_move = float(np.mean((ang > 20.0) & (rig_ang < 5.0)))
            out["per"].setdefault(m, {})[nm] = {
                "eig_rel_err_p99": pct(eig, 99),
                "eig_rel_err_max": float(eig.max()),
                "shape_err": brief(shape_err),
                "rigid_err": brief(ang),
                "false_move_frac": false_move,
                "applied_deg": brief(
                    rotation_angle_deg(np.broadcast_to(eye, (n, 3, 3)), r_delta)),
            }
            r = out["per"][m][nm]
            print("   %-18s %10.2e %10.4f %10.4f %10.3f %9.2f%%"
                  % (m, r["eig_rel_err_p99"], r["shape_err"]["p50"],
                     r["shape_err"]["p90"], r["rigid_err"]["p50"],
                     100.0 * false_move))

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("");
    print("saved", OUT)


if __name__ == "__main__":
    main()
