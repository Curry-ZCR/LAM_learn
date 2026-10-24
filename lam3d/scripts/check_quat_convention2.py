#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decide the axis convention of LAM's 3DGS rasteriser, with the image
v-axis sign MEASURED rather than assumed.

Why a second script.  check_quat_convention.py compared the measured image
major-axis angle against predictions built under the assumption v ~ +y.
check_axis_convention.py used a y-up angle convention.  The two agreed on
the raw pixels and disagreed on the verdict, because neither one measured
which way the image v axis points relative to world +y.  This script does
that first, then applies the sign.

  step 1  render one gaussian at (0, +Y, z) and one at (0, -Y, z);
          dv/dy > 0 iff the +y one lands at the LARGER row index.
          also render ( +X, 0, z) / (-X, 0, z) for du/dx (sanity).
  step 2  with sy = sign(dv/dy) known, the image-space angle relates to the
          world-space angle by theta_img = sy * theta_world, so compare
          sy * theta_meas against both hypotheses evaluated in world space.

    COLUMN   Sigma = R D R^T      axis i is COLUMN i of R   (canonical 3DGS)
    ROW      Sigma = R^T D R      axis i is ROW    i of R
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402

from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam.models.rendering.gaussian_model import GaussianModel  # noqa: E402

IMAGE = "/workspace/LAM/assets/sample_input/messi.png"
FOCAL = 400.0
SIZE = 512


def quat_z(deg):
    h = math.radians(deg) * 0.5
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def moments(img, thr=0.35):
    w = img.astype(np.float64)
    w = np.clip(w - thr * w.max(), 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return None
    vv, uu = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    cu = (w * uu).sum() / tot
    cv = (w * vv).sum() / tot
    du = uu - cu
    dv = vv - cv
    suu = (w * du * du).sum() / tot
    svv = (w * dv * dv).sum() / tot
    suv = (w * du * dv).sum() / tot
    return dict(cu=float(cu), cv=float(cv), suu=suu, svv=svv, suv=suv,
                tot=float(tot))


def theta(suu, svv, suv):
    return math.degrees(0.5 * math.atan2(2.0 * suv, suu - svv))


def elong(m):
    ev = np.linalg.eigvalsh(np.array([[m["suu"], m["suv"]], [m["suv"], m["svv"]]]))
    return math.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12))


def predicted_angles(R, scales):
    d = np.diag(np.square(np.asarray(scales, dtype=np.float64)))
    Rn = np.asarray(R, dtype=np.float64)
    out = {}
    for name, S in (("column", Rn @ d @ Rn.T), ("row", Rn.T @ d @ Rn)):
        out[name] = math.degrees(0.5 * math.atan2(2.0 * S[0, 1], S[0, 0] - S[1, 1]))
    return out


def wrap(x):
    while x <= -90.0:
        x += 180.0
    while x > 90.0:
        x -= 180.0
    return x


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/conv4_job", job_id="conv4")
    renderer = pipe.arkit_session["renderer"]

    intr = np.array([[FOCAL, 0, SIZE / 2.0], [0, FOCAL, SIZE / 2.0], [0, 0, 1.0]])
    c2w = np.eye(4)

    def render(xyz, quat, sc):
        g = GaussianModel(
            xyz=torch.tensor([xyz], dtype=torch.float32, device="cuda"),
            opacity=torch.ones(1, 1, device="cuda"),
            rotation=torch.tensor([quat], dtype=torch.float32, device="cuda"),
            scaling=torch.tensor([sc], dtype=torch.float32, device="cuda"),
            shs=torch.full((1, 3), 1.5, dtype=torch.float32, device="cuda"))
        with torch.no_grad():
            out = renderer.forward_single_batch(
                [g], torch.from_numpy(c2w).float().cuda()[None],
                torch.from_numpy(intr).float().cuda()[None],
                SIZE, SIZE, torch.zeros(1, 3, device="cuda"))
        return out["comp_rgb"][0].clamp(0, 1).mean(-1).cpu().numpy()

    ident = [1.0, 0.0, 0.0, 0.0]
    small = [0.02, 0.02, 0.02]
    off = 0.15

    print("");
    print("=== step 1: which way do u and v point? ===")
    mp = moments(render([0.0, +off, 0.5], ident, small))
    mn = moments(render([0.0, -off, 0.5], ident, small))
    if mp is None or mn is None:
        print("    no pixels -- abort");
        return
    dy = mp["cv"] - mn["cv"]
    print("    centre at world y=+0.15 -> row %.3f ; at y=-0.15 -> row %.3f"
          % (mp["cv"], mn["cv"]))
    print("    row(+y) - row(-y) = %.3f  =>  dv/dy %s 0  (sy = %+d)"
          % (dy, ">" if dy > 0 else "<", 1 if dy > 0 else -1))
    sy = 1 if dy > 0 else -1
    mx = moments(render([+off, 0.0, 0.5], ident, small))
    nx = moments(render([-off, 0.0, 0.5], ident, small))
    dx = mx["cu"] - nx["cu"]
    print("    col(+x) - col(-x) = %.3f  =>  du/dx %s 0  (sx = %+d)"
          % (dx, ">" if dx > 0 else "<", 1 if dx > 0 else -1))

    print("");
    print("=== step 2: major axis, measurement corrected by sy ===")
    cases = [
        ("axis0(long) 45 deg about z", [0.05, 0.005, 0.005], 45.0),
        ("axis1(long) 45 deg about z", [0.005, 0.05, 0.005], 45.0),
        ("axis0(long) 60 deg about z", [0.05, 0.005, 0.005], 60.0),
        ("axis1(long) 30 deg about z", [0.005, 0.05, 0.005], 30.0),
        ("axis0(long) -45 deg about z", [0.05, 0.005, 0.005], -45.0),
    ]
    print("    %-30s %8s %8s | %8s %8s | %7s %7s"
          % ("case", "th_meas", "th_corr", "col", "row", "d_col", "d_row"))
    votes = {"column": 0, "row": 0}
    for name, sc, deg in cases:
        q = quat_z(deg)
        m = moments(render([0.0, 0.0, 0.5], q, sc))
        if m is None:
            print("    %-30s  no pixels" % name);
            continue
        th_img = theta(m["suu"], m["svv"], m["suv"])
        th_world = sy * th_img
        pred = predicted_angles(quat_to_matrix(torch.tensor([q])).numpy()[0], sc)
        dc = abs(wrap(th_world - pred["column"]))
        dr = abs(wrap(th_world - pred["row"]))
        win = "column" if dc + 1.0 < dr else ("row" if dr + 1.0 < dc else "tie")
        if win in votes:
            votes[win] += 1
        print("    %-30s %8.2f %8.2f | %8.2f %8.2f | %7.2f %7.2f  %s  (elong %.2f)"
              % (name, th_img, wrap(th_world), pred["column"], pred["row"],
                 dc, dr, win, elong(m)))

    print("");
    print("    votes:", votes)
    best = max(votes, key=votes.get) if max(votes.values()) else "inconclusive"
    print("    VERDICT:", best.upper())
    if best == "column":
        print("      Sigma = R D R^T, axis i = COLUMN i of R == canonical 3DGS");
        print("      => q_new = delta_q * q_old applies the world rotation as intended");
    elif best == "row":
        print("      Sigma = R^T D R, axis i = ROW i of R");
        print("      => q_new = delta_q * q_old applies the INVERSE rotation");
    print("")


if __name__ == "__main__":
    main()
