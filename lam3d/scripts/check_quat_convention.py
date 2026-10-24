#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decide the ellipsoid axis convention of LAM's 3DGS rasteriser.

The rasteriser never sees a matrix: it takes (scaling, rotation) and builds
the 3D covariance in CUDA.  Two conventions are possible:

    COLUMN   Sigma = R D R^T      axis i is COLUMN i of R   (canonical 3DGS)
    ROW      Sigma = R^T D R      axis i is ROW    i of R

Both are predicted analytically in IMAGE space and compared against the
second moments actually measured from a rendered frame.  Working in image
space avoids every hand-wavy sign argument about which way y points.

Prediction.  Camera at the origin looking down +z, pixel = 256 + f*x/z,
row index growing downward.  At a gaussian centred on the optical axis the
projection Jacobian is J = (f/z) * [1 0 0; 0 1 0], so

    Sigma_img = J Sigma_world J^T = (f/z)^2 * [[Sxx, Sxy], [Sxy, Syy]]

and the major axis angle, measured from +u toward +v (down), is

    theta = 0.5 * atan2(2*Sxy, Sxx - Syy)

The same formula applied to the measured image moments gives theta_meas.
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


def predicted_angle(R, scales):
    """theta from the analytic projection of Sigma under both conventions."""
    d = np.diag(np.square(np.asarray(scales, dtype=np.float64)))
    Rn = np.asarray(R, dtype=np.float64)
    out = {}
    for name, S in (("column", Rn @ d @ Rn.T), ("row", Rn.T @ d @ Rn)):
        sxy = 0.5 * math.atan2(2.0 * S[0, 1], S[0, 0] - S[1, 1])
        out[name] = math.degrees(sxy)
    return out


def measured_angle(img, thr=0.35):
    """theta of the principal axis of the rendered blob, u right / v down."""
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
    ang = 0.5 * math.atan2(2.0 * suv, suu - svv)
    ev = np.linalg.eigvalsh(np.array([[suu, suv], [suv, svv]]))
    return dict(
        ang=math.degrees(ang), cu=float(cu), cv=float(cv),
        elong=float(math.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12))))


def wrap(x):
    while x <= -90.0:
        x += 180.0
    while x > 90.0:
        x -= 180.0
    return x


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/conv3_job", job_id="conv3")
    renderer = pipe.arkit_session["renderer"]

    intr = np.array([[FOCAL, 0, SIZE / 2.0], [0, FOCAL, SIZE / 2.0], [0, 0, 1.0]])
    c2w = np.eye(4)

    def render(quat, sc):
        g = GaussianModel(
            xyz=torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device="cuda"),
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

    print("");
    print("=== sanity: matrix_to_quat inverts quat_to_matrix, no transpose ===");
    Rz = torch.from_numpy(np.array([
        [math.cos(math.radians(45)), -math.sin(math.radians(45)), 0.0],
        [math.sin(math.radians(45)), math.cos(math.radians(45)), 0.0],
        [0.0, 0.0, 1.0]])).float()[None]
    qq = matrix_to_quat(Rz)
    back = quat_to_matrix(qq)
    print("    |quat_to_matrix(matrix_to_quat(R)) - R| max = %.3e"
          % float((back - Rz).abs().max()));
    print("    |... - R^T| max = %.3e"
          % float((back - Rz.transpose(-1, -2)).abs().max()));

    cases = [
        ("axis0 (long) 45deg about z", [0.05, 0.005, 0.005], 45.0),
        ("axis1 (long) 45deg about z", [0.005, 0.05, 0.005], 45.0),
        ("axis0 (long) 60deg about z", [0.05, 0.005, 0.005], 60.0),
        ("axis2 (long) 45deg about z", [0.005, 0.005, 0.05], 45.0),
    ]

    print("");
    print("=== image-space major axis: measured vs the two hypotheses ===");
    print("    %-28s %8s | %8s %8s | %8s %8s"
          % ("case", "meas", "col", "row", "d_col", "d_row"));
    votes = {"column": 0, "row": 0}
    for name, sc, deg in cases:
        q = quat_z(deg)
        img = render(q, sc)
        m = measured_angle(img)
        if m is None:
            print("    %-28s  no pixels" % name);
            continue
        pred = predicted_angle(quat_to_matrix(torch.tensor([q])).numpy()[0], sc)
        dc = abs(wrap(m["ang"] - pred["column"]))
        dr = abs(wrap(m["ang"] - pred["row"]))
        win = "column" if dc + 1.0 < dr else ("row" if dr + 1.0 < dc else "tie")
        if win in votes:
            votes[win] += 1
        print("    %-28s %8.2f | %8.2f %8.2f | %8.2f %8.2f   -> %s"
              % (name, m["ang"], pred["column"], pred["row"], dc, dr, win));

    print("");
    print("    votes:", votes);
    verdict = max(votes, key=votes.get) if max(votes.values()) > 0 else "inconclusive"
    print("    VERDICT:", verdict.upper());
    if verdict == "column":
        print("      Sigma = R D R^T, axis i = COLUMN i of R -- matches canonical 3DGS.");
        print("      => R(a*b) = R(a) R(b), so q_new = delta_q * q_old applies the");
        print("         intended WORLD rotation.  Current code is CORRECT.");
    elif verdict == "row":
        print("      Sigma = R^T D R, axis i = ROW i of R.");
        print("      => the current composition order applies the INVERSE rotation.");
    print("");


if __name__ == "__main__":
    main()
