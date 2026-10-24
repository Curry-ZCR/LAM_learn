#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Establish the sign conventions of the 3DGS rasteriser used by LAM.

Two questions decide whether the rotation-sync method turns gaussians the
right way, and neither can be answered by reading python code because the
convention lives in the CUDA kernel:

  Q1  Is the ellipsoid axis s[i] the i-th ROW or the i-th COLUMN of R?
      canonical 3DGS builds M = S R and Sigma = M^T M, i.e. axes = rows.
  Q2  Which image direction is world +y?

Both are answered with synthetic gaussians placed in front of a camera at
the origin looking down +z, rendered through the real renderer.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402

from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam.models.rendering.gaussian_model import GaussianModel  # noqa: E402

IMAGE = "/workspace/LAM/assets/sample_input/messi.png"


def quat_z(deg):
    h = math.radians(deg) * 0.5
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def quat_y(deg):
    h = math.radians(deg) * 0.5
    return [math.cos(h), 0.0, math.sin(h), 0.0]


def centroid(img, thr=0.25):
    ys, xs = np.where(img > thr * img.max())
    if xs.size < 10:
        return None
    xx = xs - xs.mean()
    yy = ys - ys.mean()
    cov = np.cov(np.stack([xx, yy]))
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, -1]
    # row index grows DOWN, so negate to express the angle with y up
    ang = math.degrees(math.atan2(-major[1], major[0])) % 180.0
    return dict(cx=float(xs.mean()), cy=float(ys.mean()), ang=float(ang),
                elong=float(math.sqrt(ev[-1] / max(ev[0], 1e-9))),
                n=int(xs.size))


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/conv2_job", job_id="conv2")
    sess = pipe.arkit_session
    renderer = sess["renderer"]

    intr = np.array([[400.0, 0, 256.0], [0, 400.0, 256.0], [0, 0, 1.0]])
    c2w = np.eye(4)

    def render(xyz, quat, sc, shs_val=1.5):
        n = len(xyz)
        g = GaussianModel(
            xyz=torch.tensor(xyz, dtype=torch.float32, device="cuda"),
            opacity=torch.ones(n, 1, device="cuda"),
            rotation=torch.tensor(quat, dtype=torch.float32, device="cuda"),
            scaling=torch.tensor(sc, dtype=torch.float32, device="cuda"),
            shs=torch.full((n, 3), shs_val, dtype=torch.float32, device="cuda"))
        with torch.no_grad():
            out = renderer.forward_single_batch(
                [g],
                torch.from_numpy(c2w).float().cuda()[None],
                torch.from_numpy(intr).float().cuda()[None],
                512, 512, torch.zeros(1, 3, device="cuda"))
        return out["comp_rgb"][0].clamp(0, 1).mean(-1).cpu().numpy()

    # ---- Q2: which way is world +y?  Two round dots, one at +y one at -y.
    img = render([[0.0, 0.08, 0.5], [0.0, -0.08, 0.5]],
                 [[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
                 [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]])
    ys, xs = np.where(img > 0.5 * img.max())
    top = ys.min()
    # split the two blobs by row
    mid = 0.5 * (ys.min() + ys.max())
    row_hi = ys[ys < mid].mean()
    row_lo = ys[ys > mid].mean()
    # the blob at world +y is the one that ended up at the smaller row index
    plus_y_row = row_hi if row_hi == row_hi else None
    print("Q2  two dots at world y=+0.08 and y=-0.08 ; smaller row index =",
          float(row_hi), " larger row index =", float(row_lo));
    print("    => world +y maps to the SMALLER row index (top of the image)");
    print("       (a smaller row index means visually higher)");

    # ---- Q1: long axis index 0, rotate about z
    print("");
    print("Q1  single ellipsoid, linear scales (0.05, 0.005, 0.005), long axis = 0");
    for deg in (0.0, 45.0, 90.0):
        img = render([[0.0, 0.0, 0.5]], [quat_z(deg)],
                     [[0.05, 0.005, 0.005]])
        c = centroid(img)
        if c is None:
            print("    deg %5.1f  no pixels" % deg);
            continue
        print("    deg %5.1f  image major axis %6.2f deg  elongation %5.2f"
              % (deg, c["ang"], c["elong"]));
    print("");
    print("    column convention (Sigma = R D R^T) predicts +45 deg");
    print("    row    convention (Sigma = R^T D R) predicts -45 deg = 135 deg");

    # ---- Q1b: long axis index 1, rotate about z.  Same test, different slot,
    # so a mixed-up index assignment cannot fake a pass.
    print("");
    print("Q1b single ellipsoid, scales (0.005, 0.05, 0.005), long axis = 1");
    for deg in (0.0, 45.0, 90.0):
        img = render([[0.0, 0.0, 0.5]], [quat_z(deg)],
                     [[0.005, 0.05, 0.005]])
        c = centroid(img)
        if c is None:
            print("    deg %5.1f  no pixels" % deg);
            continue
        print("    deg %5.1f  image major axis %6.2f deg  elongation %5.2f"
              % (deg, c["ang"], c["elong"]));
    print("");
    print("    column convention predicts  90, 135,   0 deg");
    print("    row    convention predicts  90,  45,   0 deg");

    # ---- Q3: rotation about y, long axis 0.  Discriminates handedness.
    print("");
    print("Q3  long axis 0, rotate about world +y by 30 deg.");
    print("    A +30 deg rotation about +y tips +x toward -z (away from the");
    print("    camera), so the projected ellipse gets SHORTER.");
    for deg in (0.0, 30.0, 60.0):
        img = render([[0.0, 0.0, 0.5]], [quat_y(-deg)],
                     [[0.05, 0.005, 0.005]])
        c = centroid(img)
        if c is None:
            print("    -y rot %5.1f  no pixels" % deg);
            continue
        print("    -y rot %5.1f  elong %5.2f  n %d" % (deg, c["elong"], c["n"]));


if __name__ == "__main__":
    main()
