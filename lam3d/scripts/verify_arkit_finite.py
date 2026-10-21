"""Tensor-level verification of the ARKit covariance modulation on the REAL
checkpoint session (the dry-run uses a random-init model).

Replicates exactly what the lam3d server does for a job, then inspects the
modulated GaussianModel attributes + the rasterized image for finiteness.
"""
import sys
import time

import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core import arkit_live  # noqa: E402

IMAGE = Path("/workspace/LAM/assets/sample_input/messi.png")
JOB_DIR = Path("/tmp/arkit_verify_job")

ATTRS = ("xyz", "rotation", "scaling", "opacity", "shs", "offset")


def quat_to_matrix(q):
    """wxyz -> 3x3 rotation matrix (pytorch3d convention)."""
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0 / (q * q).sum(dim=-1).clamp(min=1e-12).unsqueeze(-1)
    return torch.stack((
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def main():
    print("== building pipeline (same env/path as the server) ==", flush=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    t0 = time.time()
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id="verify")
    print("pipeline.run: %.1f s" % (time.time() - t0), flush=True)

    sess = pipe.arkit_session
    assert sess is not None, "session was not created"
    gs0 = sess["gs_model"]
    print("session ready: job=%s n_gaussians=%d render_hw=%s"
          % (sess["job_id"], gs0.xyz.shape[0], sess["render_hw"]), flush=True)

    def vec(**kw):
        v = np.zeros(52, dtype=np.float32)
        for k, val in kw.items():
            v[int(k)] = val
        return v

    cases = [
        ("all-zero (neutral)", vec()),
        ("v[43]=mouthSmileLeft=0.8", vec(**{"43": 0.8})),
        ("all-ones (52x1.0)", np.ones(52, dtype=np.float32)),
        ("jawOpen+smile preset", vec(**{"24": 1.0, "43": 1.0, "44": 1.0})),
    ]

    imgs = {}
    for name, arkit in cases:
        a = torch.from_numpy(np.ascontiguousarray(arkit, dtype=np.float32))
        with torch.no_grad():
            t0 = time.time()
            gs = sess["renderer"].animate_gs_model(
                sess["gs_model"], sess["query_points"], sess["flame_data"],
                arkit_params=a)[0]
            torch.cuda.synchronize()
            t_anim = (time.time() - t0) * 1e3
            fin = {k: bool(torch.isfinite(getattr(gs, k).float()).all()) for k in ATTRS}
            qn = gs.rotation.float().norm(dim=-1)
            sc = gs.scaling.float()
            t0 = time.time()
            img = arkit_live.render_frame(sess, arkit, size=320)
            t_frame = (time.time() - t0) * 1e3
        imgs[name] = img
        print("\n--- %s ---" % name)
        print("  animate_gs_model      : %7.3f ms" % t_anim)
        print("  render_frame (320)    : %7.3f ms" % t_frame)
        print("  all attrs finite      : %s" % fin)
        print("  image finite          : %s  dtype=%s shape=%s"
              % (bool(np.isfinite(img.astype(np.float64)).all()), img.dtype, img.shape))
        print("  |q| min/max           : %.6f / %.6f" % (qn.min().item(), qn.max().item()))
        print("  s  min/max            : %.6e / %.6e  (all > 0: %s)"
              % (sc.min().item(), sc.max().item(), bool((sc > 0).all())))
        print("  image min/max/mean/std: %d / %d / %.2f / %.2f"
              % (img.min(), img.max(), img.mean(), img.std()))

    from PIL import Image
    ref = np.asarray(Image.open("/tmp/fneu1.png"))
    mine = imgs["all-zero (neutral)"]
    print("\n=== standalone all-zero render vs HTTP neutral PNG (/tmp/fneu1.png) ===")
    print("  bit-identical: %s" % (ref.shape == mine.shape and bool((ref == mine).all())))

    print("\n=== all-ones: Σ = R diag(s^2) Rᵀ well posed (decoupled q,s only) ===")
    with torch.no_grad():
        gs = sess["renderer"].animate_gs_model(
            sess["gs_model"], sess["query_points"], sess["flame_data"],
            arkit_params=torch.ones(52, dtype=torch.float32))[0]
    M = quat_to_matrix(gs.rotation.float())
    gram = torch.bmm(M.transpose(1, 2), M)
    eye = torch.eye(3, device=gram.device).expand_as(gram)
    s2 = gs.scaling.float() ** 2
    print("  R orthonormal max|RRᵀ-I|: %.3e" % (gram - eye).abs().max().item())
    print("  s^2 min/max              : %.3e / %.3e (all > 0: %s)"
          % (s2.min().item(), s2.max().item(), bool((s2 > 0).all())))
    print("  Σ eigenvalues == s^2     : finite %s"
          % bool(torch.isfinite(s2).all()))
    print("\nDONE")


if __name__ == "__main__":
    main()
