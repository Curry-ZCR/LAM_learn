#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""protrusion（薄片出皮量）的可辨识形式。

关键点：高斯中心 x 由 FLAME 形变 xb 决定，**四个方法都只改 (q,s)，不改 x**
（probe_dep3d.py 已证：四方法的中性/表情 3D 法向偏移逐位相同）。
因此任何基于「中心相对网格的偏移」的指标对本课题是**不可辨识**的，
真正可辨识且与「薄片斜插」对应的量是：椭球沿薄轴 s_max 在法向上的投影

    protr_i = s_max_i * sin(theta_i),  theta_i = angle(thin_axis_i, n_i)

正确解 theta -> 0（薄轴对齐法向，圆盘躺在切平面上）。用局部平均一环边长
e_bar = 5.308 mm 归一：protrusion_over_edge = protr / e_bar。
"""
import sys, json
from pathlib import Path
import numpy as np

# NOTE: LAM 内部会对 sys.argv 再 parse 一次（flame_tracking_single_image._parse_args），
# 所以必须在 import LAM 之前取走自己的参数并清空 sys.argv。
_ASSET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "messi"
sys.argv = [sys.argv[0]]

sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def quat_to_R(q):
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def pctl(x, qs=(50, 90, 99)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {("p%d" % q): float(np.percentile(x, q)) for q in qs}


def main():
    import eval_cov_methods as E
    import eval_cov_protocol_v2 as V
    from lam3d.core import cov_switch
    from lam3d.core.pipeline import SingleImage3DPipeline
    asset = _ASSET
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    img = Path("/workspace/LAM/assets/sample_input/%s" % asset)
    if not img.exists():
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            if Path(str(img) + ext).exists():
                img = Path(str(img) + ext); break
    pipe.run(image_path=img, job_dir="/tmp/v2_job_%s" % asset, job_id="v2_%s" % asset)
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    meta = V.arkit_live.arkit_meta()
    base = V.tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    gs = sess["gs_model"]
    xyz = gs.xyz.detach().unsqueeze(0).float()
    q0t = gs.rotation.detach().float(); s0t = gs.scaling.detach().float()
    q0 = q0t.detach().cpu().numpy().astype(np.float64)
    s0 = s0t.detach().cpu().numpy().astype(np.float64)

    x_rest = E.mesh_vertices(renderer, sess["flame_data"], E.vector_for(meta, "zeros"), base)
    # 局部平均一环边长
    tri = x_rest[faces]
    el = np.concatenate([np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
                         np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                         np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)])
    e_bar = float(np.median(el))

    presets = ["zeros", "ones", "half", "bigSmile", "smile", "mouthOpen", "surprise"]
    rep = {"asset": asset, "e_bar_mm": e_bar, "per": {}}
    for name in V.METHODS:
        cov_switch.set_method(name, renderer=renderer)
        mod = renderer._cov_modulator
        rec = {}
        for pv in presets:
            vec = E.vector_for(meta, pv)
            q1, s1, _ = E.modulate(mod, renderer, sess["flame_data"], vec, xyz, q0t, s0t)
            q1 = np.asarray(q1, np.float64).reshape(-1, 4)
            s1 = np.asarray(s1, np.float64).reshape(-1, 3)
            nB, _ = E.vertex_normals(E.mesh_vertices(renderer, sess["flame_data"], vec, base), faces)
            R1 = quat_to_R(q1)
            th = np.argmin(s1, -1)
            u = R1[np.arange(len(q1)), :, th]
            cosv = np.clip(np.abs((u * nB).sum(-1)), 0.0, 1.0)
            ang = np.degrees(np.arccos(cosv))
            prot = s1.max(-1) * np.sqrt(np.maximum(1.0 - cosv ** 2, 0.0)) * 1000.0  # mm
            poe = prot / e_bar
            planar = (s0.max(-1) / np.maximum(s0.min(-1), 1e-12)) > 2.0
            rec[pv] = {"theta_deg": pctl(ang[planar]), "prot_mm": pctl(prot[planar]),
                       "poe": pctl(poe[planar]),
                       "frac_poe_gt1": float((poe[planar] > 1.0).mean()),
                       "frac_poe_gt2": float((poe[planar] > 2.0).mean()),
                       "frac_theta_lt15": float((ang[planar] < 15).mean()),
                       "frac_planar": float(planar.mean())}
        rep["per"][name] = rec
        print("[prot] %-17s %s" % (name, " ".join(
            "%s theta50 %.1f poe50 %.2f >1:%.3f" % (pv, rec[pv]["theta_deg"]["p50"],
                                                    rec[pv]["poe"]["p50"], rec[pv]["frac_poe_gt1"])
            for pv in presets)), flush=True)
    Path("/tmp/cov_v2/protrusion.json").write_text(json.dumps(rep, indent=1))
    print("[prot] wrote /tmp/cov_v2/protrusion.json e_bar=%.3f mm" % e_bar, flush=True)


main()
