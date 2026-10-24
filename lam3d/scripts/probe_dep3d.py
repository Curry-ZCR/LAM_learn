#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DEP-3D 标定：为什么图像空间 comp_depth 差值不可用，以及 3D 法向偏移（含中性基线差分）如何可用。

输出三块：
 A) 图像空间（旧定义）的 corr(gs_depth, mesh_z) —— 已在 probe_dep.py 里证明只有 0.14
 B) 3D 每高斯法向偏移 s_i = (c_i - x_i)·n_i 的绝对量（含 LAM 烘焙 offset，方法无关的部分）
 C) 表情态相对中性态的差分 Δs_i = s_i(expr) - s_i(neutral)  ← 协议应当使用的量
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def vertex_normals(v, faces):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    acc = np.zeros_like(v)
    for k in range(3):
        np.add.at(acc, faces[:, k], fn)
    n = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return n


def main():
    import eval_cov_methods as E
    import eval_cov_protocol_v2 as V
    from lam3d.core import cov_switch
    pipe = None
    from lam3d.core.pipeline import SingleImage3DPipeline
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=Path("/workspace/LAM/assets/sample_input/messi.png"),
             job_dir="/tmp/v2_job_messi", job_id="v2_messi")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    meta = V.arkit_live.arkit_meta()
    base = V.tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    x_rest = E.mesh_vertices(renderer, sess["flame_data"], E.vector_for(meta, "zeros"), base)
    n_rest = vertex_normals(x_rest, faces)

    out = {}
    for name in V.METHODS:
        cov_switch.set_method(name, renderer=renderer)
        rec = {}
        for vec_name in ("zeros", "ones", "bigSmile"):
            vec = E.vector_for(meta, vec_name)
            fr = E.render_full(sess, vec, yaw=0.0, size=V.SIZE)
            xyz = fr["xyz"].astype(np.float64)
            mv = (x_rest if vec_name == "zeros"
                  else E.mesh_vertices(renderer, sess["flame_data"], vec, base)).astype(np.float64)
            nv = n_rest if vec_name == "zeros" else vertex_normals(mv, faces)
            d = xyz - mv
            s = np.einsum("ij,ij->i", d, nv) * 1000.0        # mm，沿顶点法向
            tan = np.linalg.norm(d - s[:, None] * nv / 1000.0, axis=1) * 1000.0
            rec[vec_name] = dict(s=s, offset_mm=np.linalg.norm(d, axis=1) * 1000.0, tan_mm=tan)
        z = rec["zeros"]["s"]
        for k, r in rec.items():
            s = r["s"]
            ds = s - z
            out[name + "|" + k] = dict(
                s_med=float(np.median(s)), s_p05=float(np.percentile(s, 5)),
                s_p95=float(np.percentile(s, 95)),
                frac_abs_gt_1p5=float(np.mean(np.abs(s) > 1.5)),
                ds_med=float(np.median(ds)),
                ds_p05=float(np.percentile(ds, 5)),
                ds_p95=float(np.percentile(ds, 95)),
                frac_ds_gt_1p5=float(np.mean(ds > 1.5)),
                frac_ds_lt_m1p5=float(np.mean(ds < -1.5)),
                offset_med_mm=float(np.median(r["offset_mm"])),
                tan_med_mm=float(np.median(r["tan_mm"])),
            )
    print("DEP3D_JSON " + json.dumps(out, indent=1), flush=True)


main()
