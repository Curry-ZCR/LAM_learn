#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DEP 标定探针：搞清 3DGS 的 comp_depth 与网格相机 z 之间的系统偏移来源。
只跑一次会话（~80 s），print 完即退出。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    import eval_cov_methods as E
    import eval_cov_protocol_v2 as V
    from lam3d.core import cov_switch
    from lam3d.core.pipeline import SingleImage3DPipeline
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=Path("/workspace/LAM/assets/sample_input/messi.png"),
             job_dir="/tmp/v2_job_messi", job_id="v2_messi")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    meta = V.arkit_live.arkit_meta()
    base = V.tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    vec = E.vector_for(meta, "ones")
    mv = E.mesh_vertices(renderer, sess["flame_data"], vec, base)
    faces = renderer.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    raster = E.MeshRaster(faces, V.SIZE, V.SIZE)
    out = {}
    for name in V.METHODS:
        cov_switch.set_method(name, renderer=renderer)
        fr = E.render_full(sess, vec, yaw=0.0, size=V.SIZE)
        cols = np.clip(fr["col"], 0, 1)
        best = None
        for sy in (-1.0, 1.0):
            raster.set_view(fr["c2w"], fr["intr"], sy)
            img, mask, _ = raster.render(mv, cols)
            gm = fr["alpha"] > 0.5
            iou = float((mask & gm).sum()) / max(float((mask | gm).sum()), 1.0)
            if best is None or iou > best[0]:
                best = (iou, sy, mask)
        _, sy, mask = best
        raster.set_view(fr["c2w"], fr["intr"], sy)
        _, mm, mz = raster.render(mv, cols)
        a = fr["depth"][mask].astype(np.float64)
        b = mz[mask].astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
        a, b = a[ok], b[ok]
        r = float(np.corrcoef(a, b)[0, 1])
        d = (b - a) * 1000.0
        out[name] = dict(iou=best[0], sign_y=sy, n=int(ok.sum()),
                         gs_depth_med=float(np.median(a)),
                         mesh_z_med=float(np.median(b)),
                         corr=r, d_med=float(np.median(d)),
                         d_p05=float(np.percentile(d, 5)),
                         d_p95=float(np.percentile(d, 95)))
        # gs 中心与网格顶点的偏离（判断「中心 vs 表面」是否就是主因）
        xyz = fr["xyz"]
        dist = np.linalg.norm(xyz - mv, axis=1) * 1000.0
        out[name]["center_vs_vert_med_mm"] = float(np.median(dist))
        out[name]["center_vs_vert_p95_mm"] = float(np.percentile(dist, 95))
    print(json.dumps(out, indent=1), flush=True)

main()
