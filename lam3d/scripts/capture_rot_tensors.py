# -*- coding: utf-8 -*-
"""一次性捕获真实会话的 x0/xB/gaussians 到磁盘。

完全复刻 gs_renderer.animate_gs_model 的调制分支（gs_renderer.py:629-674）：
    v_cano = gs_attr.xyz      # 注意：不是网格顶点，而是 20018 个高斯中心
    x0 = animation_forward(v_cano, ...)                      # dynamic_offset=None
    xB = animation_forward(v_cano, ..., dynamic_offset=off)  # ARKit 偏移

输出 /tmp/rot_capture.pt:
    x0 [N,3] / xB [M,N,3] / names / faces [F,3] / gs_quat [N,4] / gs_scale [N,3] / N
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
from lam3d.core.pipeline import SingleImage3DPipeline          # noqa: E402
from lam3d.core.tga_covariance import _base_module            # noqa: E402

OUT = "/tmp/rot_capture.pt"

VECTORS = {
    "ones": np.ones(52, dtype=np.float32),
    "half": np.full(52, 0.5, dtype=np.float32),
    "extreme": np.full(52, 2.0, dtype=np.float32),
}
rng = np.random.default_rng(20261022)
VECTORS["rand"] = rng.uniform(0, 1, 52).astype(np.float32)
VECTORS["rand_sparse"] = (
    (rng.uniform(0, 1, 52) > 0.75).astype(np.float32) * rng.uniform(0, 1.2, 52).astype(np.float32)
)


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path="/workspace/LAM/assets/sample_input/messi.png",
             job_dir="/tmp/cap_rot_job", job_id="caprot")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    fd = sess["flame_data"]
    gs = sess["gs_model"]
    print("flame_data keys:", sorted(fd.keys()), flush=True)

    lam = _base_module().DynamicCovarianceModulator(renderer.flame_model)
    lam.to("cuda").eval()

    v_cano = gs.xyz.detach().unsqueeze(0).float()

    def fwd(dynamic_offset=None):
        return renderer.flame_model.animation_forward(
            v_cano=v_cano,
            shape=fd["betas"].repeat(1, 1),
            expr=fd["expr"],
            rotation=fd["rotation"],
            neck=fd["neck_pose"],
            jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"],
            translation=fd["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=False,
            static_offset=None,
            dynamic_offset=dynamic_offset,
        )["animated"]

    xBs, names = [], []
    with torch.no_grad():
        x0 = fwd(None)
        for name, vec in VECTORS.items():
            p = lam.prepare_params(torch.from_numpy(vec).cuda(), 1, "cuda", torch.float32)
            o = lam.arkit_offsets(p)
            xB = fwd(o)
            xBs.append(xB[0].cpu())
            names.append(name)
            print("captured %-12s |xB-x0| mean=%.5f max=%.5f"
                  % (name, (xB - x0).abs().mean().item(), (xB - x0).abs().max().item()),
                  flush=True)

        payload = {
            "x0": x0[0].cpu(),
            "xB": torch.stack(xBs),
            "names": names,
            "faces": renderer.flame_model.faces_up.cpu(),
            "gs_quat": gs.rotation.detach().float().cpu(),
            "gs_scale": gs.scaling.detach().float().cpu(),
            "N": int(x0.shape[1]),
        }
    torch.save(payload, OUT)
    print("saved", OUT, "N=%d F=%d M=%d" % (payload["N"], payload["faces"].shape[0], len(names)),
          flush=True)


main()
