# -*- coding: utf-8 -*-
"""rotation_field 内部分步计时：找出 15.8 ms 里 10 ms 的缺口。"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
from lam3d.core import arkit_live, rot_sync, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.rot_sync import *  # noqa

IMAGE = "/workspace/LAM/assets/sample_input/messi.png"
MARK = {}


def mark(tag):
    MARK.setdefault(tag, []).append(torch.cuda.Event(True))


def stamp(tag):
    e = torch.cuda.Event(True)
    e.record()
    MARK.setdefault(tag, []).append(e)


def main():
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/steprot", job_id="steprot")
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    mod = rot_sync.install(renderer, force=True)
    ones = np.ones(52, dtype=np.float32)

    cap = {}
    orig = rot_sync.RotSyncModulator.rotation_field

    def spy(self, x0, xB):
        cap.setdefault("x0", x0.detach().clone())
        cap.setdefault("xB", xB.detach().clone())
        return orig(self, x0, xB)

    rot_sync.RotSyncModulator.rotation_field = spy
    with torch.no_grad():
        arkit_live.compute_splat(sess, ones)
    rot_sync.RotSyncModulator.rotation_field = orig
    x0 = cap["x0"][:1].contiguous()
    xB = cap["xB"][:1].contiguous()

    with torch.no_grad():
        area = mod.face_areas(x0)
        Jf = mod.face_jacobians(x0, xB)
        Jv = mod.vertex_jacobians(x0, xB, Jf=Jf)
        print("Jf finite=%s  |Jf| max=%.2f  det min=%.4f max=%.4f"
              % (bool(torch.isfinite(Jf).all()), Jf.abs().max(),
                 torch.linalg.det(Jf).min(), torch.linalg.det(Jf).max()), flush=True)
        print("Jv det min=%.4f max=%.4f" % (torch.linalg.det(Jv).min(), torch.linalg.det(Jv).max()),
              flush=True)
        # 真实数据上 bad 有多少？
        for tag, J in (("Jf", Jf), ("Jv", Jv)):
            r = rot_sync.polar_rotation(J, iters=10)
            orth = (r.transpose(-1, -2) @ r - torch.eye(3, device=r.device)).abs().amax(dim=(-1, -2))
            print("  %s polar: orth max=%.2e  det<0=%d  scale=%.2e"
                  % (tag, orth.max(), int((torch.linalg.det(r) <= 0).sum()),
                     r.reshape(-1, 9).norm(dim=-1).max()), flush=True)

    # ---- 手抄 rotation_field 主体，逐步打点
    def body():
        area_ = mod.face_areas(x0); stamp("1_face_areas")
        Jf_ = mod.face_jacobians(x0, xB); stamp("2_face_jacobians")
        Jv_ = mod.vertex_jacobians(x0, xB, Jf=Jf_); stamp("3_vertex_jacobians")
        Rv = rot_sync.polar_rotation(Jv_); stamp("4_polar_Jv")
        q_ref = F.normalize(rot_sync.matrix_to_quat(Rv), dim=-1); stamp("5_m2q_Jv+nrm")
        q_face = F.normalize(rot_sync.matrix_to_quat(rot_sync.polar_rotation(Jf_)), dim=-1)
        stamp("6_polar_Jf+m2q+nrm")
        flat = mod.faces_flat.to(x0.device)
        qr3 = q_ref[:, flat, :]; stamp("7_gather_qr3")
        pair = qr3.reshape(1, -1, 3, 4).sum(dim=2); stamp("8_pair_sum")
        qf3 = q_face.repeat_interleave(3, dim=1); stamp("9_rep_qf3")
        pref = pair.repeat_interleave(3, dim=1); stamp("10_rep_pref")
        sign = torch.where((qf3 * pref).sum(-1, keepdim=True) < 0, -1.0, 1.0); stamp("11_sign")
        w3 = area_.repeat_interleave(3, dim=1).unsqueeze(-1); stamp("12_w3")
        vals = qf3 * sign * w3; stamp("13_vals")
        num = torch.zeros(1, x0.shape[1], 4, device=x0.device, dtype=torch.float32)
        idx = flat.view(1, -1, 1).expand(1, -1, 4)
        num.scatter_add_(1, idx, vals.to(torch.float32)); stamp("14_scatter_num")
        den = torch.zeros(1, x0.shape[1], 1, device=x0.device, dtype=torch.float32)
        den.scatter_add_(1, idx[..., :1], w3.to(torch.float32)); stamp("15_scatter_den")
        delta = num / den.clamp(min=mod.eps); stamp("16_div")
        bad = delta.norm(dim=-1, keepdim=True) < 1e-8
        delta = torch.where(bad, q_ref, delta)
        delta = F.normalize(delta, dim=-1); stamp("17_fix+nrm")
        angle = rot_sync.quat_angle(delta); stamp("18_angle")
        snap = (angle < rot_sync.IDENT_EPS).unsqueeze(-1)
        delta = torch.where(snap, rot_sync.identity_quat(delta), delta); stamp("19_snap")
        cap_ = mod.max_angle_deg * rot_sync.DEG
        over = angle > cap_
        t = torch.where(over, cap_ / angle.clamp(min=1e-9), torch.ones_like(angle))
        delta = rot_sync.slerp_identity(delta, t); stamp("20_clamp")
        delta = F.normalize(delta, dim=-1); stamp("21_final_nrm")

    with torch.no_grad():
        for _ in range(6):
            body()
        torch.cuda.synchronize()
        MARK.clear()
        body()
        torch.cuda.synchronize()

    print("\n== rotation_field 分步 (ms) ==", flush=True)
    total = 0.0
    prev = None
    for tag in sorted(MARK, key=lambda s: int(s.split("_")[0])):
        e = MARK[tag][-1] if isinstance(MARK[tag], list) else MARK[tag]
        if prev is not None:
            d = prev.elapsed_time(e)
            total += d
            print("  %-24s %6.3f" % (tag, d), flush=True)
        prev = e
    print("  %-24s %6.3f" % ("SUM", total), flush=True)

    # 单独测几个可疑项
    def gpu(fn, warm=8, rep=40):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(rep):
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return float(np.median(ts))

    flat = mod.faces_flat.to(x0.device)
    q_ref = F.normalize(rot_sync.matrix_to_quat(rot_sync.polar_rotation(Jv)), dim=-1)
    q_face = F.normalize(rot_sync.matrix_to_quat(rot_sync.polar_rotation(Jf)), dim=-1)
    print("\n== 可疑项 ==", flush=True)
    print("  gather q_ref[:,flat,:]   %6.3f" % gpu(lambda: q_ref[:, flat, :]), flush=True)
    print("  repeat_interleave x3     %6.3f" % gpu(lambda: q_face.repeat_interleave(3, dim=1)), flush=True)
    print("  scatter_add num          %6.3f" % gpu(lambda: torch.zeros(1, x0.shape[1], 4, device=x0.device).scatter_add_(
        1, flat.view(1, -1, 1).expand(1, -1, 4), torch.zeros(1, flat.numel(), 4, device=x0.device))), flush=True)
    print("  polar(Jf) exact(svd64)   %6.3f" % gpu(lambda: rot_sync.polar_rotation(Jf, exact=True), warm=1, rep=3), flush=True)
    print("  polar(Jf) default        %6.3f" % gpu(lambda: rot_sync.polar_rotation(Jf)), flush=True)
    print("  polar(Jv) default        %6.3f" % gpu(lambda: rot_sync.polar_rotation(Jv)), flush=True)
    print("  polar(Jf) iters=8        %6.3f" % gpu(lambda: rot_sync.polar_rotation(Jf, iters=8)), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
