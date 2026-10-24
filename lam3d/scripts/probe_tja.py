#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TJA / 时域协议实测：只走 modulate（不动光栅器），因此 60 帧 x 4 方法 x 11 条轨迹只要几十秒。

输出每 (轨迹, 方法) 的：
  in_jerk   驱动信号的归一化三阶差分能量（单位 1/帧^3）
  out_jerk  薄轴角速度信号的三阶差分能量（度/帧^3）
  amp       out_jerk / in_jerk   ← 抖动放大率（协议 §5 的核心量）
  step_over 阶跃轨迹的过冲比 (max - final)/|final - initial|
  hyst      往返轨迹的去程/回程在同幅度处的最大差（迟滞）
"""
import sys, json
from pathlib import Path
import numpy as np

# LAM 会对 sys.argv 再 parse 一次 -> 先取走自己的参数再清空。
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


def jerk_energy(sig, dt=1.0):
    """三阶中心差分的均方（对时间归一），sig: T x D。"""
    s = np.asarray(sig, np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if len(s) < 4:
        return 0.0
    j = (s[3:] - 3 * s[2:-1] + 3 * s[1:-2] - s[:-3]) / (dt ** 3)
    return float(np.mean(np.sum(j * j, axis=1)))


def main():
    import eval_cov_methods as E
    import eval_cov_protocol_v2 as V
    from lam3d.core import cov_switch
    from lam3d.core.pipeline import SingleImage3DPipeline

    lib = dict(np.load("/tmp/cov_v2/trajectories.npz"))
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    _p = Path("/workspace/LAM/assets/sample_input/%s" % _ASSET)
    if not _p.exists():
        for _e in (".png", ".jpg", ".jpeg", ".webp"):
            if Path(str(_p) + _e).exists():
                _p = Path(str(_p) + _e); break
    pipe.run(image_path=_p, job_dir="/tmp/v2_job_%s" % _ASSET, job_id="v2_%s" % _ASSET)
    sess = pipe.arkit_session
    renderer = sess["renderer"]
    meta = V.arkit_live.arkit_meta()
    base = V.tga_covariance._base_module().DynamicCovarianceModulator(renderer.flame_model)
    base.to("cuda").eval()
    gs = sess["gs_model"]
    xyz = gs.xyz.detach().unsqueeze(0).float()
    q0 = gs.rotation.detach().float()
    s0 = gs.scaling.detach().float()
    q0n = q0.detach().cpu().numpy().astype(np.float64)
    s0n = s0.detach().cpu().numpy().astype(np.float64)
    R0 = quat_to_R(q0n)
    thin0 = np.argmin(s0n, -1)
    u0 = R0[np.arange(len(q0n)), :, thin0]          # 中性态薄轴方向

    rep = {"asset": "messi", "n_traj": len(lib), "per": {}}
    for tname, traj in lib.items():
        in_sig = np.asarray(traj, np.float64)
        # 输入信号：驱动通道的 L2 范数序列（各通道已在 [0,1]，尺度一致）
        in_j = jerk_energy(in_sig)
        rec = {"frames": int(in_sig.shape[0]), "in_jerk": in_j,
               "channels": int((np.abs(in_sig).sum(0) > 1e-6).sum())}
        for name in V.METHODS:
            cov_switch.set_method(name, renderer=renderer)   # 重建实例 → EMA 状态清零
            mod = renderer._cov_modulator
            ang = []
            for t in range(in_sig.shape[0]):
                v = in_sig[t].astype(np.float32)
                q1, s1, _ = E.modulate(mod, renderer, sess["flame_data"], v, xyz, q0, s0)
                q1 = np.asarray(q1, np.float64).reshape(-1, 4)
                s1 = np.asarray(s1, np.float64).reshape(-1, 3)
                R1 = quat_to_R(q1)
                th = np.argmin(s1, -1)
                u = R1[np.arange(len(q1)), :, th]
                c = np.clip(np.abs((u * u0).sum(-1)), 0.0, 1.0)
                ang.append(np.degrees(np.arccos(c)).mean())    # 薄轴平均角位移（度）
            a = np.asarray(ang)
            out_j = jerk_energy(a)
            amp = out_j / max(in_j, 1e-20)
            rec[name] = {"thin_axis_deg_max": float(a.max()),
                         "thin_axis_deg_final": float(a[-1]),
                         "out_jerk": out_j, "amp": amp,
                         "monotone_frac": float(np.mean(np.diff(a) >= -1e-9)) if len(a) > 1 else float("nan")}
            if tname.startswith("step"):
                half = len(a) // 2
                tgt = float(a[-1])
                overshoot = (float(a[half:].max()) - tgt) / max(abs(tgt), 1e-9)
                rec[name]["step_overshoot"] = overshoot
                rec[name]["step_ring_frames"] = int(np.sum(np.abs(np.diff(a[half:])) > 0.02 * max(abs(tgt), 1e-9)))
            if tname.startswith("ramp"):
                half = len(a) // 2
                fwd, bwd = a[:half], a[half:][::-1]
                n = min(len(fwd), len(bwd))
                rec[name]["hyst_deg"] = float(np.max(np.abs(fwd[:n] - bwd[:n])))
        rep["per"][tname] = rec
        print("[tja] %-22s in_jerk %.3e | amp %s"
              % (tname, in_j, " ".join("%s:%.3e" % (V.SHORT[m], rec[m]["amp"]) for m in V.METHODS)),
              flush=True)
    Path("/tmp/cov_v2/tja.json").write_text(json.dumps(rep, indent=1))
    print("[tja] wrote /tmp/cov_v2/tja.json", flush=True)


main()
