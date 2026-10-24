#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N1 penetration / N4 coverage Ω / N5 saturate_frac —— 四方法 × 9 preset 试点。

背景（docs/gt_free_eval_protocol.md §1.3）：
  * 四个协方差方法（lam / tga / rot_sync / vertex_then_polar）只改 (q, s)，
    不改高斯中心 ⇒ 任何"中心 vs 网格"的绝对量都不可辨识。
  * N1 因此取**方法间差分**：同一方法下 protrusion(表情) − protrusion(中性)，
    逐高斯；同时用 7 点 SDF 采样给出 penetration_rate / flip_rate 的差分。
  * N4 覆盖率守恒 Ω（防"缩小椭球换低突出"作弊）。
  * N5 45° 硬限幅饱和率（RotSync 专属风险）。

本脚本只新增文件，不修改任何既有 core/scripts 代码。
probe_protrusion.py 的 quat_to_R / pctl 以"源码 exec 且去掉末尾 main() 调用"的
方式复用（该文件没有 if __name__ 守卫，直接 import 会顺带建一次会话）；
它的 protrusion 解析式与本脚本一致，但它**没有** BVH/SDF 实现（全文无 SDF），
所以 7 点 SDF 采样在本脚本内实现：scipy cKDTree(k=8 三角面质心) + 向量化
Ericson 点-三角形最近点 + 伪法向定号，符号约定 d<0 = 网格内部。
相对 trimesh.proximity.signed_distance 的自检见 selfcheck_sdf()。

跑法（必须后台，建会话约 90–160 s）：
  cd /workspace/lam3d && nohup /usr/local/miniconda3/envs/lam/bin/python \
      scripts/exp_n1n4n5.py > /tmp/exp_n1n4n5.log 2>&1 &
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---- LAM 内部会二次 parse sys.argv（flame_tracking_single_image._parse_args），
#      所以先把参数取走并清空（与 probe_protrusion.py 相同的做法）。
_ARGV = list(sys.argv[1:])
sys.argv = [sys.argv[0]]

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, "/workspace/lam3d/scripts")

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

EPS = 1e-12


def to_np(x):
    """torch.Tensor(可能在 cuda) / ndarray -> float64 ndarray。"""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float64)
K_NN = 8                      # 每个查询点取 8 个最近三角面
CHUNK = 5000                  # SDF 分块，控制内存
E_BAR_FALLBACK_MM = 5.308     # probe_protrusion.py 记录的 meshi 局部平均一环边长

METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
SHORT = {"lam": "M-LAM", "tga": "M-TGA", "rot_sync": "M-RS",
         "vertex_then_polar": "M-VTP"}
PRESETS = ["smile", "bigSmile", "laugh", "surprise", "angry", "sad", "kiss",
           "mouthOpen", "blink"]
MAX_ANGLE_DEG = 45.0          # cov_switch.PARAMS 默认 max_angle_deg
SAT_TOL_DEG = 0.5             # |angle - 45| < 0.5° 视为饱和
N4_LO, N4_HI = 0.97, 1.03

OUT_JSON = Path("/tmp/cov_v2/exp_n1n4n5.json")
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# --------------------------------------------------------------------- 复用
def load_probe_functions():
    """复用 probe_protrusion.py 的 quat_to_R / pctl，不触发它的 main()。"""
    p = Path("/workspace/lam3d/scripts/probe_protrusion.py")
    src = p.read_text(encoding="utf-8")
    # 该脚本最后一行是裸 main()（没有 __name__ 守卫）；去掉这一行只取函数。
    cut = src.rstrip().rsplit("\nmain()", 1)
    body = cut[0] if len(cut) == 2 else src
    g = {"__name__": "probe_protrusion_stub", "__file__": str(p)}
    exec(compile(body, str(p), "exec"), g)  # noqa: S102 — 只读复用，不改文件
    return g["quat_to_R"], g["pctl"], g["EPS"] if "EPS" in g else None


quat_to_R, pctl, _ = load_probe_functions()


# --------------------------------------------------------------- SDF 采样器
def _point_tri_dist(P, A, B, C):
    """向量化 Ericson 点到三角形最近点距离。P/A/B/C: [...,3]。"""
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = P - B
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = P - C
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    q = A + v[..., None] * ab + w[..., None] * ac
    # 边/顶点区域钳位（顺序无关，互斥集合）
    t = np.clip(d1 / (d1 - d3 + 1e-30), 0.0, 1.0)
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], A + t[..., None] * ab, q)
    t = np.clip(d2 / (d2 - d6 + 1e-30), 0.0, 1.0)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], A + t[..., None] * ac, q)
    t = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30), 0.0, 1.0)
    q = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
                 B + t[..., None] * (C - B), q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], A, q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], B, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], C, q)
    return np.linalg.norm(P - q, axis=-1), q


class MeshSDF:
    """三角面质心 cKDTree(k=8) + 精确点-三角距离 + 伪法向定号的混合 SDF。

    符号约定：d < 0 表示点在网格内部（与 docs 里 d(p)<0 = embedded 一致；
    trimesh 的 signed_distance 恰好相反，自检里按 -trimesh 比较）。
    """

    def __init__(self, verts, faces, k=K_NN, chunk=CHUNK):
        self.tri = np.ascontiguousarray(verts[faces], dtype=np.float64)
        n = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        nrm = np.linalg.norm(n, axis=1, keepdims=True)
        # 退化面（面积≈0）给一个零法向，定号时用"离质心方向"兜底
        self.fn = n / np.maximum(nrm, 1e-30) * (nrm > 1e-18)
        self.k = int(k)
        self.chunk = int(chunk)
        self.tree = cKDTree(self.tri.mean(axis=1))

    def signed(self, P):
        P = np.ascontiguousarray(P, dtype=np.float64)
        out = np.empty(len(P), dtype=np.float64)
        for i in range(0, len(P), self.chunk):
            Pc = P[i:i + self.chunk]
            _, idx = self.tree.query(Pc, k=self.k)
            T = self.tri[idx]                                   # [m,k,3,3]
            Pd = np.repeat(Pc[:, None, :], self.k, axis=1)      # [m,k,3]
            dd, qq = _point_tri_dist(Pd, T[:, :, 0], T[:, :, 1], T[:, :, 2])
            j = np.argmin(dd, axis=1)
            ar = np.arange(len(Pc))
            dmin = dd[ar, j]
            cp = qq[ar, j]
            nn = self.fn[idx[ar, j]]
            sg = np.sign(((Pc - cp) * nn).sum(-1))
            sg = np.where(np.abs(sg) < EPS, 1.0, sg)
            out[i:i + self.chunk] = dmin * sg
        return out


def selfcheck_sdf(verts, faces, n=300, seed=0):
    """对 trimesh.proximity.signed_distance 做 |d| 与符号自检（可选依赖）。"""
    try:
        import trimesh
    except Exception:  # noqa: BLE001
        return {"ok": None, "reason": "trimesh unavailable"}
    rng = np.random.default_rng(seed)
    lo, hi = verts.min(0), verts.max(0)
    P = lo + (hi - lo) * rng.uniform(size=(n, 3))
    mine = MeshSDF(verts, faces).signed(P)
    from trimesh.proximity import ProximityQuery
    ref = np.asarray(ProximityQuery(trimesh.Trimesh(vertices=verts, faces=faces,
                                                    process=False)).signed_distance(P),
                     dtype=np.float64)
    mag = np.abs(np.abs(mine) - np.abs(ref))
    return {"ok": bool(np.percentile(mag, 99) < 1e-4),
            "sign_agree_vs_neg_trimesh": float((np.sign(mine) == np.sign(-ref)).mean()),
            "absdiff_p99_m": float(np.percentile(mag, 99)),
            "absdiff_max_m": float(mag.max()),
            "n": int(n)}


# ---------------------------------------------------------------- 几何工具
def protrusion(q, s, nrm):
    """probe_protrusion.py 的 protrusion 定义（mm）。

    prot = s_max · sinθ, θ = angle(thin_axis, n), thin = argmin(s)。
    返回 (prot_mm, theta_deg, u_thin, thin_idx)。
    """
    R = quat_to_R(q)                                          # [N,3,3]
    th = np.argmin(s, axis=-1)                                # [N]
    u = R[np.arange(len(s)), :, th]                           # 薄轴（列向量）
    cos = np.clip(np.abs((u * nrm).sum(-1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    prot = s.max(-1) * np.sqrt(np.maximum(1.0 - cos ** 2, 0.0)) * 1000.0
    return prot, ang, u, th


def axis7_points(x, q, s):
    """正文 N1 采样：μ 与 沿三条轴 ±σ 共 7 点。返回 [N,7,3]。"""
    R = quat_to_R(q)
    pts = [x]
    for j in range(3):
        uj = R[:, :, j] * s[:, j:j + 1]
        pts.append(x + uj)
        pts.append(x - uj)
    return np.stack(pts, axis=1)


def thin7_points(x, u_thin, s_thin):
    """协议文档 N1 采样：沿薄轴 k·σ, k=-3..3 共 7 点。返回 [N,7,3]。"""
    k = np.arange(-3, 4, dtype=np.float64)
    return x[:, None, :] + (k[None, :, None] * s_thin[:, None, None]
                            * u_thin[:, None, :])


# ------------------------------------------------------------------- 主流程
def main():
    import torch
    import eval_cov_methods as E
    import eval_cov_protocol_v2 as V
    from lam3d.core import cov_switch, tga_covariance
    from lam3d.core.pipeline import SingleImage3DPipeline

    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rep = {"meta": {"method": "N1/N4/N5", "methods": METHODS, "presets": PRESETS,
                    "max_angle_deg": MAX_ANGLE_DEG, "sat_tol_deg": SAT_TOL_DEG,
                    "eps": EPS, "k_nn": K_NN},
           "per": {}}

    log("[n1n4n5] 建会话 ...")
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=Path("/workspace/LAM/assets/sample_input/messi.png"),
             job_dir="/tmp/v2_job_messi", job_id="v2_messi")
    sess = pipe.arkit_session
    r = sess["renderer"]
    gs = sess["gs_model"]
    fd = sess["flame_data"]
    meta_arkit = V.arkit_live.arkit_meta()
    base = V.tga_covariance._base_module().DynamicCovarianceModulator(r.flame_model)
    base.to("cuda").eval()
    faces = r.flame_model.faces_up.detach().cpu().numpy().astype(np.int64)
    x0t = gs.xyz.detach().unsqueeze(0).float()
    q0t = gs.rotation.detach().float()
    s0t = gs.scaling.detach().float()
    q0 = q0t.detach().cpu().numpy().astype(np.float64)
    s0 = s0t.detach().cpu().numpy().astype(np.float64)
    N = int(q0.shape[0])
    log("[n1n4n5] 会话就绪 %.1fs  N=%d faces=%d" % (time.time() - t0, N, len(faces)))

    # ---- 中性网格：算 e_bar（局部平均一环边长）与 static planar 掩码
    v_neu = E.mesh_vertices(r, fd, E.vector_for(meta_arkit, "zeros"), base)
    tri = v_neu[faces]
    el = np.concatenate([np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
                         np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                         np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)])
    e_bar_mm = float(np.median(el) * 1000.0)
    if not np.isfinite(e_bar_mm) or e_bar_mm <= 0:
        e_bar_mm = E_BAR_FALLBACK_MM
    planar = (s0.max(-1) / np.maximum(s0.min(-1), EPS)) > 2.0
    rep["meta"].update({"e_bar_mm": e_bar_mm, "n_gauss": N, "n_faces": int(len(faces)),
                        "frac_planar": float(planar.mean()),
                        "session_s": round(time.time() - t0, 1)})
    log("[n1n4n5] e_bar=%.3f mm  planar=%.3f" % (e_bar_mm, planar.mean()))

    # ---- SDF 自检（中性网格，300 点）
    rep["sdf_selfcheck"] = selfcheck_sdf(v_neu, faces)
    log("[n1n4n5] SDF selfcheck: %s" % json.dumps(rep["sdf_selfcheck"], ensure_ascii=False))

    # ---- 每个 preset 一份网格 + 一份 SDF（SDF 只依赖网格，与方法是正交的）
    vectors = [("zeros", E.vector_for(meta_arkit, "zeros"))]
    for pv in PRESETS:
        vectors.append((pv, E.vector_for(meta_arkit, pv)))

    grids = {}
    for name, vec in vectors:
        v = v_neu if name == "zeros" else E.mesh_vertices(r, fd, vec, base)
        nrm, _ = E.vertex_normals(v, faces)
        grids[name] = {"verts": v, "nrm": nrm, "sdf": MeshSDF(v, faces)}
        log("[n1n4n5] mesh+SDF %-10s (%.1fs)" % (name, time.time() - t0))

    # ---- 四方法 × {中性 + 9 preset}
    per = {}
    for m in METHODS:
        cov_switch.set_method(m, renderer=r)
        mod = r._cov_modulator
        per[m] = {}
        for name, vec in vectors:
            tn = time.time()
            q1, s1, xb = E.modulate(mod, r, fd, vec, x0t, q0t, s0t)
            q1 = to_np(q1).reshape(N, 4)
            s1 = to_np(s1).reshape(N, 3)
            xb = to_np(xb).reshape(N, 3)
            g = grids[name]
            prot, theta, u_thin, th = protrusion(q1, s1, g["nrm"])
            s_thin = s1[np.arange(N), th]

            # --- N1-a: 轴 7 点采样（任务书口径）
            P7 = axis7_points(xb, q1, s1).reshape(-1, 3)
            d7 = g["sdf"].signed(P7).reshape(N, 7)
            emb = (d7 < 0).mean(axis=1)
            pen_depth = np.maximum(0.0, -d7.min(axis=1)) / (e_bar_mm / 1000.0)
            # 薄轴两端：pts 里 +σ_thin 在 1+2*th，-σ_thin 在 2+2*th
            i_p = 1 + 2 * th
            i_m = 2 + 2 * th
            d_plus = d7[np.arange(N), i_p]
            d_minus = d7[np.arange(N), i_m]
            flip7 = (np.sign(d_plus) != np.sign(d_minus)) & (emb > 0)

            # --- N1-b: 薄轴 ±3σ 7 点采样（协议文档口径）
            Pt = thin7_points(xb, u_thin, s_thin).reshape(-1, 3)
            dt = g["sdf"].signed(Pt).reshape(N, 7)
            emb_t = (dt < 0).mean(axis=1)
            flip_t = ((dt[:, 0] < 0) != (dt[:, 6] < 0)) & (emb_t > 0)

            # --- N4: 高斯中心到网格表面距离（该指标只依赖中心，与 (q,s) 无关）
            dcen = np.abs(g["sdf"].signed(xb))

            # --- N5: 饱和率（只有 rot_sync / vertex_then_polar 有 45° 限幅）
            # 应用转角（与实现无关，直接由 q1 vs q0 反解）：2·acos(|<q1,q0>|)
            dtheta = np.degrees(2.0 * np.arccos(np.clip(
                np.abs((q1 * q0).sum(-1)) / np.maximum(
                    np.linalg.norm(q1, axis=-1) * np.linalg.norm(q0, axis=-1), EPS),
                0.0, 1.0)))
            # 与会中"最大轴"的夹角（用于看限幅是否只作用于极少数）
            diag = getattr(mod, "_last_diag", None)
            if diag is not None and "angle_deg" in diag:
                a_pre = diag["angle_deg"].detach().cpu().numpy().astype(np.float64).reshape(N)
                clipped = diag["clipped"].detach().cpu().numpy().astype(bool).reshape(N)
                a_apply = np.minimum(a_pre, MAX_ANGLE_DEG)
                sat = a_apply >= (MAX_ANGLE_DEG - SAT_TOL_DEG)
            else:
                a_pre = a_apply = sat = None
                clipped = None

            per[m][name] = {
                "prot_mm": prot, "theta_deg": theta, "poe": prot / e_bar_mm,
                "s_thin": s_thin,
                "det_s": s1.prod(-1),
                "d7": d7, "emb7": emb, "pen_depth": pen_depth,
                "flip7": flip7, "emb7t": emb_t, "flip7t": flip_t,
                "d_center": dcen,
                "sat": sat, "clipped": clipped, "a_apply": a_apply, "a_pre": a_pre,
                "dtheta_deg": dtheta,
                "t_s": time.time() - tn,
            }
            log("[n1n4n5] %-18s %-10s prot50 %.2fmm emb7 %.3f flip7 %.4f | %.2fs"
                % (SHORT[m], name, float(np.median(prot[planar])),
                   float(emb.mean()), float(flip7.mean()), time.time() - tn))

    # ------------------------------------------------------------ 汇总统计
    def agg(x, mask=None):
        x = np.asarray(x, np.float64)
        if mask is not None:
            x = x[mask]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {"p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
                "p99": float(np.percentile(x, 99)), "mean": float(x.mean()),
                "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}

    summary = {}
    for m in METHODS:
        neu = per[m]["zeros"]
        rows = {}
        dprot_all, dpoe_all = [], []
        for name, _ in vectors[1:]:
            cur = per[m][name]
            dprot = cur["prot_mm"] - neu["prot_mm"]
            dpoe = dprot / e_bar_mm
            dprot_all.append(dprot)
            dpoe_all.append(dpoe)
            rows[name] = {
                # N1 差分（方法内：表情 − 中性，逐高斯）
                "n1_dprot_mm": agg(dprot, planar),
                "n1_dpoe": agg(dpoe, planar),
                "n1_frac_abs_dpoe_gt1": float((np.abs(dpoe[planar]) > 1.0).mean()),
                "n1_frac_dpoe_neg": float((dpoe[planar] < 0).mean()),
                "n1_d_pen_rate": float((cur["emb7"] > 0.5).mean() - (neu["emb7"] > 0.5).mean()),
                "n1_d_flip_rate": float(cur["flip7"].mean() - neu["flip7"].mean()),
                "n1_pen_rate": float((cur["emb7"] > 0.5).mean()),
                "n1_flip_rate": float(cur["flip7"].mean()),
                "n1_pen_rate_t": float((cur["emb7t"] > 0.5).mean()),
                "n1_flip_rate_t": float(cur["flip7t"].mean()),
                "n1_emb7_mean": float(cur["emb7"].mean()),
                "n1_emb7t_mean": float(cur["emb7t"].mean()),
                "n1_pen_depth": agg(cur["pen_depth"]),
                "n1_theta_deg": agg(cur["theta_deg"], planar),
                "n1_prot_mm": agg(cur["prot_mm"], planar),
                # N4 覆盖率守恒
                "n4_omega_med": float(np.median(cur["d_center"]) / max(np.median(neu["d_center"]), EPS)),
                "n4_omega_mean": float(cur["d_center"].mean() / max(neu["d_center"].mean(), EPS)),
                "n4_omega_det": float(cur["det_s"].sum() / max(neu["det_s"].sum(), EPS)),
                "n4_dcenter_mm": agg(cur["d_center"] * 1000.0),
                # N5 饱和
                "n5_saturate_frac": (None if cur["sat"] is None else float(cur["sat"].mean())),
                "n5_clipped_frac": (None if cur["clipped"] is None else float(cur["clipped"].mean())),
                "n5_angle_apply_deg": (None if cur["a_apply"] is None else agg(cur["a_apply"])),
                "n5_dtheta_deg": agg(cur["dtheta_deg"]),
                "n5_frac_dtheta_ge_44_5": float((cur["dtheta_deg"] >= MAX_ANGLE_DEG - SAT_TOL_DEG).mean()),
                "sec": round(cur["t_s"], 2),
            }
        dprot_cat = np.concatenate(dprot_all, axis=0)
        dpoe_cat = np.concatenate(dpoe_all, axis=0)
        pl_cat = np.tile(planar, len(dprot_all))
        summary[m] = {
            "rows": rows,
            "n1_dpoe_all": agg(dpoe_cat, pl_cat),
            "n1_dprot_all_mm": agg(dprot_cat, pl_cat),
            "n1_frac_abs_dpoe_gt1_all": float((np.abs(dpoe_cat[pl_cat]) > 1.0).mean()),
            "n1_d_pen_rate_mean": float(np.mean([rows[k]["n1_d_pen_rate"] for k in rows])),
            "n1_d_flip_rate_mean": float(np.mean([rows[k]["n1_d_flip_rate"] for k in rows])),
            "n1_d_pen_rate_max": float(np.max([rows[k]["n1_d_pen_rate"] for k in rows])),
            "n1_pen_rate_mean": float(np.mean([rows[k]["n1_pen_rate"] for k in rows])),
            "n1_flip_rate_mean": float(np.mean([rows[k]["n1_flip_rate"] for k in rows])),
            "n4_omega_med_range": [float(min(rows[k]["n4_omega_med"] for k in rows)),
                                   float(max(rows[k]["n4_omega_med"] for k in rows))],
            "n4_omega_det_range": [float(min(rows[k]["n4_omega_det"] for k in rows)),
                                   float(max(rows[k]["n4_omega_det"] for k in rows))],
            "n4_out_of_band_frac_med": float(np.mean(
                [not (N4_LO <= rows[k]["n4_omega_med"] <= N4_HI) for k in rows])),
            "n4_out_of_band_frac_det": float(np.mean(
                [not (N4_LO <= rows[k]["n4_omega_det"] <= N4_HI) for k in rows])),
            "n5_dtheta_p90_max": float(np.max([rows[k]["n5_dtheta_deg"]["p90"] for k in rows])),
            "n5_frac_dtheta_ge_44_5_max": float(np.max([rows[k]["n5_frac_dtheta_ge_44_5"] for k in rows])),
            "n5_saturate_frac_mean": (None if rows[PRESETS[0]]["n5_saturate_frac"] is None else
                                      float(np.mean([rows[k]["n5_saturate_frac"] for k in rows]))),
            "n5_saturate_frac_max": (None if rows[PRESETS[0]]["n5_saturate_frac"] is None else
                                     float(np.max([rows[k]["n5_saturate_frac"] for k in rows]))),
            "n5_frac_presets_fail": (None if rows[PRESETS[0]]["n5_saturate_frac"] is None else
                                     float(np.mean([rows[k]["n5_saturate_frac"] > 0.02 for k in rows]))),
        }
        log("[n1n4n5] == %-18s dpoe p50 %.3f p90 %.3f | d_pen_rate %.4f d_flip %.4f "
            "| omega_det %.4f | sat %.4f"
            % (SHORT[m], summary[m]["n1_dpoe_all"]["p50"], summary[m]["n1_dpoe_all"]["p90"],
               summary[m]["n1_d_pen_rate_mean"], summary[m]["n1_d_flip_rate_mean"],
               float(np.mean([rows[k]["n4_omega_det"] for k in rows])),
               -1.0 if summary[m]["n5_saturate_frac_mean"] is None
               else summary[m]["n5_saturate_frac_mean"]))

    rep["summary"] = summary
    # 中性态基线（四方法应逐位一致，可作为"只改 (q,s) 不改中心"的自检）
    rep["neutral_check"] = {
        "max_abs_diff_q": float(max(np.abs(per[m]["zeros"]["prot_mm"] - per["lam"]["zeros"]["prot_mm"]).max()
                                    for m in METHODS)),
        "max_abs_diff_det_s": float(max(np.abs(per[m]["zeros"]["det_s"] - per["lam"]["zeros"]["det_s"]).max()
                                        for m in METHODS)),
    }
    rep["meta"]["total_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(rep, indent=1))
    log("[n1n4n5] wrote %s  total %.1fs" % (OUT_JSON, time.time() - t0))
    log("[n1n4n5] neutral_check %s" % json.dumps(rep["neutral_check"]))


if __name__ == "__main__":
    main()
