# -*- coding: utf-8 -*-
"""s-a 判决实验：Δε = ε_ROT − ε_TGA 法向差异分布（法向协变修正支路取舍）。

问题
----
RSB（方法四）候选支路「法向协变修正」用 n1 = normalize(cof(J) n0) = normalize(J^{-T} n0)
替代 R·n0 作为预测法向，再做最短弧对齐。若 RotSync 的旋转法向与严格协变法向的
夹角 Δε 的 p90 < 2 度，则该支路收益太小，砍掉（连同 SPD 测地插值候选）。

本脚本只读张量、只写 /tmp/*.json，不改 lam3d/core 与 LAM/ 任何文件。

口径
----
* 网格：faces_up [F=39904,3]，顶点 x0 [N=20018,3]（LAM 规范高斯中心，与 rot_capture.pt 逐位相同）。
* 逐面 J_f = E_b E^{-1}（proto_m4_lib.face_jacobians，同一实现），面积加权 scatter 到顶点得 J_v。
* ε_ROT 两个来源，都算：
    ROT_M = Markley 聚合（逐面 polar(J_f) → 四元数 → 面积加权 4x4 外积主特征向量）——rot_sync.py 生产口径
    ROT_V = polar(J_v)（先平均 J 再极分解）——VTP 对照口径
* ε_TGA = normalize(J_v^{-T} n0)（float64 torch.linalg.solve，失败回退 pinv）。
* n0 = 规范网格逐面法向面积加权单位化（与 nv 同一条通道，可比）。
* Δε = arccos(clip(dot(ε_ROT, ε_TGA), -1, 1))，度。

判决结论（详见 docs/exp_de_normal_cov.md）
-----------------------------------------
1. m4_base.pt 的 nv **不是**协变法向，而是「形变后曲面法向」nv_surf（ones 上最大差 0.0442°）；
   它既不是 ε_ROT 也不是 ε_TGA。故本脚本把 nv_surf 当作「真实几何」的参照。
2. 真实 ARKit preset 合并 p90(Δε)=1.178° < 2° → 预声明判据满足，砍该支路。
3. 决定性旁证：协变法向相对 nv_surf 的残差反而更大（p90 15.468° vs ROT 14.296°，
   仅 16.5% 顶点更近）→ 该支路是**负收益**，判决不依赖判据宽严。
脚本同时输出这两组证据（branch_identity / res_* / adv_*），并给出 conf 十分位表。

可复现
------
    /usr/local/miniconda3/envs/lam/bin/python /workspace/lam3d/scripts/exp_de_normal_cov.py
    # 强制重算张量（走 LAM 管线约 90 s）：
    /usr/local/miniconda3/envs/lam/bin/python /workspace/lam3d/scripts/exp_de_normal_cov.py --recapture
    # 只读已有缓存（无 GPU 也能跑后处理）：
    /usr/local/miniconda3/envs/lam/bin/python /workspace/lam3d/scripts/exp_de_normal_cov.py --reuse

硬约束遵守：无沿 N 的 Python 循环（全部 matmul/einsum/scatter/index_add）；
dot 后 clip；除零 eps=1e-12；单卡 GPU，主体 <60 s（首帧 capture 除外）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")
sys.path.insert(0, "/workspace/lam3d/scripts")
os.environ.setdefault("LAM_ROOT", "/workspace/LAM")

from lam3d.core.rot_sync import polar_rotation                     # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat  # noqa: E402
import proto_m4_lib as L                                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12
TAU = 0.30          # rot_sync 生产的 conf 门控 tau（docs/method4_rsb_design.md:826）
TH_CAP = 45.0       # rot_sync 生产的逐高斯旋转角上限（度）
VECTORS = ["ones", "half", "extreme", "rand", "rand_sparse",
           "bigSmile", "laugh", "surprise", "mouthOpen"]
CACHE = "/tmp/nc_capture9.pt"
REF_BASE = "/tmp/rot_capture.pt"
OUT_JSON = "/tmp/exp_de_normal_cov.json"


# --------------------------------------------------------------- 数据捕获
def arkit_vectors(meta):
    """按 capture_rot_tensors.py 的原始抽取顺序构造 9 个 ARKit 向量。

    注意 rand_sparse 必须与 rand 共用同一个 rng 流（先 52 次 uniform 抽 rand，
    再在同一流上抽 mask 与幅值），否则与 rot_capture.pt 不一致。
    """
    n = int(meta["count"])
    rng = np.random.default_rng(20261022)
    v = {
        "ones": np.ones(n, np.float32),
        "half": np.full(n, 0.5, np.float32),
        "extreme": np.full(n, 2.0, np.float32),
    }
    v["rand"] = rng.uniform(0.0, 1.0, n).astype(np.float32)
    v["rand_sparse"] = ((rng.uniform(0.0, 1.0, n) > 0.75).astype(np.float32)
                        * rng.uniform(0.0, 1.2, n).astype(np.float32))
    for name in ("bigSmile", "laugh", "surprise", "mouthOpen"):
        vec = np.zeros(n, np.float32)
        for k, val in meta["presets"][name].items():
            vec[meta["names"].index(k)] = float(val)
        v[name] = vec
    return v


def capture(vectors):
    """复刻 gs_renderer.animate_gs_model 的调制分支，抓 x0 / xB 与规范高斯属性。"""
    from lam3d.core.pipeline import SingleImage3DPipeline
    from lam3d.core.tga_covariance import _base_module
    from lam3d.core import arkit_live

    t0 = time.time()
    # LAM 的跟踪阶段要反向传播，这里必须显式开 grad；抓张量本身用 no_grad。
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    with torch.enable_grad():
        pipe.run(image_path="/workspace/LAM/assets/sample_input/messi.png",
                 job_dir="/tmp/cap_nc_job", job_id="capnc")
    sess = pipe.arkit_session
    renderer, fd, gs = sess["renderer"], sess["flame_data"], sess["gs_model"]
    lam = _base_module().DynamicCovarianceModulator(renderer.flame_model).to("cuda").eval()
    v_cano = gs.xyz.detach().unsqueeze(0).float()

    def fwd(off=None):
        return renderer.flame_model.animation_forward(
            v_cano=v_cano, shape=fd["betas"].repeat(1, 1), expr=fd["expr"],
            rotation=fd["rotation"], neck=fd["neck_pose"], jaw=fd["jaw_pose"],
            eyes=fd["eyes_pose"], translation=fd["translation"],
            zero_centered_at_root_node=False, return_landmarks=False,
            return_verts_cano=False, static_offset=None, dynamic_offset=off)["animated"]

    outs = {}
    x0 = fwd(None)
    for nm, vec in vectors.items():
        p = lam.prepare_params(torch.from_numpy(vec).cuda(), 1, "cuda", torch.float32)
        xb = fwd(lam.arkit_offsets(p))[0]
        print("  captured %-12s |xB-x0|=%.5f" % (nm, (xb - x0[0]).abs().mean()), flush=True)
        outs[nm] = xb.cpu()
    payload = {
        "x0": x0[0].cpu(), "xB": torch.stack([outs[k] for k in vectors]),
        "names": list(vectors.keys()), "faces": renderer.flame_model.faces_up.cpu(),
        "gs_quat": gs.rotation.detach().float().cpu(),
        "gs_scale": gs.scaling.detach().float().cpu(),
    }
    torch.save(payload, CACHE)
    print("[capture] %.1fs -> %s" % (time.time() - t0, CACHE), flush=True)
    return payload


def load_capture(reuse):
    if reuse:
        d = torch.load(CACHE, map_location="cpu", weights_only=False)
        print("[capture] reuse %s" % CACHE)
        return d
    from lam3d.core import arkit_live
    print("[capture] building 9-vector capture (LAM pipeline, ~90 s) ...", flush=True)
    return capture(arkit_vectors(arkit_live.arkit_meta()))


# --------------------------------------------------------------- 几何工具
def vsum(pf, w, faces, N, M):
    """[M,F,3] 逐面量 × 面积 → [M,N,3]（index_add，无 N 循环）。"""
    s = torch.zeros(M, N, 3, device=pf.device)
    for k in range(3):
        s.index_add_(1, faces[:, k], pf * w[:, :, None])
    return s


def stats(x, qs=(0.5, 0.9, 0.99)):
    x = x.reshape(-1).double()
    tabs = torch.tensor(list(qs), device=x.device, dtype=torch.float64)
    q = torch.quantile(x, tabs)
    return {"n": int(x.numel()), "mean": float(x.mean()),
            "quantiles": {("%g" % p): float(v) for p, v in zip(qs, q)},
            "p50": float(torch.quantile(x, 0.5)) if 0.5 in qs else None,
            "p90": float(torch.quantile(x, 0.9)) if 0.9 in qs else None,
            "p99": float(torch.quantile(x, 0.99)) if 0.99 in qs else None,
            "max": float(x.max()),
            "frac_gt2deg": float((x > 2.0).double().mean()),
            "frac_gt5deg": float((x > 5.0).double().mean())}


def spearman(a, b):
    def rk(t):
        o = torch.argsort(torch.argsort(t))
        return o.double()
    ra, rb = rk(a.reshape(-1)), rk(b.reshape(-1))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=EPS))


def angle(u, v):
    d = (u * v).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(d) * (180.0 / math.pi)


def slerp_id(q, t):
    q = torch.where(q[..., 0:1] < 0, -q, q).clone()
    ang = 2.0 * torch.acos(q[..., 0].clamp(-1.0, 1.0))
    s = torch.sin(ang).clamp(min=1e-8)
    w = torch.where(ang < 1e-6, torch.ones_like(ang), torch.sin(ang * t) / s)
    return F.normalize(w.unsqueeze(-1) * q, dim=-1)


def rotv(q, v):
    u, w = q[..., 1:], q[..., 0:1]
    return v + 2 * w * torch.cross(u, v, dim=-1) + 2 * torch.cross(u, torch.cross(u, v, dim=-1), dim=-1)


# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="只有缓存时直接复用，不重跑 LAM 管线")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    t_all = time.time()
    torch.set_grad_enabled(False)      # 后续全部是纯前向几何
    d = load_capture(args.reuse)
    x0 = d["x0"].to(DEV).float()
    xB = d["xB"].to(DEV).float()
    faces = d["faces"].to(DEV).long()
    names = list(d["names"])
    M, N = xB.shape[0], x0.shape[0]
    Fn = faces.shape[0]
    print("[data] M=%d N=%d F=%d  vectors=%s" % (M, N, Fn, names), flush=True)

    # 基线对照：与官方 rot_capture.pt 的 5 个向量逐位核对（只读）
    integrity = {}
    if os.path.isfile(REF_BASE):
        ref = torch.load(REF_BASE, map_location="cpu", weights_only=False)
        integrity["x0_maxdiff"] = float((ref["x0"] - d["x0"]).abs().max())
        for nm in ref["names"]:
            if nm in names:
                integrity["xB_maxdiff_%s" % nm] = float(
                    (ref["xB"][ref["names"].index(nm)] - d["xB"][names.index(nm)]).abs().max())
            else:
                integrity["xB_maxdiff_%s" % nm] = None
        integrity["faces_equal"] = bool((ref["faces"] == d["faces"]).all())

    # ---- 张量
    t = time.time()
    area = L.face_areas(x0, faces, M)                                  # [M,F]
    Jf, n1f = L.face_jacobians(x0, xB, faces)                          # [M,F,3,3], [M,F,3]
    Rf = polar_rotation(Jf)                                            # [M,F,3,3]
    Jv9, _ = L.scatter_vertex(Jf, area, faces, N)
    Jv = Jv9.view(M, N, 3, 3)                                          # [M,N,3,3]
    q_face = F.normalize(matrix_to_quat(Rf.reshape(-1, 3, 3)), dim=-1).reshape(M, Fn, 4)
    qR, conf = L.markley(q_face, area, faces, N)                       # [M,N,4], [M,N]
    print("[t] geometry %.1fs" % (time.time() - t), flush=True)

    fa, fb, fc = faces.unbind(-1)
    n0f = F.normalize(torch.cross(x0[fb] - x0[fa], x0[fc] - x0[fa], dim=-1), dim=-1, eps=EPS)
    n0f = n0f.unsqueeze(0).expand(M, -1, -1).contiguous()
    n0 = F.normalize(vsum(n0f, area, faces, N, M), dim=-1, eps=EPS)              # 规范顶点法向
    nv_surf = F.normalize(vsum(n1f, area, faces, N, M), dim=-1, eps=EPS)         # 形变后曲面法向

    # ---- 三个法向
    RvM = quat_to_matrix(F.normalize(qR, dim=-1))                                # RotSync 生产口径
    RvV = polar_rotation(Jv)                                                     # VTP 口径
    n_ROT_M = F.normalize((RvM @ n0.unsqueeze(-1)).squeeze(-1), dim=-1, eps=EPS)
    n_ROT_V = F.normalize((RvV @ n0.unsqueeze(-1)).squeeze(-1), dim=-1, eps=EPS)

    Jvd = Jv.double()
    try:
        sol = torch.linalg.solve(Jvd.transpose(-1, -2), n0.double().unsqueeze(-1)).squeeze(-1)
    except Exception:
        sol = torch.linalg.pinv(Jvd, rtol=1e-6, atol=0.0).transpose(-1, -2) @ n0.double().unsqueeze(-1)
        sol = sol.squeeze(-1)
    n_TGA = F.normalize(sol.float(), dim=-1, eps=EPS)                            # 严格协变法向

    # 恒等式核对：n_TGA 与「形变后曲面法向 nv」的关系（判定 m4_base.nv 属于哪一支）
    cov_check = {
        "angle(n_TGA, nv_surf)": stats(angle(n_TGA, nv_surf), (0.5, 0.9, 0.99)),
        "angle(n_ROT_M, nv_surf)": stats(angle(n_ROT_M, nv_surf), (0.5, 0.9, 0.99)),
        "angle(n0, nv_surf)": stats(angle(n0, nv_surf), (0.5, 0.9, 0.99)),
    }
    # |cof(J) n0| —— 近奇异顶点上 normalize 的放大倍数（数值可靠性）
    cof_norm = sol.norm(dim=-1).float()

    # ---- 生产口径下真正施加的 delta_q（conf 门控 + 45° 限幅 + 单次 slerp）
    qang = 2.0 * torch.atan2(qR[..., 1:].norm(dim=-1), qR[..., 0].abs().clamp(max=1.0))
    th = qang * (180.0 / math.pi)
    g = ((conf - TAU) / (1.0 - TAU)).clamp(0.0, 1.0) ** 1.5                     # docs:826 推荐 p=1.5
    tcap = torch.where(th > 0, TH_CAP / th.clamp(min=1e-9), torch.ones_like(th)).clamp(max=1.0)
    dq = slerp_id(qR, tcap * g)
    n_ROT_APPLIED = F.normalize(rotv(dq, n0), dim=-1, eps=EPS)

    # ---- Δε
    de_M = angle(n_ROT_M, n_TGA)          # 主口径（判据）
    de_V = angle(n_ROT_V, n_TGA)          # VTP 对照
    de_A = angle(n_ROT_APPLIED, n_TGA)    # 门控后实际施加

    # ---- 决定性旁证：两个法向谁离「真实形变后曲面法向 nv_surf」更近？
    #      Δε 只说明两支不同；真正决定支路是否有收益的是「换成协变法向能否减小
    #      与真实几何的残差」。res_TGA < res_ROT 才说明该支路有实际收益。
    res_ROT = angle(n_ROT_M, nv_surf)
    res_TGA = angle(n_TGA, nv_surf)
    adv = res_ROT - res_TGA                      # >0 表示协变法向更准

    per = {}
    for k, nm in enumerate(names):
        per[nm] = {
            "dE_ROT_Markley": stats(de_M[k]),
            "dE_ROT_VertexJPolar": stats(de_V[k]),
            "dE_ROT_applied_gated": stats(de_A[k]),
            "conf": {"mean": float(conf[k].mean()), "p50": float(conf[k].median()),
                     "frac_gt_tau": float((conf[k] > TAU).float().mean())},
            "spearman_dE_conf": spearman(de_M[k], conf[k]),
            "res_ROT_to_nvsurf": stats(res_ROT[k], (0.5, 0.9)),
            "res_TGA_to_nvsurf": stats(res_TGA[k], (0.5, 0.9)),
            "adv_TGA_over_ROT_deg": stats(adv[k], (0.5, 0.9)),
            "frac_TGA_closer": float((adv[k] > 0).float().mean()),
        }
        # 门控后仍在场（g>0）的子集，以及高置信子集
        mask = conf[k] > TAU
        per[nm]["dE_ROT_Markley_gated_subset"] = stats(de_M[k][mask]) if int(mask.sum()) > 8 else None
        hi = conf[k] > 0.70
        per[nm]["dE_ROT_Markley_hi_conf_gt0.70"] = stats(de_M[k][hi]) if int(hi.sum()) > 8 else None

        # conf 十分位上的 Δε p90（看大 Δε 是否集中在低 conf）
        dec = torch.clamp((conf[k] * 10).long(), 0, 9)
        dec_tab = {}
        for b in range(10):
            sel = dec == b
            if int(sel.sum()) > 8:
                dec_tab["%d" % b] = float(torch.quantile(de_M[k][sel].double(), 0.9))
        per[nm]["dE_p90_by_conf_decile"] = dec_tab

    pooled = {
        "dE_ROT_Markley": stats(de_M),
        "dE_ROT_VertexJPolar": stats(de_V),
        "dE_ROT_applied_gated": stats(de_A),
    }
    # 排除外推型压力向量（ones/half/extreme 的逐通道常值并非真实表情）后的合并分布
    real_idx = [names.index(n) for n in ("rand", "rand_sparse", "bigSmile", "laugh", "surprise", "mouthOpen")
                if n in names]
    pooled["dE_ROT_Markley_real_presets_only"] = stats(de_M[real_idx])
    pooled["res_ROT_to_nvsurf"] = stats(res_ROT)
    pooled["res_TGA_to_nvsurf"] = stats(res_TGA)
    pooled["adv_TGA_over_ROT_deg"] = stats(adv)
    pooled["frac_TGA_closer"] = float((adv > 0).float().mean())
    pooled["res_ROT_to_nvsurf_real_only"] = stats(res_ROT[real_idx])
    pooled["res_TGA_to_nvsurf_real_only"] = stats(res_TGA[real_idx])
    pooled["dE_ROT_Markley_hi_conf_real_only"] = stats(
        de_M[real_idx][conf[real_idx] > 0.70])

    # ---- 判据：p90 < 2° → 砍
    CRIT = 2.0
    verdict_rows = {nm: {"p90": per[nm]["dE_ROT_Markley"]["p90"],
                         "cut": bool(per[nm]["dE_ROT_Markley"]["p90"] < CRIT)} for nm in names}
    verdict_pooled_cut = bool(pooled["dE_ROT_Markley"]["p90"] < CRIT)
    verdict_real_cut = bool(pooled["dE_ROT_Markley_real_presets_only"]["p90"] < CRIT)

    # ---- 稳健性：近奇异顶点（|J^-T n0| 极小/极大）与大角度翻转是否主导分布
    degen = (cof_norm < 0.5) | (cof_norm > 2.0)          # 数值可疑区
    flip = (n_ROT_M * n_TGA).sum(-1) < 0                 # 法向翻转（>90°）
    robust = {
        "frac_degen": float(degen.float().mean()),
        "frac_flip_gt90": float(flip.float().mean()),
        "dE_p99_excl_degen": stats(de_M[~degen]),
        "dE_nonflip_p90": stats(de_M[~flip]),
        "per_vector": {},
    }
    for k, nm in enumerate(names):
        robust["per_vector"][nm] = {
            "frac_degen": float(degen[k].float().mean()),
            "frac_flip_gt90": float(flip[k].float().mean()),
            "p99_excl_degen": float(torch.quantile(de_M[k][~degen[k]], 0.99)) if int((~degen[k]).sum()) > 10 else None,
            "p90_nonflip": float(torch.quantile(de_M[k][~flip[k]], 0.90)) if int((~flip[k]).sum()) > 10 else None,
        }

    res = {
        "meta": {"tau": TAU, "th_cap_deg": TH_CAP, "criterion_p90_deg": CRIT,
                 "N": N, "F": Fn, "M": M, "vectors": names, "device": DEV,
                 "wall_sec_compute": round(time.time() - t_all, 1)},
        "integrity_vs_rot_capture": integrity,
        "branch_identity": cov_check,
        "cof_n0_norm": {"p50": float(cof_norm.median()), "p01": float(torch.quantile(cof_norm, 0.01)),
                        "min": float(cof_norm.min())},
        "robustness": robust,
        "per_vector": per,
        "pooled": pooled,
        "verdict": {"criterion": "p90(Δε) < 2 deg -> cut the normal-covariance branch",
                    "per_vector_cut": verdict_rows,
                    "pooled_all_cut": verdict_pooled_cut,
                    "pooled_real_presets_cut": verdict_real_cut},
    }
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)

    # ---- 控制台表
    print()
    print("delta_eps = angle(normalize(R_v n0), normalize(J_v^-T n0))   [deg]")
    print("%-12s %8s %8s %8s %8s %8s | %8s %8s | %8s" %
          ("vector", "p50", "p90", "p99", "max", ">2deg", "p90_VTP", "sp(conf)", "gated_p90"))
    for nm in names:
        s = per[nm]["dE_ROT_Markley"]; sv = per[nm]["dE_ROT_VertexJPolar"]
        gs = per[nm]["dE_ROT_Markley_gated_subset"]
        print("%-12s %8.3f %8.3f %8.3f %8.2f %7.1f%% | %8.3f %8.3f | %8.3f" %
              (nm, s["p50"], s["p90"], s["p99"], s["max"], 100 * s["frac_gt2deg"],
               sv["p90"], per[nm]["spearman_dE_conf"], gs["p90"] if gs else float("nan")))
    print()
    print("who is closer to the true deformed surface normal (nv_surf)?  [deg, p50 / p90]")
    print("%-12s %9s %9s %9s %9s %8s %8s" % ("vector", "ROT p50", "ROT p90", "TGA p50", "TGA p90",
                                             "adv_p50", "TGA<ROT"))
    for nm in names:
        p = per[nm]
        print("%-12s %9.3f %9.3f %9.3f %9.3f %8.3f %7.1f%%" %
              (nm, p["res_ROT_to_nvsurf"]["p50"], p["res_ROT_to_nvsurf"]["p90"],
               p["res_TGA_to_nvsurf"]["p50"], p["res_TGA_to_nvsurf"]["p90"],
               p["adv_TGA_over_ROT_deg"]["p50"], 100 * p["frac_TGA_closer"]))
    print("%-12s %9.3f %9.3f %9.3f %9.3f %8.3f %7.1f%%" %
          ("POOLED", pooled["res_ROT_to_nvsurf"]["p50"], pooled["res_ROT_to_nvsurf"]["p90"],
           pooled["res_TGA_to_nvsurf"]["p50"], pooled["res_TGA_to_nvsurf"]["p90"],
           pooled["adv_TGA_over_ROT_deg"]["p50"], 100 * pooled["frac_TGA_closer"]))
    print()
    print("POOLED all vectors        : Δε p50 %.3f p90 %.3f p99 %.3f | >2deg %.1f%%" %
          (pooled["dE_ROT_Markley"]["p50"], pooled["dE_ROT_Markley"]["p90"],
           pooled["dE_ROT_Markley"]["p99"], 100 * pooled["dE_ROT_Markley"]["frac_gt2deg"]))
    print("POOLED real presets only  : Δε p50 %.3f p90 %.3f p99 %.3f | >2deg %.1f%%" %
          (pooled["dE_ROT_Markley_real_presets_only"]["p50"],
           pooled["dE_ROT_Markley_real_presets_only"]["p90"],
           pooled["dE_ROT_Markley_real_presets_only"]["p99"],
           100 * pooled["dE_ROT_Markley_real_presets_only"]["frac_gt2deg"]))
    print()
    print("%-12s %8.3f %8.3f %8.3f %8.2f %7.1f%%" % ("POOLED", pooled["dE_ROT_Markley"]["p50"],
          pooled["dE_ROT_Markley"]["p90"], pooled["dE_ROT_Markley"]["p99"],
          pooled["dE_ROT_Markley"]["max"], 100 * pooled["dE_ROT_Markley"]["frac_gt2deg"]))
    print()
    print("verdict: pooled_all cut=%s | real_presets_only cut=%s" % (verdict_pooled_cut, verdict_real_cut))
    print("json -> %s   total %.1fs" % (args.out, time.time() - t_all))


if __name__ == "__main__":
    main()
