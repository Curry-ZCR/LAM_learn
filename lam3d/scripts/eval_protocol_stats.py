#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Protocol feasibility statistics for a ground-truth-free evaluation.

Three questions this answers, all of which the protocol document needs numbers
for instead of hand-waving:

  S1  What is the effective sample size of a per-gaussian metric?
      20018 gaussians are NOT 20018 independent observations: the one-ring of a
      vertex carries the same local deformation.  We partition the mesh into
      contiguous patches, estimate the intra-class correlation, and report both
      the design effect and the patch-level bootstrap CI versus the naive
      i.i.d. per-gaussian one.  This decides what the unit of analysis is.

  S2  Are the headline differences significant, and how many units would a
      study need?  Paired patch bootstrap for the three discriminating
      contrasts (rigid_err / shape_err / eig_rel_err), plus a power calculation.

  S3  A new metric family that does not need a photograph:
        normal_track   change in the angle between the gaussian's THIN axis and
                       the mesh vertex normal, in degrees, relative to neutral.
                       Negative = the gaussian turns to follow the skin better.
        protrusion     s_max * sin(theta): how far the thin sheet sticks out of
                       the skin along the vertex normal (1-sigma half length),
                       normalised by the local mean edge length.

Run (background, builds the pipeline ~70 s):
  cd /workspace/lam3d && nohup env PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_protocol_stats.py \
    > /tmp/protocol_stats.log 2>&1 &
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/lam3d")
sys.path.insert(0, "/workspace/LAM")

import torch  # noqa: E402

from lam3d.core import cov_switch, rot_sync  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix  # noqa: E402

CAPTURE = "/tmp/rot_capture.pt"
OUT = "/tmp/cov_protocol_stats.json"
NPZ = "/tmp/cov_protocol_pergaussian.npz"
IMAGE = "/workspace/LAM/assets/sample_input/messi.png"
METHODS = ["lam", "tga", "rot_sync", "vertex_then_polar"]
DEV = "cuda"
N_PATCH = 256
N_BOOT = 2000
RNG = np.random.default_rng(0)


def log(*a):
    print(*a, flush=True)


def pct(x, q):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def brief(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    return {"mean": float(x.mean()), "p50": pct(x, 50), "p90": pct(x, 90),
            "p99": pct(x, 99), "max": float(x.max())}


def rot_angle_deg(a, b):
    tr = np.einsum("nij,nij->n", np.swapaxes(a, 1, 2), b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))


def vertex_normals(x, f):
    e1 = x[f[:, 1]] - x[f[:, 0]]
    e2 = x[f[:, 2]] - x[f[:, 0]]
    fn = np.cross(e1, e2)
    vn = np.zeros_like(x)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def kabsch_one_ring(x0, xb, faces):
    n = x0.shape[0]
    f = faces.astype(np.int64)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    p = x0[dst] - x0[src]
    q = xb[dst] - xb[src]
    h = np.zeros((n, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            h[:, i, j] = np.bincount(src, weights=p[:, i] * q[:, j], minlength=n)
    u, _, vt = np.linalg.svd(h.transpose(0, 2, 1))
    r = np.einsum("nji,nkj->nik", vt, u)
    bad = np.linalg.det(r) < 0
    if bad.any():
        vt2 = vt.copy()
        vt2[bad, -1, :] *= -1.0
        r = np.einsum("nji,nkj->nik", vt2, u)
    return r


def build_patches(faces, n_vert, n_patch):
    """Greedy BFS partition of the vertex graph into contiguous patches."""
    f = faces.astype(np.int64)
    nbr = [[] for _ in range(n_vert)]
    for tri in f:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            nbr[a].append(b)
            nbr[b].append(a)
    nbr = [np.unique(np.asarray(v, dtype=np.int64)) if v else np.zeros(0, np.int64)
           for v in nbr]
    taken = np.zeros(n_vert, bool)
    target = max(1, n_vert // n_patch)
    labels = -np.ones(n_vert, np.int64)
    # seeds spread over the vertex index space (the mesh is smoothly ordered)
    seeds = np.linspace(0, n_vert - 1, n_patch).astype(np.int64)
    pid = 0
    for s in seeds:
        if taken[s]:
            continue
        stack = [int(s)]
        labels[s] = pid
        taken[s] = True
        cnt = 1
        while stack and cnt < target:
            v = stack.pop()
            for u in nbr[v]:
                if not taken[u] and cnt < target:
                    taken[u] = True
                    labels[u] = pid
                    stack.append(int(u))
                    cnt += 1
        pid += 1
    # leftovers -> nearest labelled neighbour, else their own patch
    for v in range(n_vert):
        if labels[v] < 0:
            got = [labels[u] for u in nbr[v] if labels[u] >= 0]
            labels[v] = got[0] if got else pid
    return labels, len(np.unique(labels))


def patch_bootstrap_mean(x, labels, n_boot=N_BOOT):
    K = labels.max() + 1
    sums = np.bincount(labels, weights=x, minlength=K)
    cnts = np.bincount(labels, minlength=K).astype(np.float64)
    ok = cnts > 0
    means = sums[ok] / cnts[ok]
    idx = RNG.integers(0, means.size, size=(n_boot, means.size))
    boot = means[idx].mean(axis=1)
    return float(means.mean()), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5))


def patch_paired(x, y, labels, n_boot=N_BOOT):
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    K = labels.max() + 1
    sums = np.bincount(labels, weights=d, minlength=K)
    cnts = np.bincount(labels, minlength=K).astype(np.float64)
    ok = cnts > 0
    pm = sums[ok] / cnts[ok]
    idx = RNG.integers(0, pm.size, size=(n_boot, pm.size))
    boot = pm[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # naive iid-per-gaussian interval for the same difference
    dd = d[np.isfinite(d)]
    se_iid = dd.std(ddof=1) / np.sqrt(dd.size)
    return {
        "diff": float(dd.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_iid_halfwidth": float(1.96 * se_iid),
        "design_effect": float(((hi - lo) / (2 * 1.96)) / se_iid) ** 2,
        "n_patch": int(pm.size),
        "power_n_patch_80pct": float(
            ((1.96 + 0.8416) ** 2) * (pm.std(ddof=1) ** 2)
            / max(dd.mean() ** 2, 1e-30)),
    }


def icc(x, labels):
    K = labels.max() + 1
    sums = np.bincount(labels, weights=x, minlength=K)
    cnts = np.bincount(labels, minlength=K).astype(np.float64)
    ok = cnts > 0
    m = sums[ok] / cnts[ok]
    c = cnts[ok]
    grand = x.mean()
    ssb = float((c * (m - grand) ** 2).sum())
    ssw = float(((x - m[labels]) ** 2).sum())
    dfb, dfw = ok.sum() - 1, x.size - ok.sum()
    msb, msw = ssb / max(dfb, 1), ssw / max(dfw, 1)
    m0 = c.mean()
    denom = msb + (m0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else 0.0


def main():
    cap = torch.load(CAPTURE)
    x0 = cap["x0"].numpy().astype(np.float64)
    BX = cap["xB"].numpy().astype(np.float64)
    names = [str(v) for v in cap["names"]]
    faces = cap["faces"].numpy().astype(np.int64)
    q0 = cap["gs_quat"].numpy().astype(np.float64)
    s0 = cap["gs_scale"].numpy().astype(np.float64)
    n = x0.shape[0]
    log("capture: N=%d F=%d vectors=%s" % (n, faces.shape[0], names))

    pipe = SingleImage3DPipeline(device=DEV, lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir="/tmp/protocol_job", job_id="protocol")
    renderer = pipe.arkit_session["renderer"]
    mods = {m: cov_switch._make(m, renderer, {}) for m in METHODS}
    log("modulators:", {k: type(v).__name__ for k, v in mods.items()})

    labels, K = build_patches(faces, n, N_PATCH)
    sizes = np.bincount(labels)
    log("patches: K=%d  size min/med/max = %d/%d/%d"
        % (K, sizes.min(), int(np.median(sizes)), sizes.max()))

    R0 = quat_to_matrix(torch.from_numpy(q0).float()).numpy()
    D0 = np.square(s0)
    i_min0 = np.argmin(s0, axis=1)
    u_min0 = R0[np.arange(n), :, i_min0]
    s_max0 = s0.max(axis=1)
    n0v = vertex_normals(x0, faces)
    cos0 = np.abs(np.einsum("ni,ni->n", u_min0, n0v))
    theta0 = np.degrees(np.arccos(np.clip(cos0, -1.0, 1.0)))
    prot0 = s_max0 * np.sqrt(np.maximum(0.0, 1.0 - cos0 ** 2))
    # local mesh scale: mean one-ring edge length
    f = faces
    e = np.concatenate([np.linalg.norm(x0[f[:, 1]] - x0[f[:, 0]], axis=1),
                        np.linalg.norm(x0[f[:, 2]] - x0[f[:, 1]], axis=1),
                        np.linalg.norm(x0[f[:, 0]] - x0[f[:, 2]], axis=1)])
    edge = float(e.mean())
    log("neutral: thin-axis vs vertex normal p50=%.2f deg  protrusion/edge p50=%.4f"
        % (pct(theta0, 50), pct(prot0, 50) / edge))
    log("        |cos| with y/x/z axis: %.3f / %.3f / %.3f"
        % (np.abs(u_min0[:, 1]).mean(), np.abs(u_min0[:, 0]).mean(),
           np.abs(u_min0[:, 2]).mean()))

    x0t = torch.from_numpy(x0).float().to(DEV)
    q0t = torch.from_numpy(q0).float().to(DEV)
    s0t = torch.from_numpy(s0).float().to(DEV)
    lam0 = np.sort(D0, axis=-1)
    eye3 = np.eye(3)

    per = {}
    store = {}
    for k, nm in enumerate(names):
        xbt = torch.from_numpy(BX[k]).float().to(DEV)
        r_ref = kabsch_one_ring(x0, BX[k], faces)
        rig_ang = rot_angle_deg(np.broadcast_to(eye3, (n, 3, 3)), r_ref)
        ntv = vertex_normals(BX[k], faces)

        jv = mods["tga"].vertex_jacobians(x0t.unsqueeze(0), xbt).detach()
        jv = jv.cpu().numpy()[0].astype(np.float64)
        jv = np.clip(np.nan_to_num(jv, nan=0.0, posinf=0.0, neginf=0.0), -64, 64)
        sig_exact = (jv * D0[:, None, :]) @ jv.transpose(0, 2, 1)

        log("")
        log("== vector %s   rigid motion p50=%.3f p90=%.3f deg"
            % (nm, pct(rig_ang, 50), pct(rig_ang, 90)))
        log("   %-18s %10s %10s %10s %10s %10s %10s"
            % ("method", "rigid p50", "shape p50", "eig p99",
               "normal d", "prot/edge", "dprot"))
        for m in METHODS:
            with torch.no_grad():
                dq, payload = mods[m].delta_rotation_scale(
                    x0t.unsqueeze(0), xbt.unsqueeze(0))
                scaled = mods[m].delta_scale(payload, s0t)
                q1, s1 = mods[m].apply(dq, scaled, q0t, s0t)
            q1 = q1.detach().float().cpu().numpy()
            s1 = s1.detach().float().cpu().numpy()
            if q1.ndim == 3:
                q1, s1 = q1[0], s1[0]
            q1 = q1.astype(np.float64)
            s1 = s1.astype(np.float64)
            R1 = quat_to_matrix(torch.from_numpy(q1).float()).numpy()
            r_delta = R1 @ np.swapaxes(R0, 1, 2)
            ang = rot_angle_deg(r_ref, r_delta)
            D1 = np.square(s1)
            lam1 = np.sort(D1, axis=-1)
            eig = np.abs(lam1 / np.maximum(lam0, 1e-30) - 1.0)
            sig1 = (R1 * D1[:, None, :]) @ np.swapaxes(R1, 1, 2)
            shape = (np.linalg.norm(sig1 - sig_exact, axis=(1, 2))
                     / np.maximum(np.linalg.norm(sig_exact, axis=(1, 2)), 1e-30))

            i_min1 = np.argmin(s1, axis=1)
            u_min1 = R1[np.arange(n), :, i_min1]
            cos1 = np.abs(np.einsum("ni,ni->n", u_min1, ntv))
            theta1 = np.degrees(np.arccos(np.clip(cos1, -1.0, 1.0)))
            prot1 = s1.max(axis=1) * np.sqrt(np.maximum(0.0, 1.0 - cos1 ** 2))

            d = per.setdefault(m, {}).setdefault(nm, {})
            d["rigid_err"] = brief(ang)
            d["shape_err"] = brief(shape)
            d["eig_rel_err_p99"] = pct(eig, 99)
            d["normal_track_deg"] = brief(theta1 - theta0)
            d["theta_abs_deg"] = brief(theta1)
            d["protrusion_over_edge"] = brief(prot1 / edge)
            d["delta_protrusion_over_edge"] = brief((prot1 - prot0) / edge)
            d["prot_frac_over_0p1edge"] = float(np.mean(prot1 / edge > 0.1))
            d["icc_rigid"] = icc(ang, labels)
            d["icc_shape"] = icc(shape, labels)

            store["%s|%s|rigid" % (m, nm)] = ang
            store["%s|%s|shape" % (m, nm)] = shape
            store["%s|%s|eig" % (m, nm)] = eig
            store["%s|%s|dprot" % (m, nm)] = (prot1 - prot0) / edge
            log("   %-18s %10.3f %10.4f %10.2e %10.2f %10.4f %+10.4f"
                % (m, d["rigid_err"]["p50"], d["shape_err"]["p50"],
                   d["eig_rel_err_p99"], d["normal_track_deg"]["mean"],
                   d["protrusion_over_edge"]["p50"],
                   d["delta_protrusion_over_edge"]["mean"]))

    # ---- S1/S2: design effect and paired contrasts on the strongest vector
    key = "extreme" if "extreme" in names else names[0]
    stats = {"patch_K": K, "patch_size_med": int(np.median(sizes)),
             "edge_len": edge, "vector": key,
             "icc": {}, "paired": {}}
    for m in METHODS:
        stats["icc"][m] = {"rigid": icc(store["%s|%s|rigid" % (m, key)], labels),
                           "shape": icc(store["%s|%s|shape" % (m, key)], labels)}
    pairs = [("tga", "rot_sync"), ("rot_sync", "lam"), ("lam", "rot_sync")]
    for a, b in pairs:
        for met in ("rigid", "shape"):
            xa = store["%s|%s|%s" % (a, key, met)]
            xb = store["%s|%s|%s" % (b, key, met)]
            stats["paired"]["%s-%s|%s" % (a, b, met)] = patch_paired(xa, xb, labels)
    for met in ("rigid", "shape"):
        for m in METHODS:
            mu, lo, hi = patch_bootstrap_mean(
                store["%s|%s|%s" % (m, key, met)], labels)
            stats.setdefault("mean_ci", {}).setdefault(m, {})[met] = {
                "mean": mu, "lo": lo, "hi": hi}
    # ranking stability across the five vectors
    stats["ranking"] = {}
    for met in ("rigid_err", "shape_err", "delta_protrusion_over_edge"):
        order = []
        for nm in names:
            vals = {m: per[m][nm][met]["mean"] for m in METHODS}
            order.append(sorted(vals, key=vals.get))
        stats["ranking"][met] = order

    out = {"methods": METHODS, "vectors": names, "patch": {
        "K": K, "median_size": int(np.median(sizes)), "edge_len": edge},
        "neutral": {"theta_p50": pct(theta0, 50), "prot_p50": pct(prot0, 50) / edge},
        "per": per, "stats": stats}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(NPZ, labels=labels, **store)
    log("")
    log("saved %s and %s" % (OUT, NPZ))


if __name__ == "__main__":
    main()
