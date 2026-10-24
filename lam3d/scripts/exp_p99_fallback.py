#!/usr/bin/env python
"""exp_p99_fallback.py -- p99 tail fallback study for lam3d inference-time covariance modulation.

Question (task s-b): can any per-gaussian rotation fallback push the rigid-error p99
below the pre-declared G2 threshold of 60 deg on all five presets?

Arms
  F0  RotSync as-is (no fallback)                                     -- control
  F1  coherence_gate: delta_q <- slerp(identity, delta_q, t), t = clamp((R-tau)/(1-tau), 0, 1)
  F2  fall back to the LAM delta rotation where coherence is low
  F3  fall back to VTP (vertex_then_polar) rotation where coherence is low
  F4  divergence gate: t = clamp(1 - angle(method, rot_sync)/theta, 0, 1)     (extra)
  F5  global constant blend alpha*rot_sync + (1-alpha)*vtp                    (extra)
  ORACLE  per-gaussian min over {lam, rot_sync, vtp}     (unreachable upper bound)

Everything is offline and fully vectorised: no per-gaussian python loop, no LAM session.
The tensors it consumes were captured once from a real GPU session
(lam3d.core.cov_switch._make on a live SingleImage3DPipeline renderer); see --stage capture.
The offline stage takes about 30 s on CPU.

Metric
  rigid_err[i] = angle(r_ref[i], r_delta[i]), r_delta = R1 @ R0^T, exactly the convention of
  lam3d/scripts/eval_cov_offline.py. See angle_npz() for the transposition subtlety: the
  kabsch helper returns R^T and the angle helper computes tr(a.b) instead of tr(a^T.b), so the
  two transpositions cancel and the pair is self-consistent. Never mix one half with the other
  -- doing so silently doubles the reported angle in the tail.

Usage
  python exp_p99_fallback.py                     # offline, CPU only
  python exp_p99_fallback.py --stage splitting   # GSDeformer splitting estimate, CPU only
  python exp_p99_fallback.py --stage capture     # re-run the GPU capture (needs a free GPU)
  python exp_p99_fallback.py --cache-dir DIR --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PRESETS = ("ones", "half", "extreme", "rand", "rand_sparse")
METHODS = ("lam", "rot_sync", "vertex_then_polar")
# 0.3/0.5/0.7 are the task's grid; 0.9/0.95/0.99 are added to show the saturation of the gate,
# because tau -> 1 is the only way a gate can approach "use the fallback verbatim everywhere".
TAUS = (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
SPLIT_BEND_DEG = 5.0
G2_DEG = 60.0
P50_TOL = 0.5

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CACHE = os.path.join(REPO, "experiments", "p99_fallback", "cache")
DEFAULT_OUT = os.path.join(REPO, "experiments", "p99_fallback")


# ----------------------------------------------------------------------------------
# vectorised geometry helpers
# ----------------------------------------------------------------------------------
def mat2quat(R):
    """Batched matrix -> (w,x,y,z) quaternion, sign-normalised so that w >= 0."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    eps = 1e-12

    def norm(q):
        return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)

    t0 = np.sqrt(np.maximum(tr + 1.0, eps)) * 2
    q0 = np.stack([0.25 * t0, (m21 - m12) / t0, (m02 - m20) / t0, (m10 - m01) / t0], 1)
    t1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, eps)) * 2
    q1 = np.stack([(m21 - m12) / t1, 0.25 * t1, (m01 + m10) / t1, (m02 + m20) / t1], 1)
    t2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, eps)) * 2
    q2 = np.stack([(m02 - m20) / t2, (m01 + m10) / t2, 0.25 * t2, (m12 + m21) / t2], 1)
    t3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, eps)) * 2
    q3 = np.stack([(m10 - m01) / t3, (m02 + m20) / t3, (m12 + m21) / t3, 0.25 * t3], 1)
    cand = np.stack([q0, q1, q2, q3], 1)
    best = np.argmax(np.stack([tr, m00, m11, m22], 1), 1)
    q = norm(cand[np.arange(cand.shape[0]), best])
    return np.where(q[:, :1] < 0, -q, q)


def quat2mat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def angle_npz(a, b):
    """The exact convention of lam3d/scripts/eval_cov_offline.py:72-74.

    tr = sum_ij a_ji b_ij = tr(a @ b), NOT tr(a^T b). For true rotations this equals the
    textbook geodesic angle between a and b^T, so it is only a real angle when one operand is
    transposed. kabsch() below returns R^T (that is what eval_cov_offline.py:77-118 produces),
    which is why this pairing is self-consistent. If you feed a correctly oriented reference
    here, transpose it first, or the reported angle comes out roughly doubled in the tail.
    """
    tr = np.einsum("nij,nij->n", np.swapaxes(a, 1, 2), b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))


def slerp(qa, qb, t):
    """Per-row slerp; t is a per-row weight, broadcast from shape (N,) into [0, 1].

    The interpolation angle is the angle *between the two unit quaternions*, i.e.
    arccos(|qa . qb|), which is half the rotation angle between the two orientations. Using the
    rotation angle here instead (2 * arccos) gets the endpoints and the midpoint right but
    distorts every intermediate t, so for t = clamp((coh - tau) / (1 - tau)) it would silently
    change the meaning of tau. For qa = identity this must reproduce the shipped
    rot_sync.slerp_identity, which is Exp(t * Log q); see check_slerp_identity().
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    d = np.abs(np.einsum("ni,ni->n", qa, qb))
    om = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(om)
    par = s < 1e-7
    w = np.where(par, 1.0 - t, np.sin((1.0 - t) * om) / np.maximum(s, 1e-12))
    u = np.where(par, t, np.sin(t * om) / np.maximum(s, 1e-12))
    out = w[:, None] * qa + u[:, None] * qb
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def check_slerp_identity(n=4096, seed=0):
    """Self-check: slerp(identity, q, t) must equal Exp(t * Log q), the shipped slerp_identity.

    Returns the max absolute deviation in degrees between angle(slerp(identity, q, t)) and
    t * angle(q). A wrong interpolation angle shows up here as a large, t-dependent error.
    """
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    th = rng.uniform(0.0, np.pi, size=n)
    q = np.concatenate([np.cos(0.5 * th)[:, None], ax * np.sin(0.5 * th)[:, None]], axis=1)
    ident = np.zeros((n, 4))
    ident[:, 0] = 1.0
    worst = 0.0
    for t in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        R = quat2mat(slerp(ident, q, np.full(n, t)))
        got = angle_npz(R, np.repeat(np.eye(3)[None], n, axis=0))
        worst = max(worst, float(np.abs(got - t * np.degrees(th)).max()))
    return worst


def stats(e):
    q = np.percentile(e, [50, 90, 99])
    return dict(
        p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), mx=float(e.max()),
        n60=int((e > G2_DEG).sum()), n90=int((e > 90.0).sum()),
        g2_pass=bool(q[2] <= G2_DEG),
    )


# ----------------------------------------------------------------------------------
# topology helpers
# ----------------------------------------------------------------------------------
def directed_edges(faces, n_verts):
    """Both directions of every face edge, deduplicated as a (src, dst) pair set."""
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    key = np.unique(src.astype(np.int64) * n_verts + dst.astype(np.int64))
    return key // n_verts, key % n_verts


def ring_edges(src, dst, n_verts, ring):
    """Expand a 1-ring directed edge set to the given number of hops, dropping self edges.

    Only the hop count is looped over (three iterations at most); the vertex dimension is
    handled entirely by gather/scatter, so this stays cheap on 20k vertices.
    """
    if ring <= 1:
        return src, dst
    deg = np.bincount(src, minlength=n_verts).astype(np.int64)
    order = np.argsort(src, kind="stable")
    nbr = dst[order]
    starts = np.zeros(n_verts, dtype=np.int64)
    starts[1:] = np.cumsum(deg)[:-1]
    s, d = src, dst
    for _ in range(ring - 1):
        reps = deg[d]
        head = np.repeat(s, reps)
        group_start = np.repeat(np.cumsum(reps) - reps, reps)
        offs = np.arange(head.size) - group_start
        tail = nbr[np.repeat(starts[d], reps) + offs]
        ns = np.concatenate([s, head])
        nd = np.concatenate([d, tail])
        keep = ns != nd
        key = np.unique(ns[keep].astype(np.int64) * n_verts + nd[keep].astype(np.int64))
        s, d = key // n_verts, key % n_verts
    return s, d


def kabsch(x_rest, x_def, src, dst, n_verts):
    """Per-vertex least-squares rotation over a directed edge set.

    Reproduces eval_cov_offline.py:77-118 verbatim, including the fact that the returned
    matrix is the transpose of the rotation mapping rest edges onto deformed edges.
    """
    p = x_rest[dst] - x_rest[src]
    q = x_def[dst] - x_def[src]
    H = np.stack(
        [np.stack([np.bincount(src, weights=p[:, i] * q[:, j], minlength=n_verts)
                   for j in range(3)], 1) for i in range(3)], 1)
    u, _, vt = np.linalg.svd(np.swapaxes(H, 1, 2))
    r = np.einsum("nji,nkj->nik", vt, u)
    bad = np.linalg.det(r) < 0
    if bad.any():
        vt2 = vt.copy()
        vt2[bad, -1, :] *= -1
        r = np.einsum("nji,nkj->nik", vt2, u)
    return r


def rigid_residual(x_rest, x_def, R, src, dst, n_verts):
    """|q - R p| / |p| per edge, reduced to max and mean per vertex.

    A large value means the neighbourhood admits no single rotation at all -- the mechanism
    that defeats every rotation-only fallback in the tail.
    """
    p = x_rest[dst] - x_rest[src]
    q = x_def[dst] - x_def[src]
    Rp = np.einsum("nij,nj->ni", R[src], p)
    res = np.linalg.norm(q - Rp, axis=1) / np.maximum(np.linalg.norm(p, axis=1), 1e-30)
    mx = np.zeros(n_verts)
    np.maximum.at(mx, src, res)
    tot = np.zeros(n_verts)
    np.add.at(tot, src, res)
    cnt = np.maximum(np.bincount(src, minlength=n_verts), 1)
    return mx, tot / cnt


def vertex_rotations(x_rest, x_def, faces, n_verts):
    """Vertex-then-polar rotation: per-face 2D polar rotation, area-weighted Markley mean.

    This is the VTP arm; it also feeds the local-rotation-gradient proxy below.
    """
    f = faces
    e1 = x_rest[f[:, 1]] - x_rest[f[:, 0]]
    e2 = x_rest[f[:, 2]] - x_rest[f[:, 0]]
    cr = np.cross(e1, e2)
    nlen = np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(nlen, 1e-30)[:, None]
    b1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-30)[:, None]
    b2 = np.cross(nrm, b1)

    def coords(P):
        d1 = P[f[:, 1]] - P[f[:, 0]]
        d2 = P[f[:, 2]] - P[f[:, 0]]
        return (np.stack([np.einsum("fi,fi->f", d1, b1), np.einsum("fi,fi->f", d1, b2)], 1),
                np.stack([np.einsum("fi,fi->f", d2, b1), np.einsum("fi,fi->f", d2, b2)], 1))

    c1, c2 = coords(x_rest)
    d1, d2 = coords(x_def)
    P = np.stack([c1, c2], 2)
    Q = np.stack([d1, d2], 2)
    A = np.einsum("fij,fjk->fik", Q, np.linalg.inv(P))
    u, _, vt = np.linalg.svd(A)
    R2 = np.einsum("fij,fjk->fik", u, vt)
    th = np.arctan2(R2[:, 1, 0], R2[:, 0, 0])
    area = 0.5 * nlen
    qf = np.stack([np.cos(th / 2), np.sin(th / 2) * nrm[:, 0],
                   np.sin(th / 2) * nrm[:, 1], np.sin(th / 2) * nrm[:, 2]], 1)
    wq = np.einsum("fi,fj,f->fij", qf, qf, area)
    M = np.empty((n_verts, 4, 4))
    flat = f.reshape(-1)
    for i in range(4):
        for j in range(4):
            M[:, i, j] = np.bincount(flat, weights=np.repeat(wq[:, i, j], 3), minlength=n_verts)
    _, vec = np.linalg.eigh(M)
    return quat2mat(vec[:, :, -1]), th, P


def splitting_proxy(s0, Rv, src, dst, x_rest):
    """Crude local-rotation-gradient proxy for the GSDeformer splitting criterion.

    GSDeformer (arXiv 2405.15491) splits a gaussian along an axis when, after deformation, the
    two half-axes (x1'-c') and (x2'-c') subtend an angle below a threshold (175 deg, skipping
    axes shorter than 1e-2); it halves that axis and rescales the other two so that volume is
    preserved (k1*k2*k3 = (1/2)^(1/3)). We have no general non-rigid deformation field for this
    asset, so instead of the MVC criterion this computes 2*s0*|d angle(R_src, R_dst)| / |edge|:
    the end-to-end bend of an axis induced by the per-vertex rotation field. It OVER-estimates
    splitting (a gaussian is larger than an edge in this asset) and is only an order-of-magnitude
    check, not an implementation of the paper.
    """
    dth = angle_npz(Rv[src], Rv[dst])
    L = np.linalg.norm(x_rest[dst] - x_rest[src], axis=1)
    grad = dth / np.maximum(L, 1e-9)
    gmax = np.zeros(Rv.shape[0])
    np.maximum.at(gmax, src, grad)
    return 2.0 * s0 * gmax[:, None], gmax


# ----------------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------------
DIVERGENCE_THETAS = (30.0, 60.0, 90.0, 120.0)


def tau_tag(tau):
    """Stable arm-name suffix for a gate threshold: 0.3 -> "0.3", 0.95 -> "0.95".

    %.1f would collapse 0.95 and 0.99 onto the same "1.0" key and silently overwrite an arm.
    """
    return ("%.2f" % float(tau)).rstrip("0").rstrip(".")


def build_arms(Rref, Rd, coh):
    """Return {arm_name: rotation matrices (N,3,3)} for every fallback arm.

    F1 shrinks rot_sync toward identity, F2/F3 blend a reference method in where the coherence
    gate opens, F4 swaps in a method when it disagrees with rot_sync by less than theta, F5 is a
    global constant blend. All blends go through the quaternion geodesic.
    """
    N = Rref.shape[0]
    Qd = {m: mat2quat(Rd[m]) for m in METHODS}
    ident = np.zeros((N, 4))
    ident[:, 0] = 1.0
    arms = {"F0_rotsync": Rd["rot_sync"]}
    for tau in TAUS:
        t = np.clip((coh - tau) / max(1.0 - tau, 1e-9), 0.0, 1.0)
        arms["F1_gate_tau%s" % tau_tag(tau)] = quat2mat(slerp(ident, Qd["rot_sync"], t))
        arms["F2_lam_tau%s" % tau_tag(tau)] = quat2mat(slerp(Qd["lam"], Qd["rot_sync"], t))
        arms["F3_vtp_tau%s" % tau_tag(tau)] = quat2mat(slerp(Qd["vertex_then_polar"], Qd["rot_sync"], t))
    for m, tag in (("lam", "F4_lam"), ("vertex_then_polar", "F4_vtp")):
        dis = angle_npz(Rd[m], Rd["rot_sync"])
        for th in DIVERGENCE_THETAS:
            t = np.clip(1.0 - dis / th, 0.0, 1.0)
            arms["%s_th%d" % (tag, int(th))] = quat2mat(slerp(Qd[m], Qd["rot_sync"], t))
    for a in ALPHAS:
        arms["F5_global_alpha%.1f" % a] = quat2mat(
            slerp(Qd["vertex_then_polar"], Qd["rot_sync"], np.full(N, a)))
    return arms


# F5 with alpha 0 or 1 is not a fallback at all, it is one of the baseline methods verbatim.
DEGENERATE_ARMS = ("F5_global_alpha0.0", "F5_global_alpha1.0")

# arms whose tail vertex ids get dumped, so step 3 of the task is reproducible per scheme
TAIL_ID_ARMS = tuple(
    ["F0_rotsync"]
    + ["F1_gate_tau%s" % tau_tag(t) for t in TAUS]
    + ["F2_lam_tau%s" % tau_tag(t) for t in TAUS]
    + ["F3_vtp_tau%s" % tau_tag(t) for t in TAUS]
    + ["ORACLE3_best_of_three"]
)


NAMED_PREFIXES = ("F1", "F2", "F3", "F4")


def _rank_rows(per_preset_errs, presets, skip_degenerate, only_named=False):
    base_p50 = {p: float(np.percentile(per_preset_errs[p]["F0_rotsync"], 50)) for p in presets}
    rows = []
    for a in sorted(per_preset_errs[presets[0]]):
        if a.startswith("ORACLE"):
            continue
        if skip_degenerate and a in DEGENERATE_ARMS:
            continue
        if only_named and a != "F0_rotsync" and not a.startswith(NAMED_PREFIXES):
            continue
        worst = max(float(np.percentile(per_preset_errs[p][a], 99)) for p in presets)
        dmax = max(float(np.percentile(per_preset_errs[p][a], 50)) - base_p50[p] for p in presets)
        rows.append(dict(arm=a, worst_p99=worst, p50_delta_max=dmax, feasible=bool(dmax <= P50_TOL)))
    rows.sort(key=lambda r: (not r["feasible"], r["worst_p99"]))
    return rows


def select_best(per_preset_errs, presets):
    """Rank the arms by worst-preset p99, splitting off the degenerate global-blend endpoints.

    F5 with alpha = 0 is literally the vertex_then_polar baseline and alpha = 1 is literally
    rot_sync, so they cannot count as a fallback mechanism. We therefore report two winners:
    best_arm (any non-oracle arm, degenerate endpoints excluded) and best_any_arm (endpoints
    included, which the monotone global-blend sweep will win by simply abandoning rot_sync).

    Returns (best_named_arm, best_arm, best_any_arm, rows, rows_any, rows_named).
    best_named_arm is the headline answer: the best of the three schemes the task actually
    names (F1/F2/F3, plus the F4 extra), excluding the F5 diagnostic sweep. A row is
    (arm, worst_p99, feasible, p50_delta_max).
    """
    rows_named = _rank_rows(per_preset_errs, presets, skip_degenerate=True, only_named=True)
    rows = _rank_rows(per_preset_errs, presets, skip_degenerate=True)
    rows_any = _rank_rows(per_preset_errs, presets, skip_degenerate=False)
    return (rows_named[0]["arm"], rows[0]["arm"], rows_any[0]["arm"],
            rows, rows_any, rows_named)


def g2_table(errs, presets, taus=TAUS):
    """Per-arm G2 verdict: rigid p99 <= 60 deg on every preset, p50 not worse by more than 0.5 deg."""
    methods = ("F0_rotsync", "F1", "F2", "F3")
    table = {}
    for m in methods:
        if m == "F0_rotsync":
            names = ["F0_rotsync"]
        else:
            names = [k for k in errs[presets[0]] if k.startswith(m + "_") and "tau" in k]
        for nm in names:
            per = {}
            for p in presets:
                e = errs[p][nm]
                per[p] = dict(p99=float(np.percentile(e, 99)), p50=float(np.percentile(e, 50)),
                              n60=int((e > G2_DEG).sum()))
            p50d = {p: per[p]["p50"] - float(np.percentile(errs[p]["F0_rotsync"], 50)) for p in presets}
            table[nm] = dict(
                per_preset=per,
                pass_all=bool(all(per[p]["p99"] <= G2_DEG for p in presets)),
                p50_delta_max=float(max(p50d.values())),
                p50_ok=bool(max(p50d.values()) <= P50_TOL),
            )
    return table


# ----------------------------------------------------------------------------------
# offline stage
# ----------------------------------------------------------------------------------
def load_cache(cache_dir, presets):
    cap = np.load(os.path.join(cache_dir, "rot_capture.npz"))
    data = dict(x0=cap["x0"], xB=cap["xB"], faces=cap["faces"].astype(np.int64),
                gs_scale=cap["gs_scale"], gs_quat=cap["gs_quat"],
                names=[str(v) for v in cap["names"]])
    per = {}
    for p in presets:
        d = {}
        for m in METHODS:
            d[m] = np.load(os.path.join(cache_dir, "p99_rdel_%s_%s.npy" % (m, p)))
        d["rref"] = np.load(os.path.join(cache_dir, "p99_rref_%s.npy" % p))
        d["coh"] = np.load(os.path.join(cache_dir, "p99_cohreal_%s.npy" % p)).reshape(-1)
        per[p] = d
    npz_path = os.path.join(cache_dir, "cov_protocol_pergaussian.npz")
    npz = np.load(npz_path) if os.path.exists(npz_path) else None
    return data, per, npz


def arm_diagnostics(Rref, Rd, coh, x0, xB, faces, s0, gs, tail):
    """Reference stability, non-rigidity, coherence description and splitting proxy for one preset."""
    N = x0.shape[0]
    src, dst = gs["src"], gs["dst"]
    diag = {}

    # -- reference stability: the 1-ring Kabsch reference itself moves when the ring grows
    refs, ring_sizes = {}, {}
    for r in (1, 2, 3):
        s, d = ring_edges(src, dst, N, r)
        ring_sizes[r] = int(s.size)
        refs[r] = kabsch(x0, xB, s, d, N)
    diag["reference_stability"] = {
        "ref1_vs_ref2": stats(angle_npz(refs[1], refs[2])),
        "ref2_vs_ref3": stats(angle_npz(refs[2], refs[3])),
        "rotsync_vs_ref2": stats(angle_npz(refs[2], Rd["rot_sync"])),
        "vtp_vs_ref2": stats(angle_npz(refs[2], Rd["vertex_then_polar"])),
        "lam_vs_ref2": stats(angle_npz(refs[2], Rd["lam"])),
        # ring r means every directed pair within graph distance r (self pairs dropped), which is
        # wider than the "keep the source at 1-ring, extend only the destination" construction,
        # so the counts are 1x / 3x / 6x the 1-ring size rather than 1x / 2x / 3x
        "edge_counts": {str(r): ring_sizes[r] for r in ring_sizes},
    }

    # -- non-rigidity: does a single rotation exist for this neighbourhood at all
    Rv, th, _ = vertex_rotations(x0, xB, faces, N)
    mx, mean = rigid_residual(x0, xB, Rref, src, dst, N)
    diag["nonrigid"] = dict(
        resid_p50=float(np.median(mx)), resid_p99=float(np.percentile(mx, 99)),
        resid_max=float(mx.max()),
        head_resid_p50=float(np.median(mx[~tail])),
        tail_resid_p50=float(np.median(mx[tail])) if tail.any() else None,
        tail_frac_resid_gt_0p2=float((mx[tail] > 0.2).mean()) if tail.any() else None,
        tail_frac_resid_gt_0p5=float((mx[tail] > 0.5).mean()) if tail.any() else None,
    )

    # -- coherence description: is the gate a selector or a global shrink
    diag["coherence"] = dict(
        p1=float(np.percentile(coh, 1)), p50=float(np.median(coh)),
        frac_lt_0p3=float((coh < 0.3).mean()), frac_lt_0p5=float((coh < 0.5).mean()),
        frac_lt_0p7=float((coh < 0.7).mean()), frac_lt_1=float((coh < 1.0).mean()),
        mean_t={tau_tag(tau): float(np.mean(np.clip((coh - tau) / max(1.0 - tau, 1e-9), 0, 1)))
                for tau in TAUS},
        # a gate only "falls back" where it fires; tau -> 1 makes t -> 1 almost everywhere, so
        # the arm degenerates into the fallback method verbatim. This is the saturation number
        # that separates a real fallback from a rename of VTP.
        frac_fully_gated={tau_tag(tau): float(np.mean(
            np.clip((coh - tau) / max(1.0 - tau, 1e-9), 0, 1) >= 0.99)) for tau in TAUS},
        tail_coh_p50=float(np.median(coh[tail])) if tail.any() else None,
        tail_frac_coh_lt_0p7=float((coh[tail] < 0.7).mean()) if tail.any() else None,
    )

    # -- applied rotation size: how far each method actually rotates a gaussian away from rest.
    # RotSync/VTP are capped by max_angle_deg=45, LAM is not, so this separates "the method is
    # shy" from "the method is wrong" when reading the fallback numbers.
    ident = np.repeat(np.eye(3)[None], N, axis=0)
    diag["applied_deg"] = {
        m: dict(p50=float(np.median(a)), p99=float(np.percentile(a, 99)), mx=float(a.max()),
                frac_at_cap=float((a >= 44.99).mean()))
        for m, a in ((m, angle_npz(Rd[m], ident)) for m in METHODS)
    }

    # -- splitting proxy and how far from rigid the tail is
    bend, _ = splitting_proxy(s0, Rv, src, dst, x0)
    split_axis = bend > SPLIT_BEND_DEG
    diag["splitting"] = dict(
        face_angle_p50=float(np.median(np.abs(th))), face_angle_p99=float(np.percentile(np.abs(th), 99)),
        bend_p50=float(np.median(bend)), bend_p99=float(np.percentile(bend, 99)),
        split_frac_per_axis=[float(x) for x in split_axis.mean(0)],
        split_any=float(split_axis.any(1).mean()),
        split_any_in_tail=float(split_axis[tail].any(1).mean()) if tail.any() else None,
        exp_children=float(np.prod(1.0 + split_axis.mean(0))),
    )
    # how many halvings would be needed to bring the median tail residual down to the head level
    head = diag["nonrigid"]["head_resid_p50"]
    tmed = diag["nonrigid"]["tail_resid_p50"]
    if tmed and head > 0:
        ratio = tmed / head
        nsplit = float(np.log2(ratio)) if ratio > 1 else 0.0
        diag["splitting"]["resid_tail_over_head"] = float(ratio)
        diag["splitting"]["splits_needed"] = nsplit
        diag["splitting"]["children_needed"] = float(2.0 ** nsplit)
    return diag


def run_offline(cache_dir, out_dir, presets=PRESETS, write_tail_ids=True):
    os.makedirs(out_dir, exist_ok=True)
    data, per, npz = load_cache(cache_dir, presets)
    N = data["x0"].shape[0]

    # shared topology, computed once
    src, dst = directed_edges(data["faces"], N)
    gs = dict(src=src, dst=dst)
    print("[topology] %d directed 1-ring edges over %d vertices" % (src.size, N), flush=True)

    errs, blocks = {}, {}
    for p in presets:
        d = per[p]
        Rref, Rd, coh = d["rref"], {m: d[m] for m in METHODS}, d["coh"]
        arms = build_arms(Rref, Rd, coh)
        e = {a: angle_npz(Rref, R) for a, R in arms.items()}
        base = e["F0_rotsync"]
        tail = base > G2_DEG

        # oracle upper bounds: perfect per-gaussian selection among the three methods
        Em = {m: angle_npz(Rref, Rd[m]) for m in METHODS}
        e["ORACLE3_best_of_three"] = np.minimum(np.minimum(Em["lam"], Em["rot_sync"]),
                                                Em["vertex_then_polar"])
        e["ORACLE2_best_of_rs_vtp"] = np.minimum(Em["rot_sync"], Em["vertex_then_polar"])

        diag = arm_diagnostics(Rref, Rd, coh, data["x0"], data["xB"][data["names"].index(p)],
                              data["faces"], data["gs_scale"], gs, tail)

        # cross-check against the frozen protocol npz where it is comparable
        xcheck = {}
        if npz is not None:
            for m in METHODS:
                k = "%s|%s|rigid" % (m, p)
                if k in npz.files:
                    ref = npz[k]
                    xcheck[m] = dict(npz=stats(ref), here=stats(Em[m]),
                                     match=False)
                    xcheck[m]["match"] = bool(
                        abs(xcheck[m]["npz"]["p50"] - xcheck[m]["here"]["p50"]) < 0.02
                        and abs(xcheck[m]["npz"]["p99"] - xcheck[m]["here"]["p99"]) < 0.05)

        errs[p] = e
        blocks[p] = dict(
            stats={a: stats(errs[p][a]) for a in sorted(errs[p])},
            diagnostics=diag,
            tail_count=int(tail.sum()),
            tail_frac=float(tail.mean()),
            npz_crosscheck=xcheck,
        )
        print("[%s] F0 p50=%.2f p99=%.2f n60=%d | ORACLE3 p99=%.2f"
              % (p, blocks[p]["stats"]["F0_rotsync"]["p50"], blocks[p]["stats"]["F0_rotsync"]["p99"],
                 blocks[p]["tail_count"], blocks[p]["stats"]["ORACLE3_best_of_three"]["p99"]),
              flush=True)

    best_named_arm, best_arm, best_any_arm, rank, rank_any, rank_named = select_best(errs, presets)
    g2 = g2_table(errs, presets)

    # which arms actually clear the bar on all five presets
    passing = sorted([a for a, v in g2.items() if v["pass_all"] and v["p50_ok"]])
    passing_non_degenerate = [a for a in passing if a not in DEGENERATE_ARMS]

    if write_tail_ids:
        for p in presets:
            for arm in TAIL_ID_ARMS:
                idx = np.flatnonzero(errs[p][arm] > G2_DEG).astype(np.int64)
                np.save(os.path.join(out_dir, "tail_ids_%s_%s.npy" % (p, arm)), idx)

    res = dict(
        meta=dict(
            script="scripts/exp_p99_fallback.py",
            presets=list(presets), methods=list(METHODS), taus=list(TAUS),
            alphas=list(ALPHAS), divergence_thetas=list(DIVERGENCE_THETAS),
            split_bend_deg=SPLIT_BEND_DEG, g2_deg=G2_DEG, p50_tol=P50_TOL,
            n_vertices=int(N), n_faces=int(data["faces"].shape[0]),
            n_edge_directed=int(src.size),
            metric="rigid_err = angle(r_ref, R1 R0^T) in the eval_cov_offline.py convention",
            cache_dir=os.path.abspath(cache_dir),
        ),
        presets=blocks,
        best_named_arm=best_named_arm,
        best_named_arm_stats={p: blocks[p]["stats"][best_named_arm] for p in presets},
        best_named_arm_p50_delta_max=max(g2[best_named_arm]["p50_delta_max"], 0.0)
        if best_named_arm in g2 else None,
        arm_ranking_named=rank_named[:8],
        best_arm=best_arm,
        best_arm_stats={p: blocks[p]["stats"][best_arm] for p in presets},
        best_arm_p50_delta_max=max(g2[best_arm]["p50_delta_max"], 0.0) if best_arm in g2 else None,
        best_any_arm=best_any_arm,
        best_any_arm_stats={p: blocks[p]["stats"][best_any_arm] for p in presets},
        best_any_arm_is_degenerate=bool(best_any_arm in DEGENERATE_ARMS),
        arm_ranking=rank[:12],
        arm_ranking_including_degenerate=rank_any[:6],
        g2=g2,
        g2_arms_passing=passing,
        g2_arms_passing_non_degenerate=passing_non_degenerate,
        g2_verdict=("PASS" if passing else "FAIL"),
        g2_criterion=("rigid p99 <= %.0f deg on all %d presets, p50 not worse than F0 by more than "
                      "%.1f deg" % (G2_DEG, len(presets), P50_TOL)),
    )
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("\nbest of the three named schemes: %s  (worst-preset p99 %.2f deg)"
          % (best_named_arm, max(v["p99"] for v in res["best_named_arm_stats"].values())))
    print("best non-degenerate fallback arm: %s  (worst-preset p99 %.2f deg, p50 delta max %.2f deg)"
          % (best_arm, max(v["p99"] for v in res["best_arm_stats"].values()),
             res["best_arm_p50_delta_max"] or 0.0))
    print("best arm overall (degenerate endpoints allowed): %s  (worst-preset p99 %.2f deg)"
          % (best_any_arm, max(v["p99"] for v in res["best_any_arm_stats"].values())))
    print("arms evaluated: %d   arms clearing G2: %s"
          % (len(rank_any), passing if passing else "none"))
    print("G2 verdict: %s" % res["g2_verdict"])
    print("wrote %s" % path)
    return res


# ----------------------------------------------------------------------------------
# splitting stage: the GSDeformer criterion applied to this asset (literature check only)
# ----------------------------------------------------------------------------------
def _smooth_displacement_field(x_rest, disp, k=8, power=2.0):
    """Inverse-distance blend of the per-vertex displacement field, in rest coordinates.

    Returns f(P) -> deformed P. Unlike a per-face barycentric interpolation this has no
    triangle-to-triangle kink, and unlike a per-vertex rotation it does not assume the
    deformation is locally rigid. It is still a proxy for GSDeformer's mean-value-coordinate
    cage deformation, because this asset has no coarse cage; but it is driven by the true
    deformed vertex positions, so it is the most faithful field available here.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(x_rest)

    def f(P):
        d, idx = tree.query(P, k=k)
        w = 1.0 / np.maximum(d, 1e-12) ** power
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        return P + np.einsum("mk,mki->mi", w, disp[idx])

    return f


def splitting_criterion(x_rest, axes, f, threshold_deg=175.0):
    """GSDeformer's axis-splitting test (arXiv 2405.15491) under an arbitrary deformation field.

    For each gaussian take the centre c and the two endpoints c +/- a_j of each principal axis.
    After deformation the two half-axes are a1 = f(c+a_j) - f(c) and a2 = f(c-a_j) - f(c); the
    axis is split when the angle between them falls below the threshold (the paper uses 175 deg
    and skips axes shorter than 1e-2, which never triggers on this asset: median axis 1.2e-3).
    Splitting one axis in two with the other two rescaled by k keeps the volume, k1*k2*k3 =
    (1/2)^(1/3), so the child count of one split is the number of split axes times k^3-weighted.
    Here we only count children, i.e. the multiplicative gaussian inflation, not the geometry.
    """
    c = x_rest
    cp = f(c)
    frac, aperture = [], []
    for j in range(3):
        a1 = f(c + axes[:, :, j]) - cp
        a2 = f(c - axes[:, :, j]) - cp
        n1 = np.linalg.norm(a1, axis=1)
        n2 = np.linalg.norm(a2, axis=1)
        cosv = np.einsum("ni,ni->n", a1, a2) / np.maximum(n1 * n2, 1e-30)
        ang = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
        frac.append(float((ang < threshold_deg).mean()))
        aperture.append(180.0 - ang)
    frac = np.asarray(frac)                 # (3,) split fraction per axis
    aperture = np.stack(aperture, 1)        # (N,3) degrees of axis aperture loss
    return dict(
        split_frac_per_axis=[float(v) for v in frac],
        split_any=float(1.0 - np.prod(1.0 - frac)),
        exp_children=float(np.prod(1.0 + frac)),
        bend_p50=float(np.median(aperture)), bend_p99=float(np.percentile(aperture, 99)),
        bend_max=float(aperture.max()),
        bend_per_gaussian=aperture.max(1),      # worst axis of each gaussian, (N,), degrees
        k_per_split=float(0.5 ** (1.0 / 3.0)),
    )


def recursive_splits(x_rest, axes, f, threshold_deg=175.0, rounds=6):
    """Apply the splitting rule repeatedly until it converges, and count the children.

    GSDeformer allows a gaussian to be split more than once. Each split halves one axis; when the
    two children are uniformly shrunk copies of the parent, 2 * k^3 = 1 gives k = 2^(-1/3), which
    is the factor the paper reports as slightly better than exact per-axis rescaling. The shrunk
    child is retested on the next round, so the only loop here is over rounds, never over
    gaussians.

    The convergence flag matters: on a smooth field the aperture of a small axis is dominated by
    the local *curvature* of the field, which does not shrink as fast as the axis does, so some
    gaussians keep splitting every round. The 2^n count is then a property of the round cap, not
    of the geometry, and must not be quoted as a cost. The honest single-round number is
    exp_children from splitting_criterion().
    """
    k = 0.5 ** (1.0 / 3.0)       # each child is a uniformly shrunk copy: 2*k^3 = 1
    N = x_rest.shape[0]
    scale = np.ones(N)           # uniform linear scale of the current children
    nsplits = np.zeros(N, dtype=np.int64)
    cp = f(x_rest)
    first_frac = np.zeros(N)
    last_hit = np.zeros(N, dtype=bool)
    any_split = False
    for _ in range(rounds):
        any_split = False
        for j in range(3):
            half = axes[:, :, j] * scale[:, None]
            a1 = f(x_rest + half) - cp
            a2 = f(x_rest - half) - cp
            n1 = np.linalg.norm(a1, axis=1)
            n2 = np.linalg.norm(a2, axis=1)
            cosv = np.einsum("ni,ni->n", a1, a2) / np.maximum(n1 * n2, 1e-30)
            ang = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
            hit = ang < threshold_deg
            if nsplits.sum() == 0:
                first_frac += hit / 3.0
            if hit.any():
                any_split = True
                last_hit |= hit
                nsplits += hit
                scale = np.where(hit, scale * k, scale)
        if not any_split:
            break
    children = 2.0 ** nsplits.astype(np.float64)
    return dict(children_mean=float(children.mean()), children_p50=float(np.median(children)),
                children_p99=float(np.percentile(children, 99)),
                children_max=float(children.max()),
                frac_split_any=float((nsplits > 0).mean()),
                mean_splits_per_gaussian=float(nsplits.mean()),
                rounds_used=int(rounds),
                # a criterion that has not converged after the round cap is a local flag, not a
                # cost model: the count then depends on the budget, not on the geometry
                converged=bool(not any_split),
                frac_still_splitting=float(last_hit.mean()),
                # first round only: 3 * first_frac is the number of split axes of that gaussian
                children_if_capped_at_1_round=float(np.mean(2.0 ** (3.0 * first_frac))))


def run_splitting(cache_dir, out_dir, presets=PRESETS):
    """Quantify how much gaussian inflation the GSDeformer splitting rule would cost here.

    Literature evaluation only: nothing is implemented and no arm is added to the fallback
    study. Two fields are evaluated because they bracket the answer -- the per-vertex rotation
    field (assumes local rigidity, under-estimates bending) and the blended true displacement
    field (smooth, no rigidity assumption).
    """
    from scipy.spatial import cKDTree

    os.makedirs(out_dir, exist_ok=True)
    cap = np.load(os.path.join(cache_dir, "rot_capture.npz"))
    x0 = cap["x0"].astype(np.float64)
    s0 = cap["gs_scale"].astype(np.float64)
    q0 = cap["gs_quat"].astype(np.float64)
    faces = cap["faces"].astype(np.int64)
    names = [str(v) for v in cap["names"]]
    N = x0.shape[0]
    axes = quat2mat(q0) * s0[:, None, :]
    src, dst = directed_edges(faces, N)

    out = {}
    print("%-11s %-8s %-8s %-8s %-9s %-8s" % ("preset", "rot_ch", "disp_ch", "split_any",
                                              "tail_bend", "fixable"), flush=True)
    for nm in presets:
        xB = cap["xB"][names.index(nm)].astype(np.float64)
        disp = xB - x0
        # (a) per-vertex rotation field, anchored at each point's own nearest rest vertex:
        #     assumes the neighbourhood is rigid, so it can only bend an axis through the
        #     rotation *difference* between the two endpoints. This is a LOWER bound.
        Rv, _, _ = vertex_rotations(x0, xB, faces, N)
        kt = cKDTree(x0)

        def f_rot(P, Rv=Rv, kt=kt):
            v = kt.query(P)[1]
            return x0[v] + np.einsum("nij,nj->ni", Rv[v], P - x0[v])

        r_rot = splitting_criterion(x0, axes, f_rot)
        # (b) smooth blended true displacement field -- the headline number
        f_disp = _smooth_displacement_field(x0, disp)
        r_disp = splitting_criterion(x0, axes, f_disp)

        # does the footprint argument actually explain the fallback tail? The tail is defined by
        # the F0 rot_sync error; if the whole axis span of those gaussians already sweeps more
        # than 60 deg of rotation, no per-gaussian rotation can describe them and only splitting
        # (a smaller footprint) can. "fixable" counts tail gaussians whose error would drop below
        # the G2 threshold after subtracting the full aperture bend, an optimistic upper bound.
        Rref = np.load(os.path.join(cache_dir, "p99_rref_%s.npy" % nm))
        Rs = np.load(os.path.join(cache_dir, "p99_rdel_rot_sync_%s.npy" % nm))
        err = angle_npz(Rref, Rs)
        tail = err > G2_DEG
        span = r_disp["bend_per_gaussian"]
        fixable = float(((err[tail] - span[tail]) <= G2_DEG).mean()) if tail.any() else None
        out[nm] = dict(
                       rotation_field={k: v for k, v in r_rot.items()
                                       if k != "bend_per_gaussian"},
                       displacement_field={k: v for k, v in r_disp.items()
                                           if k != "bend_per_gaussian"},
                       axis_len_p50=float(np.median(np.linalg.norm(axes, axis=1))),
                       disp_p50=float(np.median(np.linalg.norm(disp, axis=1))),
                       tail_n=int(tail.sum()),
                       tail_span_p50=float(np.median(span[tail])) if tail.any() else None,
                       tail_span_p10=float(np.percentile(span[tail], 10)) if tail.any() else None,
                       span_gt60_frac=float((span > G2_DEG).mean()),
                       tail_err_p50=float(np.median(err[tail])) if tail.any() else None,
                       tail_fixable_frac=fixable)
        # the quantity that decides whether a footprint can be described by one rotation at all:
        # the skinning rotation difference across the gaussian diameter, which is what a gaussian
        # of this size actually straddles. Independent of any interpolation field, so it is the
        # cleanest diagnostic here.
        strad = []
        for j in range(3):
            v1 = kt.query(x0 + axes[:, :, j])[1]
            v2 = kt.query(x0 - axes[:, :, j])[1]
            strad.append(angle_npz(Rv[v1], Rv[v2]))
        strad = np.stack(strad, 1)
        out[nm]["straddle_deg"] = dict(
            p50=[float(np.median(strad[:, j])) for j in range(3)],
            p90=float(np.percentile(strad, 90)), mx=float(strad.max()),
            frac_axis_gt_5=float((strad > 5.0).mean()),
            frac_axis_gt_45=float((strad > 45.0).mean()),
            frac_gauss_gt_45=float((strad > 45.0).any(1).mean()),
            frac_gauss_gt_60=float((strad > G2_DEG).any(1).mean()))

        rec = recursive_splits(x0, axes, f_disp)
        out[nm]["recursive_splitting"] = rec
        print("%-11s %-8.4f %-8.4f %-8.4f %-9.2f %-8s  rec: mean %.3f max %.0f splits %.2f"
              % (nm, r_rot["exp_children"], r_disp["exp_children"], r_disp["split_any"],
                 out[nm]["tail_span_p50"] or float("nan"),
                 "%.3f" % fixable if fixable is not None else "-",
                 rec["children_mean"], rec["children_max"], rec["mean_splits_per_gaussian"]),
              flush=True)

    path = os.path.join(out_dir, "splitting.json")
    with open(path, "w") as fh:
        json.dump(dict(meta=dict(
            paper="GSDeformer, arXiv 2405.15491", threshold_deg=175.0,
            note="literature evaluation only, no arm is added to the fallback study",
            fields="rotation_field (lower bound) / displacement_field (headline)"),
            presets=out), fh, indent=1)
    print("wrote %s" % path)
    return out


# ----------------------------------------------------------------------------------
# GPU capture stage (only needed to refresh the cache; about 90 s, needs a free GPU)
# ----------------------------------------------------------------------------------
def run_capture(cache_dir, presets=PRESETS):
    """Re-run the one-and-only GPU session and overwrite the cache tensors.

    This drives the real lam3d pipeline rather than reimplementing it, so the captured
    R1 R0^T tensors are exactly what the inference code would apply. It needs the LAM
    checkpoints and a free GPU; everything else in this file is CPU-only.
    """
    import torch
    for p in ("/workspace/lam3d", "/workspace/LAM"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lam3d.core.pipeline import SingleImage3DPipeline
    from lam3d.core import cov_switch
    from lam3d.core.tga_covariance import quat_to_matrix

    os.makedirs(cache_dir, exist_ok=True)
    pipe = SingleImage3DPipeline(device="cuda", lazy_load=True)
    pipe.run(image_path=os.path.join("/workspace/LAM/assets/sample_input/messi.png"),
             job_dir="/tmp/p99_job", job_id="p99")
    renderer = pipe.arkit_session["renderer"]
    mods = {m: cov_switch._make(m, renderer, {}) for m in METHODS}

    cap = np.load(os.path.join(cache_dir, "rot_capture.npz"))
    x0 = torch.as_tensor(cap["x0"], dtype=torch.float32)
    q0 = torch.as_tensor(cap["gs_quat"], dtype=torch.float32)
    s0 = torch.as_tensor(cap["gs_scale"], dtype=torch.float32)
    R0 = quat_to_matrix(q0).detach().cpu().numpy()
    names = [str(v) for v in cap["names"]]
    for k, nm in enumerate(names):
        xB = torch.as_tensor(cap["xB"][k], dtype=torch.float32)
        for m in METHODS:
            dq, payload = mods[m].delta_rotation_scale(x0.unsqueeze(0), xB.unsqueeze(0))
            sc = mods[m].delta_scale(payload, s0)
            q1, _ = mods[m].apply(dq, sc, q0, s0)
            R1 = quat_to_matrix(q1[0].float()).detach().cpu().numpy()
            np.save(os.path.join(cache_dir, "p99_rdel_%s_%s.npy" % (m, nm)),
                    R1 @ np.swapaxes(R0, 1, 2))
        coh = mods["rot_sync"]._last_diag["coherence"].detach().cpu().numpy()
        np.save(os.path.join(cache_dir, "p99_cohreal_%s.npy" % nm), coh)
        print("[capture] %s done" % nm, flush=True)

    x0n = cap["x0"].astype(np.float64)
    faces = cap["faces"].astype(np.int64)
    src, dst = directed_edges(faces, x0n.shape[0])
    for k, nm in enumerate(names):
        Rref = kabsch(x0n, cap["xB"][k].astype(np.float64), src, dst, x0n.shape[0])
        np.save(os.path.join(cache_dir, "p99_rref_%s.npy" % nm), Rref)
    print("[capture] reference rotations written", flush=True)


# ----------------------------------------------------------------------------------
# reference-stability stage: why the rigid metric has a tail at all
# ----------------------------------------------------------------------------------
REF_STABILITY_DEG = 5.0


def run_reference_stability(cache_dir, out_dir, presets=PRESETS):
    """Quantify how much of the rigid metric's tail is a property of the reference itself.

    rigid_err compares the modulator's rotation against r_ref, the 1-ring Kabsch rotation of the
    rest frame. That reference is estimated from a median of 6 directed edges, so it is itself an
    estimator with a wide variance: the same vertex scored against the 2-ring Kabsch rotation
    moves by tens of degrees. This stage measures that instability and, more importantly, checks
    whether the tail of the metric is *contained* in it. If every gaussian with rigid_err > 60 deg
    also has an unstable reference, then the 60 deg bar is not measuring modulator quality there.
    """
    data, per, npz = load_cache(cache_dir, presets)
    x0 = data["x0"].astype(np.float64)
    faces = data["faces"].astype(np.int64)
    N = x0.shape[0]
    src, dst = directed_edges(faces, N)
    rings = {r: ring_edges(src, dst, N, r) for r in (1, 2, 3)}
    out = {}

    def _ref(r, xB):
        s, d = rings[r]
        return kabsch(x0, xB, s, d, N)

    print("%-12s %-9s %8s %8s %6s %6s %6s" %
          ("preset", "probe", "inst_p50", "inst_p90", "tail", "tailX", "p99|stbl"), flush=True)
    for p in presets:
        xB = data["xB"][data["names"].index(p)].astype(np.float64)
        R1, R2, R3 = _ref(1, xB), _ref(2, xB), _ref(3, xB)
        inst12 = angle_npz(R1, R2)
        inst13 = angle_npz(R1, R3)
        d = per[p]
        Rref = d["rref"]
        arms = build_arms(Rref, {m: d[m] for m in METHODS}, d["coh"])
        errs = {a: angle_npz(Rref, R) for a, R in arms.items()}

        probes = {}
        for tag, inst in (("r1_vs_r2", inst12), ("r1_vs_r3", inst13)):
            stable = inst < REF_STABILITY_DEG
            # the frozen protocol npz is scored against its own reference, so evaluate it too:
            # if the containment holds for both, it is a property of the metric, not of our rerun
            sources = {"here_" + a: e for a, e in errs.items()}
            for m in METHODS:
                k = "%s|%s|rigid" % (m, p)
                if k in npz.files:
                    sources["npz_" + m] = npz[k].reshape(-1)
            rows = {}
            for a, e in sources.items():
                tail = e > G2_DEG
                rows[a] = dict(
                    n_tail=int(tail.sum()),
                    # tails that survive an "only keep stable references" filter: with the
                    # inversion fixed, this is the count the containment claim rests on
                    n_tail_on_stable=int((tail & stable).sum()),
                    p99=float(np.percentile(e, 99)),
                    p99_on_stable=float(np.percentile(e[stable], 99)) if stable.any() else None,
                    p90_on_stable=float(np.percentile(e[stable], 90)) if stable.any() else None,
                    p50_on_stable=float(np.percentile(e[stable], 50)) if stable.any() else None,
                    p50=float(np.percentile(e, 50)))
            probes[tag] = dict(
                inst_p50=float(np.median(inst)), inst_p90=float(np.percentile(inst, 90)),
                frac_stable=float(stable.mean()),
                tail_contained_in_unstable=bool(
                    all(v["n_tail_on_stable"] == 0 for v in rows.values())),
                max_tails_surviving_stable_filter=int(
                    max(v["n_tail_on_stable"] for v in rows.values())),
                arms=rows)
            outside = max(v["n_tail_on_stable"] for v in rows.values())
            covered = all(v["n_tail_on_stable"] == 0 for v in rows.values())
            print("%-12s %-9s %8.2f %8.2f %6d %8.2f  (%d arms, max %d tails survive a "
                  "stable-reference filter, %s)" %
                  (p, tag, float(np.median(inst)), float(np.percentile(inst, 90)),
                   rows["here_F0_rotsync"]["n_tail"],
                   rows["here_F0_rotsync"]["p99_on_stable"] or float("nan"),
                   len(rows), outside,
                   "contained" if covered else "NOT contained"), flush=True)

        # how aggressive a stability filter would have to be before the tail survives it
        sweep = []
        for th in (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0):
            keep = inst12 < th
            tail = errs["F0_rotsync"] > G2_DEG
            sweep.append(dict(th=th, frac_kept=float(keep.mean()),
                              tail_kept=int(tail[keep].sum()),
                              tail_rate_among_kept=float(tail[keep].mean()) if keep.any() else None))

        # why the reference is unstable: neighbourhood size and conditioning of the 1-ring fit
        deg1 = np.bincount(src, minlength=N)
        deg2 = np.bincount(rings[2][0], minlength=N)
        pe = x0[src] - x0[dst]
        H = np.zeros((N, 3, 3))
        np.add.at(H, (src, slice(None), slice(None)), np.einsum("ni,nj->nij", pe, pe))
        sv = np.linalg.svd(H, compute_uv=False)
        cond = sv[:, 0] / np.maximum(sv[:, 2], 1e-30)
        unstable = inst12 >= REF_STABILITY_DEG
        mech = dict(
            deg1_p50=int(np.median(deg1)), deg1_min=int(deg1.min()), deg1_max=int(deg1.max()),
            deg2_p50=int(np.median(deg2)),
            cond_p50=float(np.median(cond)), cond_p90=float(np.percentile(cond, 90)),
            cond_p99=float(np.percentile(cond, 99)),
            corr_err_vs_inst=float(np.corrcoef(np.log1p(errs["F0_rotsync"]),
                                               np.log1p(inst12))[0, 1]),
            p99_stable=float(np.percentile(errs["F0_rotsync"][~unstable], 99)),
            p99_unstable=float(np.percentile(errs["F0_rotsync"][unstable], 99)),
        )
        out[p] = dict(probes=probes, stability_sweep=sweep, mechanism=mech)

    res = dict(
        meta=dict(
            stage="reference_stability", script="scripts/exp_p99_fallback.py",
            stability_threshold_deg=REF_STABILITY_DEG, g2_deg=G2_DEG,
            presets=list(presets), n_vertices=int(N),
            definition=("a gaussian's reference is unstable when the 1-ring and 2-ring Kabsch "
                        "rotations of the same rest neighbourhood differ by >= %.0f deg"
                        % REF_STABILITY_DEG),
        ),
        presets=out,
    )
    path = os.path.join(out_dir, "refstability.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)

    print("")
    for p in presets:
        m = out[p]["mechanism"]
        print("%-12s F0 p99 on stable refs %.2f deg | p99 on unstable %.2f deg | corr(log err, log "
              "instability) %.3f" % (p, m["p99_stable"], m["p99_unstable"], m["corr_err_vs_inst"]))
    print("wrote %s" % path)
    return res


# ----------------------------------------------------------------------------------
# local rotation smoothness: is the "1-ring face rotations contradict each other" premise true
# ----------------------------------------------------------------------------------
def face_frames(x, faces):
    """Orthonormal per-face frame (b1 = edge1, b2 = normal, b3 = b1 x b2) as columns.

    The frame must be ORTHONORMAL. Using the raw (e1, e2, e1 x e2) columns instead makes the
    "rotation" F1 F0^-1 a non-orthogonal matrix whose angle is a meaningless 115-160 deg for
    nearly every face, which is exactly the artefact that a 176 deg pairwise-angle diagnosis
    would report.
    """
    e1 = x[faces[:, 1]] - x[faces[:, 0]]
    e2 = x[faces[:, 2]] - x[faces[:, 0]]
    n = np.cross(e1, e2)
    b1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-30)[:, None]
    b2 = n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    b3 = np.cross(b1, b2)
    b3 /= np.maximum(np.linalg.norm(b3, axis=1), 1e-30)[:, None]
    return np.stack([b1, b2, b3], 2)


def face_rotations(x_rest, x_def, faces):
    """Per-face rotation, plus the rest/deformed areas used to weight it."""
    R = np.einsum("fij,fkj->fik", face_frames(x_def, faces), face_frames(x_rest, faces))
    bad = np.linalg.det(R) < 0
    if bad.any():
        raise RuntimeError("face frame rotation has negative determinant: %d faces" % bad.sum())
    e1 = x_rest[faces[:, 1]] - x_rest[faces[:, 0]]
    e2 = x_rest[faces[:, 2]] - x_rest[faces[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    return R, area


def _auc(pos, neg):
    """Rank-based AUC without scipy: fraction of (pos, neg) pairs where pos > neg."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # average ranks over ties so the statistic stays unbiased on discrete values
    uniq, inv, cnts = np.unique(allv, return_inverse=True, return_counts=True)
    tie = cnts > 1
    if tie.any():
        sums = np.bincount(inv, weights=ranks, minlength=uniq.size)
        ranks = (sums / cnts)[inv]
    return (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def vertex_incidence(faces, n_verts):
    """Padded vertex -> incident-face table: (pad[N,K], valid[N,K], deg[N], K).

    Pad entries are -1 and must be clamped before indexing ("np.maximum(pad, 0)").
    """
    vflat = faces.reshape(-1)
    fflat = np.repeat(np.arange(faces.shape[0]), 3)
    order = np.argsort(vflat, kind="stable")
    vs, fs = vflat[order], fflat[order]
    cnt = np.bincount(vs, minlength=n_verts)
    K = int(cnt.max())
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    off = np.arange(vs.size) - np.repeat(start, cnt)
    pad = np.full(n_verts * K, -1, dtype=np.int64)
    pad[np.repeat(np.arange(n_verts), cnt) * K + off] = fs
    pad = pad.reshape(n_verts, K)
    return pad, pad >= 0, cnt, K


def incident_face_angles(R_face, faces, n_verts):
    """Per-vertex median pairwise angle among the rotations of its incident faces.

    This is the direct test of the claim that tail vertices are vertices whose 1-ring face
    rotations disagree pairwise by ~176 deg. Fully vectorised: incidence is padded to (N, K)
    and all K*(K-1)/2 pairs are evaluated with one einsum.
    """
    pad, valid, cnt, K = vertex_incidence(faces, n_verts)
    iu = np.triu_indices(K, 1)
    vmask = valid[:, iu[0]] & valid[:, iu[1]]
    Rp = R_face[np.maximum(pad, 0)]
    tr = np.einsum("naji,nbji->nab", Rp, Rp)
    ang = np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
    a = np.where(vmask, ang[:, iu[0], iu[1]], np.nan)
    a.sort(axis=1)
    nv = vmask.sum(1)
    # true median: average the two middle values when the pair count is even
    lo = np.clip((nv - 1) // 2, 0, None)
    hi = np.clip(nv // 2, 0, None)
    med = 0.5 * (np.take_along_axis(a, lo[:, None], axis=1)[:, 0]
                 + np.take_along_axis(a, hi[:, None], axis=1)[:, 0])
    mx = np.nanmax(np.where(vmask, ang[:, iu[0], iu[1]], -1.0), axis=1)
    return med, mx, cnt, int(K)


def run_local_rotation(cache_dir, out_dir, presets=PRESETS):
    """Record how smooth the per-vertex rotation field actually is, tail vs head."""
    data, per, _ = load_cache(cache_dir, presets)
    x0, faces, N = data["x0"], data["faces"], data["x0"].shape[0]
    xB_all = data["xB"]
    src, dst = directed_edges(faces, N)
    out = {"meta": dict(
        stage="local_rotation", script="scripts/exp_p99_fallback.py",
        question="do the rotations of the faces incident to one vertex actually disagree?",
        n_vertices=int(N), n_faces=int(faces.shape[0]), presets=list(presets))}
    out["presets"] = {}
    print("%-11s %8s %8s %8s %8s %8s" % ("preset", "face50", "face99", "pair50", "tail50", "hi_area"), flush=True)
    for p in presets:
        d = per[p]
        # tail = rot_sync rigid error above the G2 line, the set every fallback is trying to fix
        tail = angle_npz(d["rref"], d["rot_sync"]) > G2_DEG
        Rf, area = face_rotations(x0, xB_all[data["names"].index(p)], faces)
        th = angle_npz(Rf, np.repeat(np.eye(3)[None], Rf.shape[0], axis=0))
        med, mx, cnt, K = incident_face_angles(Rf, faces, N)
        rec = dict(
            face_deg=dict(p50=float(np.median(th)), p90=float(np.percentile(th, 90)),
                          p99=float(np.percentile(th, 99)), mx=float(th.max()),
                          frac_gt_90=float((th > 90).mean()), frac_gt_170=float((th > 170).mean())),
            incident=dict(max_faces=K, deg_p50=float(np.median(cnt)),
                          pair_med_p50=float(np.median(med)), pair_med_p90=float(np.percentile(med, 90)),
                          pair_med_p99=float(np.percentile(med, 99)), pair_med_mx=float(med.max()),
                          frac_gt_90=float((med > 90).mean()), frac_gt_150=float((med > 150).mean()),
                          frac_gt_170=float((med > 170).mean())),
            tail=dict(n=int(tail.sum()),
                      pair_med_p50=float(np.median(med[tail])) if tail.any() else None,
                      pair_med_p90=float(np.percentile(med[tail], 90)) if tail.any() else None,
                      pair_max_p50=float(np.median(mx[tail])) if tail.any() else None,
                      pair_med_frac_gt_90=float((med[tail] > 90).mean()) if tail.any() else None,
                      pair_med_frac_gt_150=float((med[tail] > 150).mean()) if tail.any() else None,
                      pair_med_frac_gt_170=float((med[tail] > 170).mean()) if tail.any() else None,
                      pair_max_frac_gt_150=float((mx[tail] > 150).mean()) if tail.any() else None),
            head=dict(pair_med_p50=float(np.median(med[~tail])) if (~tail).any() else None,
                      pair_max_p50=float(np.median(mx[~tail])) if (~tail).any() else None,
                      pair_max_frac_gt_150=float((mx[~tail] > 150).mean()) if (~tail).any() else None),
            separation=dict(
                auc_rank=float(_auc(med[tail], med[~tail])) if tail.any() and (~tail).any() else None,
                note="probability that a random tail vertex outranks a random head vertex on "
                     "ring-disagreement; 0.5 = no predictive power, 1.0 = perfect separator"),
            flipped_faces=dict(n=int((th > 90).sum()),
                               area_p50=float(np.median(area)),
                               area_p50_flipped=float(np.median(area[th > 90])) if (th > 90).any() else None),
        )
        # -- reference audit: how much of the "error" is the choice of evaluation reference?
        # The shipped protocol (eval_cov_offline.py:150-193) fits r_ref by Kabsch over directed
        # edge pairs weighted p[:,i]*q[:,j]; rot_sync itself builds its field as an area-weighted
        # Markley mean of face rotations. Those two are different estimators, so part of the
        # measured p99 is reference disagreement, not method error.
        pad, valid, deg, _K = vertex_incidence(faces, N)
        w = np.where(valid, area[np.maximum(pad, 0)], 0.0)
        Rm = np.einsum("nk,nkij->nij", w / np.maximum(w.sum(1, keepdims=True), 1e-30),
                       Rf[np.maximum(pad, 0)])
        u_, s_, vt_ = np.linalg.svd(Rm)
        Rmk = np.einsum("nij,njk->nik", u_, vt_)
        badmk = np.linalg.det(Rmk) < 0
        if badmk.any():
            u2_ = u_.copy()
            u2_[badmk, :, -1] *= -1.0
            Rmk = np.einsum("nij,njk->nik", u2_, vt_)
        e_ref = angle_npz(d["rref"], Rmk)
        audit = {"kabsch_vs_markley": dict(
            p50=float(np.median(e_ref)), p90=float(np.percentile(e_ref, 90)),
            p99=float(np.percentile(e_ref, 99)), n60=int((e_ref > G2_DEG).sum()),
            frac=float((e_ref > G2_DEG).mean()))}
        for m in METHODS:
            ek = angle_npz(d["rref"], d[m])
            em = angle_npz(Rmk, d[m])
            audit[m] = dict(
                vs_kabsch=dict(p50=float(np.median(ek)), p99=float(np.percentile(ek, 99)),
                               n60=int((ek > G2_DEG).sum())),
                vs_markley=dict(p50=float(np.median(em)), p99=float(np.percentile(em, 99)),
                                n60=int((em > G2_DEG).sum())))
        rec["reference_audit"] = audit

        # -- what actually separates the tail? Test three candidate predictors by AUC, plus the
        # "does this vertex touch a flipped face" incidence test, which is the mechanical reading
        # of the task's premise (a 1-ring that contains folds).
        m90 = th > 90.0
        m150 = th > 150.0
        inc90 = np.zeros(N, dtype=bool)
        inc150 = np.zeros(N, dtype=bool)
        inc90[np.unique(faces[m90].reshape(-1))] = True
        inc150[np.unique(faces[m150].reshape(-1))] = True
        e_vtp = angle_npz(d["rref"], d["vertex_then_polar"])
        rec["tail_predictors"] = {
            # AUC of the "is in the tail" label predicted by 1 - coherence. High AUC here means
            # ordering is available, but the absolute value still decides whether a gate can open.
            "auc_1_minus_coherence": float(_auc((1.0 - d["coh"])[tail], (1.0 - d["coh"])[~tail])),
            "auc_borrowed_error_vtp": float(_auc(e_vtp[tail], e_vtp[~tail])),

            "touches_flipped_face": dict(
                gt_90=dict(tail_frac=float(inc90[tail].mean()), head_frac=float(inc90[~tail].mean()),
                           lift=float(inc90[tail].mean() / max(inc90[~tail].mean(), 1e-9))),
                gt_150=dict(tail_frac=float(inc150[tail].mean()),
                            head_frac=float(inc150[~tail].mean()),
                            lift=float(inc150[tail].mean() / max(inc150[~tail].mean(), 1e-9)))),
            "spearman_coherence": float(np.corrcoef(
                np.argsort(np.argsort(1.0 - d["coh"])), np.argsort(np.argsort(
                    angle_npz(d["rref"], d["rot_sync"]))))[0, 1]),
        }

        # -- is the rotation the tail needs even present in its own 1-ring?
        # ring_best = min over incident faces of angle(R_face, r_ref): the best single face
        # rotation available locally. ORACLE4 adds it to the {rs, vtp, lam} oracle.
        Rp = Rf[np.maximum(pad, 0)]
        tr_ = np.einsum("nkij,nij->nk", Rp, d["rref"])
        af = np.degrees(np.arccos(np.clip((tr_ - 1.0) * 0.5, -1.0, 1.0)))
        af = np.where(valid, af, np.inf)
        ring_best = af.min(1)
        cand = {m: angle_npz(d["rref"], d[m]) for m in METHODS}
        cand["ring_best"] = ring_best
        orc4 = np.min(np.stack([cand[m] for m in METHODS] + [ring_best], 1), axis=1)
        rec["ring_best_audit"] = {
            "ring_best": dict(p50=float(np.median(ring_best)),
                              p99=float(np.percentile(ring_best, 99)),
                              n60=int((ring_best > G2_DEG).sum())),
            "oracle4_best_of_lam_rs_vtp_ringbest": dict(
                p50=float(np.median(orc4)), p99=float(np.percentile(orc4, 99)),
                n60=int((orc4 > G2_DEG).sum()),
                per_candidate_n60={m: int((cand[m] > G2_DEG).sum()) for m in cand}),
            "tail_on_ring_best": dict(
                n=int(tail.sum()),
                p50=float(np.median(ring_best[tail])),
                p90=float(np.percentile(ring_best[tail], 90))),
        }
        out["presets"][p] = rec
        print("%-11s %8.3f %8.2f %8.3f %8.3f %8.2e" % (
            p, rec["face_deg"]["p50"], rec["face_deg"]["p99"],
            rec["incident"]["pair_med_p50"], rec["tail"]["pair_med_p50"] or float("nan"),
            rec["flipped_faces"]["area_p50_flipped"] or float("nan")), flush=True)
    path = os.path.join(out_dir, "local_rotation.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote %s" % path, flush=True)
    return out


# ----------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("offline", "splitting", "refstability",
                                        "local_rotation", "capture"),
                    default="offline")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--no-tail-ids", action="store_true")
    args = ap.parse_args(argv)
    presets = tuple(args.presets.split(","))
    if args.stage == "capture":
        run_capture(args.cache_dir, presets)
        return 0
    if args.stage == "splitting":
        run_splitting(args.cache_dir, args.out_dir, presets)
        return 0
    if args.stage == "refstability":
        run_reference_stability(args.cache_dir, args.out_dir, presets)
        return 0
    if args.stage == "local_rotation":
        run_local_rotation(args.cache_dir, args.out_dir, presets)
        return 0
    run_offline(args.cache_dir, args.out_dir, presets, not args.no_tail_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
