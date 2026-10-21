#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSPL v1 契约验收测试（独立 QA / task-3）—— CONTRACT_v2.md 第 8 节。

覆盖:
  A 组 (1-8)   后端
  C 组 (17-18) 性能 (C17 客户端 prepareMs 需真实浏览器；缺失句柄/浏览器时 SKIP 并交接 qa-browser)

本脚本是**独立验证方**的测试，只读生产代码，不修改 lam3d/lam3d/ 或 lam3d/static/。

用法:
  # 不需要后端：生成给 splat_js_test.mjs 用的 fixture（合法 + 3 个畸形 buffer）
  /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_splat_test.py \
        --make-fixture /tmp/lspl_fixture

  # 需要后端（先 `cd /workspace/lam3d && bash run.sh --daemon`）
  cd /workspace/lam3d
  /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_splat_test.py \
        --base http://127.0.0.1:8000 --sample status.png

退出码: 0 = 全部 PASS（SKIP 不算失败，但会打印）; 1 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 契约常量
LSPL_MAGIC = b"LSPL"
LSPL_VERSION = 1
LSPL_HEADER_SIZE = 144
LSPL_FLOATS_PER_GAUSSIAN = 56
EXPECTED_N = 20018
ARKIT_COUNT = 52
SEGMENTS = ("xyz", "rot", "scale", "opacity", "rgb")
SEG_WIDTH = {"xyz": 3, "rot": 4, "scale": 3, "opacity": 1, "rgb": 3}
DEFAULT_BASE = os.environ.get("LAM3D_BASE", "http://127.0.0.1:8000")
LAM3D_ROOT = Path("/workspace/lam3d")
CONTRACT = LAM3D_ROOT / "CONTRACT_v2.md"
CUDA_DIR = Path("/workspace/LAM/thirdparty_src/diff-gaussian-rasterization")
# 本轮开始前 /workspace/LAM 的既有改动（契约 §8 A8：不得出现这 3 项之外的新条目）
LAM_KNOWN_ENTRIES = {
    " M LAM/lam/models/rendering/flame_model/flame.py",
    " M LAM/lam/models/rendering/gs_renderer.py",
    "?? LAM/lam/models/rendering/covariance_modulation.py",
}


# ================================================================ LSPL 编解码
def encode_lspl(arrays: dict, meta: dict) -> bytes:
    """测试侧参考实现（fixture 用），严格照 CONTRACT_v2.md §4。"""
    n = int(meta["count"])
    flags = int(meta.get("flags", 1))
    hdr = struct.pack("<4sIII", LSPL_MAGIC, int(meta.get("version", 1)), n, flags)
    hdr += np.asarray(meta["bbox_min"], "<f4").reshape(3).tobytes()
    hdr += np.asarray(meta["bbox_max"], "<f4").reshape(3).tobytes()
    hdr += np.asarray(meta["center"], "<f4").reshape(3).tobytes()
    hdr += np.asarray(meta["ref_c2w"], "<f4").reshape(16).tobytes()
    hdr += np.asarray(meta["ref_intr"], "<f4").reshape(4).tobytes()
    hdr += np.asarray(meta["ref_size"], "<f4").reshape(2).tobytes()
    hdr += b"\x00" * 4
    assert len(hdr) == LSPL_HEADER_SIZE, len(hdr)
    body = b"".join(
        np.asarray(arrays[k], "<f4").reshape(n * SEG_WIDTH[k]).tobytes() for k in SEGMENTS
    )
    return hdr + body


def parse_lspl(body: bytes) -> dict:
    """用 struct/numpy 逐字段解析 LSPL v1（QA 独立解析，不看生产代码）。"""
    if len(body) < LSPL_HEADER_SIZE:
        raise ValueError(f"LSPL 长度不足: {len(body)} < {LSPL_HEADER_SIZE}")
    magic, version, count, flags = struct.unpack_from("<4sIII", body, 0)
    out = {
        "magic": magic,
        "version": int(version),
        "count": int(count),
        "flags": int(flags),
        "bbox_min": np.frombuffer(body, "<f4", 3, 16).copy(),
        "bbox_max": np.frombuffer(body, "<f4", 3, 28).copy(),
        "center": np.frombuffer(body, "<f4", 3, 40).copy(),
        "ref_c2w": np.frombuffer(body, "<f4", 16, 52).copy(),
        "ref_intr": np.frombuffer(body, "<f4", 4, 116).copy(),
        "ref_size": np.frombuffer(body, "<f4", 2, 132).copy(),
        "reserved": body[140:144],
    }
    want = LSPL_HEADER_SIZE + count * LSPL_FLOATS_PER_GAUSSIAN
    if len(body) != want:
        raise ValueError(f"LSPL 长度不是 144+N*56: len={len(body)} want={want}")
    off = LSPL_HEADER_SIZE
    for k in SEGMENTS:
        w = SEG_WIDTH[k]
        out[k] = np.frombuffer(body, "<f4", count * w, off).reshape(count, w).copy()
        out[f"off_{k}"] = off
        off += count * w * 4
    out["total"] = want
    return out


# ================================================================ fixture
def make_fixture(outdir: Path, n: int = 1000) -> dict:
    """写 1 个合法 + 4 个畸形 LSPL 与 expected.json（给 splat_js_test.mjs 读）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    xyz = (rng.standard_normal((n, 3)) * 0.05).astype(np.float32)
    rot = rng.standard_normal((n, 4)).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    scale = np.exp(rng.standard_normal((n, 3)) * 0.2 - 2.0).astype(np.float32)
    opacity = rng.uniform(0.05, 0.98, (n, 1)).astype(np.float32)
    rgb = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    op = opacity[:, 0].astype(np.float64)
    center = (xyz.astype(np.float64) * op[:, None]).sum(0) / max(op.sum(), 1e-9)
    # 一个合法的 4x4 c2w（旋转 + 平移），行主序
    ax = np.array([0.3, 1.0, 0.2], float)
    ax /= np.linalg.norm(ax)
    ang = 0.37
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = np.array([0.01, -0.02, -0.62], np.float32)
    meta = {
        "version": LSPL_VERSION,
        "count": n,
        "flags": 1,
        "bbox_min": xyz.min(0).tolist(),
        "bbox_max": xyz.max(0).tolist(),
        "center": center.astype(np.float32).tolist(),
        "ref_c2w": c2w.reshape(16).tolist(),
        "ref_intr": [512.0, 512.0, 256.0, 256.0],
        "ref_size": [512.0, 512.0],
    }
    arrays = {"xyz": xyz, "rot": rot, "scale": scale, "opacity": opacity, "rgb": rgb}
    valid = encode_lspl(arrays, meta)
    (outdir / "valid.lspl").write_bytes(valid)

    bad_magic = b"XXXX" + valid[4:]
    (outdir / "bad_magic.lspl").write_bytes(bad_magic)
    bad_version = valid[:4] + struct.pack("<I", 2) + valid[8:]
    (outdir / "bad_version.lspl").write_bytes(bad_version)
    (outdir / "truncated.lspl").write_bytes(valid[:-97])
    (outdir / "short.lspl").write_bytes(valid[:100])

    exp = {
        "schema": 1,
        "n": n,
        "headerSize": LSPL_HEADER_SIZE,
        "floatsPerGaussian": LSPL_FLOATS_PER_GAUSSIAN,
        "totalBytes": len(valid),
        "md5": hashlib.md5(valid).hexdigest(),
        "magic": "LSPL",
        "version": LSPL_VERSION,
        "count": n,
        "flags": 1,
        "bboxMin": [float(np.float32(v)) for v in meta["bbox_min"]],
        "bboxMax": [float(np.float32(v)) for v in meta["bbox_max"]],
        "center": [float(v) for v in meta["center"]],
        "refC2W": [float(v) for v in meta["ref_c2w"]],
        "refIntr": list(meta["ref_intr"]),
        "refSize": list(meta["ref_size"]),
        "offsets": {},
        "segments": {},
    }
    off = LSPL_HEADER_SIZE
    for k in SEGMENTS:
        w = SEG_WIDTH[k]
        exp["offsets"][k] = off
        raw = np.asarray(arrays[k], "<f4").reshape(-1).tobytes()
        exp["segments"][k] = base64.b64encode(raw).decode("ascii")
        off += n * w * 4
    (outdir / "expected.json").write_text(json.dumps(exp), encoding="utf-8")

    # QA 自检：参数化编解码必须闭环
    back = parse_lspl(valid)
    assert back["count"] == n and back["total"] == len(valid)
    for k in SEGMENTS:
        assert np.array_equal(back[k], np.asarray(arrays[k], np.float32)), k
        assert back[f"off_{k}"] == exp["offsets"][k], k
    print(f"[fixture] {outdir}  valid.lspl={len(valid)}B N={n} md5={exp['md5'][:12]}")
    print("[fixture] 畸形: bad_magic / bad_version / truncated / short")
    return exp


# ================================================================ 报告器
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def check(self, tid: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((tid, name, "PASS" if ok else "FAIL", detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {tid} {name}" + (f"  — {detail}" if detail else ""),
              flush=True)
        return bool(ok)

    def skip(self, tid: str, name: str, detail: str = "") -> None:
        self.rows.append((tid, name, "SKIP", detail))
        print(f"[SKIP] {tid} {name}" + (f"  — {detail}" if detail else ""), flush=True)

    def summary(self) -> int:
        print("\n================ 验收汇总 ================", flush=True)
        npass = sum(1 for r in self.rows if r[2] == "PASS")
        nfail = sum(1 for r in self.rows if r[2] == "FAIL")
        nskip = sum(1 for r in self.rows if r[2] == "SKIP")
        for tid, name, st, detail in self.rows:
            print(f"  {st:4s} {tid:5s} {name}" + (f"   [{detail}]" if detail and st != "PASS" else ""))
        print(f"-------- PASS={npass} FAIL={nfail} SKIP={nskip} --------")
        return 1 if nfail else 0


# ================================================================ HTTP 套件
class Suite:
    def __init__(self, base: str, sample: str, args) -> None:
        self.base = base.rstrip("/")
        self.sample = sample
        self.args = args
        self.rep = Report()
        self.job_id: str | None = None
        self.job: dict = {}
        self.session: dict = {}
        self.lspl0: bytes = b""
        self.p0: dict = {}
        self.ms_values: list[float] = []
        self.cached: dict[str, bytes] = {}

    # ------------------------------------------------------------- helpers
    def _http(self):
        import requests
        s = requests.Session()
        s.headers.update({"User-Agent": "lam3d-qa/contract-v2"})
        return s

    def get(self, path: str, **kw):
        import requests
        r = requests.get(self.base + path, timeout=kw.pop("timeout", 120), **kw)
        return r

    def post_json(self, path: str, payload, timeout: int = 300):
        import requests
        return requests.post(self.base + path, json=payload, timeout=timeout)

    def wait_job(self, job_id: str, timeout: float = 420.0) -> dict:
        t0 = time.time()
        last = ""
        while time.time() - t0 < timeout:
            r = self.get(f"/api/jobs/{job_id}", timeout=30)
            if r.status_code != 200:
                time.sleep(2)
                continue
            j = r.json()
            line = f"{j.get('state')}/{j.get('stage')} {j.get('progress')}%"
            if line != last:
                print(f"[poll] {line} {j.get('detail','')}", flush=True)
                last = line
            if j.get("state") in ("succeeded", "failed", "canceled"):
                return j
            time.sleep(2.0)
        raise TimeoutError(f"任务 {job_id} 超时未结束")

    def splat(self, arkit, expect=200) -> "object":
        # 注意：不能用 requests 的 json= 参数 —— allow_nan=False 会在**客户端**
        # 抛 InvalidJSONError，NaN 请求根本没发出去，A6u 会误判。这里手动
        # json.dumps(..., allow_nan=True) 后按原始 body 发送，测的是服务端行为。
        import requests
        body = json.dumps({"arkit": list(arkit)}, allow_nan=True).encode("utf-8")
        r = requests.post(self.base + f"/api/jobs/{self.job_id}/arkit_splat",
                          data=body, headers={"Content-Type": "application/json"},
                          timeout=300)
        if expect is not None and r.status_code != expect:
            raise AssertionError(
                f"arkit_splat 期望 {expect} 得到 {r.status_code}: {r.text[:300]}")
        if r.status_code == 200:
            try:
                self.ms_values.append(float(r.headers.get("X-LSPL-Ms", "nan")))
            except ValueError:
                self.ms_values.append(float("nan"))
        return r

    def arkit_vec(self, **kw) -> list[float]:
        v = [0.0] * ARKIT_COUNT
        names = [n for n in self.rep_names]
        for k, val in kw.items():
            v[names.index(k)] = float(val)
        return v

    # ------------------------------------------------------------- A1
    def check_a1_generate(self) -> None:
        import requests
        # 兼容性: 带 motion/skip_video 必须被接受并忽略（不得 422）
        files = None
        data = {"sample": self.sample, "motion": "__bogus_motion__", "skip_video": "true"}
        r = requests.post(self.base + "/api/generate", data=data, timeout=120)
        self.rep.check("A1a", "POST /api/generate 接受并忽略 motion/skip_video (非 422)",
                       r.status_code == 200,
                       f"HTTP {r.status_code} {r.text[:200]}")
        if r.status_code != 200:
            return
        js = r.json()
        self.rep.check("A1b", "generate 返回 {job_id,state}",
                       bool(js.get("job_id")) and "state" in js,
                       json.dumps(js, ensure_ascii=False)[:200])
        self.job_id = js.get("job_id")
        # 既没有 image 也没有 sample -> 400
        r400 = requests.post(self.base + "/api/generate", data={}, timeout=60)
        self.rep.check("A1c", "image/sample 都没有 -> 400", r400.status_code == 400,
                       f"HTTP {r400.status_code}")
        if not self.job_id:
            return
        self.job = self.wait_job(self.job_id)
        ok = self.job.get("state") == "succeeded"
        self.rep.check("A1d", f"任务 succeeded (sample={self.sample})", ok,
                       f"state={self.job.get('state')} err={self.job.get('error')}")
        if not ok:
            return
        res = self.job.get("result") or {}
        self.rep.check("A1e", "result 含 face/thumb/glb/obj/gs_ply",
                       all(k in res for k in ("face", "thumb", "glb", "obj", "gs_ply")),
                       f"keys={sorted(res.keys())}")

        from PIL import Image
        rface = self.get(f"/api/jobs/{self.job_id}/file/face", timeout=120)
        self.cached["face"] = rface.content if rface.status_code == 200 else b""
        if rface.status_code == 200:
            img = Image.open(io.BytesIO(rface.content)).convert("RGB")
            arr = np.asarray(img).astype(np.int16)
            h, w = arr.shape[:2]
            bg = np.all(arr >= 250, axis=2)
            frac = 1.0 - float(bg.mean())
            self.rep.check("A1f", "face.png 为 512x512 PNG", (w, h) == (512, 512),
                           f"{w}x{h} {img.format}")
            self.rep.check("A1g", "face.png 非全白 且 非背景像素 > 5%", frac > 0.05,
                           f"non-bg={frac*100:.2f}%")
        else:
            self.rep.check("A1f", "face.png 可下载", False, f"HTTP {rface.status_code}")
            self.rep.check("A1g", "face.png 非全白/非背景>5%", False, "face 不可下载")
        for nm in ("thumb", "glb", "gs_ply"):
            rr = self.get(f"/api/jobs/{self.job_id}/file/{nm}", timeout=300)
            self.rep.check(f"A1h.{nm}", f"file/{nm} 可下载且非空",
                           rr.status_code == 200 and len(rr.content) > 100,
                           f"HTTP {rr.status_code} len={len(rr.content)}")

    # ------------------------------------------------------------- A2
    def check_a2_files(self) -> None:
        if not self.job_id:
            self.rep.check("A2", "file/face & preview & video", False, "无 job_id")
            return
        rf = self.get(f"/api/jobs/{self.job_id}/file/face", timeout=120)
        rp = self.get(f"/api/jobs/{self.job_id}/file/preview", timeout=120)
        self.rep.check("A2a", "file/face -> 200 image/png",
                       rf.status_code == 200 and rf.headers.get("content-type", "").startswith("image/png"),
                       f"HTTP {rf.status_code} {rf.headers.get('content-type')}")
        self.rep.check("A2b", "file/preview 与 face 字节相同",
                       rp.status_code == 200 and rf.status_code == 200 and rp.content == rf.content,
                       f"face={hashlib.md5(rf.content).hexdigest()[:12]} "
                       f"preview={hashlib.md5(rp.content).hexdigest()[:12]}")
        for nm in ("video", "video_noaudio"):
            rv = self.get(f"/api/jobs/{self.job_id}/file/{nm}", timeout=60)
            self.rep.check(f"A2c.{nm}", f"file/{nm} -> 404", rv.status_code == 404,
                           f"HTTP {rv.status_code}")

    # ------------------------------------------------------------- A3
    def check_a3_schema(self) -> None:
        rc = self.get("/api/config", timeout=60)
        cfg = rc.json() if rc.status_code == 200 else {}
        self.rep.check("A3a", "/api/config 无 motions / frames_hint",
                       "motions" not in cfg and "frames_hint" not in cfg,
                       f"keys={sorted(cfg.keys())}")
        self.rep.check("A3b", "/api/config 保留 samples/gpu/device/max_upload_mb",
                       all(k in cfg for k in ("samples", "gpu", "device", "max_upload_mb")),
                       f"keys={sorted(cfg.keys())}")
        if self.job_id:
            self.rep.check("A3c", "GET /api/jobs/{id} 无 motion 键", "motion" not in self.job,
                           f"keys={sorted(self.job.keys())}")

    # ------------------------------------------------------------- A4
    def check_a4_meta(self) -> None:
        r = self.get("/api/arkit/meta", timeout=60)
        ok = r.status_code == 200
        m = r.json() if ok else {}
        names = m.get("names") or []
        self.rep_names = names or [""] * ARKIT_COUNT
        self.rep.check("A4a", "arkit/meta count=52", m.get("count") == 52,
                       f"count={m.get('count')} len(names)={len(names)}")
        self.rep.check("A4b", "names[24]=='jawOpen'",
                       len(names) > 24 and names[24] == "jawOpen",
                       f"names[24]={names[24] if len(names) > 24 else None}")

    # ------------------------------------------------------------- A5
    def check_a5_session(self) -> None:
        if not self.job_id:
            self.rep.check("A5", "arkit_session", False, "无 job_id")
            return
        r = self.get(f"/api/jobs/{self.job_id}/arkit_session", timeout=60)
        s = r.json() if r.status_code == 200 else {}
        self.session = s
        ok = s.get("ready") is True
        self.rep.check("A5a", "arkit_session ready=true", ok, json.dumps(s)[:200])
        if not ok:
            return
        self.rep.check("A5b", "center(3)", len(s.get("center") or []) == 3, f"{s.get('center')}")
        self.rep.check("A5c", "ref_c2w(16)", len(s.get("ref_c2w") or []) == 16,
                       f"len={len(s.get('ref_c2w') or [])}")
        self.rep.check("A5d", "ref_intr(4)", len(s.get("ref_intr") or []) == 4, f"{s.get('ref_intr')}")
        self.rep.check("A5e", "ref_size(2)", len(s.get("ref_size") or []) == 2, f"{s.get('ref_size')}")
        self.rep.check("A5f", "n_gaussians/arkit_count", s.get("n_gaussians") == EXPECTED_N
                       and s.get("arkit_count") == ARKIT_COUNT,
                       f"n={s.get('n_gaussians')} arkit={s.get('arkit_count')}")

    # ------------------------------------------------------------- A6
    def check_a6_splat(self) -> None:
        if not self.job_id:
            self.rep.check("A6", "arkit_splat", False, "无 job_id")
            return
        zeros = [0.0] * ARKIT_COUNT
        r1 = self.splat(zeros)
        body = r1.content
        hdr = {k: r1.headers.get(k) for k in
               ("X-LSPL-Count", "X-LSPL-Ms", "Cache-Control", "Content-Type")}
        self.rep.check("A6a", "X-LSPL-Count == 20018", str(hdr["X-LSPL-Count"]) == str(EXPECTED_N),
                       f"count={hdr['X-LSPL-Count']}")
        self.rep.check("A6b", f"len(body) == 144+20018*56 = {EXPECTED_N*56+144}",
                       len(body) == EXPECTED_N * 56 + LSPL_HEADER_SIZE,
                       f"len={len(body)}")
        self.rep.check("A6c", "Content-Type=application/octet-stream 且 Cache-Control=no-store",
                       (hdr["Content-Type"] or "").startswith("application/octet-stream")
                       and (hdr["Cache-Control"] or "").lower() == "no-store",
                       f"ct={hdr['Content-Type']} cc={hdr['Cache-Control']}")
        try:
            p = parse_lspl(body)
        except Exception as exc:
            self.rep.check("A6d", "header 可解析", False, f"{type(exc).__name__}: {exc}")
            return
        self.lspl0, self.p0 = body, p
        self.rep.check("A6d", "magic=LSPL / version=1 / count=20018 / flags=1",
                       p["magic"] == LSPL_MAGIC and p["version"] == 1
                       and p["count"] == EXPECTED_N and p["flags"] == 1,
                       f"magic={p['magic']} ver={p['version']} count={p['count']} flags={p['flags']}")
        self.rep.check("A6e", "reserved 4 字节全 0", p["reserved"] == b"\x00" * 4,
                       f"{p['reserved']!r}")
        # header 合理性
        bmin, bmax, ctr = p["bbox_min"], p["bbox_max"], p["center"]
        fx, fy, cx, cy = [float(v) for v in p["ref_intr"]]
        w, h = [float(v) for v in p["ref_size"]]
        self.rep.check("A6f", "bbox_min <= center <= bbox_max",
                       bool(np.all(bmin <= ctr + 1e-5) and np.all(ctr <= bmax + 1e-5)),
                       f"min={bmin.round(3)} c={ctr.round(3)} max={bmax.round(3)}")
        self.rep.check("A6g", "ref_intr fx,fy > 0 且 ref_size=(512,512) 宽在前",
                       fx > 0 and fy > 0 and (w, h) == (512.0, 512.0),
                       f"intr=({fx},{fy},{cx},{cy}) size=({w},{h})")
        self.rep.check("A6h", "render_hw 与 ref_size 顺序相反",
                       self.session.get("render_hw") == [int(h), int(w)]
                       if self.session.get("render_hw") else False,
                       f"render_hw={self.session.get('render_hw')} ref_size=({w},{h})")
        # payload
        segs_ok = True
        for k in SEGMENTS:
            a = p[k]
            if not np.all(np.isfinite(a)):
                segs_ok = False
        self.rep.check("A6i", "payload 5 段长度正确且全 isfinite",
                       segs_ok and all(p[k].shape == (EXPECTED_N, SEG_WIDTH[k]) for k in SEGMENTS),
                       " ".join(f"{k}{p[k].shape}" for k in SEGMENTS))
        qn = np.linalg.norm(p["rot"].astype(np.float64), axis=1)
        self.rep.check("A6j", "|q| ∈ [1-1e-5, 1+1e-5]",
                       bool(np.all(np.abs(qn - 1.0) <= 1e-5)),
                       f"|q|min={qn.min():.8f} max={qn.max():.8f}")
        smin, smax = float(p["scale"].min()), float(p["scale"].max())
        self.rep.check("A6k", "scale > 0", smin > 0, f"min={smin:.6g} max={smax:.6g}")
        omin, omax = float(p["opacity"].min()), float(p["opacity"].max())
        self.rep.check("A6l", "opacity ∈ [0,1]", 0.0 <= omin and omax <= 1.0,
                       f"min={omin:.6g} max={omax:.6g}")
        rmin, rmax = float(p["rgb"].min()), float(p["rgb"].max())
        self.rep.check("A6m", "rgb ∈ [0,1]", -1e-6 <= rmin and rmax <= 1.0 + 1e-6,
                       f"min={rmin:.6g} max={rmax:.6g}")
        # 退化自洽: header 必须由 B=0 payload 推出（等价于「B=0 == 不传 arkit 静态路径」）
        gmin = p["xyz"].min(0).astype(np.float32)
        gmax = p["xyz"].max(0).astype(np.float32)
        op = p["opacity"][:, 0].astype(np.float64)
        gctr = ((p["xyz"].astype(np.float64) * op[:, None]).sum(0)
                / max(op.sum(), 1e-9)).astype(np.float32)
        d_bmin = float(np.abs(gmin - bmin).max())
        d_bmax = float(np.abs(gmax - bmax).max())
        d_ctr = float(np.abs(gctr - ctr).max())
        self.rep.check("A6n", "退化一致性: header bbox/center == B=0 payload 实算值",
                       d_bmin < 1e-5 and d_bmax < 1e-5 and d_ctr < 1e-4,
                       f"dbbox_min={d_bmin:.3g} dbbox_max={d_bmax:.3g} dcenter={d_ctr:.3g}")
        # 幂等
        r2 = self.splat(zeros)
        self.rep.check("A6o", "B=0 两次请求逐字节相同 (sha256)",
                       hashlib.sha256(r2.content).hexdigest() == hashlib.sha256(body).hexdigest(),
                       f"{hashlib.sha256(body).hexdigest()[:16]}")
        # jawOpen=1
        names = self.rep_names
        jaw = self.arkit_vec(jawOpen=1.0)
        rj = self.splat(jaw)
        pj = parse_lspl(rj.content)
        dx = float(np.abs(pj["xyz"].astype(np.float64) - p["xyz"].astype(np.float64)).mean())
        qnj = np.linalg.norm(pj["rot"].astype(np.float64), axis=1)
        self.rep.check("A6p", "jawOpen=1 与 B=0 的 xyz mean|Δ| > 1e-5", dx > 1e-5,
                       f"mean|Δxyz|={dx:.3g}")
        self.rep.check("A6q", "jawOpen=1 时 |q|≈1 且 scale>0",
                       bool(np.all(np.abs(qnj - 1.0) <= 1e-5)) and float(pj["scale"].min()) > 0,
                       f"|q|max-1={float(np.abs(qnj-1).max()):.3g} smin={float(pj['scale'].min()):.3g}")
        # 全 1
        ones = [1.0] * ARKIT_COUNT
        ra = self.splat(ones)
        pa = parse_lspl(ra.content)
        finite = all(np.all(np.isfinite(pa[k])) for k in SEGMENTS)
        ratio = (pa["scale"].astype(np.float64) / np.maximum(p["scale"].astype(np.float64), 1e-30))
        self.rep.check("A6r", "全 1 -> 无 NaN/Inf", finite,
                       "" if finite else "存在非有限值")
        self.rep.check("A6s", "全 1 -> s <= 1.35*s0 (逐点)", bool(np.nanmax(ratio) <= 1.35 + 1e-4),
                       f"max(s/s0)={float(np.nanmax(ratio)):.6f}")
        # 非法输入
        rb = self.splat([0.0] * (ARKIT_COUNT - 1), expect=400)
        self.rep.check("A6t", "非法长度(51) -> 400", rb.status_code == 400,
                       f"HTTP {rb.status_code} {rb.text[:120]}")
        bad = [0.0] * ARKIT_COUNT
        bad[0] = float("nan")
        rn = self.splat(bad, expect=400)
        self.rep.check("A6u", "NaN -> 400", rn.status_code == 400,
                       f"HTTP {rn.status_code} {rn.text[:120]}")
        # 越界值应被 clip 而非 400（契约 §10 sanitize_arkit 语义）
        rclip = self.splat([2.0] * ARKIT_COUNT)
        self.rep.check("A6v", "越界值 [2.0]*52 -> 200 (clip 而非 400)", rclip.status_code == 200,
                       f"HTTP {rclip.status_code}")
        ru = self.post_json(f"/api/jobs/does-not-exist/arkit_splat", {"arkit": zeros})
        self.rep.check("A6w", "不存在 job -> 404", ru.status_code == 404, f"HTTP {ru.status_code}")

    # ------------------------------------------------------------- A7
    def check_a7_cross(self) -> None:
        if not self.p0:
            self.rep.check("A7", "LSPL<->arkit_frame 交叉一致性 IoU>0.5", False, "A6 未通过")
            return
        from PIL import Image
        p = self.p0
        c2w = p["ref_c2w"].astype(np.float64).reshape(4, 4)
        fx, fy, cx, cy = [float(v) for v in p["ref_intr"]]
        w, h = [float(v) for v in p["ref_size"]]
        R, t = c2w[:3, :3], c2w[:3, 3]
        rel = p["xyz"].astype(np.float64) - t[None, :]
        xc, yc, zc = rel @ R[:, 0], rel @ R[:, 1], rel @ R[:, 2]
        op = p["opacity"][:, 0].astype(np.float64)
        m = (zc > 1e-6) & (op > 1e-3)
        u = fx * xc[m] / zc[m] + cx
        v = fy * yc[m] / zc[m] + cy
        ub = (float(u.min()), float(v.min()), float(u.max()), float(v.max()))
        r = self.post_json(
            f"/api/jobs/{self.job_id}/arkit_frame",
            {"arkit": [0.0] * ARKIT_COUNT, "yaw": 0.0, "pitch": 0.0, "zoom": 1.0, "size": 512},
            timeout=300)
        if r.status_code != 200:
            self.rep.check("A7", "arkit_frame(B=0,size=512)", False,
                           f"HTTP {r.status_code} {r.text[:200]}")
            return
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.int16)
        nonwhite = ~np.all(arr >= 250, axis=2)
        ys, xs = np.nonzero(nonwhite)
        if xs.size == 0:
            self.rep.check("A7", "arkit_frame 有非白像素", False, "全白")
            return
        vb = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        iw = max(0.0, min(ub[2], vb[2]) - max(ub[0], vb[0]) + 1)
        ih = max(0.0, min(ub[3], vb[3]) - max(ub[1], vb[1]) + 1)
        inter = iw * ih
        union = ((ub[2] - ub[0] + 1) * (ub[3] - ub[1] + 1)
                 + (vb[2] - vb[0] + 1) * (vb[3] - vb[1] + 1) - inter)
        iou = inter / union if union > 0 else 0.0
        self.rep.check("A7", "LSPL 投影 bbox vs arkit_frame 非白 bbox IoU > 0.5", iou > 0.5,
                       f"IoU={iou:.3f} LSPL_uv={tuple(round(x) for x in ub)} "
                       f"frame_uv={tuple(int(x) for x in vb)} img={arr.shape[1]}x{arr.shape[0]}")
        np.save("/tmp/qa_arkit_frame_b0.npy", arr.astype(np.uint8))
        Image.fromarray(arr.astype(np.uint8)).save("/tmp/qa_arkit_frame_b0.png")

    # ------------------------------------------------------------- A8
    def check_a8_frozen(self) -> None:
        try:
            t0 = CONTRACT.stat().st_mtime
        except OSError:
            t0 = time.time()
        cmd = ["find", str(CUDA_DIR), "-newermt", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        self.rep.check("A8a", "diff-gaussian-rasterization 无新 mtime 文件（CUDA 零改动）",
                       out == "", f"新增: {out.splitlines()[:5]}")
        g = subprocess.run(["git", "-C", "/workspace", "status", "--porcelain"],
                           capture_output=True, text=True).stdout
        lam = {ln.rstrip() for ln in g.splitlines() if "LAM/" in ln}
        extra = lam - LAM_KNOWN_ENTRIES
        self.rep.check("A8b", "git status 中 LAM 仅既有改动（无新增）", not extra,
                       f"新增条目: {sorted(extra)}" if extra else f"既有 {len(lam)} 项")

    # ------------------------------------------------------------- C18
    def check_c18_ms(self) -> None:
        vals = [v for v in self.ms_values if v == v]
        if not vals:
            self.rep.check("C18", "X-LSPL-Ms <= 30", False, "无 X-LSPL-Ms 头")
            return
        mx = max(vals)
        self.rep.check("C18", "后端 X-LSPL-Ms <= 30ms", mx <= 30.0,
                       f"values={['%.2f' % v for v in vals]} max={mx:.2f}ms")

    # ------------------------------------------------------------- C17
    def check_c17_prepare(self) -> None:
        """stats().prepareMs <= 16 (N=20018)。

        app.js 没有把 view 挂到 window（无可依赖的调试句柄，见报告），因此这里
        **不猜句柄**：在真实 Chromium 里按契约 §5 的公开 API 自己 new 一个
        SplatView，把**后端真实返回**的 LSPL 字节喂进去，读它自己报的
        stats().prepareMs。这样测的是产品的排序/投影/写属性耗时本身，
        而不是刻意留给测试的后门。
        """
        if self.args.no_browser:
            self.rep.skip("C17", "stats().prepareMs <= 16", "由 --no-browser 关闭")
            return
        if not self.job_id:
            self.rep.check("C17", "stats().prepareMs <= 16", False, "无 job_id")
            return
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            self.rep.skip("C17", "stats().prepareMs <= 16", f"playwright 不可用: {exc}")
            return
        js = """
        async (jobId) => {
          const r = await fetch(`/api/jobs/${jobId}/arkit_splat`, {
              method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({arkit: new Array(52).fill(0)})});
          if (!r.ok) return {error: 'splat_http_' + r.status};
          const buf = await r.arrayBuffer();
          const mod = await import('/static/splat.js');
          if (typeof mod.parseLSPL !== 'function') return {error: 'no_parseLSPL'};
          if (typeof mod.SplatView !== 'function') return {error: 'no_SplatView_class'};
          const parsed = mod.parseLSPL(buf);
          const host = document.createElement('div');
          host.style.cssText = 'position:fixed;left:0;top:0;width:1024px;height:1024px;';
          document.body.appendChild(host);
          const v = new mod.SplatView(host, {background: 0x101014});
          const samples = [];
          for (let i = 0; i < 5; i++) {
            v.setData(parsed);           // 契约: setData 必须可连续调用且不重建上下文
            v.render();
            samples.push(v.stats().prepareMs);
          }
          const st = v.stats();
          const cw = v.canvas.width, ch = v.canvas.height;
          v.dispose(); host.remove();
          return {ms: Math.max(...samples), all: samples, count: parsed.count,
                  visible: st.visible, canvas: [cw, ch]};
        }
        """
        try:
            with sync_playwright() as pw:
                br = pw.chromium.launch(args=[
                    "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
                    "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
                pg = br.new_page(viewport={"width": 1280, "height": 900})
                pg.goto(self.base, wait_until="load", timeout=60000)
                pg.wait_for_timeout(2000)
                res = pg.evaluate(js, self.job_id)
                br.close()
        except Exception as exc:  # noqa: BLE001
            self.rep.check("C17", "SplatView 可在真实 Chromium 中驱动", False,
                           f"{type(exc).__name__}: {exc}")
            return
        if not isinstance(res, dict) or "ms" not in res:
            self.rep.check("C17", f"SplatView 公开 API 可驱动 ({res})", False, str(res))
            return
        self.rep.check("C17a", "SplatView 装载后端 LSPL 且 count=20018",
                       res.get("count") == EXPECTED_N, f"count={res.get('count')}")
        self.rep.check("C17", "stats().prepareMs <= 16ms (N=20018)", res["ms"] <= 16.0,
                       f"prepareMs max={res['ms']:.2f} samples={['%.2f' % x for x in res.get('all', [])]} "
                       f"visible={res.get('visible')} canvas={res.get('canvas')}")

    # ------------------------------------------------------------- run
    def run(self) -> int:
        steps = [
            ("A1", self.check_a1_generate),
            ("A2", self.check_a2_files),
            ("A3", self.check_a3_schema),
            ("A4", self.check_a4_meta),
            ("A5", self.check_a5_session),
            ("A6", self.check_a6_splat),
            ("A7", self.check_a7_cross),
            ("A8", self.check_a8_frozen),
            ("C17", self.check_c17_prepare),
            ("C18", self.check_c18_ms),
        ]
        for tag, fn in steps:
            print(f"\n--- {tag} ---", flush=True)
            try:
                fn()
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                self.rep.check(tag, f"{tag} 执行异常", False, tb.splitlines()[-1])
                print(tb, flush=True)
        return self.rep.summary()


def main() -> int:
    ap = argparse.ArgumentParser(description="LSPL v1 契约验收测试 (task-3)")
    ap.add_argument("--make-fixture", metavar="DIR", help="只生成 fixture（不需要后端）")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--sample", default="status.png")
    ap.add_argument("--n", type=int, default=1000, help="fixture 高斯数")
    ap.add_argument("--no-browser", action="store_true", help="跳过 C17 浏览器测量")
    args = ap.parse_args()

    if args.make_fixture:
        make_fixture(Path(args.make_fixture), n=args.n)
        return 0

    suite = Suite(args.base, args.sample, args)
    suite.rep_names = [""] * ARKIT_COUNT
    print(f"LAM3D 契约验收: base={suite.base} sample={suite.sample}")
    return suite.run()


if __name__ == "__main__":
    sys.exit(main())
