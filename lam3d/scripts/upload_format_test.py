#!/usr/bin/env python
"""task-8 独立验收：上传格式归一化（WEBP/BMP -> PNG）端到端测试。

独立验证方 qa-contract 所有。**只读生产代码**，仅从 HTTP 层打真实服务。
- 不复用 task-7 作者的脚本与结论。
- 真实 GPU 串行：本脚本**严格串行**逐个生成，绝不并发 submit。

跑法：
  # 1) 只造 fixture（不需要服务、不占 GPU）
  PYTHONPATH=/workspace/lam3d:/workspace/LAM /usr/local/miniconda3/envs/lam/bin/python \
      scripts/upload_format_test.py --fixtures-only --fixture-dir /tmp/uft_fix

  # 2) 真实端到端（生成任务多，用 nohup + flag 文件）
  nohup env PYTHONPATH=/workspace/lam3d:/workspace/LAM \
      /usr/local/miniconda3/envs/lam/bin/python scripts/upload_format_test.py \
      --base http://127.0.0.1:8000 --fixture-dir /tmp/uft_fix > /tmp/uft.log 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SAMPLE = Path("/workspace/LAM/assets/sample_input/status.png")
LAM_ROOT = Path("/workspace/LAM")
WORK_ROOT = Path(os.environ.get("LAM3D_WORK", "/workspace/lam3d/workspace"))
INDEX_HTML = Path("/workspace/lam3d/static/index.html")
README = Path("/workspace/lam3d/README.md")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# T7 开工时刻（task-7 修复开始前的基准），用于 LAM 零改动断言
T7_START = os.environ.get("T7_START", "2026-10-16 11:50:00")
T7_START_DT = datetime.datetime.strptime(T7_START, "%Y-%m-%d %H:%M:%S")

# index.html accept -> 约定扩展名（UI 承诺 = 真实能力 的对照表）
ACCEPT_TO_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, tag: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((tag, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}  — {detail}", flush=True)
        return bool(ok)

    def info(self, tag: str, detail: str) -> None:
        print(f"[INFO] {tag}  — {detail}", flush=True)

    def fail_note(self, tag: str, detail: str) -> None:
        self.rows.append((tag, False, detail))
        print(f"[FAIL] {tag}  — {detail}", flush=True)

    def summary(self) -> int:
        p = sum(1 for _, ok, _ in self.rows if ok)
        f = len(self.rows) - p
        print("\n================ 上传格式归一化 验收汇总 ================", flush=True)
        for tag, ok, _ in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'} {tag}", flush=True)
        print(f"-------- {p} passed / {f} failed --------", flush=True)
        return 1 if f else 0


# ------------------------------------------------------------------ fixtures
def make_fixtures(outdir: Path) -> dict:
    """从真实示例 PNG 转出各格式样本；每个文件都记录 sha256 与 PIL 真实格式。"""
    from PIL import Image

    outdir.mkdir(parents=True, exist_ok=True)
    base = Image.open(SAMPLE).convert("RGB")
    # 归一化到较小尺寸，加快 GPU 生成，同时保持人脸结构
    base = base.resize((384, 384), Image.LANCZOS)

    specs = [
        ("true_png", "PNG", ".png"),
        ("true_jpeg", "JPEG", ".jpg"),
        ("webp", "WEBP", ".webp"),
        ("bmp", "BMP", ".bmp"),
        ("tiff", "TIFF", ".tiff"),
        ("gif", "GIF", ".gif"),
    ]
    meta: dict = {}
    for name, fmt, ext in specs:
        p = outdir / f"{name}{ext}"
        kw = {}
        if fmt == "JPEG":
            kw["quality"] = 95
        base.save(p, format=fmt, **kw)
        meta[name] = {"path": str(p), "ext": ext, "sha256": _sha(p), "size": p.stat().st_size}

    # 内容=WEBP 但后缀=.png（证明按内容而非后缀判定）
    webp_bytes = (outdir / "webp.webp").read_bytes()
    renamed = outdir / "webp_renamed.png"
    renamed.write_bytes(webp_bytes)
    meta["webp_renamed_png"] = {
        "path": str(renamed), "ext": ".png", "sha256": _sha(renamed),
        "size": renamed.stat().st_size,
    }

    # 坏文件：README.md 改名 .png
    bad = outdir / "readme_renamed.png"
    bad.write_bytes(README.read_bytes())
    meta["bad_readme_png"] = {"path": str(bad), "ext": ".png", "sha256": _sha(bad),
                              "size": bad.stat().st_size}

    # 超限：21MB 随机字节（大小检查先于 PIL 校验，稳定得 413）
    big = outdir / "too_big.png"
    big.write_bytes(os.urandom(MAX_UPLOAD_BYTES + 1024 * 1024))
    meta["too_big"] = {"path": str(big), "ext": ".png", "sha256": _sha(big),
                       "size": big.stat().st_size}

    # 空文件
    empty = outdir / "empty.png"
    empty.write_bytes(b"")
    meta["empty"] = {"path": str(empty), "ext": ".png", "sha256": _sha(empty), "size": 0}

    # 逐文件用 PIL 复核真实格式
    for name, m in meta.items():
        if name in ("too_big", "empty"):
            continue
        try:
            with Image.open(m["path"]) as im:
                m["pil_format"] = im.format
                m["pil_size"] = list(im.size)
        except Exception as exc:  # noqa: BLE001
            m["pil_format"] = f"ERR:{type(exc).__name__}"

    (outdir / "fixtures.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ HTTP
def wait_ready(base: str, timeout: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{base}/api/status", timeout=5)
            if r.ok and r.json().get("ready"):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    return False


def submit_upload(base: str, path: Path, filename: str | None = None,
                  timeout: float = 60.0) -> requests.Response:
    fn = filename or path.name
    with open(path, "rb") as fh:
        return requests.post(
            f"{base}/api/generate",
            files={"image": (fn, fh, "application/octet-stream")},
            timeout=timeout,
        )


def poll_job(base: str, jid: str, timeout: float = 240.0, every: float = 2.0) -> dict:
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{base}/api/jobs/{jid}", timeout=10)
            if r.ok:
                last = r.json()
                if last.get("state") in ("succeeded", "failed", "cancelled"):
                    return last
        except Exception:  # noqa: BLE001
            pass
        time.sleep(every)
    last["_poll_timeout"] = True
    return last


def face_stats(job_dir: Path) -> dict:
    """读 job 目录 face.png 的尺寸/格式/非背景占比。"""
    from PIL import Image

    out: dict = {"err": None}
    p = job_dir / "face.png"
    if not p.exists():
        globbed = sorted(job_dir.glob("face.*"))
        out["err"] = f"face.png 不存在（目录: {[q.name for q in globbed]}）"
        return out
    with Image.open(p) as im:
        out["format"] = im.format
        out["size"] = list(im.size)
        rgb = im.convert("RGB")
        px = list(rgb.getdata())
    n = len(px)
    nonwhite = sum(1 for r, g, b in px if not (r > 245 and g > 245 and b > 245))
    out["non_bg_ratio"] = nonwhite / max(n, 1)
    out["path"] = str(p)
    return out


def job_dir_of(jid: str) -> Path:
    return WORK_ROOT / "jobs" / jid


def input_files(jid: str) -> list[Path]:
    d = job_dir_of(jid)
    return sorted(d.glob("input.*")) if d.exists() else []


# ------------------------------------------------------------------ suites
def suite_fixtures(rep: Report, meta: dict) -> None:
    print("\n--- F: fixture 与基准证据 ---", flush=True)
    for name in ("true_png", "true_jpeg", "webp", "bmp"):
        rep.check(f"F.{name} fixture 可用", meta[name]["size"] > 0,
                  f"{meta[name]['path']} {meta[name]['size']}B pil={meta[name]['pil_format']}")
    rep.check("F.bad_readme_png 是 README 改名", meta["bad_readme_png"]["size"] > 0,
              f"{meta['bad_readme_png']['size']}B")
    rep.check("F.too_big > 20MB", meta["too_big"]["size"] > MAX_UPLOAD_BYTES,
              f"{meta['too_big']['size']}B")
    rep.check("F.empty 0 字节", meta["empty"]["size"] == 0, "0B")

    # 修复前证据（只读旧产物 + /api/jobs）
    for jid in ("09ef12e402a4", "98e7e5c4c438"):
        d = job_dir_of(jid)
        olds = sorted(d.glob("input.*")) if d.exists() else []
        names = [p.name for p in olds]
        rep.check(f"F.bug_repro.{jid} 修复前原样按后缀落盘 input.webp",
                  names == ["input.webp"], f"{jid} inputs={names}")


def suite_reject(rep: Report, base: str, meta: dict) -> None:
    print("\n--- R: 坏文件 / 超限 / 空文件（不发生成，不占 GPU） ---", flush=True)
    # README.md 改名 .png -> 400 且文案干净
    r = submit_upload(base, Path(meta["bad_readme_png"]["path"]))
    body = _safe_text(r)
    rep.check("R.bad_readme_400", r.status_code == 400, f"HTTP {r.status_code} {body[:180]}")
    lowered = body.lower()
    banned = ["unsupported image file", "torchvision", "traceback"]
    hit = [b for b in banned if b.lower() in lowered]
    rep.check("R.bad_readme_text_clean", not hit,
              f"命中内部术语: {hit}" if hit else "文案不含 Unsupported image file/torchvision/Traceback")

    # >20MB -> 413
    r = submit_upload(base, Path(meta["too_big"]["path"]), timeout=120)
    rep.check("R.too_big_413", r.status_code == 413,
              f"HTTP {r.status_code} {_safe_text(r)[:120]}")

    # 0 字节 -> 400
    r = submit_upload(base, Path(meta["empty"]["path"]))
    rep.check("R.empty_400", r.status_code == 400,
              f"HTTP {r.status_code} {_safe_text(r)[:120]}")

    # TIFF/GIF：不在 ALLOWED_SUFFIX ⇒ 按设计拒绝(400)，不算 FAIL，但必须如实记录
    for name, ext in (("tiff", ".tiff"), ("gif", ".gif")):
        r = submit_upload(base, Path(meta[name]["path"]))
        rep.check(f"R.design_reject.{name}", r.status_code == 400,
                  f"{ext} -> HTTP {r.status_code}（按设计拒绝，非 FAIL）: {_safe_text(r)[:100]}")


def _safe_text(r: requests.Response) -> str:
    try:
        return r.text
    except Exception:  # noqa: BLE001
        return "<no text>"


def gen_and_assert(rep: Report, base: str, tag: str, path: Path, label: str,
                   expect_success: bool = True,
                   upload_name: str | None = None) -> dict | None:
    """上传 -> 轮询终态 -> 断言。严格串行（本函数返回前不 submit 下一个）。"""
    r = submit_upload(base, path, filename=upload_name)
    if r.status_code != 200:
        rep.check(f"{tag}.http200", False, f"HTTP {r.status_code} {_safe_text(r)[:200]}")
        return None
    j = r.json()
    jid = j.get("id") or j.get("job_id")
    rep.info(f"{tag} submit", f"job={jid} state={j.get('state')} "
                              f"input_size={j.get('input_size')} name={label}")
    final = poll_job(base, jid)
    state = final.get("state")
    err = final.get("error")
    if not expect_success:
        rep.check(f"{tag}.state_failed", state == "failed", f"state={state} err={err}")
        return final
    ok = rep.check(f"{tag}.succeeded", state == "succeeded",
                   f"job={jid} state={state} err={err}")
    if not ok:
        return final
    fs = face_stats(job_dir_of(jid))
    rep.check(f"{tag}.face_512_png",
              fs.get("err") is None and fs.get("size") == [512, 512]
              and (fs.get("format") or "").upper() == "PNG",
              f"size={fs.get('size')} format={fs.get('format')} err={fs.get('err')}")
    ratio = fs.get("non_bg_ratio", 0.0)
    rep.check(f"{tag}.face_nonbg_gt5pct", ratio > 0.05, f"non-bg={ratio*100:.2f}%")
    fs["job_id"] = jid
    return {"job": final, "face": fs, "jid": jid}


def suite_core(rep: Report, base: str, meta: dict) -> dict:
    print("\n--- C: 曾失败格式 WEBP / BMP 端到端（核心） ---", flush=True)
    out: dict = {}
    for name in ("webp", "bmp"):
        res = gen_and_assert(rep, base, f"C.{name}", Path(meta[name]["path"]),
                             meta[name]["path"])
        if res:
            out[name] = res
    return out


def suite_zero_regression(rep: Report, base: str, meta: dict) -> dict:
    print("\n--- Z: 真 PNG / 真 JPEG 逐字节零回归（关键） ---", flush=True)
    out: dict = {}
    for name in ("true_png", "true_jpeg"):
        m = meta[name]
        src = Path(m["path"])
        res = gen_and_assert(rep, base, f"Z.{name}", src, src.name)
        if not res:
            continue
        jid = res["jid"]
        infiles = input_files(jid)
        rep.check(f"Z.{name}.input_exists", len(infiles) == 1,
                  f"input files={[p.name for p in infiles]}")
        if len(infiles) == 1:
            stored = infiles[0]
            stored_sha = _sha(stored)
            rep.check(f"Z.{name}.bytes_identical", stored_sha == m["sha256"],
                      f"uploaded sha256={m['sha256'][:16]} stored({stored.name})="
                      f"{stored_sha[:16]} identical={stored_sha == m['sha256']}")
        out[name] = res
    return out


def suite_content_over_ext(rep: Report, base: str, meta: dict) -> dict:
    print("\n--- X: 内容≠后缀（WEBP 改名 .png） ---", flush=True)
    from PIL import Image

    m = meta["webp_renamed_png"]
    src = Path(m["path"])
    res = gen_and_assert(rep, base, "X.webp_as_png", src, "webp_renamed.png")
    if not res:
        return {}
    jid = res["jid"]
    infiles = input_files(jid)
    rep.check("X.input_exists", len(infiles) >= 1, f"input files={[p.name for p in infiles]}")
    if infiles:
        with Image.open(infiles[0]) as im:
            fmt = im.format
        rep.check("X.stored_content_is_png", (fmt or "").upper() == "PNG",
                  f"stored={infiles[0].name} PIL.format={fmt}（按内容判定生效）")
    return {"webp_as_png": res}


def suite_sample(rep: Report, base: str) -> dict:
    print("\n--- S: 示例图路径回归 sample=status.png ---", flush=True)
    r = requests.post(f"{base}/api/generate", data={"sample": "status.png"}, timeout=60)
    rep.check("S.sample_http200", r.status_code == 200,
              f"HTTP {r.status_code} {_safe_text(r)[:120]}")
    if r.status_code != 200:
        return {}
    j = r.json()
    jid = j.get("id") or j.get("job_id")
    rep.info("S.sample submit", f"job={jid} state={j.get('state')}")
    final = poll_job(base, jid)
    rep.check("S.sample_succeeded", final.get("state") == "succeeded",
              f"job={jid} state={final.get('state')} err={final.get('error')}")
    return {"status_png": {"jid": jid, "job": final}}


def suite_accept_consistency(rep: Report, base: str, meta: dict,
                             core: dict, zero: dict) -> None:
    print("\n--- A: index.html accept == 真实能力 一致性 ---", flush=True)
    html = INDEX_HTML.read_text(encoding="utf-8", errors="replace")
    mm = re.search(r'accept="([^"]+)"', html)
    rep.check("A.accept_found", bool(mm), f"accept={mm.group(1) if mm else None}")
    if not mm:
        return
    accept = [x.strip() for x in mm.group(1).split(",") if x.strip()]
    rep.info("A.accept", str(accept))
    missing = [a for a in accept if a not in ACCEPT_TO_SUFFIX]
    rep.check("A.accept_all_known", not missing, f"未知项={missing}")
    # README 格式说明与 accept 一致
    readme = README.read_text(encoding="utf-8", errors="replace")
    rep.info("A.readme_has_webp", f"'WEBP' in README -> {'WEBP' in readme.upper()}")
    rep.check("A.readme_mentions_webp", "WEBP" in readme.upper(),
              "README 提到 WEBP（与 accept 一致）")
    rep.check("A.readme_mentions_png_jpg",
              "PNG" in readme.upper() and ("JPG" in readme.upper() or "JPEG" in readme.upper()),
              "README 提到 PNG/JPG")

    # accept 里每种格式都必须真能 succeeded（复用已跑过的结果，避免重复占 GPU）
    have = {"image/png": zero.get("true_png"), "image/jpeg": zero.get("true_jpeg"),
            "image/webp": core.get("webp"), "image/bmp": core.get("bmp")}
    for mime in accept:
        res = have.get(mime)
        rep.check(f"A.cap.{mime}", bool(res and res.get("face")),
                  f"reuse job={res.get('jid') if res else None}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="task-8 上传格式归一化独立验收")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--fixture-dir", default="/tmp/uft_fix")
    ap.add_argument("--fixtures-only", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true", help="只跑非生成类断言")
    ap.add_argument("--only-invariants", action="store_true", help="只跑 I 组产品级不变量")
    args = ap.parse_args()

    rep = Report()
    fdir = Path(args.fixture_dir)
    print(f"task-8 上传格式归一化验收 base={args.base} fixtures={fdir}", flush=True)
    try:
        meta = make_fixtures(fdir)
    except Exception as exc:  # noqa: BLE001
        print(f"fixture 生成失败: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(f"fixture 就绪: {json.dumps({k: v.get('pil_format') for k, v in meta.items()}, ensure_ascii=False)}",
          flush=True)

    suite_fixtures(rep, meta)
    if args.only_invariants:
        print("\n--- I: 产品级不变量（仅 I 组复跑） ---", flush=True)
        suite_invariants(rep)
        return rep.summary()
    if args.fixtures_only:
        return rep.summary()

    if not wait_ready(args.base):
        print("服务未 ready，终止", flush=True)
        rep.fail_note("P.service_ready", "服务未 ready")
        return rep.summary()

    suite_reject(rep, args.base, meta)
    if args.skip_gpu:
        return rep.summary()

    core = suite_core(rep, args.base, meta)
    zero = suite_zero_regression(rep, args.base, meta)
    suite_content_over_ext(rep, args.base, meta)
    suite_sample(rep, args.base)
    suite_accept_consistency(rep, args.base, meta, core, zero)

    # 产品级不变量
    print("\n--- I: 产品级不变量 ---", flush=True)
    suite_invariants(rep)
    return rep.summary()


def suite_invariants(rep: Report) -> None:
    import subprocess

    def sh(cmd: str) -> str:
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=120).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<ERR {exc}>"

    porcelain = sh("git -C /workspace status --porcelain")
    rep.info("I.porcelain", porcelain.replace("\n", " | ")[:600])
    bad_static = [ln for ln in porcelain.splitlines() if "lam3d/static/" in ln]
    rep.check("I.no_static_changes", not bad_static, f"static 改动={bad_static}")
    # 基线不靠硬编码 allowlist（写错路径就会误报），而是按**开工时刻**判定：
    # 任何 LAM 条目的文件 mtime 都早于 T7_START ⇒ 是 T7 之前就存在的既有脏文件（与本次修复无关）；
    # 只要出现 mtime >= T7_START 的 LAM 条目，才是真回归。这条比 allowlist 严格且不会因路径笔误误判。
    bad_lam, pre_lam = [], []
    for ln in porcelain.splitlines():
        if "LAM/" not in ln:
            continue
        # 不能用固定偏移 ln[3:]：# 首行若被 strip 掉前导空格，偏移就会错位。
        # 用 split(maxsplit=1) 去掉状态码（M/??/AM 等），拿到路径本体。
        parts = ln.split(maxsplit=1)
        if len(parts) < 2:
            continue
        rel = parts[1].strip().strip('"')
        full = os.path.join("/workspace", rel)
        try:
            mt = datetime.datetime.fromtimestamp(os.stat(full).st_mtime)
        except OSError:
            bad_lam.append(f"{ln} <missing>")
            continue
        (pre_lam if mt < T7_START_DT else bad_lam).append(f"{ln} (mtime={mt:%H:%M:%S})")
    rep.info("I.lam_preexisting", f"{pre_lam}")
    rep.check("I.no_new_lam_changes", not bad_lam,
              f"T7 开工后新增/改动的 LAM 条目={bad_lam}")

    newt = sh(f'find /workspace/LAM -newermt "{T7_START}" -type f '
              f'-not -path "*/__pycache__/*" 2>/dev/null | head -20')
    rep.check("I.lam_zero_touch", not newt, f"LAM 改动文件={newt!r} (since {T7_START})")

    n_html = sh("find /workspace/lam3d/static -maxdepth 1 -name '*.html' | wc -l")
    rep.check("I.static_one_html", n_html.strip() == "1", f"html 数={n_html.strip()}")

    pc = sh("cd /workspace/lam3d && /usr/local/miniconda3/envs/lam/bin/python -m py_compile "
            "lam3d/server/app.py && echo OK")
    rep.check("I.app_compiles", pc.strip().endswith("OK"), f"py_compile={pc[:120]}")


if __name__ == "__main__":
    sys.exit(main())
