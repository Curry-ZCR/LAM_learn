#!/usr/bin/env python
"""T9 — 浏览器真实「选本机文件 → 点生成」路径端到端验证（WEBP / BMP / PNG 零回归）。

为什么需要这个脚本（与 HTTP 层直打字节的 T8 不同）：
    用户真实失败路径是「在浏览器里点『选择文件』→ 点生成」。此前所有验证都是
    HTTP 层直接 POST 字节，**没有走过前端的 `#file` input accept 过滤 / `change`
    事件 / FormData 组装 / File 对象传递**这条链路。本脚本用真实无头 Chromium +
    `page.setInputFiles('#file', <本机文件>)`（Playwright 模拟真实选文件的标准做法，
    会触发前端的 change 事件）走完整流程。

只读生产代码：本脚本不修改 `lam3d/lam3d/**`、`lam3d/static/**`、`/workspace/LAM/**`。

用法::

    /usr/local/miniconda3/envs/lam/bin/python scripts/e2e_upload_formats.py \
        [--cases webp,bmp,png,mismatch] [--tag t9]

产物：
- 截图 `workspace/e2e/t9_<case>_ok.png`
- 结构化报告 `/tmp/t9/t9_report.json`
- 结束标志 `/tmp/t9/t9_upload_formats.done`（内容 `exit=<code>`），供 20s 超时的 rw_exec 轮询
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("LAM3D_BASE", "http://127.0.0.1:8000")
SHOT_DIR = ROOT / "workspace" / "e2e"
SHOT_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR = Path("/tmp/t9")
FIX_DIR.mkdir(parents=True, exist_ok=True)
DONE_FLAG = FIX_DIR / "t9_upload_formats.done"
REPORT = FIX_DIR / "t9_report.json"

SRC = Path("/workspace/LAM/assets/sample_input/status.png")
BG = (0x0F, 0x13, 0x1C)          # 画布背景色（CONTRACT_v2 §8 口径）
FG_TOL = 6                        # 任一通道差 > 6 记为前景
VIEWPORT = {"width": 1600, "height": 950}
DSF = 2

RESULTS: list[dict] = []
CASES: list[dict] = []
NOTE: list[str] = []


# ────────────────────────────────────────────────────────── 记录 / 断言
def check(cid: str, name: str, ok: bool, extra: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {cid} {name}" + (f"  — {extra}" if extra else ""), flush=True)
    RESULTS.append({"id": cid, "name": name, "ok": bool(ok), "tag": tag, "extra": extra})
    return ok


def note(msg: str) -> None:
    print(f"       · {msg}", flush=True)
    NOTE.append(msg)


def get_json(url: str, timeout: float = 8.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def wait_ready(seconds: int = 240) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        st = get_json(f"{BASE}/api/status")
        if st and st.get("ready"):
            note(f"服务就绪：{st}")
            return True
        if st and st.get("error"):
            note(f"引擎预热失败：{st['error']}")
            return False
        time.sleep(3)
    return False


# ────────────────────────────────────────────────────────── 像素工具
def img_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def canvas_shot(canvas_loc) -> Image.Image:
    """拿画布的真实上屏像素。

    优先 `canvas.toDataURL()`；WebGL 未开 preserveDrawingBuffer 时合成后缓冲区
    会被清空，此时退回**元素截图**（同样是真实上屏像素），并记录所用通道。
    """
    try:
        url = canvas_loc.evaluate("(c) => c.toDataURL('image/png')")
        if url and url.startswith("data:image/png"):
            im = img_from_bytes(base64.b64decode(url.split(",", 1)[1]))
            if len(im.getcolors(maxcolors=1 << 24) or []) > 1:
                im.info["via"] = "toDataURL"
                return im
    except Exception as e:  # noqa: BLE001
        note(f"toDataURL 失败（回退元素截图）：{e}")
    im = img_from_bytes(canvas_loc.screenshot())
    im.info["via"] = "element_screenshot"
    return im


def fg_stats(im: Image.Image, bg: tuple = BG, tol: int = FG_TOL) -> dict:
    """非背景像素覆盖率。背景 = #0f131c；前景 = 与背景任一通道差 > tol。

    同时给出「出现最多的颜色」作为对照（防画布背景被别的颜色主导时误判）。
    """
    w, h = im.size
    colors = im.getcolors(maxcolors=1 << 24) or []
    modal = max(colors, key=lambda c: c[0])[1][:3] if colors else (0, 0, 0)
    px = im.load()
    step = max(1, min(w, h) // 500)
    tot = lit = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y][:3]
            tot += 1
            if abs(r - bg[0]) > tol or abs(g - bg[1]) > tol or abs(b - bg[2]) > tol:
                lit += 1
    return {"w": w, "h": h, "n_colors": len(colors), "modal": list(modal),
            "coverage": round(lit / max(tot, 1), 4), "sampled": tot, "fg_px": lit}


# ────────────────────────────────────────────────────────── 素材
def make_fixtures() -> dict:
    """从 status.png 转出真 WEBP / 真 BMP / 真 PNG + 「后缀 .png 内容是 webp」错配件。

    素材只放 /tmp/t9/，不进仓库。
    """
    with Image.open(SRC) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    fx = {
        "webp": FIX_DIR / "t9_true.webp",
        "bmp": FIX_DIR / "t9_true.bmp",
        "png": FIX_DIR / "t9_true.png",
        "mismatch": FIX_DIR / "t9_webp_mismatch.png",   # 内容是 WEBP，后缀写 .png
    }
    rgb.save(fx["webp"], format="WEBP", quality=92)
    rgb.save(fx["bmp"], format="BMP")
    rgb.save(fx["png"], format="PNG")
    rgb.save(fx["mismatch"], format="WEBP", quality=92)
    print("=== 0. 素材（由 status.png 转出，只落 /tmp/t9/）===", flush=True)
    out = {}
    for k, p in fx.items():
        with Image.open(p) as im2:
            fmt = im2.format
            size = im2.size
        out[k] = {"path": str(p), "bytes": p.stat().st_size, "format": fmt,
                  "size": f"{size[0]}x{size[1]}",
                  "suffix": p.suffix, "mismatch": fmt != p.suffix.lstrip(".").upper()}
        print(f"       {k:9s} {p}  format={fmt:5s} bytes={out[k]['bytes']:>9,} "
              f"{out[k]['size']} suffix={p.suffix} "
              f"{'⚠ 后缀/内容不一致' if out[k]['mismatch'] else ''}", flush=True)
        check("F0", f"素材 {k} 真实格式 == {fmt}", True,
              f"{p.name} {out[k]['bytes']}B {out[k]['size']}")
    return out


# ────────────────────────────────────────────────────────── 单格式流程
def run_case(page, ctx, key: str, fx: dict, watch: dict) -> dict:
    """走一遍完整的「选文件 → 点生成 → 等 succeeded → 断言渲染」流程。"""
    print(f"\n=== 前端流程：{key.upper()}（{fx['path']}）===", flush=True)
    rec: dict = {"case": key, "file": fx, "ok": False}
    c = f"T9-{key}"
    res_console = len(watch["console"])
    res_pageerr = len(watch["pageerror"])

    page.goto(BASE, wait_until="load", timeout=90_000)
    page.wait_for_timeout(1500)

    # ---- accept 过滤（前端宣传的格式必须覆盖服务端 ALLOWED_SUFFIX）
    acc = page.locator("#file").get_attribute("accept")
    rec["accept"] = acc

    # ---- 真实选文件（Playwright 模拟文件选择器，触发 change 事件）
    page.set_input_files("#file", fx["path"])
    page.wait_for_timeout(1200)

    picked = page.evaluate("""() => {
      const f = document.getElementById('file');
      const p = document.getElementById('preview');
      const g = document.getElementById('generate');
      return {n_files: f ? f.files.length : -1,
              file_name: f && f.files[0] ? f.files[0].name : null,
              file_size: f && f.files[0] ? f.files[0].size : null,
              file_type: f && f.files[0] ? f.files[0].type : null,
              preview_hidden: p ? p.hidden : null,
              preview_nw: p ? p.naturalWidth : null,
              preview_nh: p ? p.naturalHeight : null,
              preview_src: p ? (p.src || '').slice(0, 20) : null,
              generate_disabled: g ? g.disabled : null,
              dropHint_hidden: (document.getElementById('dropHint') || {}).hidden,
              errorBox_visible: (() => { const e = document.getElementById('errorBox');
                  return !!e && !e.hidden && e.offsetParent !== null; })(),
    }; }""")
    rec["picked"] = picked
    rec["error_box_after_pick"] = bool(picked.get("errorBox_visible"))
    check(f"{c}.1", f"[{key}] setInputFiles 后 File 对象进入 input",
          picked["n_files"] == 1 and picked["file_name"] == Path(fx["path"]).name,
          f"files={picked['n_files']} name={picked['file_name']} size={picked['file_size']} "
          f"type={picked['file_type']!r}")
    check(f"{c}.2", f"[{key}] 左栏缩略图出现（#preview 可见且 naturalWidth>0）",
          picked["preview_hidden"] is False and (picked["preview_nw"] or 0) > 0,
          f"hidden={picked['preview_hidden']} {picked['preview_nw']}x{picked['preview_nh']} "
          f"dropHint.hidden={picked['dropHint_hidden']}")
    check(f"{c}.3", f"[{key}] #generate 变为可用（disabled=false）",
          picked["generate_disabled"] is False,
          f"disabled={picked['generate_disabled']}")
    page.screenshot(path=str(FIX_DIR / f"t9_{key}_picked.png"))

    if picked["generate_disabled"]:
        check(f"{c}", f"[{key}] 前端流程走通并 succeeded", False, "#generate 不可用，流程中断")
        return rec

    # ---- 点生成（抓 POST /api/generate 响应）
    t0 = time.time()
    try:
        with page.expect_response(
                lambda r: r.url.rstrip("/").endswith("/api/generate")
                and r.request.method == "POST", timeout=60_000) as ri:
            page.locator("#generate").click()
        resp = ri.value
        status = resp.status
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"_raw": resp.text()[:400]}
    except Exception as e:  # noqa: BLE001
        check(f"{c}.4", f"[{key}] POST /api/generate 返回 job id", False,
              f"{type(e).__name__}: {e}")
        return rec
    job_id = body.get("id") or body.get("job_id") if isinstance(body, dict) else None
    rec["generate_api"] = {"status": status, "body": body}
    rec["job_id"] = job_id
    note(f"POST /api/generate → HTTP {status} {json.dumps(body, ensure_ascii=False)[:180]}")
    check(f"{c}.4", f"[{key}] POST /api/generate 200 且返回 job id",
          status == 200 and bool(job_id), f"status={status} job_id={job_id}")

    # ---- 等终态（自己轮询 API，不依赖前端）
    last = None
    deadline = time.time() + 420
    while time.time() < deadline:
        j = get_json(f"{BASE}/api/jobs/{job_id}") if job_id else None
        if j:
            last = {k: j.get(k) for k in ("state", "stage", "progress", "error", "elapsed")}
            if j.get("state") in ("succeeded", "failed", "canceled"):
                break
        time.sleep(2)
    elapsed = round(time.time() - t0, 1)
    state = (last or {}).get("state")
    rec.update({"state": state, "elapsed_s": elapsed, "last": last})
    check(f"{c}.5", f"[{key}] 任务终态 == succeeded", state == "succeeded",
          f"state={state} err={(last or {}).get('error')!r} queue→done {elapsed}s")

    # ---- 前端结果区 + 错误框
    try:
        page.wait_for_selector("#result:not([hidden])", timeout=30_000)
    except Exception as e:  # noqa: BLE001
        note(f"#result 未显示：{e}")
    result_visible = page.locator("#result").is_visible()
    errbox_upto_now = page.evaluate(
        "() => { const e = document.getElementById('errorBox');"
        " return !!e && !e.hidden && e.offsetParent !== null; }")
    rec["result_visible"] = result_visible
    rec["error_box_after_job"] = bool(errbox_upto_now)
    check(f"{c}.6", f"[{key}] #result 显示 且 #errorBox 不显示",
          result_visible and not errbox_upto_now,
          f"result={result_visible} errorBox={errbox_upto_now}")

    # ---- #faceImg 512×512
    face = {"nw": None, "nh": None}
    try:
        page.wait_for_function(
            "() => { const i = document.getElementById('faceImg');"
            " return i && i.complete && i.naturalWidth > 0; }", timeout=30_000)
        face = page.evaluate("""() => { const i = document.getElementById('faceImg');
            return {nw: i.naturalWidth, nh: i.naturalHeight,
                    src: i.src, visible: !document.getElementById('faceCard').hidden,
                    card_hidden: document.getElementById('faceCard').hidden}; }""")
    except Exception as e:  # noqa: BLE001
        note(f"#faceImg 未就绪：{e}")
    rec["face"] = face
    check(f"{c}.7", f"[{key}] #faceImg naturalWidth==512 && naturalHeight==512",
          face.get("nw") == 512 and face.get("nh") == 512,
          f"{face.get('nw')}x{face.get('nh')} card_hidden={face.get('card_hidden')} "
          f"src={str(face.get('src'))[-40:]!r}")

    # ---- 画布真实渲染
    badge = ""
    try:
        page.wait_for_selector("#viewer canvas", timeout=45_000)
        try:
            page.wait_for_function(
                "() => { const b = document.getElementById('viewerBadge');"
                " return b && /已加载/.test(b.textContent || ''); }", timeout=45_000)
            page.wait_for_timeout(2500)
        except Exception as e:  # noqa: BLE001
            note(f"#viewerBadge 未变为「已加载」：{e}")
        badge = page.locator("#viewerBadge").inner_text()
    except Exception as e:  # noqa: BLE001
        note(f"#viewer canvas 未出现：{e}")
    rec["badge"] = badge

    canvas = page.locator("#viewer canvas")
    if canvas.count() == 0:
        check(f"{c}.8", f"[{key}] 画布非背景覆盖率 > 5%", False, "无 #viewer canvas")
        check(f"{c}.9", f"[{key}] #viewerBadge 显示已加载", False, f"badge={badge!r}")
    else:
        im = canvas_shot(canvas)
        st = fg_stats(im)
        rec["canvas"] = st | {"via": im.info.get("via")}
        check(f"{c}.8", f"[{key}] 画布非背景覆盖率 > 5%（bg=#0f131c，通道差>{FG_TOL}）",
              st["coverage"] > 0.05,
              f"coverage={st['coverage']} fg={st['fg_px']}/{st['sampled']} "
              f"colors={st['n_colors']} modal={st['modal']} via={im.info.get('via')} "
              f"canvas={st['w']}x{st['h']}")
        check(f"{c}.9", f"[{key}] #viewerBadge 显示已加载", "已加载" in (badge or ""),
              f"badge={badge!r}")

    # ---- 终检：错误框 / 本次用例新增的 console 错误
    final_err = page.evaluate(
        "() => { const e = document.getElementById('errorBox');"
        " return {hidden: e.hidden, visible: !e.hidden && e.offsetParent !== null,"
        "         text: (document.getElementById('errorText') || {}).textContent || ''}; }")
    rec["error_box_final"] = final_err
    new_console = watch["console"][res_console:]
    new_pageerr = watch["pageerror"][res_pageerr:]
    rec["console_errors_new"] = new_console
    rec["pageerrors_new"] = new_pageerr
    check(f"{c}.10", f"[{key}] #errorBox 全程 hidden",
          (not rec["error_box_after_pick"]) and (not rec["error_box_after_job"])
          and final_err["hidden"] is True,
          f"pick={rec['error_box_after_pick']} job={rec['error_box_after_job']} "
          f"final.hidden={final_err['hidden']} text={final_err['text']!r}")
    check(f"{c}.11", f"[{key}] 本用例新增 console error 0 条",
          not new_console, f"{len(new_console)} 条：{json.dumps(new_console[:5], ensure_ascii=False)}")
    check(f"{c}.12", f"[{key}] 本用例新增 pageerror 0 条",
          not new_pageerr, f"{len(new_pageerr)} 条：{json.dumps(new_pageerr[:3], ensure_ascii=False)}")

    shot = SHOT_DIR / f"t9_{key}_ok.png"
    page.screenshot(path=str(shot), full_page=True)
    with Image.open(shot) as im2:
        size = f"{im2.size[0]}x{im2.size[1]}"
    rec["screenshot"] = {"path": str(shot), "bytes": shot.stat().st_size, "size": size}
    print(f"       📸 {shot} ({size}, {shot.stat().st_size:,} bytes)", flush=True)
    rec["ok"] = bool(state == "succeeded" and face.get("nw") == 512
                     and rec.get("canvas", {}).get("coverage", 0) > 0.05
                     and final_err["hidden"] and not new_console and not new_pageerr)
    return rec


# ────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="webp,bmp,png,mismatch",
                    help="逗号分隔，默认 webp,bmp,png,mismatch（mismatch=内容是 WEBP 但后缀 .png）")
    ap.add_argument("--tag", default="t9")
    args = ap.parse_args()
    order = [s.strip() for s in args.cases.split(",") if s.strip()]

    print(f"### T9 浏览器文件选择上传 — tag={args.tag} base={BASE} cases={order}", flush=True)
    if DONE_FLAG.exists():
        DONE_FLAG.unlink()

    if not wait_ready(240):
        check("S0", "服务就绪（/api/status ready:true）", False, "240s 内未就绪")
        return finish(1)
    check("S0", "服务就绪（/api/status ready:true）", True, str(get_json(f"{BASE}/api/status")))

    fx = make_fixtures()
    watch = {"console": [], "pageerror": [], "dialogs": [], "popups": [], "failed": []}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=DSF)
        ctx.set_default_timeout(60_000)
        page = ctx.new_page()
        page.on("pageerror", lambda e: watch["pageerror"].append(str(e)))
        page.on("console", lambda m: watch["console"].append(
            f"{m.type}: {m.text} @ {m.location.get('url')}") if m.type == "error" else None)
        page.on("dialog", lambda d: (watch["dialogs"].append(f"{d.type}:{d.message}"), d.dismiss()))
        ctx.on("page", lambda p: watch["popups"].append(p.url))
        page.on("requestfailed", lambda r: watch["failed"].append(
            f"{r.method} {r.url} :: {r.failure}"))

        # ---- 1. 页面加载健康（首次加载，仅一次，避免把各用例的导航混进来）
        print("\n=== 1. 页面加载 ===", flush=True)
        try:
            page.goto(BASE, wait_until="load", timeout=90_000)
            page.wait_for_timeout(2500)
            title = page.title()
            check("P1", "页面加载成功", True, f"title={title!r} url={page.url}")
        except Exception as e:  # noqa: BLE001
            check("P1", "页面加载成功", False, f"{type(e).__name__}: {e}")
            title = None
        n_load_err = len(watch["console"])
        n_load_perr = len(watch["pageerror"])
        check("P2", "首屏 console error 0 条", n_load_err == 0,
              f"{n_load_err} 条：{json.dumps(watch['console'][:6], ensure_ascii=False)}")
        check("P3", "首屏 pageerror 0 条", n_load_perr == 0,
              f"{n_load_perr} 条：{json.dumps(watch['pageerror'][:4], ensure_ascii=False)}")

        baseline = {}
        for key in order:
            try:
                rec = run_case(page, ctx, key, fx[key], watch)
            except Exception as e:  # noqa: BLE001
                import traceback
                note(f"用例 {key} 未捕获异常：{e}\n{traceback.format_exc()}")
                check(f"T9-{key}", f"[{key}] 流程无异常", False, f"{type(e).__name__}: {e}")
                rec = {"case": key, "file": fx[key], "ok": False, "exception": str(e)}
            CASES.append(rec)
            baseline[key] = {k: rec.get(k) for k in
                             ("job_id", "state", "elapsed_s", "face", "canvas", "badge",
                              "screenshot", "picked", "generate_api")}

        # ---- 全局健康
        print("\n=== 2. 全局运行时健康 ===", flush=True)
        check("H1", "全流程 console error 0 条", not watch["console"],
              f"{len(watch['console'])} 条：{json.dumps(watch['console'][:8], ensure_ascii=False)}")
        check("H2", "全流程 pageerror 0 条", not watch["pageerror"],
              f"{len(watch['pageerror'])} 条：{json.dumps(watch['pageerror'][:5], ensure_ascii=False)}")
        check("H3", "无 alert/confirm/prompt 对话框、无新标签页",
              not watch["dialogs"] and not watch["popups"],
              f"dialogs={watch['dialogs']} popups={watch['popups']}")
        check("H4", "无失败的网络请求", not watch["failed"],
              f"{len(watch['failed'])} 条：{json.dumps(watch['failed'][:5], ensure_ascii=False)}")

        try:
            ctx.close()
            browser.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- 汇总表
    print("\n=== 3. 逐格式汇总 ===", flush=True)
    print(f"{'case':9s} {'upload filename':22s} {'job id':14s} {'state':10s} {'耗时s':>7s} "
          f"{'#faceImg':>10s} {'覆盖率':>7s} {'conErr':>6s} {'结果':>5s}", flush=True)
    for rec in CASES:
        p = rec.get("picked") or {}
        face = rec.get("face") or {}
        cv = rec.get("canvas") or {}
        fn = Path(rec["file"]["path"]).name
        dims = f"{face.get('nw')}x{face.get('nh')}" if face.get("nw") else "-"
        print(f"{rec['case']:9s} {fn:22s} {str(rec.get('job_id')):14s} "
              f"{str(rec.get('state')):10s} {str(rec.get('elapsed_s')):>7s} {dims:>10s} "
              f"{str(cv.get('coverage')):>7s} {len(rec.get('console_errors_new') or []):>6d} "
              f"{'OK' if rec.get('ok') else 'BAD':>5s}", flush=True)
        note(f"{rec['case']}: {rec['file']['format']}(内容) suffix={rec['file']['suffix']} "
             f"→ job={rec.get('job_id')} {rec.get('state')} {rec.get('elapsed_s')}s | "
             f"face={dims} cov={cv.get('coverage')} via={cv.get('via')} "
             f"badge={rec.get('badge')!r} | {rec.get('screenshot', {}).get('path')}")

    # ---- 结论
    print("\n=== 4. 结论 ===", flush=True)
    ok_cases = [r["case"] for r in CASES if r.get("ok")]
    bad_cases = [r["case"] for r in CASES if not r.get("ok")]
    check("C1", "所有用例（含 webp/bmp 真机路径）端到端成功", not bad_cases,
          f"ok={ok_cases} bad={bad_cases}")
    for want in ("webp", "bmp", "png"):
        shot = SHOT_DIR / f"t9_{want}_ok.png"
        check("C2", f"截图 t9_{want}_ok.png 存在", shot.exists(),
              f"{shot} {shot.stat().st_size if shot.exists() else 0} bytes")

    n_fail = sum(1 for r in RESULTS if r["tag"] == "FAIL")
    return finish(1 if n_fail else 0)


def finish(code: int) -> int:
    rep = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "base": BASE,
        "viewport": VIEWPORT, "device_scale_factor": DSF,
        "bg": "#%02x%02x%02x" % BG, "fg_tol": FG_TOL,
        "passed": sum(1 for r in RESULTS if r["tag"] == "PASS"),
        "failed": sum(1 for r in RESULTS if r["tag"] == "FAIL"),
        "results": RESULTS, "cases": CASES, "notes": NOTE, "exit_code": code,
    }
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n==== {rep['passed']} passed / {rep['failed']} failed ====", flush=True)
    for r in RESULTS:
        if r["tag"] == "FAIL":
            print(f"  FAIL  [{r['id']}] {r['name']} — {r['extra']}", flush=True)
    print(f"report → {REPORT}", flush=True)
    DONE_FLAG.write_text(f"exit={code}\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
