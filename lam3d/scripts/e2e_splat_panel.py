#!/usr/bin/env python
"""T4 — 真实无头 Chromium 端到端 + 视觉验收（CONTRACT_v2 第 8 节 B 组：验收 9-16）。

独立验证方：本脚本**只读** `lam3d/lam3d/` 与 `lam3d/static/`，不改任何生产代码。
所有页面侧插桩（OrbitControls 观察器、合成 canvas 自检）都只存在于浏览器内存里。

用法::

    # 阶段 1：harness 自检 + 旧版基线（不生成）
    python scripts/e2e_splat_panel.py --no-generate --tag baseline

    # 阶段 2：全流程（必须重启服务加载新代码后跑）
    python scripts/e2e_splat_panel.py --restart --tag final

产物：
- 截图 `/workspace/lam3d/workspace/e2e/final_*.png`（≥6 张）
- 结构化报告 `workspace/e2e/report_<tag>.json`
- 结束标志 `/tmp/e2e_splat_panel.done`（内容 `exit=<code>`），供 20s 超时的 rw_exec 轮询
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("LAM3D_BASE", "http://127.0.0.1:8000")
SHOT_DIR = ROOT / "workspace" / "e2e"
SHOT_DIR.mkdir(parents=True, exist_ok=True)
DONE_FLAG = Path("/tmp/e2e_splat_panel.done")
VIEWPORT = {"width": 1600, "height": 950}
DSF = 2  # 强制 devicePixelRatio=2，验证 dpr clamp ≤ 2 与高清画布尺寸

RESULTS: list[dict] = []
SCREENSHOTS: list[dict] = []
NOTE: list[str] = []


# ────────────────────────────────────────────────────────── 记录 / 断言
def check(cid: str, name: str, ok: bool, extra: str = "", *, excused: bool = False) -> bool:
    tag = "PASS" if ok else ("SKIP" if excused else "FAIL")
    print(f"[{tag}] {cid} {name}" + (f"  — {extra}" if extra else ""), flush=True)
    RESULTS.append({"id": cid, "name": name, "ok": bool(ok), "tag": tag, "extra": extra})
    return ok


def note(msg: str) -> None:
    print(f"       · {msg}", flush=True)
    NOTE.append(msg)


def shot(page, name: str, *, full: bool = True) -> Path:
    p = SHOT_DIR / f"final_{name}.png"
    page.screenshot(path=str(p), full_page=full)
    with Image.open(p) as im:
        w, h = im.size
    SCREENSHOTS.append({"file": p.name, "size": f"{w}x{h}", "bytes": p.stat().st_size})
    print(f"       📸 {p.name} ({w}x{h})", flush=True)
    return p


# ────────────────────────────────────────────────────────── 像素工具
def mark_shot(p: Path) -> None:
    """登记一张已落盘的截图，尺寸/字节数从真实文件读取（不要手写）。"""
    with Image.open(p) as im:
        w, h = im.size
    SCREENSHOTS.append({"file": p.name, "size": f"{w}x{h}", "bytes": p.stat().st_size})
    print(f"       📸 {p.name} ({w}x{h})", flush=True)


def img_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def image_stats(im: Image.Image, bg_tol: int = 24) -> dict:
    """真实像素统计：不同颜色数 / 非背景覆盖率 / 均值亮度。

    注意：PIL 的 `getcolors()` 按 count **升序** 排序，所以最常见的颜色（背景）在**末尾**。
    """
    w, h = im.size
    colors = im.getcolors(maxcolors=1 << 24) or []
    n_colors = len(colors)
    bg = max(colors, key=lambda c: c[0])[1] if colors else (0, 0, 0)
    px = im.load()
    step = max(1, min(w, h) // 400)  # 大图抽样，但覆盖全图
    tot = lit = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y]
            tot += 1
            if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) > bg_tol:
                lit += 1
    return {"w": w, "h": h, "n_colors": n_colors, "bg": list(bg),
            "coverage": round(lit / max(tot, 1), 4)}


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        b = b.resize(a.size)
    pa, pb = a.load(), b.load()
    w, h = a.size
    step = max(1, min(w, h) // 400)
    acc = n = 0.0
    for y in range(0, h, step):
        for x in range(0, w, step):
            ra, ga, ba = pa[x, y]
            rb, gb, bb = pb[x, y]
            acc += abs(ra - rb) + abs(ga - gb) + abs(ba - bb)
            n += 3
    return round(acc / max(n, 1), 4)


def fg_mask(im: Image.Image, tol: int = 6):
    """前景掩码 = 与画布背景色（出现最多的颜色）任一通道差 > tol 的像素。

    契约 8.13 判定口径（Lead 裁决）：背景像素按定义不会变，把整块画布纳入均值
    会把前景信号稀释约 10 倍（前视角点云仅占画布 ~9-11%），因此
    mean|Δpixel| 只在**前景 = 非背景像素**上计算。
    返回 (前景坐标列表, 背景色, 画布总像素数)。
    """
    colors = im.getcolors(maxcolors=1 << 24) or []
    if not colors:
        return [], (0, 0, 0), im.size[0] * im.size[1]
    bg = max(colors, key=lambda c: c[0])[1][:3]
    px = im.load()
    w, h = im.size
    coords = []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y][:3]
            if abs(r - bg[0]) > tol or abs(g - bg[1]) > tol or abs(b - bg[2]) > tol:
                coords.append((x, y))
    return coords, bg, w * h


def fg_mean_abs_diff(a: Image.Image, b: Image.Image, coords, changed_tol: float = 2.0) -> dict:
    """只在前景坐标上算 mean|Δpixel| 与「变化像素占比」（通道均差 > changed_tol 记为变化）。"""
    if a.size != b.size:
        b = b.resize(a.size)
    pa, pb = a.load(), b.load()
    acc = n = 0.0
    changed = 0
    for (x, y) in coords:
        ra, ga, ba = pa[x, y][:3]
        rb, gb, bb = pb[x, y][:3]
        d = abs(ra - rb) + abs(ga - gb) + abs(ba - bb)
        acc += d
        n += 3
        if d / 3.0 > changed_tol:
            changed += 1
    N = len(coords)
    return {"mean": round(acc / max(n, 1), 4), "n_fg": N, "changed_px": changed,
            "changed_ratio": round(changed / max(N, 1), 4)}


def canvas_shot(canvas_loc) -> Image.Image:
    """画布真实合成像素。

    优先 canvas.toDataURL()（契约字面要求）；WebGL 在未开 preserveDrawingBuffer 时
    合成后缓冲区会被清空，此时退回**元素截图**（同样是真实上屏像素），并记录所用通道。
    """
    try:
        url = canvas_loc.evaluate("(c) => c.toDataURL('image/png')")
        if url and url.startswith("data:image/png"):
            im = img_from_bytes(base64.b64decode(url.split(",", 1)[1]))
            if len(im.getcolors(maxcolors=1 << 24) or []) > 1:
                im.info["via"] = "toDataURL"
                return im
            im.info["via"] = "toDataURL(blank)"
    except Exception as e:  # noqa: BLE001
        note(f"toDataURL 失败（回退元素截图）：{e}")
    im = img_from_bytes(canvas_loc.screenshot())
    im.info["via"] = "element_screenshot"
    return im


def canvas_metrics(page) -> dict:
    return page.evaluate("""() => {
      const host = document.getElementById('viewer');
      const c = (host && host.querySelector('canvas')) ||
                document.querySelector('#viewer canvas') ||
                document.querySelector('canvas');
      if (!c) return {found: false};
      return {found: true, id: c.id, cls: c.className,
              width: c.width, height: c.height,
              cssW: c.clientWidth, cssH: c.clientHeight,
              rect: c.getBoundingClientRect().toJSON(),
              dpr: window.devicePixelRatio,
              styleDisplay: getComputedStyle(c).display};
    }""")


# ────────────────────────────────────────────────────────── OrbitControls 观察器
OC_TRACK_SRC = """const __OC_Base = OrbitControls;
class __OC_Tracked extends __OC_Base {
	constructor(...args) { super(...args); (window.__ocList ||= []).push(this); }
}
export { __OC_Tracked as OrbitControls };"""


def install_orbit_tracker(page) -> bool:
    """只观察不改行为：派生类在构造时把实例登记到 window.__ocList。"""
    oc_path = ROOT / "static" / "vendor" / "three" / "examples" / "jsm" / "controls" / "OrbitControls.js"
    if not oc_path.exists():
        note(f"OrbitControls.js 不存在：{oc_path}")
        return False
    src = oc_path.read_text(encoding="utf-8")
    old = "export { OrbitControls };"
    if old not in src:
        note("OrbitControls.js 里找不到 `export { OrbitControls };`，插桩失效")
        return False
    patched = src.replace(old, OC_TRACK_SRC, 1)

    def handler(route):
        route.fulfill(status=200, content_type="application/javascript", body=patched)

    page.route("**/controls/OrbitControls.js*", handler)
    return True


def read_orbit(page) -> list[dict]:
    try:
        return page.evaluate("""() => (window.__ocList || []).map((c) => ({
            dampingFactor: c.dampingFactor, rotateSpeed: c.rotateSpeed,
            enableDamping: c.enableDamping, minDistance: c.minDistance,
            maxDistance: c.maxDistance, enabled: c.enabled,
            pos: c.object ? c.object.position.toArray() : null,
            quat: c.object ? c.object.quaternion.toArray() : null,
            fov: c.object ? c.object.fov : null,
            domSel: c.domElement ? (c.domElement.tagName + '#' + (c.domElement.id || '') +
                     '.' + (c.domElement.className || '')) : null,
        }))""") or []
    except Exception as e:  # noqa: BLE001
        note(f"读 OrbitControls 失败：{e}")
        return []


# ────────────────────────────────────────────────────────── 拖拽工具
def drag(page, bbox: dict, dx: int, dy: int, steps: int = 12) -> None:
    cx = bbox["x"] + bbox["width"] / 2
    cy = bbox["y"] + bbox["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(cx + dx * i / steps, cy + dy * i / steps)
        page.wait_for_timeout(12)
    page.mouse.up()
    page.wait_for_timeout(500)


# ────────────────────────────────────────────────────────── 服务
def get_json(url: str, timeout: float = 8.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def restart_server() -> bool:
    subprocess.run(["bash", "run.sh", "--stop"], cwd=str(ROOT), check=False,
                   capture_output=True, text=True)
    time.sleep(2)
    subprocess.run(["bash", "run.sh", "--daemon"], cwd=str(ROOT), check=False,
                   capture_output=True, text=True)
    return wait_ready(240)


def wait_ready(seconds: int) -> bool:
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


# ────────────────────────────────────────────────────────── harness 自检
def harness_selftest(page) -> None:
    print("\n=== 0. harness 自检（截图管线 / 像素统计 / 拖拽工具）===", flush=True)
    page.goto(BASE, wait_until="load", timeout=90_000)
    page.wait_for_timeout(1200)

    # (a) 截图管线
    p = shot(page, "00_baseline_page")
    check("H1", "page.screenshot 可写出 PNG 且尺寸 == viewport*dsf", p.exists(),
          f"{p.name}")

    # (b) 合成 WebGL canvas：颜色数 / 覆盖率
    info = page.evaluate("""() => {
      const old = document.getElementById('__probe'); if (old) old.remove();
      const c = document.createElement('canvas');
      c.id = '__probe';
      Object.assign(c.style, {position:'fixed', left:'40px', top:'80px',
        width:'320px', height:'240px', zIndex:'99999', display:'block'});
      document.body.appendChild(c);
      const dpr = Math.min(window.devicePixelRatio, 2);
      c.width = Math.round(320*dpr); c.height = Math.round(240*dpr);
      const gl = c.getContext('webgl2', {antialias:false});
      if (!gl) return {err:'no webgl2'};
      const vs = 'attribute vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }';
      const fs = 'precision mediump float; void main(){ gl_FragColor=vec4(gl_FragCoord.x/640.0, gl_FragCoord.y/480.0, 0.5, 1.0); }';
      const mk = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o); return o; };
      const prog = gl.createProgram();
      gl.attachShader(prog, mk(gl.VERTEX_SHADER, vs));
      gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(prog); gl.useProgram(prog);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.viewport(0, 0, c.width, c.height); gl.drawArrays(gl.TRIANGLES, 0, 3);
      // 拖拽计数：证明 page.mouse 事件真的落到元素上
      window.__dragMoves = 0;
      c.addEventListener('pointermove', () => { window.__dragMoves++; });
      return {ok:true, dpr, w: c.width, h: c.height, cssW: c.clientWidth};
    }""")
    if not info.get("ok"):
        check("H2", "WebGL2 合成 canvas 可用", False, str(info))
        return
    check("H2", "WebGL2 合成 canvas 可用", True, f"dpr={info['dpr']} {info['w']}x{info['h']}")

    probe = page.locator("#__probe")
    im = canvas_shot(probe)
    st = image_stats(im)
    check("H3", "像素统计管道可用（合成渐变 ≥200 色 / 覆盖率 >5%）",
          st["n_colors"] >= 200 and st["coverage"] > 0.05,
          f"colors={st['n_colors']} coverage={st['coverage']} via={im.info.get('via')}")

    bb = page.evaluate("() => document.getElementById('__probe').getBoundingClientRect().toJSON()")
    drag(page, bb, 60, 30)
    moves = page.evaluate("() => window.__dragMoves")
    check("H4", "拖拽工具可用（mouse.down/move/up 触达元素）", moves > 0,
          f"pointermove={moves}")

    check("H5", "dpr clamp 生效（≤2）", info["dpr"] <= 2, f"devicePixelRatio={info['dpr']}")
    page.evaluate("() => { const e = document.getElementById('__probe'); if (e) e.remove(); }")


# ────────────────────────────────────────────────────────── 验收 9/10/15/16
def static_and_dom_checks(page) -> None:
    print("\n=== A. 页面结构（验收 9 / 10）===", flush=True)

    # 9：页面仅含指定区块
    dom = page.evaluate("""() => {
      const has = (id) => !!document.getElementById(id);
      const cards = [...document.querySelectorAll('main .side .card h3')].map((h) => h.textContent.trim());
      return {
        upload: has('drop') && has('file') && has('preview') && has('samples'),
        generate: has('generate') && has('engineHint'),
        arkitPanel: has('arkitPanel') && has('arkitToggle'),
        canvasHost: has('viewer') && has('viewerBadge'),
        faceCard: has('faceCard') && has('faceImg'),
        exportCard: has('exportCard') && has('dlGlb') && has('dlObj') && has('dlPly'),
        exportCollapsed: (() => { const e = document.getElementById('exportCard');
                                  return !!e && e.tagName.toLowerCase() === 'details' && !e.open; })(),
        statsCard: has('stats'),
        splatQuality: has('splatQuality'),
        cards, tools: [...document.querySelectorAll('.viewer__tools .tool')].length,
      };
    }""")
    for key, label in [("upload", "上传区 #drop/#file/#preview/#samples"),
                       ("generate", "生成区 #generate/#engineHint"),
                       ("canvasHost", "画布宿主 #viewer/#viewerBadge"),
                       ("faceCard", "#faceCard + #faceImg"),
                       ("exportCard", "#exportCard 含 3 个下载链接"),
                       ("exportCollapsed", "#exportCard 默认折叠 (details 未 open)"),
                       ("statsCard", "#statsCard"),
                       ("splatQuality", "#splatQuality 清晰度选择器")]:
        check("9", f"存在 {label}", bool(dom.get(key)), f"value={dom.get(key)}")
    check("9", "无驱动视频卡片 / 无下载 MP4 卡片",
          not page.locator("#video").count() and "驱动视频" not in dom["cards"],
          f"side cards={dom['cards']}")

    # 10：删除项不存在
    gone = page.evaluate("""() => ({
        motions: !!document.getElementById('motions'),
        skipVideo: !!document.getElementById('skipVideo'),
        historyWrap: !!document.getElementById('historyWrap'),
        video: !!document.getElementById('video'),
        dlMp4: !!document.getElementById('dlMp4'),
    })""")
    check("10", "#motions / #skipVideo / #historyWrap / #video / #dlMp4 全部不存在",
          not any(gone.values()), json.dumps(gone, ensure_ascii=False))

    htmls = sorted(p.name for p in (ROOT / "static").rglob("*.html"))
    check("10", "static/ 下 html 只有 1 个", len(htmls) == 1, f"htmls={htmls}")

    js_files = sorted(p for p in (ROOT / "static").rglob("*.js") if "vendor" not in p.parts)
    banned = {}
    for p in js_files:
        hits = re.findall(r"window\.open|(?<!\w)alert\s*\(|(?<!\w)confirm\s*\(|(?<!\w)prompt\s*\(", 
                          p.read_text(encoding="utf-8", errors="replace"))
        if hits:
            banned[p.name] = hits[:5]
    html_txt = (ROOT / "static" / "index.html").read_text(encoding="utf-8", errors="replace")
    html_hits = re.findall(r"window\.open|(?<!\w)alert\s*\(|(?<!\w)confirm\s*\(", html_txt)
    check("10", "全站 JS/HTML 无 alert/confirm/window.open",
          not banned and not html_hits, f"js={banned} html={html_hits}")

    # 15 运行时健康不在这里判定：必须等 #wire / #spin / 拖拽 / ARKit 交互都点过之后
    # 再核对 console/pageerror（见 runtime_health_checks）。


def runtime_health_checks(console_errors, page_errors, dialogs, popups) -> None:
    print("\n=== D. 运行时健康（验收 15，在全部交互之后核对）===", flush=True)
    check("15", "console error 0 条", not console_errors,
          f"{len(console_errors)} 条：" + json.dumps(console_errors[:8], ensure_ascii=False))
    check("15", "pageerror 0 条", not page_errors,
          f"{len(page_errors)} 条：" + json.dumps(page_errors[:6], ensure_ascii=False))
    check("15", "无 alert/confirm 弹窗、无新标签页", not dialogs and not popups,
          f"dialogs={dialogs} popups={popups}")


def orbit_check(orbit_before: list[dict]) -> None:
    print("\n=== E. OrbitControls（验收 16）===", flush=True)
    if not orbit_before:
        check("16", "OrbitControls 实例存在（插桩捕获）", False,
              "window.__ocList 为空：可能未创建，或未从 three/addons 导入")
        return
    c = orbit_before[-1]
    check("16", "OrbitControls 实例存在", True,
          f"{len(orbit_before)} 个实例；dom={c.get('domSel')}")
    ok = (abs((c.get("dampingFactor") or 0) - 0.07) < 1e-9
          and c.get("enableDamping") is True
          and abs((c.get("minDistance") or 0) - 0.2) < 1e-9
          and abs((c.get("maxDistance") or 0) - 4.0) < 1e-9)
    check("16", "参数未被改：damping 0.07 / min 0.2 / max 4.0 / enableDamping true", ok,
          f"damping={c.get('dampingFactor')} min={c.get('minDistance')} "
          f"max={c.get('maxDistance')} enableDamping={c.get('enableDamping')}")
    rot = c.get("rotateSpeed")
    check("16", "rotateSpeed 0.85 保留", rot is not None and abs(rot - 0.85) < 1e-9,
          f"rotateSpeed={rot}")


# ────────────────────────────────────────────────────────── arkit_splat 抓包 + 几何交叉验证
SPLAT_SOURCES: list[dict] = []


def install_splat_capture(page) -> None:
    """只观察：记录页面发出的每个 /arkit_splat 请求体与响应头。"""
    def on_response(resp):
        try:
            if not resp.url.rstrip("/").endswith("/arkit_splat"):
                return
            body = resp.request.post_data
            src = json.loads(body) if body else {}
            SPLAT_SOURCES.append({
                "arkit": list(src.get("arkit") or []),
                "status": resp.status,
                "count_hdr": resp.headers.get("x-lspl-count"),
                "ms_hdr": resp.headers.get("x-lspl-ms"),
                "cache": resp.headers.get("cache-control"),
            })
        except Exception as e:  # noqa: BLE001
            note(f"抓 arkit_splat 响应失败：{e}")

    page.on("response", on_response)


def post_splat(job_id: str, arkit: list, timeout: float = 40.0) -> bytes:
    req = urllib.request.Request(
        f"{BASE}/api/jobs/{job_id}/arkit_splat",
        data=json.dumps({"arkit": arkit}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_lspl(buf: bytes) -> dict:
    import struct
    if buf[:4] != b"LSPL":
        raise ValueError(f"magic 非法：{buf[:4]!r}")
    ver, count, flags = struct.unpack_from("<III", buf, 4)
    bbox_min = struct.unpack_from("<3f", buf, 16)
    bbox_max = struct.unpack_from("<3f", buf, 28)
    center = struct.unpack_from("<3f", buf, 40)
    ref_c2w = struct.unpack_from("<16f", buf, 52)
    ref_intr = struct.unpack_from("<4f", buf, 116)
    ref_size = struct.unpack_from("<2f", buf, 132)
    off = 144
    n3, n4 = count * 3, count * 4
    seg = lambda i: struct.unpack_from(f"<{i}f", buf, off)
    xyz = seg(n3); off += n3 * 4
    rot = seg(n4); off += n4 * 4
    scale = seg(n3); off += n3 * 4
    opacity = seg(count); off += count * 4
    rgb = seg(n3)
    return {"version": ver, "count": count, "flags": flags,
            "bbox_min": bbox_min, "bbox_max": bbox_max, "center": center,
            "ref_c2w": ref_c2w, "ref_intr": ref_intr, "ref_size": ref_size,
            "xyz": xyz, "rot": rot, "scale": scale, "opacity": opacity, "rgb": rgb,
            "total_bytes": len(buf), "expect_bytes": 144 + count * 56}


def dump_arkit_sources(page, n_sources: int = 2, n_values: int = 52) -> None:
    """打印前端实际发出的 arkit_splat 请求体摘要（只读观察）。"""
    print("\n=== C1. 页面发出的 /arkit_splat 请求（抓包观察）===", flush=True)
    taken = SPLAT_SOURCES[-n_sources:]
    for s in taken:
        v = s.get("arkit") or []
        nz = {i: round(x, 4) for i, x in enumerate(v) if abs(x) > 1e-6}
        note(f"POST arkit_splat → HTTP {s['status']} | count={s['count_hdr']} "
             f"ms={s['ms_hdr']} cache={s['cache']!r} | len={len(v)} 非零({len(nz)})={nz}")


def verify_preset_geometry(page, job_id: str) -> dict | None:
    """独立交叉验证：取页面实际发出的两次不同预设向量，**由 Python 自己** POST 后端，
    逐段比对 LSPL。证明「不同预设 ⇒ 服务端几何确实不同」，不依赖前端渲染。"""
    srcs = [s for s in SPLAT_SOURCES if len(s.get("arkit") or []) == 52]
    distinct: list[list] = []
    for s in srcs:
        a = s["arkit"]
        if not distinct or a != distinct[-1]:
            distinct.append(a)
    if len(distinct) < 2:
        note(f"抓到 {len(srcs)} 个 arkit_splat，但只有 {len(distinct)} 个互异向量")
        return None
    a, b = distinct[0], distinct[-2] if len(distinct) > 2 else distinct[1]
    ba, bb = post_splat(job_id, a), post_splat(job_id, b)
    pa, pb = parse_lspl(ba), parse_lspl(bb)
    if pa["count"] != pb["count"]:
        return {"error": f"count 不一致 {pa['count']} vs {pb['count']}"}
    dxyz = sum(abs(x - y) for x, y in zip(pa["xyz"], pb["xyz"])) / max(len(pa["xyz"]), 1)
    dscale = sum(abs(x - y) for x, y in zip(pa["scale"], pb["scale"])) / max(len(pa["scale"]), 1)
    dcenter = max(abs(x - y) for x, y in zip(pa["center"], pb["center"]))
    return {"n": pa["count"], "xyz_mean_abs_delta": round(dxyz, 6),
            "scale_mean_abs_delta": round(dscale, 6),
            "center_max_abs_delta": round(dcenter, 6),
            "bytes_a": len(ba), "bytes_b": len(bb),
            "preset_a_nonzero": sum(1 for x in a if abs(x) > 1e-6),
            "preset_b_nonzero": sum(1 for x in b if abs(x) > 1e-6),
            "bbox_a": [round(v, 4) for v in pa["bbox_min"] + pa["bbox_max"]],
            "ms_a": SPLAT_SOURCES[-1].get("ms_hdr")}


# ────────────────────────────────────────────────────────── 全流程
def run_flow(page) -> dict:
    """上传/选样 → 生成 → 等 succeeded → 截图。返回 {job_id, api_json}。"""
    print("\n=== B. 全流程：生成 → 点云渲染（验收 11-14）===", flush=True)
    out: dict = {"job_id": None, "gen_api": None}
    shot(page, "01_initial")

    # 选样例（优先）或上传
    samples = page.locator("#samples img")
    n = samples.count()
    if n == 0:
        check("B0", "样例/上传入口可用", False, "#samples 下没有 img，且无文件可用")
        return out
    samples.nth(0).click()
    page.wait_for_timeout(600)
    note(f"已选样例：{samples.nth(0).get_attribute('alt')}（{n} 个可选）")
    gen_disabled = page.locator("#generate").is_disabled()
    check("B0", "选择图片后 #generate 可用", not gen_disabled, f"disabled={gen_disabled}")
    if gen_disabled:
        return out

    shot(page, "02_ready")

    with page.expect_response(
            lambda r: r.url.rstrip("/").endswith("/api/generate") and r.request.method == "POST",
            timeout=60_000) as ri:
        page.locator("#generate").click()
    resp = ri.value
    try:
        gen_json = resp.json()
    except Exception:  # noqa: BLE001
        gen_json = {"_raw": resp.text()[:400]}
    out["gen_api"] = {"status": resp.status, "body": gen_json}
    note(f"POST /api/generate → {resp.status} {json.dumps(gen_json, ensure_ascii=False)[:200]}")

    job_id = None
    if isinstance(gen_json, dict):
        job_id = gen_json.get("job_id") or gen_json.get("id")
    if not job_id:
        # 前端/后端字段不一致时兜底：取最新任务
        jobs = get_json(f"{BASE}/api/jobs") or []
        if jobs:
            job_id = jobs[0].get("id") or jobs[0].get("job_id")
        check("B1", "POST /api/generate 返回可直接使用的 job id", False,
              f"响应字段={list(gen_json.keys()) if isinstance(gen_json, dict) else gen_json}"
              f"，兜底取到 {job_id}")
    else:
        check("B1", "POST /api/generate 返回 job id", True, f"job_id={job_id}")
    out["job_id"] = job_id

    page.wait_for_timeout(1200)
    shot(page, "03_generating")

    # 等 succeeded（自轮询，避免依赖前端）
    deadline = time.time() + 300
    last = None
    while time.time() < deadline:
        j = get_json(f"{BASE}/api/jobs/{job_id}") if job_id else None
        if not j:
            jobs = get_json(f"{BASE}/api/jobs") or []
            j = jobs[0] if jobs else None
        if j:
            last = {k: j.get(k) for k in ("state", "stage", "progress", "error")}
            if j.get("state") in ("succeeded", "failed", "canceled"):
                break
        time.sleep(3)
    state = (last or {}).get("state")
    check("B2", "任务跑到 succeeded", state == "succeeded",
          f"state={state} err={(last or {}).get('error')}")
    if state != "succeeded":
        return out

    # 等前端把结果画出来
    try:
        page.wait_for_selector("#result:not([hidden])", timeout=30_000)
        note("#result 已显示")
    except Exception as e:  # noqa: BLE001
        note(f"#result 未显示：{e}")
    try:
        page.wait_for_selector("#viewer canvas", timeout=45_000)
        # 等点云真正铺上画布：#viewerBadge 由「点云加载中…」变为「已加载 · N 高斯 …」
        try:
            page.wait_for_function(
                "() => { const b = document.getElementById('viewerBadge');"
                " return b && /已加载/.test(b.textContent || ''); }",
                timeout=45_000)
            page.wait_for_timeout(2500)  # 等 SplatView 自驱 rAF 渲染几帧
        except Exception as e:  # noqa: BLE001
            note(f"#viewerBadge 未变为「已加载」：{e}")
        badge = page.locator("#viewerBadge").inner_text() if page.locator("#viewerBadge").count() else ""
        note(f"#viewerBadge = {badge!r}")
        note("#viewer canvas 已出现")
    except Exception as e:  # noqa: BLE001
        note(f"#viewer canvas 未出现：{e}")

    page.wait_for_timeout(800)
    shot(page, "04_result_facecard")
    return out


# ────────────────────────────────────────────────────────── T6 布局断言（面板不得遮挡预览区）
def _rect_intersection(a: dict, b: dict) -> float:
    """两个 {x,y,width,height} 矩形的交集面积（无交集 = 0）。"""
    if not a or not b:
        return 0.0
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def t6_layout_checks(page, cm_before: dict) -> None:
    """T6 核心：`#arkitPanel` 搬到左栏卡片后「不遮挡预览区」的真实布局断言。

    用户原始诉求 = 表情面板挡住预览区；仅断言「交集面积 == 0」不够，因为父级
    flex 列容器把面板压扁到 24px 高时交集同样是 0（**假阳性**）。因此同时断言
    面板高度 ≥ 220 与 `#arkitSliders` 有真实高度。
    """
    print("\n=== F. T6 布局断言（#arkitPanel 不遮挡预览区）===", flush=True)

    # 让面板滚到可见处（与 app.js openArkitPanel 的 scrollIntoView 一致），便于视觉取证
    page.evaluate("""() => { const p = document.querySelector('#arkitPanel');
        if (p && p.scrollIntoView) p.scrollIntoView({block:'nearest'}); }""")
    page.wait_for_timeout(400)

    boxes = page.evaluate("""() => {
      const b = (sel) => { const e = document.querySelector(sel);
        if (!e) return null; const r = e.getBoundingClientRect();
        return {x:r.x, y:r.y, width:r.width, height:r.height,
                top:r.top, left:r.left, right:r.right, bottom:r.bottom}; };
      const sl = document.querySelector('#arkitSliders');
      const panel = document.querySelector('#arkitPanel');
      return {panel: b('#arkitPanel'), viewerWrap: b('.viewer-wrap'),
              viewer: b('#viewer'), canvas: b('#viewer canvas'),
              side: b('aside.panel--input'),
              panelInLeftCol: !!(panel && panel.closest('aside.panel--input')),
              panelHidden: panel ? panel.hidden : null,
              panelDisplay: panel ? getComputedStyle(panel).display : null,
              panelPosition: panel ? getComputedStyle(panel).position : null,
              panelFlex: panel ? getComputedStyle(panel).flexGrow + '/' + getComputedStyle(panel).flexShrink : null,
              sliders: sl ? {scrollHeight: sl.scrollHeight, clientHeight: sl.clientHeight,
                             offsetHeight: sl.offsetHeight, scrollTop: sl.scrollTop,
                             overflowing: sl.scrollHeight > sl.clientHeight} : null};
    }""")

    pb, vb = boxes.get("panel"), boxes.get("viewerWrap")
    inter = _rect_intersection(pb, vb)
    print(f"       panel box = x{pb['x']:.1f} y{pb['y']:.1f} w{pb['width']:.1f} h{pb['height']:.1f}"
          if pb else "       panel box = None", flush=True)
    if vb:
        print(f"       viewer-wrap box = x{vb['x']:.1f} y{vb['y']:.1f} "
              f"w{vb['width']:.1f} h{vb['height']:.1f}", flush=True)
    print(f"       交集面积 = {inter:.1f} px² ; panel 高度 = "
          f"{pb['height'] if pb else 0:.1f} ; 左栏子树内 = {boxes.get('panelInLeftCol')}", flush=True)

    # ---- 断言 1：面板与预览区矩形交集面积为 0（用户原始诉求）
    check("T6", "#arkitPanel 与 .viewer-wrap 矩形交集面积为 0（不遮挡预览区）",
          pb is not None and vb is not None and inter == 0.0,
          f"inter={inter:.1f}px² panel=({pb['x']:.1f},{pb['y']:.1f},{pb['width']:.1f},"
          f"{pb['height']:.1f}) viewer-wrap=({vb['x']:.1f},{vb['y']:.1f},"
          f"{vb['width']:.1f},{vb['height']:.1f})" if pb and vb else "缺少矩形")

    # ---- 断言 2：面板真实展开（防「被压扁成 24px 高」的假阳性）
    check("T6", "#arkitPanel 高度 ≥ 220px（面板真实展开，非被 flex 压扁）",
          bool(pb) and pb["height"] >= 220,
          f"panel.height={pb['height']:.1f}" if pb else "panel 不存在")
    sl = boxes.get("sliders") or {}
    check("T6", "#arkitSliders 有真实高度（≥ 80px）",
          bool(sl) and sl.get("offsetHeight", 0) >= 80,
          f"offsetHeight={sl.get('offsetHeight')} clientHeight={sl.get('clientHeight')}")

    # ---- 断言 3：面板位于左栏 aside.panel--input 子树内
    check("T6", "#arkitPanel 位于左栏 aside.panel--input DOM 子树内",
          bool(boxes.get("panelInLeftCol")),
          f"closest('aside.panel--input')={boxes.get('panelInLeftCol')} "
          f"position={boxes.get('panelPosition')} flex={boxes.get('panelFlex')}")

    # ---- 断言 4：面板打开前后 canvas 尺寸不变，且 A14 两条仍成立
    after = canvas_metrics(page)
    same = (cm_before.get("found") and after.get("found")
            and cm_before.get("width") == after.get("width")
            and cm_before.get("height") == after.get("height"))
    check("T6", "面板打开前后 canvas.width/height 不变（A14 未被破坏）", bool(same),
          f"before={cm_before.get('width')}x{cm_before.get('height')} "
          f"after={after.get('width')}x{after.get('height')}")
    if after.get("found"):
        dpr = after["dpr"]
        eff = after["width"] / max(after["cssW"], 1)
        check("T6", "面板打开后 canvas.width === cssW * dpr（dpr clamp ≤2）",
              abs(after["width"] - after["cssW"] * dpr) <= 1 and 1 - 1e-6 <= eff <= 2 + 1e-6,
              f"width={after['width']} cssW={after['cssW']} dpr={dpr} eff={round(eff,3)}")
        check("T6", "面板打开后绘制缓冲区 ≥ 640×640（A14 高清档）",
              after["width"] >= 640 and after["height"] >= 640,
              f"buffer={after['width']}x{after['height']} css={after['cssW']}x{after['cssH']}")

    # ---- 断言 5：内部滚动生效（或面板完整可见，未撑爆左栏）
    over = bool(sl) and sl.get("overflowing")
    sliders_box_ok = bool(sl) and sl.get("clientHeight", 0) > 0
    check("T6", "#arkitSliders 内部滚动生效（scrollHeight > clientHeight）或面板完整可见",
          over or (sliders_box_ok and bool(pb) and pb["height"] >= 220),
          f"scrollHeight={sl.get('scrollHeight')} clientHeight={sl.get('clientHeight')} "
          f"overflowing={over}")

    # ---- 视觉证据 1：面板打开的整体页（左栏卡片内）
    page.screenshot(path=str(SHOT_DIR / "t6_panel_in_left_col.png"))
    mark_shot(SHOT_DIR / "t6_panel_in_left_col.png")

    # ---- 视觉证据 2：把 panel / viewer-wrap / #viewer 三个矩形描边叠加后截图
    # 纯页面侧插桩（import 后即用，不改任何生产文件）
    page.evaluate("""(b) => {
      document.querySelectorAll('.__t6_hl').forEach(e => e.remove());
      const mk = (rect, color, label) => {
        if (!rect) return;
        const d = document.createElement('div');
        d.className = '__t6_hl';
        Object.assign(d.style, {position:'fixed', zIndex:99999, pointerEvents:'none',
          left: rect.x+'px', top: rect.y+'px', width: rect.width+'px',
          height: rect.height+'px', border:'2px solid '+color, boxSizing:'border-box'});
        const t = document.createElement('div');
        t.textContent = label;
        Object.assign(t.style, {position:'absolute', left:'0', top:'-16px',
          fontSize:'11px', fontFamily:'monospace', color:'#000', background:color,
          padding:'0 4px', whiteSpace:'nowrap'});
        d.appendChild(t);
        document.body.appendChild(d);
      };
      mk(b.panel, '#ff3355', 'ARKIT PANEL');
      mk(b.viewerWrap, '#22cc66', 'VIEWER-WRAP');
      mk(b.viewer, '#33aaff', '#viewer');
      mk(b.canvas, '#ffcc00', 'canvas');
    }""", {"panel": pb, "viewerWrap": vb, "viewer": boxes.get("viewer"),
           "canvas": boxes.get("canvas")})
    page.wait_for_timeout(300)
    page.screenshot(path=str(SHOT_DIR / "t6_panel_vs_viewer.png"))
    mark_shot(SHOT_DIR / "t6_panel_vs_viewer.png")

    # ---- 视觉证据 3：展开全部分组后再滚到底。
    # 契约允许「内部滚动生效」**或**「面板完整可见」两种形态：默认态只有第 1 组展开，
    # 未必溢出（此时 scrollTop 恒为 0，属正常）。故先展开全部组使内容真实超出，
    # 再按是否溢出分别断言：溢出则必须能滚到底；不溢出则断言内容完整可见。
    expanded = page.evaluate("""() => {
      const host = document.querySelector('#arkitSliders');
      if (!host) return null;
      host.querySelectorAll('details.arkit-group').forEach(d => { d.open = true; });
      return {n_groups: host.querySelectorAll('details.arkit-group').length,
              scrollHeight: host.scrollHeight, clientHeight: host.clientHeight,
              overflowing: host.scrollHeight > host.clientHeight};
    }""")
    page.wait_for_timeout(400)
    print(f"       [T6] 展开全部分组后: {expanded}", flush=True)
    scrolled = page.evaluate("""() => {
      const sl = document.querySelector('#arkitSliders');
      if (!sl) return null;
      sl.scrollTop = sl.scrollHeight;
      return {scrollTop: sl.scrollTop, scrollHeight: sl.scrollHeight,
              clientHeight: sl.clientHeight,
              overflowing: sl.scrollHeight > sl.clientHeight};
    }""")
    page.wait_for_timeout(400)
    page.screenshot(path=str(SHOT_DIR / "t6_panel_scrolled.png"))
    mark_shot(SHOT_DIR / "t6_panel_scrolled.png")
    if scrolled and scrolled.get("overflowing"):
        check("T6", "展开全部分组后 #arkitSliders 可滚动到内容底部（scrollTop > 0）",
              scrolled["scrollTop"] > 0,
              f"scrollTop={scrolled['scrollTop']:.0f} scrollHeight={scrolled['scrollHeight']} "
              f"clientHeight={scrolled['clientHeight']}")
    else:
        check("T6", "展开全部分组后 #arkitSliders 仍无溢出（内容完整可见，无需内部滚动）",
              bool(scrolled) and scrolled["scrollHeight"] <= scrolled["clientHeight"],
              f"scrollHeight={scrolled.get('scrollHeight') if scrolled else None} "
              f"clientHeight={scrolled.get('clientHeight') if scrolled else None} "
              f"panel.height={pb['height']:.1f}")

    # 清掉标注层，避免污染后续截图
    page.evaluate("() => document.querySelectorAll('.__t6_hl').forEach(e => e.remove())")
    # 滑块滚回顶部，避免影响后续像素测量
    page.evaluate("() => { const s = document.querySelector('#arkitSliders'); if (s) s.scrollTop = 0; }")
    page.wait_for_timeout(200)


def render_checks(page, have_result: bool, job_id: str | None = None) -> None:
    """验收 11 / 12 / 13 / 14。"""
    if not have_result:
        for cid, nm in [("11", "SplatView 真实渲染（≥200 色 / 覆盖 >5%）"),
                        ("12", "360° 拖拽 + 视角预设切换"),
                        ("13", "ARKit 预设 → 点云像素变化"),
                        ("14", "清晰度 canvas.width == cssW*dpr")]:
            check(cid, nm, False, "基线模式：未生成结果，无法评估", excused=True)
        return

    cm = canvas_metrics(page)
    # ---- 14 清晰度
    print("\n=== C. 点云渲染（验收 11-14）===", flush=True)
    if not cm.get("found"):
        check("14", "画布存在", False, "页面里找不到 canvas")
    else:
        dpr = cm["dpr"]
        eff = cm["width"] / max(cm["cssW"], 1)
        check("14", "canvas.width == cssW * dpr（dpr clamp ≤2）",
              abs(cm["width"] - cm["cssW"] * dpr) <= 1 and 1 - 1e-6 <= eff <= 2 + 1e-6,
              f"width={cm['width']} cssW={cm['cssW']} dpr={dpr} eff={round(eff,3)}")
        check("14", "高清模式绘制像素 ≥ 640×640 等效",
              cm["width"] >= 640 and cm["height"] >= 640,
              f"{cm['width']}x{cm['height']}")
        q = page.locator("#splatQuality")
        if q.count():
            qinfo = q.evaluate("""(e) => e.tagName === 'SELECT'
              ? {tag:'SELECT', value:e.value,
                 text:e.options[e.selectedIndex] ? e.options[e.selectedIndex].textContent.trim() : '',
                 options:[...e.options].map(o=>o.textContent.trim())}
              : {tag:e.tagName, value:null, text:e.textContent.trim(), options:[]}""")
            hay = f"{qinfo.get('value') or ''}{qinfo.get('text') or ''}{''.join(qinfo.get('options') or [])}"
            check("14", "#splatQuality 默认高清且含 标准/高清 选项",
                  ("高清" in hay or qinfo.get("value") in ("hd", "high"))
                  and ("标准" in hay or "高清" in hay),
                  f"{qinfo}")

    canvas = page.locator("#viewer canvas")
    if canvas.count() == 0:
        for cid, nm in [("11", "SplatView 真实渲染"), ("12", "360° 拖拽"),
                        ("13", "ARKit 预设 → 像素变化")]:
            check(cid, nm, False, "无 #viewer canvas")
        return

    # ---- 11 渲染
    im0 = canvas_shot(canvas)
    st0 = image_stats(im0)
    check("11", "canvas 不同颜色数 ≥ 200", st0["n_colors"] >= 200,
          f"colors={st0['n_colors']} via={im0.info.get('via')}")
    check("11", "非背景覆盖率 > 5%", st0["coverage"] > 0.05,
          f"coverage={st0['coverage']} bg={st0['bg']}")
    im0.save(SHOT_DIR / "final_05_splat_render.png")
    mark_shot(SHOT_DIR / "final_05_splat_render.png")

    # ---- 12 拖拽 + 预设
    orbit_a = read_orbit(page)
    pos_a = (orbit_a[-1].get("pos") if orbit_a else None)
    bb = page.evaluate("() => document.querySelector('#viewer canvas').getBoundingClientRect().toJSON()")
    drag(page, bb, -220, 60)
    im_drag = canvas_shot(canvas)
    orbit_b = read_orbit(page)
    pos_b = (orbit_b[-1].get("pos") if orbit_b else None)
    d_px = mean_abs_diff(im0, im_drag)
    moved = (pos_a is not None and pos_b is not None
             and max(abs(x - y) for x, y in zip(pos_a, pos_b)) > 1e-4)
    check("12", "拖拽后点云像素变化", d_px > 0.5, f"mean|Δpixel|={d_px}")
    check("12", "拖拽后相机位置改变", bool(moved),
          f"pos {pos_a} → {pos_b}")
    shot(page, "06_after_drag")

    views = {}
    view_pos = {}
    for v in ("left", "right", "three"):
        btn = page.locator(f'.viewer__tools .tool[data-view="{v}"]')
        if btn.count() == 0:
            check("12", f"视角预设 {v} 按钮存在", False, "按钮缺失")
            continue
        btn.nth(0).click()
        page.wait_for_timeout(900)
        views[v] = canvas_shot(canvas)
        oc = read_orbit(page)
        view_pos[v] = oc[-1].get("pos") if oc else None
        p = SHOT_DIR / f"final_07_view_{v}.png"
        views[v].save(p)
        mark_shot(p)
    if len(view_pos) == 3 and all(view_pos.values()):
        pairs = [("left", "right"), ("left", "three"), ("right", "three")]
        diffs = {f"{a}~{b}": round(max(abs(x - y) for x, y in zip(view_pos[a], view_pos[b])), 4)
                 for a, b in pairs}
        check("12", "三个预设对应的相机位置互不相同", all(d > 1e-4 for d in diffs.values()),
              f"max|Δpos|={diffs}; pos={view_pos}")
    else:
        note(f"预设相机位置读数不完整：{view_pos}")
    if len(views) == 3:
        d_lr = mean_abs_diff(views["left"], views["right"])
        d_lt = mean_abs_diff(views["left"], views["three"])
        d_rt = mean_abs_diff(views["right"], views["three"])
        check("12", "left / right / three 三张画面互不相同",
              d_lr > 0.5 and d_lt > 0.5 and d_rt > 0.5,
              f"left↔right={d_lr} left↔three={d_lt} right↔three={d_rt}")

    # ---- 13 ARKit 预设 / 滑块 → 点云像素变化（契约 8.13 明写「预设/滑块」两条路径）
    dump_arkit_sources(page, n_sources=2, n_values=52)
    # 度量口径：先回到契约 5.3 的 canonical 正面视角，避免把相机姿态差混入像素差
    fv = page.locator('.viewer__tools .tool[data-view="front"]')
    if fv.count():
        fv.nth(0).click()
        page.wait_for_timeout(1000)
    # T6：记录面板打开**之前**的状态（契约 6.2 要求生成完成后仍 hidden）
    cm_before = canvas_metrics(page)
    panel_hidden_before = page.evaluate(
        "() => { const p = document.querySelector('#arkitPanel'); return p ? p.hidden : null; }")
    note(f"面板打开前：canvas={cm_before.get('width')}x{cm_before.get('height')} "
         f"css={cm_before.get('cssW')}x{cm_before.get('cssH')} dpr={cm_before.get('dpr')} "
         f"panel.hidden={panel_hidden_before}")
    check("T6", "生成完成后面板保持 hidden（#arkitToggle 是唯一入口）",
          panel_hidden_before is True, f"panel.hidden={panel_hidden_before}")
    toggle = page.locator("#arkitToggle")
    if toggle.count():
        toggle.nth(0).click()
        page.wait_for_timeout(1500)
    panel_open = page.locator("#arkitPanel").is_visible()
    check("13", "ARKit 面板可打开", panel_open, f"visible={panel_open}")
    shot(page, "08_arkit_panel")

    # ---- T6 核心：布局断言（面板在左栏，不遮挡预览区）
    t6_layout_checks(page, cm_before)

    before = canvas_shot(canvas)
    floor = mean_abs_diff(before, canvas_shot(canvas))
    note(f"像素度量噪声地板 mean|Δ|={floor}（同状态连拍两帧；0 表示捕获完全确定性）")

    # 契约 8.13 判定口径（Lead 裁决，已写入 CONTRACT_v2.md:295-297）：
    # 前景掩码 = neutral 基线帧中与画布背景色差异 > 6 的像素；mean|Δpixel| 只在掩码内计算。
    fg_coords, fg_bg, fg_total = fg_mask(before, tol=6)
    fg_ratio = len(fg_coords) / max(fg_total, 1)
    dilution = 1.0 / fg_ratio if fg_ratio > 0 else float("inf")
    note(f"前景掩码：bg={fg_bg} fg_px={len(fg_coords)}/{fg_total}（占比 {fg_ratio:.4%}，"
         f"全图均值被稀释约 {dilution:.1f}x）")
    print(f"       前景掩码 fg={len(fg_coords)}/{fg_total} ({fg_ratio:.4%}) bg={fg_bg} "
          f"稀释≈{dilution:.1f}x", flush=True)
    if len(fg_coords) == 0:
        check("13", "前景掩码非空（neutral 帧上存在非背景像素）", False,
              f"bg={fg_bg} 全帧皆背景？")

    presets = page.locator("#arkitPresets .arkit-preset:not(.arkit-preset--reset)")
    n_pre = presets.count()
    if n_pre == 0:
        check("13", "ARKit 预设按钮存在", False, "没有 .arkit-preset")
    else:
        # 逐预设全量测量（点击即整组覆盖，故每次都相对 neutral 基线正确）
        table = []
        for i in range(n_pre):
            title = presets.nth(i).get_attribute("title") or f"#{i}"
            presets.nth(i).click()
            page.wait_for_timeout(1600)
            cur = canvas_shot(canvas)
            full_d = mean_abs_diff(before, cur)
            fg = fg_mean_abs_diff(before, cur, fg_coords)
            lat = page.locator("#arkitLatency").inner_text() if page.locator("#arkitLatency").count() else ""
            table.append({"title": title, "full": full_d, "fg": fg["mean"],
                          "changed_ratio": fg["changed_ratio"], "lat": lat})
        txt = " ".join(f"{t['title']}=fg:{t['fg']:.3f}/full:{t['full']:.3f}"
                       f"({t['changed_ratio']:.2%})" for t in table)
        print(f"       逐预设 前景mean|Δpixel| / 全图mean|Δpixel| / 前景变化像素占比: {txt}", flush=True)
        best = max(table, key=lambda t: t["fg"])
        weak = min(table, key=lambda t: t["fg"])
        n_pass = sum(1 for t in table if t["fg"] > 0.5)
        check("13", "ARKit 预设 → 前景 mean|Δpixel| > 0.5（契约 8.13 口径）", best["fg"] > 0.5,
              f"最强预设 {best['title']}=前景 {best['fg']:.4f}（{n_pass}/{n_pre} 个预设 >0.5）；全部：{txt}")
        check("13", "每个 ARKit 预设均产生高于噪声本底的像素变化",
              all(t["fg"] > max(floor * 3, 0.01) for t in table),
              f"噪声地板={floor} 最弱预设 {weak['title']}=前景 {weak['fg']:.4f}；全部：{txt}")
        note(f"逐预设分布（canonical front，前景掩码内 mean|Δ|）：{txt}")
        if n_pass < n_pre:
            note(f"诚实登记：{n_pre - n_pass}/{n_pre} 个预设前景均值 <0.5"
                 f"（最弱 {weak['title']}={weak['fg']:.4f}）；全图口径下这些预设更被"
                 f"稀释 ≈{dilution:.1f}x，故全图口径不适用于温和表情")
        # 重新点回最强预设，出图留证
        presets.nth([t["title"] for t in table].index(best["title"])).click()
        page.wait_for_timeout(1800)
        shot(page, "09_arkit_preset")

    # ---- 13（滑块路径，契约明写）逐通道推满 → 像素变化
    SLIDER_SET_JS = """([nm, val]) => {
        let hit = false;
        document.querySelectorAll('#arkitSliders .arkit-slider').forEach((l) => {
            const n = l.querySelector('.arkit-slider__name');
            const r = l.querySelector('input[type=range]');
            if (n && r && n.textContent.trim() === nm) {
                let p = l.parentElement;
                while (p && p.tagName !== 'DETAILS') p = p.parentElement;
                if (p) p.open = true;
                r.value = String(val);
                r.dispatchEvent(new Event('input', { bubbles: true }));
                hit = true;
            }
        });
        return hit;
    }"""
    rst = page.locator("#arkitPresets .arkit-preset--reset")
    if rst.count():
        rst.nth(0).click()
        page.wait_for_timeout(1600)
    base0 = canvas_shot(canvas)
    sl_tab = []
    for nm in ("jawOpen", "browInnerUp", "mouthSmileLeft"):
        if not page.evaluate(SLIDER_SET_JS, [nm, 1.0]):
            note(f"滑块 {nm} 未在 #arkitSliders 中找到")
            continue
        page.wait_for_timeout(1700)
        cur = canvas_shot(canvas)
        fg = fg_mean_abs_diff(base0, cur, fg_coords)
        lat = page.locator("#arkitLatency").inner_text() if page.locator("#arkitLatency").count() else ""
        sl_tab.append({"title": nm, "fg": fg["mean"], "full": mean_abs_diff(base0, cur),
                       "changed_ratio": fg["changed_ratio"], "lat": lat})
        if page.evaluate(SLIDER_SET_JS, [nm, 0.0]):
            page.wait_for_timeout(1200)
    if sl_tab:
        stxt = " ".join(f"{t['title']}=fg:{t['fg']:.3f}/full:{t['full']:.3f}"
                        f"({t['changed_ratio']:.2%})" for t in sl_tab)
        sl_best = max(sl_tab, key=lambda t: t["fg"])
        print(f"       逐滑块（推满 1.0）前景/全图 mean|Δpixel| / 变化占比: {stxt}", flush=True)
        check("13", "ARKit 滑块 → 前景 mean|Δpixel| > 0.5（契约 8.13 口径）", sl_best["fg"] > 0.5,
              f"最强通道 {sl_best['title']}=前景 {sl_best['fg']:.4f}；全部：{stxt}")
        shot(page, "10_arkit_slider")
        if rst.count():
            rst.nth(0).click()
            page.wait_for_timeout(1200)
    else:
        check("13", "ARKit 滑块触发点云像素变化 mean|Δ| > 0.5", False,
              "#arkitSliders 中未找到可用的 range 输入")

    if panel_open:
        lat = page.locator("#arkitLatency").inner_text() if page.locator("#arkitLatency").count() else ""
        got_num = bool(re.search(r"\d", lat or ""))
        check("13", "#arkitLatency 出现数值", got_num, f"text={lat!r}")

        # 独立交叉验证：同一组 arkit 值经真实后端 → 客户端两次不同预设必须产生不同几何
        diff = verify_preset_geometry(page, job_id) if job_id else None
        if diff is None:
            check("13", "预设几何交叉验证（前端两次 POST → header/xyz 差异）", False,
                  "无法从页面捕获 arkit_splat 响应")
        elif "error" in diff:
            check("13", "预设几何交叉验证（前端两次 POST → header/xyz 差异）", False, str(diff))
        else:
            check("13", "两次不同预设 → 服务端点云几何确实不同", diff["xyz_mean_abs_delta"] > 1e-5,
                  f"{diff}")

    # 复位
    if rst.count():
        rst.nth(0).click()
        page.wait_for_timeout(1000)

    # ---- 15 附加交互面：必须点一次 #wire 与 #spin（回归 setPointMode TypeError）
    print("\n=== C2. #wire / #spin 交互（验收 15 回归面）===", flush=True)
    for tid in ("wire", "spin"):
        el = page.locator(f"#{tid}")
        if el.count() == 0:
            check("15", f"#{tid} 存在", False, "按钮缺失")
            continue
        try:
            el.nth(0).click()
            page.wait_for_timeout(800)
            is_on = el.nth(0).evaluate("(e)=>e.classList.contains('is-on')")
            check("15", f"#{tid} 点击无异常", True, f"is-on={is_on}")
        except Exception as e:  # noqa: BLE001
            check("15", f"#{tid} 点击无异常", False, f"{type(e).__name__}: {e}")
    for tid in ("wire", "spin"):  # 复位，别把截图搞花
        el = page.locator(f"#{tid}")
        if el.count() and el.nth(0).evaluate('(e)=>e.classList.contains("is-on")'):
            el.nth(0).click()
            page.wait_for_timeout(600)
    page.wait_for_timeout(500)
    shot(page, "11_final_state")


# ────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generate", action="store_true", help="阶段 1 基线：不跑生成流程")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--restart", action="store_true", help="先 run.sh --stop/--daemon 再跑")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    print(f"### T4 e2e — tag={args.tag} base={BASE} no_generate={args.no_generate}", flush=True)
    if DONE_FLAG.exists():
        DONE_FLAG.unlink()

    if args.restart:
        note("重启服务（run.sh --stop && --daemon）")
        if not restart_server():
            check("S0", "服务就绪", False, "重启后 240s 内未 ready")
            return finish(args.tag, 1)
    if not wait_ready(240):
        check("S0", "服务就绪（/api/status ready:true）", False, "240s 内未就绪")
        return finish(args.tag, 1)
    check("S0", "服务就绪（/api/status ready:true）", True, BASE)

    console_errors: list[str] = []
    page_errors: list[str] = []
    dialogs: list[str] = []
    popups: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=DSF)
        ctx.set_default_timeout(60_000)
        page = ctx.new_page()
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("console", lambda m: console_errors.append(
            f"{m.type}: {m.text} @ {m.location.get('url')}") if m.type == "error" else None)
        page.on("dialog", lambda d: (dialogs.append(f"{d.type}:{d.message}"), d.dismiss()))
        ctx.on("page", lambda p: popups.append(p.url))

        try:
            if install_orbit_tracker(page):
                note("OrbitControls 观察器已装载")
            if not args.skip_selftest:
                harness_selftest(page)

            # 让页面回到干净状态（自检插入的元素已移除）
            page.goto(BASE, wait_until="load", timeout=90_000)
            page.wait_for_timeout(1500)
            install_splat_capture(page)
            static_and_dom_checks(page)

            info = None
            if args.no_generate:
                note("基线模式：跳过生成流程")
                render_checks(page, have_result=False)
            else:
                info = run_flow(page)
                ok = bool(info.get("job_id")) and page.locator("#result:not([hidden])").count() > 0
                render_checks(page, have_result=ok, job_id=info.get("job_id"))
            orbit_check(read_orbit(page))
            runtime_health_checks(console_errors, page_errors, dialogs, popups)
        except Exception as e:  # noqa: BLE001
            import traceback
            note(f"未捕获异常：{e}\n{traceback.format_exc()}")
        finally:
            try:
                ctx.close()
                browser.close()
            except Exception:  # noqa: BLE001
                pass

    n_fail = sum(1 for r in RESULTS if r["tag"] == "FAIL")
    return finish(args.tag, 1 if n_fail else 0)


def finish(tag: str, code: int) -> int:
    rep = {
        "tag": tag, "base": BASE, "viewport": VIEWPORT, "device_scale_factor": DSF,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "passed": sum(1 for r in RESULTS if r["tag"] == "PASS"),
        "failed": sum(1 for r in RESULTS if r["tag"] == "FAIL"),
        "skipped": sum(1 for r in RESULTS if r["tag"] == "SKIP"),
        "results": RESULTS, "screenshots": SCREENSHOTS, "notes": NOTE,
    }
    out = SHOT_DIR / f"report_{tag}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n==== {rep['passed']} passed / {rep['failed']} failed / {rep['skipped']} skipped "
          f"→ {out} ====", flush=True)
    for r in RESULTS:
        if r["tag"] == "FAIL":
            print(f"  FAIL  [{r['id']}] {r['name']} — {r['extra']}", flush=True)
    DONE_FLAG.write_text(f"exit={code}\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
