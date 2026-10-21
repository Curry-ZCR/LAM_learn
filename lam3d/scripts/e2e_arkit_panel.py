#!/usr/bin/env python
"""端到端验证：ARKit 表情面板是否**集成在原单图 3D 查看器页面**里（不是新页面 / 弹窗），
以及是否破坏原有相机控制。

用真实无头 Chromium 打开 /workspace/lam3d 的**同一个**页面，在真实生成任务的结果上取证：

  A 阶段（判定，针对当前仓库）
    1. 服务就绪（/api/status ready:true，最长 180s）
    2. 原页面可用：status.png + Anti_Drugs + 跳过视频 → 生成 → #result → 画布非空
    3. 约束 5 的 DOM 事实：面板存在、是 .viewer-wrap 的后代、初始隐藏；
       始终 1 个 canvas / 1 个 .viewer-wrap / 1 个 context.pages / URL 仍是 "/"
    4. 点 #arkitToggle → 面板可见、52 行滑块、预设 >1、
       **面板 bbox 完全落在 .viewer-wrap bbox 内**（是覆盖层）
    5. smile / neutral / mouthOpen：等 blob: src 真的变化，取真实 PNG 负载 SHA-256
    6. 关面板：hidden、#arkitFrame 移出 DOM、controls.enabled === true；
       拖拽画布 → 像素变化；.tool[data-view=left] → 再变一次
    7. 无 console error / pageerror / dialog / 新标签页
    8. /static/app.js 0 处 window.open|alert(|confirm(|prompt(|cv2

  B 阶段（诊断，**不参与判定**，输出 [PASS-DIAG]/[FAIL-DIAG]）
    用 page.route 把 /static/app.js 换成「把 init() 调用挪到模块末尾」的补丁版，
    重新加载后用历史任务回放（不重新生成）跑同一套面板断言。若 B 阶段全绿而 A 阶段
    在打开面板处失败，就证明失败原因是 app.js 的顶层执行顺序（TDZ），而面板本身的
    DOM / 覆盖层布局 / 帧渲染 / 相机恢复逻辑都是正确的。

用法::

    /usr/local/miniconda3/envs/lam/bin/python scripts/e2e_arkit_panel.py
"""
from __future__ import annotations

import hashlib
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
SHOT = ROOT / "workspace" / "e2e"
SHOT.mkdir(parents=True, exist_ok=True)

PASSED: list[str] = []
FAILED: list[str] = []
DIAG_PASSED: list[str] = []
DIAG_FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "", diag: bool = False) -> bool:
    tag = "PASS-DIAG" if (diag and ok) else "FAIL-DIAG" if diag else "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  — {extra}" if extra else ""), flush=True)
    if diag:
        (DIAG_PASSED if ok else DIAG_FAILED).append(name)
    else:
        (PASSED if ok else FAILED).append(name)
    return ok


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_json(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------- 服务生命周期
def ensure_server() -> bool:
    st = get_json(f"{BASE}/api/status")
    if st and st.get("ready"):
        print(f"      服务已就绪：{st}", flush=True)
        return True
    print("      启动服务：bash run.sh --daemon", flush=True)
    subprocess.run(["bash", "run.sh", "--daemon"], cwd=str(ROOT), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 180
    while time.time() < deadline:
        st = get_json(f"{BASE}/api/status")
        if st and st.get("ready"):
            print(f"      就绪：{st}", flush=True)
            return True
        if st and st.get("error"):
            print(f"      预热失败：{st['error']}", flush=True)
            return False
        time.sleep(3)
    return False


def stop_server() -> None:
    p = subprocess.run(["bash", "run.sh", "--stop"], cwd=str(ROOT), check=False,
                       capture_output=True, text=True)
    print(f"      run.sh --stop → {p.stdout.strip()} {p.stderr.strip()}".rstrip(), flush=True)


def canvas_lit_ratio(png: Path) -> float:
    """截图里「明显亮于背景」的像素占比。

    不能用 gl.readPixels：preserveDrawingBuffer 关闭时合成后 drawingBuffer 已清空，
    读回来恒为 0。画布背景 #0f131c 三通道合计仅约 62。
    """
    with Image.open(png) as im:
        im = im.convert("RGB")
        px = im.load()
        w, h = im.size
        lit = tot = 0
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                r, g, b = px[x, y]
                tot += 1
                if r + g + b > 150:
                    lit += 1
    return lit / max(tot, 1)


ARKIT_PATCH_NOTE = "TDZ 诊断补丁：把顶层 init() 调用挪到模块末尾"

# OrbitControls 观察器（测试侧插桩，不改仓库文件）。
# 注意：OrbitControls.js:172 在**构造函数里**写了 `this.update = function () {...}`，
# 这是实例自有属性，会遮蔽原型上的 update()，所以 patch 原型抓不到实例。
# 改为把被服务的模块里的 `export { OrbitControls };` 换成「记录实例的派生类」再导出。
OC_OLD_EXPORT = "export { OrbitControls };"
OC_NEW_EXPORT = """const __OC_Base = OrbitControls;
class __OC_Tracked extends __OC_Base {
	constructor(...args) { super(...args); (window.__ocList ||= []).push(this); }
}
export { __OC_Tracked as OrbitControls };"""


def build_patched_orbitcontrols(src: str) -> str:
    """只做观察：派生类在构造时把实例登记到 window.__ocList，行为完全不变。"""
    if OC_OLD_EXPORT not in src:
        raise RuntimeError("OrbitControls.js 里找不到导出语句，插桩失效")
    return src.replace(OC_OLD_EXPORT, OC_NEW_EXPORT, 1)


def build_patched_app_js(src: str) -> str:
    """把 `init();` 从模块顶部挪到末尾 —— 让 const arkitState 先完成初始化。

    这正是修复 app.js TDZ 的最小改法；补丁只存在于浏览器内存里，
    仓库文件不被改动（写权限限制：只允许改本脚本 + workspace/e2e 截图）。
    """
    if "\ninit();\n" not in src:
        raise RuntimeError("app.js 里找不到顶层 'init();' 调用，补丁失效")
    return src.replace("\ninit();\n", "\n", 1) + "\ninit();\n"


def main() -> int:  # noqa: C901
    if not ensure_server():
        check("服务就绪（/api/status ready:true）", False, "180s 内未就绪")
        stop_server()
        print(f"\n==== {len(PASSED)} passed, {len(FAILED)} failed ====", flush=True)
        return 1
    check("服务就绪（/api/status ready:true）", True, BASE)

    meta = get_json(f"{BASE}/api/arkit/meta", timeout=20) or {}
    names = meta.get("names") or []
    presets = list((meta.get("presets") or {}).keys())
    print(f"      ARKit meta：{len(names)} 通道，预设 {presets}", flush=True)
    check("/api/arkit/meta 提供 52 通道 + 多个预设", len(names) == 52 and len(presets) > 1,
          f"names={len(names)} presets={len(presets)}")

    # app.js 静态检查（在 Python 侧做，浏览器内只用来跑页面）
    src_js = urllib.request.urlopen(f"{BASE}/static/app.js", timeout=20).read().decode("utf-8")
    static_hits = re.findall(r"window\.open|alert\(|confirm\(|prompt\(|cv2", src_js)
    check("/static/app.js 无 window.open|alert(|confirm(|prompt(|cv2", not static_hits,
          f"{len(static_hits)} 处命中 {static_hits[:5]}；{len(src_js)} 字节")
    patched_js = build_patched_app_js(src_js)

    console_errors: list[str] = []
    page_errors: list[str] = []
    dialogs: list[str] = []
    popups: list[str] = []
    diag_console: list[str] = []
    diag_page_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
                  "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1600, "height": 950})
        context.set_default_timeout(60_000)
        page = context.new_page()

        page.on("pageerror", lambda e: page_errors.append(f"pageerror: {e}"))
        page.on("console",
                lambda m: console_errors.append(f"console.{m.type}: {m.text} @ {m.location.get('url')}")
                if m.type == "error" else None)
        page.on("dialog", lambda d: (dialogs.append(f"{d.type}: {d.message}"), d.dismiss()))
        context.on("page", lambda p: popups.append(p.url))

        panel_ok = False
        panel_shot: dict[str, str] = {}
        try:
            # ══════════════ A 阶段：当前仓库的真实页面 ══════════════
            # 先把 OrbitControls 换成「构造时登记实例」的派生类（行为不变，只做观察）。
            oc_src = urllib.request.urlopen(
                f"{BASE}/static/vendor/three/examples/jsm/controls/OrbitControls.js",
                timeout=20).read().decode("utf-8")
            oc_patched = build_patched_orbitcontrols(oc_src)
            page.route("**/controls/OrbitControls.js", lambda route: route.fulfill(
                status=200, content_type="text/javascript", body=oc_patched))
            page.goto(BASE, wait_until="networkidle", timeout=90_000)
            check("打开原页面（原 URL、原标题、原控件）",
                  page.title().startswith("LAM") and page.locator("#generate").count() == 1,
                  f"title={page.title()!r} url={page.url}")
            check("初始 context.pages == 1", len(context.pages) == 1, f"{len(context.pages)} 个")

            # OrbitControls 实例在结果出现后才会创建（initViewer 由结果触发），
            # 这里只确认插桩模块已被加载；真正的实例断言放到画布出现之后。
            hook = page.evaluate("""() => (window.__ocList && window.__ocList.length)
                ? 'ok：已捕获 ' + window.__ocList.length + ' 个实例'
                : '待结果出现后才创建实例'""")
            print(f"      OrbitControls 取证插桩：{hook}", flush=True)

            page.wait_for_selector(".samples img", timeout=30_000)
            page.locator('.samples img[alt="status.png"]').click()
            page.locator("#motions .motion", has_text="Anti_Drugs").click()
            page.locator("#skipVideo").check()
            check("已选 status.png + Anti_Drugs + 跳过视频",
                  page.locator("#preview").is_visible() and page.locator("#generate").is_enabled())

            t0 = time.time()
            page.locator("#generate").click()
            page.wait_for_selector("#result:not([hidden])", timeout=240_000)
            gen_s = time.time() - t0
            print(f"      生成完成，用时 {gen_s:.1f}s", flush=True)
            check("结果区出现（原页面内联，未发生跳转）", True, f"{gen_s:.1f}s")
            check("URL 仍是根路径 /（没有换页）", page.url.rstrip("/") == BASE.rstrip("/"), page.url)

            page.wait_for_selector("#viewer canvas", timeout=60_000)
            page.wait_for_timeout(4000)
            badge = page.locator("#viewerBadge").inner_text()
            check("#viewerBadge 含「已加载」", "已加载" in badge, badge)

            canvas = page.locator("#viewer canvas")
            page.wait_for_function(
                "() => window.__ocList && window.__ocList.length > 0", timeout=30_000)
            n_oc = page.evaluate("() => window.__ocList.length")
            check("OrbitControls 插桩生效（构造时登记，行为不变）", n_oc > 0,
                  f"已捕获 {n_oc} 个实例，controls.enabled="
                  f"{page.evaluate('() => window.__ocList[0].enabled')}")
            shot_pre = SHOT / "arkit_pre_canvas.png"
            canvas.screenshot(path=str(shot_pre))
            ratio = canvas_lit_ratio(shot_pre)
            check("原 3D 画布非空（像素级，非 gl.readPixels）", ratio > 0.02,
                  f"前景占比 {ratio*100:.1f}%（阈值 2%）；bg=#0f131c")
            page.screenshot(path=str(SHOT / "arkit_result_full.png"), full_page=True)

            # ── 约束 5 的 DOM 事实 ──
            dom = page.evaluate("""() => {
                const p = document.getElementById('arkitPanel');
                const w = document.querySelector('.viewer-wrap');
                return {
                    exists: !!p, tag: p ? p.tagName : null,
                    inWrap: !!(p && w && w.contains(p)),
                    hidden: !!(p && p.hidden),
                    display: p ? getComputedStyle(p).display : null,
                    wraps: document.querySelectorAll('.viewer-wrap').length,
                    canvases: document.querySelectorAll('canvas').length,
                    toggleInTools: !!document.querySelector('.viewer__tools #arkitToggle'),
                    toggleHandler: !!(document.getElementById('arkitToggle') || {}).onclick,
                    sliders: document.querySelectorAll('#arkitSliders .arkit-slider').length,
                    presets: document.querySelectorAll('#arkitPresets button').length,
                };
            }""")
            check("#arkitPanel 存在（<section>）且是 .viewer-wrap 的后代、初始隐藏",
                  dom["exists"] and dom["tag"] == "SECTION" and dom["inWrap"]
                  and dom["hidden"] and dom["display"] == "none",
                  json.dumps({k: dom[k] for k in ("tag", "inWrap", "hidden", "display")},
                             ensure_ascii=False))
            check("页面只有 1 个 canvas / 1 个 .viewer-wrap（没新建视图、没有副本）",
                  dom["canvases"] == 1 and dom["wraps"] == 1,
                  f"canvas={dom['canvases']} wrap={dom['wraps']}")
            check("#arkitToggle 挂在原工具栏 .viewer__tools 里", dom["toggleInTools"])

            # ── 打开面板（A 阶段的关键判定） ──
            page.locator("#arkitToggle").click()
            opened = True
            try:
                page.wait_for_selector("#arkitPanel:not([hidden])", timeout=8_000)
            except Exception:  # noqa: BLE001
                opened = False
            if not opened:
                pwerr = [e for e in page_errors if "arkitState" in e or "before initialization" in e]
                check("#arkitToggle 点击后 #arkitPanel 可见", False,
                      f"点击无效：toggleHandler={dom['toggleHandler']} "
                      f"sliders={dom['sliders']} presets={dom['presets']} "
                      f"pageerror={pwerr[0] if pwerr else '(无)'}")
                check("#arkitSliders 有 52 行（@runtime，A 阶段）", dom["sliders"] == 52,
                      f"实测 {dom['sliders']} 行")
                check("#arkitPresets 按钮 > 1（@runtime，A 阶段）", dom["presets"] > 1,
                      f"实测 {dom['presets']} 个")
            if opened:
                panel_ok = True

            wrap = page.locator(".viewer-wrap")
            n_sliders = page.locator("#arkitSliders .arkit-slider").count()
            n_presets = page.locator("#arkitPresets button").count()
            if panel_ok:
                check("#arkitSliders 有 52 行", n_sliders == 52, f"{n_sliders} 行")
                check("#arkitPresets 按钮 > 1", n_presets > 1, f"{n_presets} 个按钮")

                boxes = page.evaluate("""() => {
                    const r = (el) => { const b = el.getBoundingClientRect();
                        return {x: b.x, y: b.y, w: b.width, h: b.height, r: b.right, b: b.bottom}; };
                    const p = document.getElementById('arkitPanel');
                    const w = document.querySelector('.viewer-wrap');
                    return {panel: r(p), wrap: r(w), canvas: r(document.querySelector('#viewer canvas')),
                            panelPos: getComputedStyle(p).position};
                }""")
                pb, wb, cb = boxes["panel"], boxes["wrap"], boxes["canvas"]
                inside = (pb["x"] >= wb["x"] - 0.5 and pb["y"] >= wb["y"] - 0.5
                          and pb["r"] <= wb["r"] + 0.5 and pb["b"] <= wb["b"] + 0.5
                          and pb["w"] < wb["w"])
                print(f"      .viewer-wrap bbox = x{wb['x']:.1f} y{wb['y']:.1f} "
                      f"w{wb['w']:.1f} h{wb['h']:.1f} (right {wb['r']:.1f}, bottom {wb['b']:.1f})",
                      flush=True)
                print(f"      #arkitPanel bbox = x{pb['x']:.1f} y{pb['y']:.1f} "
                      f"w{pb['w']:.1f} h{pb['h']:.1f} (right {pb['r']:.1f}, bottom {pb['b']:.1f}) "
                      f"position={boxes['panelPos']}", flush=True)
                check("面板 bbox 完全落在 .viewer-wrap bbox 内（覆盖层，不是新页面）", inside)
                check("WebGL 画布仍在且仍有尺寸",
                      cb["w"] > 100 and cb["h"] > 100, f"canvas {cb['w']:.0f}x{cb['h']:.0f}")

                def digest() -> dict:
                    return page.evaluate("""async () => {
                        const i = document.getElementById('arkitFrame');
                        if (!i || !i.src) return null;
                        const b = new Uint8Array(await (await fetch(i.src)).arrayBuffer());
                        const h = await crypto.subtle.digest('SHA-256', b);
                        const bb = i.getBoundingClientRect();
                        return {len: b.length, nw: i.naturalWidth, nh: i.naturalHeight,
                                sha: [...new Uint8Array(h)].map(x => x.toString(16).padStart(2,'0')).join(''),
                                x: bb.x, y: bb.y, r: bb.right, b: bb.bottom, src: i.src};
                    }""")

                page.wait_for_selector("#arkitFrame", timeout=60_000)
                page.wait_for_function(
                    "() => { const i = document.getElementById('arkitFrame');"
                    " return i && i.src.startsWith('blob:') && i.complete && i.naturalWidth > 0; }",
                    timeout=60_000)
                d0 = digest()
                check("#arkitFrame 以 blob: 出现且 PNG 已解码",
                      bool(d0) and d0["src"].startswith("blob:") and d0["nw"] > 0,
                      f"{d0['nw']}x{d0['nh']}, {d0['len']} B, sha={d0['sha'][:16]}…")

                page.locator('.arkit-preset[title="smile"]').click()
                page.wait_for_function(
                    "(o) => { const i = document.getElementById('arkitFrame');"
                    " return i && i.src && i.src !== o && i.complete && i.naturalWidth > 0; }",
                    arg=d0["src"], timeout=30_000)
                d1 = digest()
                lat = page.locator("#arkitLatency").inner_text()
                gau = page.locator("#arkitGaussians").inner_text()
                check("#arkitLatency / #arkitGaussians 非空",
                      lat.replace("延迟", "").strip() not in ("", "--") and gau.strip() != "",
                      f"{lat!r} / {gau!r}")
                covers = (d1["x"] <= cb["x"] + 1 and d1["r"] >= cb["r"] - 1
                          and d1["y"] <= cb["y"] + 1 and d1["b"] >= cb["b"] - 1)
                check("#arkitFrame bbox 覆盖 WebGL 画布 bbox", covers,
                      f"frame x{d1['x']:.0f}-{d1['r']:.0f} y{d1['y']:.0f}-{d1['b']:.0f} vs "
                      f"canvas x{cb['x']:.0f}-{cb['r']:.0f} y{cb['y']:.0f}-{cb['b']:.0f}")

                wrap.screenshot(path=str(SHOT / "arkit_panel_open.png"))
                page.screenshot(path=str(SHOT / "arkit_panel_full.png"), full_page=True)
                panel_shot = {"open": str(SHOT / "arkit_panel_open.png")}

                page.locator(".arkit-preset--reset").click()   # neutral = 全 0
                page.wait_for_function(
                    "(o) => { const i = document.getElementById('arkitFrame');"
                    " return i && i.src && i.src !== o && i.complete && i.naturalWidth > 0; }",
                    arg=d1["src"], timeout=30_000)
                d2 = digest()
                page.locator("#arkitFrame").screenshot(path=str(SHOT / "arkit_neutral.png"))

                other = "mouthOpen" if "mouthOpen" in presets else next(
                    p for p in presets if p not in ("neutral", "smile"))
                page.locator(f'.arkit-preset[title="{other}"]').click()
                page.wait_for_function(
                    "(o) => { const i = document.getElementById('arkitFrame');"
                    " return i && i.src && i.src !== o && i.complete && i.naturalWidth > 0; }",
                    arg=d2["src"], timeout=30_000)
                d3 = digest()
                page.locator("#arkitFrame").screenshot(path=str(SHOT / "arkit_mouthopen.png"))

                f_neu, f_mou = SHOT / "arkit_neutral.png", SHOT / "arkit_mouthopen.png"
                print(f"      帧负载 SHA-256： 首帧={d0['sha'][:24]}…  smile={d1['sha'][:24]}…  "
                      f"neutral={d2['sha'][:24]}…  {other}={d3['sha'][:24]}…", flush=True)
                check("smile → neutral 的 blob: src 与 PNG 负载均变化",
                      d2["sha"] != d1["sha"] and d2["src"] != d1["src"])
                check(f"neutral → {other} 的 blob: src 与 PNG 负载均变化",
                      d3["sha"] != d2["sha"] and d3["src"] != d2["src"])
                check("arkit_neutral.png 与 arkit_mouthopen.png 字节级不同",
                      sha(f_neu) != sha(f_mou),
                      f"neutral sha256={sha(f_neu)[:16]}… vs {other} sha256={sha(f_mou)[:16]}…")
            else:
                check("#arkitSliders 有 52 行（运行时）", False, "面板打不开，无法验证")
                check("#arkitPresets 按钮 > 1（运行时）", False, "面板打不开，无法验证")
                check("#arkitFrame 为真实 ARKit 渲染且随表情变化", False, "面板打不开，无法验证")
                check("面板 bbox 落在 .viewer-wrap 内（覆盖层）", False, "面板打不开，无法验证")
                check("arkit_neutral.png 与 arkit_mouthopen.png 字节级不同", False,
                      "面板打不开，无法生成")

            # ── 关闭面板 + 原相机控制 ──
            cb = page.evaluate("""() => { const b = document.querySelector('#viewer canvas').getBoundingClientRect();
                return {x: b.x, y: b.y, w: b.width, h: b.height, r: b.right, b: b.bottom}; }""")
            if panel_ok:
                page.locator("#arkitClose").click()
                page.wait_for_timeout(600)
            closed = page.evaluate("""() => ({
                exists: !!document.getElementById('arkitPanel'),
                hidden: document.getElementById('arkitPanel') ? document.getElementById('arkitPanel').hidden : null,
                display: document.getElementById('arkitPanel') ? getComputedStyle(document.getElementById('arkitPanel')).display : null,
                frameInDom: !!document.getElementById('arkitFrame'),
                oc: !!(window.__ocList && window.__ocList[0]),
                enabled: (window.__ocList && window.__ocList[0]) ? window.__ocList[0].enabled : null,
            })""")
            if panel_ok:
                check("#arkitClose 后面板隐藏", closed["hidden"] and closed["display"] == "none",
                      f"hidden={closed['hidden']} display={closed['display']}")
                check("#arkitFrame 已从 DOM 移除", closed["frameInDom"] is False)
            check("viewer.controls.enabled === true（相机控制已恢复/未被破坏）",
                  closed["enabled"] is True,
                  f"OrbitControls 取证={closed['oc']} enabled={closed['enabled']}")

            # 拖拽必须真的转相机：取画布 72% 宽度处，避开面板区域，作为正/负对照
            bx = cb["x"] + cb["w"] * 0.72
            by = cb["y"] + cb["h"] * 0.5
            before_png = SHOT / "arkit_canvas_before_drag.png"
            canvas.screenshot(path=str(before_png))
            cam0 = page.evaluate(
                "() => (window.__ocList && window.__ocList[0]) ? window.__ocList[0].object.position.toArray() : null")
            page.mouse.move(bx, by)
            page.mouse.down()
            for i in range(1, 13):
                page.mouse.move(bx - 14 * i, by + 4 * i)
            page.mouse.up()
            page.wait_for_timeout(1200)
            after_png = SHOT / "arkit_canvas_after_drag.png"
            canvas.screenshot(path=str(after_png))
            cam1 = page.evaluate(
                "() => (window.__ocList && window.__ocList[0]) ? window.__ocList[0].object.position.toArray() : null")
            d_px = max(abs(a - b) for a, b in zip(cam0, cam1)) if (cam0 and cam1) else None
            check("拖拽画布后渲染图像变化（OrbitControls 仍在轨道）",
                  sha(before_png) != sha(after_png),
                  f"before {sha(before_png)[:12]}… → after {sha(after_png)[:12]}…")
            check("相机位置确实被拖拽改变（不是静止画面）",
                  d_px is not None and d_px > 0.01,
                  f"cam {cam0} → {cam1}（Δmax={d_px if d_px is not None else 'n/a'}）")

            view_png = SHOT / "arkit_canvas_after_view_left.png"
            page.locator('.tool[data-view="left"]').click()
            page.wait_for_timeout(1200)
            canvas.screenshot(path=str(view_png))
            cam2 = page.evaluate(
                "() => (window.__ocList && window.__ocList[0]) ? window.__ocList[0].object.position.toArray() : null")
            check(".tool[data-view=left] 后画布再次变化",
                  sha(view_png) != sha(after_png) and cam2 != cam1, f"cam={cam2}")
            page.screenshot(path=str(SHOT / "arkit_closed_camera_ok.png"))

            # ── 运行时错误 ──
            check("无 alert/confirm/prompt 弹窗", not dialogs, "; ".join(dialogs[:3]))
            check("始终没有新标签页/新窗口", len(context.pages) == 1 and not popups,
                  f"pages={len(context.pages)} popups={popups}")
            fav = [e for e in console_errors if "favicon" in e]
            real_console = [e for e in console_errors if "favicon" not in e]
            check("无 pageerror", not page_errors, "; ".join(page_errors[:2]))
            check("无 console error", not real_console, "; ".join(real_console[:2]))
            print(f"      console error 列表 = {real_console}（favicon 404 噪声 {len(fav)} 条，"
                  f"按 e2e_frontend.py 同规则剔除）", flush=True)

            # ══════════════ B 阶段：诊断（不参与判定） ══════════════
            if not panel_ok:
                print("\n      ── B 阶段诊断：把 app.js 的顶层 init() 挪到模块末尾后重跑 ──",
                      flush=True)
                n_err_before = len(page_errors)
                page.route("**/static/app.js", lambda route: route.fulfill(
                    status=200, content_type="text/javascript", body=patched_js))
                page.on("pageerror", lambda e: diag_page_errors.append(str(e)))
                page.on("console", lambda m: diag_console.append(m.text) if m.type == "error" else None)
                page.goto(BASE, wait_until="networkidle", timeout=90_000)
                page.wait_for_timeout(3000)
                d_sl = page.locator("#arkitSliders .arkit-slider").count()
                d_pr = page.locator("#arkitPresets button").count()
                d_handler = page.evaluate("()=>!!document.getElementById('arkitToggle').onclick")
                check("补丁后：无 TDZ pageerror（新增）", len(page_errors) == n_err_before,
                      f"新增 {len(page_errors) - n_err_before} 条：{diag_page_errors[:1]}", diag=True)
                check("补丁后：#arkitSliders 52 行 / #arkitPresets >1 / toggle 有处理器",
                      d_sl == 52 and d_pr > 1 and d_handler,
                      f"sliders={d_sl} presets={d_pr} handler={d_handler}", diag=True)

                # 用历史任务回放结果（避免再花 60s 生成）
                page.wait_for_selector("#history .hitem", timeout=60_000)
                page.locator("#history .hitem").first.click()
                page.wait_for_selector("#result:not([hidden])", timeout=120_000)
                page.wait_for_selector("#viewer canvas", timeout=60_000)
                page.wait_for_timeout(3000)
                page.locator("#arkitToggle").click()
                d_opened = True
                try:
                    page.wait_for_selector("#arkitPanel:not([hidden])", timeout=15_000)
                except Exception:  # noqa: BLE001
                    d_opened = False
                check("补丁后：点 #arkitToggle 面板可见", d_opened, diag=True)
                if d_opened:
                    dv = page.evaluate("""() => {
                        const r = (el) => { const b = el.getBoundingClientRect();
                            return {x: b.x, y: b.y, w: b.width, h: b.height, r: b.right, b: b.bottom}; };
                        return {panel: r(document.getElementById('arkitPanel')),
                                wrap: r(document.querySelector('.viewer-wrap')),
                                wraps: document.querySelectorAll('.viewer-wrap').length,
                                canvases: document.querySelectorAll('canvas').length,
                                sliders: document.querySelectorAll('#arkitSliders .arkit-slider').length,
                                presets: document.querySelectorAll('#arkitPresets button').length,
                                pages: 0};
                    }""")
                    dpb, dwb = dv["panel"], dv["wrap"]
                    d_inside = (dpb["x"] >= dwb["x"] - 0.5 and dpb["y"] >= dwb["y"] - 0.5
                                and dpb["r"] <= dwb["r"] + 0.5 and dpb["b"] <= dwb["b"] + 0.5)
                    print(f"      [DIAG] wrap  bbox = x{dwb['x']:.1f} y{dwb['y']:.1f} "
                          f"w{dwb['w']:.1f} h{dwb['h']:.1f} (right {dwb['r']:.1f}, bottom {dwb['b']:.1f})",
                          flush=True)
                    print(f"      [DIAG] panel bbox = x{dpb['x']:.1f} y{dpb['y']:.1f} "
                          f"w{dpb['w']:.1f} h{dpb['h']:.1f} (right {dpb['r']:.1f}, bottom {dpb['b']:.1f})",
                          flush=True)
                    check("补丁后：面板 bbox 在 .viewer-wrap 内 且 1 canvas / 1 wrap / 1 page / URL /",
                          d_inside and dv["canvases"] == 1 and dv["wraps"] == 1
                          and len(context.pages) == 1 and page.url.rstrip("/") == BASE.rstrip("/"),
                          f"inside={d_inside} canvas={dv['canvases']} wrap={dv['wraps']} "
                          f"pages={len(context.pages)} url={page.url}", diag=True)

                    def ddigest() -> dict:
                        return page.evaluate("""async () => {
                            const i = document.getElementById('arkitFrame');
                            if (!i || !i.src) return {nw: 0, sha: '', src: ''};
                            const b = new Uint8Array(await (await fetch(i.src)).arrayBuffer());
                            const h = await crypto.subtle.digest('SHA-256', b);
                            return {nw: i.naturalWidth, len: b.length,
                                    sha: [...new Uint8Array(h)].map(x=>x.toString(16).padStart(2,'0')).join(''),
                                    src: i.src};
                        }""")
                    page.wait_for_selector("#arkitFrame", timeout=60_000)
                    page.wait_for_function(
                        "() => { const i = document.getElementById('arkitFrame');"
                        " return i && i.src.startsWith('blob:') && i.complete && i.naturalWidth > 0; }",
                        timeout=60_000)
                    e0 = ddigest()
                    page.locator('.arkit-preset[title="smile"]').click()
                    page.wait_for_function(
                        "(o) => { const i = document.getElementById('arkitFrame');"
                        " return i && i.src !== o && i.complete && i.naturalWidth > 0; }",
                        arg=e0["src"], timeout=30_000)
                    e1 = ddigest()
                    page.locator(".viewer-wrap").screenshot(path=str(SHOT / "arkit_panel_open.png"))
                    page.screenshot(path=str(SHOT / "arkit_panel_full.png"), full_page=True)
                    panel_shot["open"] = str(SHOT / "arkit_panel_open.png")
                    check("补丁后：#arkitFrame 是真实 ARKit PNG 负载且随表情变化",
                          e0["sha"] != e1["sha"] and e0["nw"] > 0,
                          f"{e0['nw']}x · {e0['len']}B · sha {e0['sha'][:16]}… → {e1['sha'][:16]}…",
                          diag=True)

                    page.locator(".arkit-preset--reset").click()
                    page.wait_for_function(
                        "(o) => { const i = document.getElementById('arkitFrame');"
                        " return i && i.src !== o && i.complete && i.naturalWidth > 0; }",
                        arg=e1["src"], timeout=30_000)
                    e2 = ddigest()
                    page.locator("#arkitFrame").screenshot(path=str(SHOT / "arkit_neutral.png"))
                    d_other = "mouthOpen" if "mouthOpen" in presets else presets[-1]
                    page.locator(f'.arkit-preset[title="{d_other}"]').click()
                    page.wait_for_function(
                        "(o) => { const i = document.getElementById('arkitFrame');"
                        " return i && i.src !== o && i.complete && i.naturalWidth > 0; }",
                        arg=e2["src"], timeout=30_000)
                    e3 = ddigest()
                    page.locator("#arkitFrame").screenshot(path=str(SHOT / "arkit_mouthopen.png"))
                    a, b_ = SHOT / "arkit_neutral.png", SHOT / "arkit_mouthopen.png"
                    check("补丁后：neutral 与 mouthOpen 两帧 PNG 字节级不同",
                          e2["sha"] != e3["sha"] and sha(a) != sha(b_),
                          f"sha {e2['sha'][:16]}… vs {e3['sha'][:16]}…", diag=True)

                    check("补丁后：#arkitLatency / #arkitGaussians 有内容",
                          page.locator("#arkitLatency").inner_text().replace("延迟", "").strip()
                          not in ("", "--") and page.locator("#arkitGaussians").inner_text().strip() != "",
                          f"{page.locator('#arkitLatency').inner_text()!r} / "
                          f"{page.locator('#arkitGaussians').inner_text()!r}", diag=True)

                    page.locator("#arkitClose").click()
                    page.wait_for_timeout(700)
                    dc = page.evaluate("""() => ({
                        hidden: document.getElementById('arkitPanel').hidden,
                        frameInDom: !!document.getElementById('arkitFrame'),
                        enabled: (window.__ocList && window.__ocList[0]) ? window.__ocList[0].enabled : null})""")
                    check("补丁后：关面板 → hidden、#arkitFrame 移除、controls.enabled === true",
                          dc["hidden"] and not dc["frameInDom"] and dc["enabled"] is True,
                          json.dumps(dc), diag=True)
                    check("补丁后：无新增 console error",
                          not [c for c in diag_console if "favicon" not in c],
                          "; ".join(diag_console[:2]), diag=True)
        finally:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass

    stop_server()
    print(f"\n      截图：{sorted(p.name for p in SHOT.glob('arkit_*.png'))}", flush=True)
    if panel_ok:
        print(f"      面板截图：{panel_shot}", flush=True)
    print(f"\n==== {len(PASSED)} passed, {len(FAILED)} failed ====", flush=True)
    if DIAG_PASSED or DIAG_FAILED:
        print(f"[诊断·不计入判定] {len(DIAG_PASSED)} passed, {len(DIAG_FAILED)} failed")
    if FAILED:
        print("失败项：" + "; ".join(FAILED), flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
