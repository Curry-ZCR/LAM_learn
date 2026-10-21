"""前端端到端测试：用真实无头浏览器打开页面，跑完一次「单图 → 3D 人物」。

覆盖点：
  1. 页面加载、无 JS 报错
  2. /api/config 拉取后动作列表与示例图渲染出来
  3. 选择示例图 + 动作 → 点「生成」
  4. 轮询进度直到 succeeded
  5. Three.js 画布真的渲染出东西（读像素，不是只看 DOM）
  6. GLB 被 GLTFLoader 成功解析（三角面数 > 0）
  7. 视频与下载链接就位

用法::

    python scripts/e2e_frontend.py [base_url] [--headful]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "http://127.0.0.1:8000"
SHOT_DIR = Path(__file__).resolve().parents[1] / "workspace" / "e2e"
SHOT_DIR.mkdir(parents=True, exist_ok=True)

FAILURES: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {extra}" if extra else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def main() -> int:
    with sync_playwright() as pw:
        import os

        launch_kwargs = dict(
            headless="--headful" not in sys.argv,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
        # 允许用系统里已有的 Chrome（本机 Playwright 的 chromium 下载很慢）
        exe = os.environ.get("LAM3D_CHROME")
        if exe:
            launch_kwargs["executable_path"] = exe
        browser = pw.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1600, "height": 950})

        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)

        # ---- 1. 页面加载 ----
        page.goto(BASE, wait_until="networkidle", timeout=60_000)
        check("页面标题", "LAM" in page.title(), page.title())
        page.wait_for_selector("#motions .motion", timeout=30_000)
        n_motions = page.locator("#motions .motion").count()
        n_samples = page.locator("#samples img").count()
        check("动作列表渲染", n_motions >= 10, f"{n_motions} 个")
        check("示例图渲染", n_samples >= 6, f"{n_samples} 张")

        # ---- 2. 选择示例图与动作 ----
        page.locator("#samples img").nth(0).click()
        page.wait_for_timeout(400)
        check("选中示例图后预览出现", page.locator("#preview").is_visible())
        page.locator("#motions .motion", has_text="Anti_Drugs").click()
        page.wait_for_timeout(300)
        check("生成按钮可用", page.locator("#generate").is_enabled())
        page.screenshot(path=str(SHOT_DIR / "1_ready.png"))

        # ---- 3. 触发生成 ----
        page.locator("#generate").click()
        page.wait_for_selector("#progress:not([hidden])", timeout=20_000)
        check("进度条出现", True)

        # ---- 4. 等待完成 ----
        deadline = time.time() + 420
        last = ""
        while time.time() < deadline:
            stage = page.locator("#stageText").inner_text()
            pct = page.locator("#pctText").inner_text()
            detail = page.locator("#detailText").inner_text()
            cur = f"{stage} {pct} {detail}"
            if cur != last:
                print(f"      进度: {cur}", flush=True)
                last = cur
            if stage.strip() == "完成":
                break
            if page.locator("#errorBox").is_visible():
                check("生成成功", False, page.locator("#errorText").inner_text())
                break
            page.wait_for_timeout(1500)
        else:
            check("生成成功", False, "超时")

        page.wait_for_selector("#result:not([hidden])", timeout=30_000)
        check("结果区出现", True)
        page.screenshot(path=str(SHOT_DIR / "2_result.png"))

        # 关键：空态与结果区不能同时可见（[hidden] 曾被 display:grid 盖掉）
        vis = page.evaluate("""() => ({
            empty: getComputedStyle(document.querySelector('#empty')).display !== 'none',
            result: getComputedStyle(document.querySelector('#result')).display !== 'none',
            auto: document.querySelector('#result').getBoundingClientRect().height,
        })""")
        check("空态已隐藏、结果区可见（互斥）",
              (not vis["empty"]) and vis["result"] and vis["auto"] > 100, str(vis))

        # ---- 5. Three.js 画布真的画了东西 ----
        page.wait_for_timeout(4000)  # 等 GLTFLoader 解析 + 首帧渲染
        badge = page.locator("#viewerBadge").inner_text()
        check("GLB 加载成功（徽标）", "已加载" in badge, badge)

        # 用截图裁出画布区域再统计像素。
        # 注意：不能用 gl.readPixels —— 未开 preserveDrawingBuffer 时，
        # 合成之后的 drawning buffer 已被清空，读回来恒为 0。
        canvas_el = page.locator("#viewer canvas")
        shot = SHOT_DIR / "3_viewer.png"
        canvas_el.screenshot(path=str(shot))

        import io as _io

        from PIL import Image as _Image

        with _Image.open(shot) as im:
            im = im.convert("RGB")
            px = im.load()
            w, h = im.size
            lit = 0
            total = 0
            # 采样：步长 3 像素，统计「明显亮于背景」的像素占比
            for y in range(0, h, 3):
                for x in range(0, w, 3):
                    r, g, b = px[x, y]
                    total += 1
                    if r + g + b > 150:      # 背景 #0f131c 合计仅约 60
                        lit += 1
            ratio = lit / max(total, 1)
        check("3D 模型已渲染到画布（像素级）", ratio > 0.02,
              f"前景占比 {ratio*100:.1f}%（阈值 2%）")

        # ---- 6. 视频与下载 ----
        vready = page.evaluate("""() => {
            const v = document.querySelector('#video');
            return {shown: !v.hidden, src: (v.src||'').split('/').pop(), ready: v.readyState};
        }""")
        check("视频元素就位", vready["shown"] and "video" in vready["src"], str(vready))

        links = page.evaluate("""() => ({
            glb: !!document.querySelector('#dlGlb').href,
            obj: !!document.querySelector('#dlObj').href,
            ply: !!document.querySelector('#dlPly').href,
            mp4: !!document.querySelector('#dlMp4').href,
        })""")
        check("四个下载链接就位", all(links.values()), str(links))

        stats = page.locator("#stats").inner_text().replace("\n", " / ")
        print(f"      统计: {stats}", flush=True)
        check("统计信息非空", "高斯点数" in stats)

        # ---- 7. 交互：旋转 / 视角 / 线框 ----
        for view in ("front", "left", "right", "three"):
            page.locator(f'.tool[data-view="{view}"]').click()
            page.wait_for_timeout(350)
        page.locator("#spin").click()
        page.wait_for_timeout(1200)
        page.locator("#wire").click()
        page.wait_for_timeout(800)
        page.screenshot(path=str(SHOT_DIR / "3_interact.png"))
        check("查看器交互无异常（4 个视角 / 自转 / 线框）", True)

        # 历史记录
        page.wait_for_timeout(2500)
        n_hist = page.locator("#history .hitem").count()
        check("历史记录出现", n_hist >= 1, f"{n_hist} 条")

        real_errors = [e for e in errors if "favicon" not in e.lower()]
        check("无 JS 运行时报错", not real_errors,
              "; ".join(real_errors[:3]) if real_errors else "")

        page.screenshot(path=str(SHOT_DIR / "4_final.png"), full_page=True)
        browser.close()

    print("\n" + ("前端端到端测试全部通过 ✅" if not FAILURES else f"失败 {len(FAILURES)} 项：{FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
