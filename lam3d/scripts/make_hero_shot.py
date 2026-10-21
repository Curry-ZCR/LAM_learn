"""生成一张最终演示截图：真人照片 → 3D 模型结果页（正面视角）。"""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path("/workspace/lam3d/workspace/e2e")
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    lk = dict(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage",
              "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    if os.environ.get("LAM3D_CHROME"):
        lk["executable_path"] = os.environ["LAM3D_CHROME"]
    browser = pw.chromium.launch(**lk)
    page = browser.new_page(viewport={"width": 1560, "height": 900}, device_scale_factor=1.5)

    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#motions .motion")

    # 选真人示例 + 动作
    page.locator("#samples img[title='messi.png']").click()
    page.wait_for_timeout(300)
    page.locator("#motions .motion", has_text="Taylor_Swift").click()
    page.wait_for_timeout(300)

    page.locator("#generate").click()
    print("已提交，等待生成…", flush=True)
    deadline = time.time() + 400
    while time.time() < deadline:
        if page.locator("#result").is_visible():
            break
        if page.locator("#errorBox").is_visible():
            print("失败:", page.locator("#errorText").inner_text())
            sys.exit(1)
        page.wait_for_timeout(1500)

    page.wait_for_timeout(5000)          # 等 GLB 加载 + 首帧
    page.screenshot(path=str(OUT / "0_hero.png"))
    print("已保存", OUT / "0_hero.png")

    # 再来一张四分之三视角
    page.locator('.tool[data-view="three"]').click()
    page.wait_for_timeout(1500)
    page.locator('.tool[data-view="left"]').click()
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "0_hero_side.png"))
    print("已保存", OUT / "0_hero_side.png")
    browser.close()
