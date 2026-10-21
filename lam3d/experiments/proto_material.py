"""用真实浏览器（Chrome + SwiftShader WebGL）对比几种材质/光照配置，
挑出与 LAM 自己渲染结果最接近的方案。

LAM 的 renderer 不做光照：gs.shs 的直流分量就是最终颜色，光栅化后直接输出。
所以顶点色里已经包含烘焙好的明暗，再叠加强光照就会"二次着色"变灰白。
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

GLB = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if GLB is None or not GLB.is_file():
    print("用法: python proto_material.py <file.glb>")
    raise SystemExit(2)

VENDOR = Path("/workspace/lam3d/static/vendor")
glb_b64 = base64.b64encode(GLB.read_bytes()).decode()
BASE = os.environ.get("LAM3D_BASE", "http://127.0.0.1:8000")

PAGE = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script type="importmap">
{{"imports": {{
  "three": "./vendor/three/build/three.module.js",
  "three/addons/": "./vendor/three/examples/jsm/"
}}}}
</script></head><body style="margin:0;background:#000">
<canvas id="c" width="360" height="360"></canvas>
<script type="module">
import * as THREE from 'three';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const CONFIGS = {json.dumps([
    {"name": "A_unlit", "desc": "无光照 MeshBasic（顶点色直出）", "mode": "unlit"},
    {"name": "B_amb1", "desc": "环境光 1.0，无色调映射", "mode": "std", "ambient": 1.0, "dir": 0.0, "tone": "none"},
    {"name": "C_amb1_dir6", "desc": "环境光 1.0 + 方向光 0.6", "mode": "std", "ambient": 1.0, "dir": 0.6, "tone": "none"},
    {"name": "D_amb24_dir12_aces", "desc": "环境光 2.4 + 方向光 1.18 + ACES", "mode": "std", "ambient": 2.4, "dir": 1.18, "tone": "aces"},
])};

const b64 = "{glb_b64}";
const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));

const renderer = new THREE.WebGLRenderer({{canvas: document.getElementById('c'), antialias: true}});
renderer.setSize(360, 360, false);

new GLTFLoader().parse(bin.buffer, '', (gltf) => {{
  const root = gltf.scene;
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  root.position.sub(center);
  const holder = new THREE.Group(); holder.add(root); holder.scale.setScalar(1 / maxDim);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  camera.position.set(0.42, 0.10, 0.88).normalize().multiplyScalar(1.75);
  camera.lookAt(0, 0, 0);   // 少了这句相机会朝 -Z 看，模型被裁到一边

  const out = [];
  for (const cfg of CONFIGS) {{
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f131c);
    holder.traverse((o) => {{
      if (!o.isMesh) return;
      const g = o.geometry;
      if (!g.attributes.normal) g.computeVertexNormals();
      o.material = cfg.mode === 'unlit'
        ? new THREE.MeshBasicMaterial({{vertexColors: true, side: THREE.DoubleSide}})
        : new THREE.MeshStandardMaterial({{vertexColors: true, color: 0xffffff,
            roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide}});
    }});
    scene.add(holder);
    if (cfg.ambient) scene.add(new THREE.AmbientLight(0xffffff, cfg.ambient));
    if (cfg.dir) {{ const d = new THREE.DirectionalLight(0xffffff, cfg.dir);
      d.position.set(0.55, 0.65, 1.0); scene.add(d); }}
    renderer.toneMapping = cfg.tone === 'aces' ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.render(scene, camera);

    const px = new Uint8Array(360*360*4);
    const gl = renderer.getContext();
    gl.readPixels(0, 0, 360, 360, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let n = 0, r = 0, g2 = 0, b = 0;
    for (let i = 0; i < px.length; i += 4) {{
      const R = px[i], G = px[i+1], B = px[i+2];
      if (R + G + B > 140) {{ n++; r += R; g2 += G; b += B; }}
    }}
    out.push({{...cfg, avg: n ? [r/n, g2/n, b/n] : [0,0,0],
               coverage: n/(360*360)*100,
               png: document.getElementById('c').toDataURL('image/png')}});
  }}
  window.__RESULT__ = out;
}});
</script></body></html>"""

with sync_playwright() as pw:
    launch = dict(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
              "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    exe = os.environ.get("LAM3D_CHROME")
    if exe:
        launch["executable_path"] = exe
    browser = pw.chromium.launch(**launch)
    page = browser.new_page(viewport={"width": 500, "height": 460})
    errs: list[str] = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

    tmp = Path(tempfile.mkdtemp()) / "mat.html"
    tmp.write_text(PAGE, encoding="utf-8")
    import shutil as _sh
    _sh.copytree("/workspace/lam3d/static/vendor", tmp.parent / "vendor", dirs_exist_ok=True)
    # file:// 下 ES module 会被 CORS 拦掉，用本地 http 服务提供测试页
    import http.server
    import threading

    os.chdir(tmp.parent)
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp.parent), **k)
    )
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    page.goto(f"http://127.0.0.1:{port}/{tmp.name}")
    page.wait_for_function("() => window.__RESULT__ !== undefined", timeout=90_000)
    res = page.evaluate("() => window.__RESULT__")
    httpd.shutdown()

    print("参考：LAM 自己渲染的视频帧 面部平均 RGB = [175, 170, 157]\n")
    for r in res:
        a = r["avg"]
        print(f"{r['name']:20s} {r['desc']:28s} RGB=[{a[0]:5.1f},{a[1]:5.1f},{a[2]:5.1f}]  "
              f"覆盖 {r['coverage']:5.1f}%")

    # 拼接对比图为一张 PNG
    from PIL import Image
    import io as _io

    tiles = []
    for r in res:
        png = base64.b64decode(r["png"].split(",", 1)[1])
        tiles.append(Image.open(_io.BytesIO(png)).convert("RGB"))
    w = sum(t.width for t in tiles)
    h = max(t.height for t in tiles)
    strip = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width
    out = Path("/tmp/material_cmp.png")
    strip.save(out)
    print(f"\n已写出 {out}（左→右：{' | '.join(r['name'] for r in res)}）")
    if errs:
        print("\n页面报错:", errs[:3])
    browser.close()
