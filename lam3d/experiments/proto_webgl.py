"""最小复现：确认 headless Chrome 里 three.js + WebGL 到底卡在哪一步。"""
import base64, json, os, sys, tempfile, threading, http.server
from pathlib import Path
from playwright.sync_api import sync_playwright

GLB = Path(sys.argv[1])
BASE = "http://127.0.0.1:8000"
b64 = base64.b64encode(GLB.read_bytes()).decode()

# 关键：vendor 必须与测试页同源，否则 ES module 会被 CORS 拦截。
# 所以把 vendor 软链到测试目录下，用相对路径导入。
PAGE = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script type="importmap">{{"imports": {{
  "three": "./vendor/three/build/three.module.js",
  "three/addons/": "./vendor/three/examples/jsm/"
}}}}</script></head><body><canvas id="c" width="200" height="200"></canvas>
<script type="module">
const log = (m) => {{ window.__LOG__ = (window.__LOG__||[]).concat([m]); }};
log('module start');
try {{
  const THREE = await import('three');
  log('three imported ' + THREE.REVISION);
  const {{ GLTFLoader }} = await import('three/addons/loaders/GLTFLoader.js');
  log('gltfloader imported');
  const r = new THREE.WebGLRenderer({{canvas: document.getElementById('c')}});
  log('renderer ok, ctx=' + (r.getContext() ? 'yes':'no'));
  const gl = r.getContext();
  log('GL_VERSION=' + gl.getParameter(gl.VERSION));
  log('GL_RENDERER=' + gl.getParameter(gl.RENDERER));
  const bin = Uint8Array.from(atob("{b64}"), c=>c.charCodeAt(0));
  log('glb bytes=' + bin.length);
  const loader = new GLTFLoader();
  loader.parse(bin.buffer, '', (g) => {{
    log('gltf parsed');
    let geo; g.scene.traverse(o=>{{ if(o.isMesh && !geo) geo=o.geometry; }});
    log('verts=' + geo.attributes.position.count + ' hasColor=' + !!geo.attributes.color);
    const sc = new THREE.Scene(); sc.background = new THREE.Color(0x223344);
    sc.add(new THREE.AmbientLight(0xffffff, 1.0));
    const box=new THREE.Box3().setFromObject(g.scene); const size=new THREE.Vector3(); box.getSize(size);
    const ctr=new THREE.Vector3(); box.getCenter(ctr); g.scene.position.sub(ctr);
    const h=new THREE.Group(); h.add(g.scene); h.scale.setScalar(1/Math.max(size.x,size.y,size.z));
    h.traverse(o=>{{ if(o.isMesh) o.material=new THREE.MeshBasicMaterial({{vertexColors:!!o.geometry.attributes.color}}); }});
    sc.add(h);
    const cam=new THREE.PerspectiveCamera(38,1,0.01,100); cam.position.set(0.4,0.1,0.9).normalize().multiplyScalar(1.75); cam.lookAt(0,0,0);
    r.setSize(200,200,false); r.render(sc,cam); log('rendered');
    const px=new Uint8Array(200*200*4); gl.readPixels(0,0,200,200,gl.RGBA,gl.UNSIGNED_BYTE,px);
    let lit=0; for(let i=0;i<px.length;i+=4) if(px[i]+px[i+1]+px[i+2]>140) lit++;
    log('lit pixels=' + lit + ' (' + (lit/(200*200)*100).toFixed(1) + '%)');
    window.__DONE__ = true;
  }}, (e) => {{ log('gltf ERROR ' + e); window.__DONE__=true; }});
}} catch (e) {{ log('EXCEPTION: ' + e.message); window.__DONE__ = true; }}
</script></body></html>"""

with sync_playwright() as pw:
    launch = dict(headless=True, args=["--no-sandbox","--disable-dev-shm-usage",
                  "--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
    if os.environ.get("LAM3D_CHROME"): launch["executable_path"] = os.environ["LAM3D_CHROME"]
    b = pw.chromium.launch(**launch)
    pg = b.new_page()
    msgs = []
    pg.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
    pg.on("pageerror", lambda e: msgs.append(f"[pageerror] {e}"))
    d = Path(tempfile.mkdtemp()); (d/"t.html").write_text(PAGE, encoding="utf-8")
    import shutil as _sh
    _sh.copytree("/workspace/lam3d/static/vendor", d/"vendor", dirs_exist_ok=True)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1",0),
        lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(d), **k))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pg.goto(f"http://127.0.0.1:{httpd.server_address[1]}/t.html")
    try:
        pg.wait_for_function("() => window.__DONE__ === true", timeout=45000)
    except Exception as e:
        print("超时:", str(e)[:80])
    print("--- 页面日志 ---")
    for l in pg.evaluate("() => window.__LOG__ || []"): print(" ", l)
    print("--- console/错误 ---")
    for m in msgs[:15]: print(" ", m[:200])
    httpd.shutdown(); b.close()
