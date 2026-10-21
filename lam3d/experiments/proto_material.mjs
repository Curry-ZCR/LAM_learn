/**
 * 用真实 offscreen WebGL 渲染对比几种材质/光照配置，挑出最接近 LAM 自己渲染结果的方案。
 *
 * 背景：LAM 的 renderer 不做光照 —— gs.shs 的直流分量就是最终颜色，
 * 光栅化后直接输出。所以顶点色里已经包含烘焙好的明暗。
 */
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';
import { readFileSync, writeFileSync } from 'node:fs';
import { createCanvas } from 'canvas';

const buf = readFileSync(process.argv[2]);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const SIZE = 360;

const configs = [
  { name: 'A_unlit',            desc: '无光照（MeshBasic）', ambient: 0, dir: 0, tone: 'none' },
  { name: 'B_amb1_none',        desc: '环境光1.0 无色调映射', ambient: 1.0, dir: 0.0, tone: 'none' },
  { name: 'C_amb1_dir0.6',      desc: '环境光1.0 + 方向光0.6', ambient: 1.0, dir: 0.6, tone: 'none' },
  { name: 'D_amb2.4_aces',      desc: '环境光2.4 方向光1.18 ACES(当前)', ambient: 2.4, dir: 1.18, tone: 'aces' },
];

new GLTFLoader().parse(ab, '', (gltf) => {
  const canvas = createCanvas(SIZE, SIZE);
  const gl = canvas.getContext('webgl2');
  if (!gl) { console.error('no webgl2 in node'); process.exit(1); }

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  renderer.setSize(SIZE, SIZE, false);

  const scene0 = gltf.scene;
  const box = new THREE.Box3().setFromObject(scene0);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  scene0.position.sub(center);

  const holder = new THREE.Group();
  holder.add(scene0);
  holder.scale.setScalar(1 / maxDim);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  camera.position.set(0.42, 0.10, 0.88).normalize().multiplyScalar(1.55);

  const results = [];
  for (const cfg of configs) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f131c);
    // 每个配置重建一份材质
    holder.traverse((o) => {
      if (!o.isMesh) return;
      const g = o.geometry;
      if (cfg.name === 'A_unlit') {
        o.material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
      } else {
        o.material = new THREE.MeshStandardMaterial({
          vertexColors: true, color: 0xffffff, roughness: 1.0, metalness: 0.0,
          side: THREE.DoubleSide,
        });
      }
      if (!g.attributes.normal) g.computeVertexNormals();
    });
    scene.add(holder);
    if (cfg.ambient > 0) scene.add(new THREE.AmbientLight(0xffffff, cfg.ambient));
    if (cfg.dir > 0) {
      const d = new THREE.DirectionalLight(0xffffff, cfg.dir);
      d.position.set(0.55, 0.65, 1.0);
      scene.add(d);
    }
    renderer.toneMapping = cfg.tone === 'aces' ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    renderer.render(scene, camera);
    const px = new Uint8Array(SIZE * SIZE * 4);
    gl.readPixels(0, 0, SIZE, SIZE, gl.RGBA, gl.UNSIGNED_BYTE, px);

    // 统计前景（非背景）平均色
    let n = 0, r = 0, g = 0, b = 0;
    for (let i = 0; i < px.length; i += 4) {
      const R = px[i], G = px[i + 1], B = px[i + 2];
      if (R + G + B > 140) { n++; r += R; g += G; b += B; }
    }
    const avg = n ? [r / n, g / n, b / n] : [0, 0, 0];
    results.push({ ...cfg, avg, coverage: (n / (SIZE * SIZE) * 100).toFixed(1), px });
    console.log(`${cfg.name.padEnd(16)} ${cfg.desc.padEnd(30)} 前景平均 RGB = [${avg.map((v) => v.toFixed(0)).join(', ')}]  覆盖 ${(n / (SIZE * SIZE) * 100).toFixed(1)}%`);
  }

  // 拼接成对比图（PNG，Y 轴翻转）
  const cols = results.length;
  const out = Buffer.alloc(SIZE * SIZE * 4 * cols);
  results.forEach((r, k) => {
    for (let y = 0; y < SIZE; y++) {
      const src = (SIZE - 1 - y) * SIZE * 4;
      r.px.copy(out, (y * SIZE * cols + k * SIZE) * 4, src, src + SIZE * 4);
    }
  });
  // 手工写 PNG：用 canvas 反向贴图
  const outCanvas = createCanvas(SIZE * cols, SIZE);
  const octx = outCanvas.getContext('2d');
  const img = octx.createImageData(SIZE * cols, SIZE);
  img.data.set(out);
  octx.putImageData(img, 0, 0);
  writeFileSync('/tmp/material_cmp.png', outCanvas.toBuffer('image/png'));
  console.log('\n已写出 /tmp/material_cmp.png');
  console.log('参考: LAM 视频帧面部平均 RGB = [175, 170, 157]');
});
