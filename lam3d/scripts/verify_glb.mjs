/**
 * 用 three.js 真实的 GLTFLoader 解析后端产出的 GLB —— 与浏览器里跑的完全同一份代码。
 * Node 里没有 DOM/WebGL，但 GLTFLoader 的解析、属性布局、顶点色、材质生成
 * 都是纯 JS 逻辑，可以完整验证。
 */
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { readFileSync } from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('用法: node verify_glb.mjs <file.glb>'); process.exit(2); }

// GLTFLoader 需要全局 FileLoader 走 fetch；直接喂 ArrayBuffer 绕过网络
const buf = readFileSync(path);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const loader = new GLTFLoader();
const fails = [];
const ok = (n, c, x = '') => {
  console.log(`[${c ? 'PASS' : 'FAIL'}] ${n}${x ? '  — ' + x : ''}`);
  if (!c) fails.push(n);
};

loader.parse(ab, '', (gltf) => {
  ok('GLTFLoader 解析成功', true);

  let meshes = 0, tris = 0, verts = 0;
  const details = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    meshes++;
    const g = o.geometry;
    const nv = g.attributes.position.count;
    const idx = g.index ? g.index.count : nv;
    verts += nv; tris += idx / 3;
    details.push({
      name: o.name,
      verts: nv,
      tris: idx / 3,
      hasColor: !!g.attributes.color,
      colorItems: g.attributes.color ? g.attributes.color.itemSize : 0,
      hasIndex: !!g.index,
      matType: o.material?.type,
      matVertexColors: !!o.material?.vertexColors,
    });
  });

  ok('场景含网格', meshes > 0, `${meshes} 个 mesh`);
  ok('三角面数 > 0', tris > 0, `${tris.toLocaleString()} 三角面`);
  ok('顶点数 > 0', verts > 0, `${verts.toLocaleString()} 顶点`);

  const m = details[0];
  console.log('      首个 mesh:', JSON.stringify(m));
  ok('顶点色属性 COLOR_0 存在', m.hasColor, m.hasColor ? `itemSize=${m.colorItems}` : '缺失');
  ok('顶点色为 RGB/RGBA', m.colorItems === 3 || m.colorItems === 4, `itemSize=${m.colorItems}`);
  ok('材质启用 vertexColors', m.matVertexColors, `material=${m.matType}`);
  ok('使用索引几何', m.hasIndex);

  // 几何数值健全性：无 NaN，包围盒合理
  const g = gltf.scene.children.find((c) => c.isMesh)?.geometry
        || (() => { let r; gltf.scene.traverse((o) => { if (o.isMesh && !r) r = o.geometry; }); return r; })();
  const pos = g.attributes.position.array;
  const nan = [...pos].filter((v) => !Number.isFinite(v)).length;
  ok('顶点坐标无 NaN/Inf', nan === 0, `${nan} 个异常值`);

  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], pos[i + k]);
      max[k] = Math.max(max[k], pos[i + k]);
    }
  }
  const size = max.map((v, i) => +(v - min[i]).toFixed(4));
  ok('包围盒尺寸合理（头部量级 0.1~0.5）',
     size.every((s) => s > 0.05 && s < 0.6), `size=${JSON.stringify(size)}`);

  // 索引范围检查 —— 正是上游 *_textured_mesh.obj 出错的地方
  const idxArr = g.index.array;
  let maxIdx = 0;
  for (let i = 0; i < idxArr.length; i++) if (idxArr[i] > maxIdx) maxIdx = idxArr[i];
  ok('索引全部在顶点范围内', maxIdx < g.attributes.position.count,
     `maxIdx=${maxIdx} verts=${g.attributes.position.count}`);
  ok('所有顶点被引用（无死数据）', maxIdx + 1 === g.attributes.position.count,
     `${maxIdx + 1} / ${g.attributes.position.count}`);

  console.log(`\n${fails.length ? '失败 ' + fails.length + ' 项: ' + fails : 'three.js GLTFLoader 验证全部通过 ✅'}`);
  process.exit(fails.length ? 1 : 0);
}, (err) => {
  console.error('[FAIL] GLTFLoader 解析抛错:', err);
  process.exit(1);
});
