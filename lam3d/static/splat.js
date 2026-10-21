/* ═══════════════════════════════════════════════════════════════
   LSPL v1 高斯点云查看器（纯 ES module，只依赖 'three'）

   - parseLSPL(buffer)  解析「144 字节头 + 5 段 float32 payload」（契约第 4 节）
   - class SplatView    InstancedBufferGeometry（每实例一个四边形）
                        + 2D 协方差投影（契约 5.1 的 J · Rw · Σ · Rwᵀ · Jᵀ）
                        + 远到近深度排序（radix，稳定、无比较回调）
                        + exp(-0.5 dᵀΣ2⁻¹d) 片元 alpha + NormalBlending

   本文件不访问除自己创建的 canvas 之外的 DOM；
   不调用任何阻塞式弹窗 API，也不新开窗口或独立页面。
   ═══════════════════════════════════════════════════════════════ */
import * as THREE from 'three';

/* ── LSPL v1 头部布局（必须与 lam3d/lam3d/core/splat.py 逐字节一致） ── */
const LSPL_MAGIC = 'LSPL';
const LSPL_VERSION = 1;
const LSPL_HEADER_SIZE = 144;   // 字节
const LSPL_STRIDE = 56;         // 每个高斯的 payload 字节数 = 14 float32

const OFF_MAGIC = 0;            // char[4]
const OFF_VERSION = 4;          // uint32
const OFF_COUNT = 8;            // uint32
const OFF_FLAGS = 12;           // uint32  bit0 = has_rgb
const OFF_BBOX_MIN = 16;        // float32[3]
const OFF_BBOX_MAX = 28;        // float32[3]
const OFF_CENTER = 40;          // float32[3]
const OFF_REF_C2W = 52;         // float32[16] 行主序
const OFF_REF_INTR = 116;       // float32[4]  [fx, fy, cx, cy]
const OFF_REF_SIZE = 132;       // float32[2]  [w, h] 宽度在前

/* payload 各段在一个高斯内占的字节数（顺序不可变） */
const SEG_XYZ = 12;             // N x 3
const SEG_ROT = 16;             // N x 4  (w, x, y, z)
const SEG_SCALE = 12;           // N x 3
const SEG_OPACITY = 4;          // N x 1
const SEG_RGB = 12;             // N x 3

/* 契约 5.1 的 εI 正则项，单位 px² */
const CONIC_EPS = 0.3;

const _now = () =>
  (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

const clampNum = (v, a, b) => (v < a ? a : v > b ? b : v);

/**
 * 解析 LSPL v1 二进制。
 * magic / version / 总长度非法一律抛 Error（message 含 "LSPL"）。
 * 返回的 TypedArray 是 buffer 上的零拷贝视图（本机小端，与契约一致）。
 */
export function parseLSPL(buffer) {
  if (ArrayBuffer.isView(buffer)) {
    buffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  }
  if (!(buffer instanceof ArrayBuffer)) {
    throw new Error('LSPL: 需要 ArrayBuffer 输入');
  }
  if (buffer.byteLength < LSPL_HEADER_SIZE) {
    throw new Error(`LSPL: 数据不足头部长度（${buffer.byteLength} < ${LSPL_HEADER_SIZE}）`);
  }

  const dv = new DataView(buffer);
  const magic = String.fromCharCode(
    dv.getUint8(OFF_MAGIC), dv.getUint8(OFF_MAGIC + 1),
    dv.getUint8(OFF_MAGIC + 2), dv.getUint8(OFF_MAGIC + 3),
  );
  if (magic !== LSPL_MAGIC) {
    throw new Error(`LSPL: magic 非法（读到 "${magic}"，期望 "${LSPL_MAGIC}"）`);
  }
  const version = dv.getUint32(OFF_VERSION, true);
  if (version !== LSPL_VERSION) {
    throw new Error(`LSPL: 版本不支持（读到 ${version}，期望 ${LSPL_VERSION}）`);
  }
  const count = dv.getUint32(OFF_COUNT, true);
  const flags = dv.getUint32(OFF_FLAGS, true);

  const expected = LSPL_HEADER_SIZE + count * LSPL_STRIDE;
  if (buffer.byteLength !== expected) {
    throw new Error(
      `LSPL: 长度不匹配（count=${count} 期望 ${expected} 字节，实际 ${buffer.byteLength} 字节）`);
  }

  const f32 = (byteOffset, length) => new Float32Array(buffer, byteOffset, length);

  const oXyz = LSPL_HEADER_SIZE;
  const oRot = oXyz + count * SEG_XYZ;
  const oScale = oRot + count * SEG_ROT;
  const oOpacity = oScale + count * SEG_SCALE;
  const oRgb = oOpacity + count * SEG_OPACITY;

  return {
    count,
    flags,
    bboxMin: f32(OFF_BBOX_MIN, 3),
    bboxMax: f32(OFF_BBOX_MAX, 3),
    center: f32(OFF_CENTER, 3),
    refC2W: f32(OFF_REF_C2W, 16),
    refIntr: f32(OFF_REF_INTR, 4),
    refSize: f32(OFF_REF_SIZE, 2),
    xyz: f32(oXyz, count * 3),
    rot: f32(oRot, count * 4),
    scale: f32(oScale, count * 3),
    opacity: f32(oOpacity, count),
    rgb: f32(oRgb, count * 3),
  };
}

/* ── shader：四边形 + 逐片元 exp(-0.5 dᵀΣ2⁻¹d) ── */
const VERTEX_SHADER = /* glsl */`
precision highp float;
attribute vec2 iCenter;   // 屏幕像素坐标（左上原点，y 向下）
attribute vec2 iAxis1;    // 第一半轴向量（px）= 3*sqrt(λ1) * e1
attribute vec2 iAxis2;    // 第二半轴向量（px）= 3*sqrt(λ2) * e2
attribute vec3 iConic;    // Σ2⁻¹ 的 (a, b, c)  ↦ a*dx² + 2b*dx*dy + c*dy²
attribute vec3 iColor;
attribute float iOpacity;
uniform vec2 uResolution; // 绘制缓冲尺寸（px）
varying vec2 vOffset;
varying vec3 vColor;
varying float vOpacity;
varying vec3 vConic;

void main() {
  // position 为基础四边形（-1..1）^2
  vec2 off = position.x * iAxis1 + position.y * iAxis2;
  vec2 px = iCenter + off;
  vec2 ndc = vec2(
    px.x / uResolution.x * 2.0 - 1.0,
    1.0 - px.y / uResolution.y * 2.0
  );
  vOffset = off;
  vColor = iColor;
  vOpacity = iOpacity;
  vConic = iConic;
  gl_Position = vec4(ndc, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */`
precision highp float;
uniform float uCutoff;
varying vec2 vOffset;
varying vec3 vColor;
varying float vOpacity;
varying vec3 vConic;

void main() {
  float dx = vOffset.x;
  float dy = vOffset.y;
  float e = vConic.x * dx * dx + 2.0 * vConic.y * dx * dy + vConic.z * dy * dy;
  float alpha = vOpacity * exp(-0.5 * e);
  if (alpha < uCutoff) discard;
  gl_FragColor = vec4(vColor, alpha);
}
`;

/* 基础四边形（两个三角形，position 是 vec2） */
const QUAD_POS = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]);
const QUAD_INDEX = [0, 1, 2, 0, 2, 3];

export class SplatView {
  /**
   * @param {HTMLElement} host 宿主元素（canvas 内部创建）
   * @param {{background?: number, maxPixelRatio?: number}} [opts]
   */
  constructor(host, opts = {}) {
    if (!host || typeof host.appendChild !== 'function') {
      throw new Error('SplatView: 需要一个宿主元素');
    }
    this.host = host;
    this._maxPixelRatio = opts.maxPixelRatio || 2;
    this._backgroundHex = opts.background != null ? opts.background : 0x0f131c;

    /* ── canvas ── */
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'display:block;width:100%;height:100%';
    host.appendChild(this.canvas);

    /* ── renderer：不做 linear→sRGB 变换（rgb 已是显示色，契约 5.1） ── */
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.autoClear = true;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 1000);
    this.camera.position.set(0, 0, 2);

    /* ── material ── */
    this._cutoff = 0.0035;
    this._splatScale = 1.0;
    this._uniforms = {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uCutoff: { value: this._cutoff },
    };
    this.material = new THREE.ShaderMaterial({
      uniforms: this._uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: true,
      blending: THREE.NormalBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });

    /* ── 状态 ── */
    this._data = null;
    this._geom = null;
    this._geomCount = -1;
    this.mesh = null;
    this._pixelRatio = clampNum(globalThis.devicePixelRatio || 1, 1, this._maxPixelRatio);
    this._cssW = 1;
    this._cssH = 1;
    this._bufW = 1;
    this._bufH = 1;
    this._dirty = true;
    this._paused = false;
    this._disposed = false;
    this._raf = 0;
    this._lastFrame = 0;
    this._stats = { count: 0, visible: 0, prepareMs: 0, renderMs: 0, fps: 0 };
    this._camCache = new Float32Array(16).fill(NaN);
    this._projCache = new Float32Array(16).fill(NaN);
    this._center = new THREE.Vector3();
    this._radius = 0.5;
    this._front = new THREE.Vector3(0, 0, 1);
    this._bitsF = new Float32Array(1);
    this._bitsU = new Uint32Array(this._bitsF.buffer);

    this.setBackground(this._backgroundHex);
    this.resize();

    /* 自驱 rAF（app.js 若自己驱动渲染，可 setPaused(true) 后手动 render()） */
    this._loop = () => {
      if (this._disposed || this._paused) return;
      this._raf = requestAnimationFrame(this._loop);
      const w = this.host.clientWidth || 0;
      const h = this.host.clientHeight || 0;
      if (w > 0 && h > 0 && (w !== this._cssW || h !== this._cssH)) this.resize();
      this.render();
    };
    this._raf = requestAnimationFrame(this._loop);
  }

  /* ═══════════════════════ 数据装载 ═══════════════════════ */

  /**
   * 装载一帧高斯；内部按需重建 GPU 缓冲，不重建 renderer / scene。
   * 可连续调用 ≥ 60 次而不泄漏 WebGL 上下文。
   */
  setData(parsed) {
    if (!parsed || typeof parsed.count !== 'number') {
      throw new Error('SplatView.setData: 需要 parseLSPL() 的返回值');
    }
    const n = parsed.count | 0;
    const d = {
      count: n,
      xyz: parsed.xyz, rot: parsed.rot, scale: parsed.scale,
      opacity: parsed.opacity, rgb: parsed.rgb,
      bboxMin: parsed.bboxMin, bboxMax: parsed.bboxMax,
      center: parsed.center, refC2W: parsed.refC2W,
      refIntr: parsed.refIntr, refSize: parsed.refSize,
    };
    this._data = d;

    /* 顶点数变化才重建几何（同 N 时复用缓冲，setData 才能廉价连调） */
    if (this._geomCount !== n) {
      this._disposeGeometry();
      this._buildGeometry(n);
      this._allocScratch(n);
    }
    this._precomputeWorldCov();
    this._computeBounds();

    this._stats.count = n;
    /* 立即做一次投影+排序，使 stats().prepareMs 在 setData 之后即可读 */
    this._prepare();
    this._cacheCamera();
  }

  /** Σ_world = R·diag(s²)·Rᵀ（只用一次构造，预处理成 6 个数：xx,xy,xz,yy,yz,zz） */
  _precomputeWorldCov() {
    const d = this._data;
    const n = d.count;
    const rot = d.rot, scl = d.scale;
    const cov = this._cov;
    for (let i = 0, i4 = 0, i3 = 0, i6 = 0; i < n; i++, i4 += 4, i3 += 3, i6 += 6) {
      const w = rot[i4], x = rot[i4 + 1], y = rot[i4 + 2], z = rot[i4 + 3];
      const il = 1 / Math.sqrt(w * w + x * x + y * y + z * z || 1);
      const qw = w * il, qx = x * il, qy = y * il, qz = z * il;

      const xx = qx * qx, yy = qy * qy, zz = qz * qz;
      const xy = qx * qy, xz = qx * qz, yz = qy * qz;
      const wx = qw * qx, wy = qw * qy, wz = qw * qz;

      // R 列主序：列为局部轴在世界系的方向
      const r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
      const r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
      const r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);

      // A = R·diag(s) ⇒ Σ = A·Aᵀ
      const s0 = scl[i3], s1 = scl[i3 + 1], s2 = scl[i3 + 2];
      const a00 = r00 * s0, a01 = r01 * s1, a02 = r02 * s2;
      const a10 = r10 * s0, a11 = r11 * s1, a12 = r12 * s2;
      const a20 = r20 * s0, a21 = r21 * s1, a22 = r22 * s2;

      cov[i6] = a00 * a00 + a01 * a01 + a02 * a02;
      cov[i6 + 1] = a00 * a10 + a01 * a11 + a02 * a12;
      cov[i6 + 2] = a00 * a20 + a01 * a21 + a02 * a22;
      cov[i6 + 3] = a10 * a10 + a11 * a11 + a12 * a12;
      cov[i6 + 4] = a10 * a20 + a11 * a21 + a12 * a22;
      cov[i6 + 5] = a20 * a20 + a21 * a21 + a22 * a22;
    }
  }

  /** 质心（头部 center，缺省用 bbox 中点）与包围球半径、参考相机朝向 */
  _computeBounds() {
    const d = this._data;
    const c = d.center;
    if (c && (c[0] || c[1] || c[2])) this._center.set(c[0], c[1], c[2]);
    else this._center.set(
      (d.bboxMin[0] + d.bboxMax[0]) * 0.5,
      (d.bboxMin[1] + d.bboxMax[1]) * 0.5,
      (d.bboxMin[2] + d.bboxMax[2]) * 0.5,
    );
    let r2 = 0;
    for (let i = 0; i < 8; i++) {
      const x = (i & 1) ? d.bboxMax[0] : d.bboxMin[0];
      const y = (i & 2) ? d.bboxMax[1] : d.bboxMin[1];
      const z = (i & 4) ? d.bboxMax[2] : d.bboxMin[2];
      const dx = x - this._center.x, dy = y - this._center.y, dz = z - this._center.z;
      const rr = dx * dx + dy * dy + dz * dz;
      if (rr > r2) r2 = rr;
    }
    this._radius = Math.sqrt(r2) || 0.5;

    const rc = this.refCamera();
    const f = new THREE.Vector3(rc.position[0], rc.position[1], rc.position[2]).sub(this._center);
    this._front.copy(f.lengthSq() > 1e-12 ? f.normalize() : new THREE.Vector3(0, 0, 1));
  }

  _buildGeometry(n) {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(QUAD_POS, 2));
    g.setIndex(QUAD_INDEX);
    const mk = (name, size) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(n * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute(name, a);
      return a;
    };
    this._aCenter = mk('iCenter', 2);
    this._aAxis1 = mk('iAxis1', 2);
    this._aAxis2 = mk('iAxis2', 2);
    this._aConic = mk('iConic', 3);
    this._aColor = mk('iColor', 3);
    this._aOpacity = mk('iOpacity', 1);
    g.instanceCount = n;
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9); // 自排序，禁用剔除

    this._geom = g;
    this._geomCount = n;
    if (this.mesh) {
      this.mesh.geometry = g;
    } else {
      this.mesh = new THREE.Mesh(g, this.material);
      this.mesh.frustumCulled = false;
      this.scene.add(this.mesh);
    }
  }

  _disposeGeometry() {
    if (this._geom) {
      this._geom.dispose();
      this._geom = null;
    }
    this._geomCount = -1;
  }

  _allocScratch(n) {
    this._cov = new Float32Array(n * 6);
    this._sx = new Float32Array(n);
    this._sy = new Float32Array(n);
    this._a1x = new Float32Array(n);
    this._a1y = new Float32Array(n);
    this._a2x = new Float32Array(n);
    this._a2y = new Float32Array(n);
    this._ca = new Float32Array(n);
    this._cb = new Float32Array(n);
    this._cc = new Float32Array(n);
    this._depth = new Float32Array(n);
    this._order = new Int32Array(n);
    this._key = new Uint32Array(n);
    this._tmpIdx = new Int32Array(n);
    this._tmpKey = new Uint32Array(n);
    this._cnt = new Uint32Array(256);
  }

  /* ═══════════════════════ 投影 + 排序 ═══════════════════════ */

  /**
   * CPU 侧「投影 + 排序 + 属性写入」。
   * 契约 5.1：p = view·x，zv = -p.z，
   *   J = [[ fx/zv, 0, -fx·p.x/zv² ], [ 0, -fy/zv, fy·p.y/zv² ]]
   *   Σ2 = J·Rw·Σ·Rwᵀ·Jᵀ + 0.3·I
   * 半轴 = 3·sqrt(λ)，逐实例特征分解。
   */
  _prepare() {
    const t0 = _now();
    const d = this._data;
    if (!d || d.count === 0 || !this._geom) {
      this._stats.prepareMs = _now() - t0;
      this._stats.visible = 0;
      return;
    }
    const n = d.count;
    const cam = this.camera;
    cam.updateMatrixWorld();
    const inv = cam.matrixWorldInverse;
    inv.copy(cam.matrixWorld).invert();
    const V = inv.elements;
    const P = cam.projectionMatrix.elements;
    const W = this._bufW, H = this._bufH;

    const fx = P[0] * 0.5 * W;
    const fy = P[5] * 0.5 * H;
    const hw = W * 0.5, hh = H * 0.5;

    /* Rw：world→camera 的旋转（列主序矩阵的前 3x3） */
    const r00 = V[0], r01 = V[4], r02 = V[8];
    const r10 = V[1], r11 = V[5], r12 = V[9];
    const r20 = V[2], r21 = V[6], r22 = V[10];
    const tx = V[12], ty = V[13], tz = V[14];

    const xyz = d.xyz, cov = this._cov, opa = d.opacity, rgb = d.rgb;
    const sxA = this._sx, syA = this._sy;
    const a1xA = this._a1x, a1yA = this._a1y, a2xA = this._a2x, a2yA = this._a2y;
    const caA = this._ca, cbA = this._cb, ccA = this._cc;
    const dep = this._depth;
    const mul = this._splatScale;
    const invMul2 = 1 / (mul * mul);
    const cut = this._cutoff;

    let visible = 0;

    for (let i = 0, i3 = 0, i6 = 0; i < n; i++, i3 += 3, i6 += 6) {
      if (opa[i] < cut) {
        dep[i] = 0; sxA[i] = 0; syA[i] = 0;
        a1xA[i] = 0; a1yA[i] = 0; a2xA[i] = 0; a2yA[i] = 0;
        caA[i] = 0; cbA[i] = 0; ccA[i] = 0;
        continue;
      }
      const wx = xyz[i3], wy = xyz[i3 + 1], wz = xyz[i3 + 2];
      const px = r00 * wx + r01 * wy + r02 * wz + tx;
      const py = r10 * wx + r11 * wy + r12 * wz + ty;
      const pz = r20 * wx + r21 * wy + r22 * wz + tz;
      const zv = -pz;
      if (!(zv > 1e-4)) {                       // 在相机后方 / 退化
        dep[i] = 0; sxA[i] = 0; syA[i] = 0;
        a1xA[i] = 0; a1yA[i] = 0; a2xA[i] = 0; a2yA[i] = 0;
        caA[i] = 0; cbA[i] = 0; ccA[i] = 0;
        continue;
      }

      /* Σ_cam = Rw · Σ_world · Rwᵀ */
      const sxx = cov[i6], sxy = cov[i6 + 1], sxz = cov[i6 + 2];
      const syy = cov[i6 + 3], syz = cov[i6 + 4], szz = cov[i6 + 5];
      const a00 = r00 * sxx + r01 * sxy + r02 * sxz;
      const a01 = r00 * sxy + r01 * syy + r02 * syz;
      const a02 = r00 * sxz + r01 * syz + r02 * szz;
      const a10 = r10 * sxx + r11 * sxy + r12 * sxz;
      const a11 = r10 * sxy + r11 * syy + r12 * syz;
      const a12 = r10 * sxz + r11 * syz + r12 * szz;
      const a20 = r20 * sxx + r21 * sxy + r22 * sxz;
      const a21 = r20 * sxy + r21 * syy + r22 * syz;
      const a22 = r20 * sxz + r21 * syz + r22 * szz;
      const gxx = a00 * r00 + a01 * r01 + a02 * r02;
      const gxy = a00 * r10 + a01 * r11 + a02 * r12;
      const gxz = a00 * r20 + a01 * r21 + a02 * r22;
      const gyy = a10 * r10 + a11 * r11 + a12 * r12;
      const gyz = a10 * r20 + a11 * r21 + a12 * r22;
      const gzz = a20 * r20 + a21 * r21 + a22 * r22;

      /* J 与 Σ2 = J·Σ_cam·Jᵀ + εI */
      const izv = 1 / zv;
      const z2 = zv * zv;
      const j00 = fx * izv;
      const j02 = -fx * px / z2;      // 契约 5.1 逐字实现
      const j11 = -fy * izv;
      const j12 = fy * py / z2;

      const m00 = j00 * j00 * gxx + 2 * j00 * j02 * gxz + j02 * j02 * gzz + CONIC_EPS;
      const m01 = j00 * j11 * gxy + j00 * j12 * gxz + j02 * j11 * gyz + j02 * j12 * gzz;
      const m11 = j11 * j11 * gyy + 2 * j11 * j12 * gyz + j12 * j12 * gzz + CONIC_EPS;

      const sxc = j00 * px + hw;
      const syc = j11 * py + hh;

      /* Σ2⁻¹ */
      let det = m00 * m11 - m01 * m01;
      if (!(det > 1e-12)) det = 1e-12;
      const idet = 1 / det;

      /* 特征分解 → 四边形半轴 = 3·sqrt(λ) */
      const tr = m00 + m11;
      const df = m00 - m11;
      let disc = df * df + 4 * m01 * m01;
      disc = disc > 0 ? Math.sqrt(disc) : 0;
      const l1 = Math.max(0.5 * (tr + disc), 0);
      const l2 = Math.max(0.5 * (tr - disc), 0);

      let e1x, e1y;
      if (Math.abs(m01) > 1e-12) {
        e1x = m01; e1y = l1 - m00;
        const il = 1 / Math.sqrt(e1x * e1x + e1y * e1y || 1);
        e1x *= il; e1y *= il;
      } else if (m00 >= m11) { e1x = 1; e1y = 0; }
      else { e1x = 0; e1y = 1; }
      const e2x = -e1y, e2y = e1x;

      const h1 = 3 * Math.sqrt(l1) * mul;
      const h2 = 3 * Math.sqrt(l2) * mul;

      /* 视口外整体剔除（半径 = 最大半轴） */
      const rr = h1 > h2 ? h1 : h2;
      if (sxc + rr < 0 || sxc - rr > W || syc + rr < 0 || syc - rr > H) {
        dep[i] = 0; sxA[i] = 0; syA[i] = 0;
        a1xA[i] = 0; a1yA[i] = 0; a2xA[i] = 0; a2yA[i] = 0;
        caA[i] = 0; cbA[i] = 0; ccA[i] = 0;
        continue;
      }

      dep[i] = zv;
      sxA[i] = sxc;
      syA[i] = syc;
      a1xA[i] = e1x * h1; a1yA[i] = e1y * h1;
      a2xA[i] = e2x * h2; a2yA[i] = e2y * h2;
      caA[i] = m11 * idet * invMul2;        // Σ2⁻¹ 缩放：全局尺度乘子等价于屏幕空间自相似缩放
      cbA[i] = -m01 * idet * invMul2;
      ccA[i] = m00 * idet * invMul2;
      visible++;
    }

    /* ── 远→近深度排序 ──
       key = 0xFFFFFFFF - sortableUint32(zv)：正浮点按 IEEE 位序单调，
       故 key 升序 == zv 降序 == 远到近。4 pass × 8bit 稳定 LSD radix。 */
    const key = this._key, order = this._order;
    const bitsF = this._bitsF, bitsU = this._bitsU;
    for (let i = 0; i < n; i++) {
      bitsF[0] = dep[i];
      let u = bitsU[0];
      u = (u & 0x80000000) ? ~u : (u | 0x80000000);
      key[i] = (0xFFFFFFFF - (u >>> 0)) >>> 0;
      order[i] = i;
    }
    this._radix(order, key, n);

    /* ── 属性写入（按排序后的顺序，绘制顺序 == 实例顺序） ── */
    const ic = this._aCenter.array, ia1 = this._aAxis1.array, ia2 = this._aAxis2.array;
    const icon = this._aConic.array, icol = this._aColor.array, iop = this._aOpacity.array;
    for (let k = 0; k < n; k++) {
      const s = order[k];
      const k2 = k * 2, k3 = k * 3, s3 = s * 3;
      ic[k2] = sxA[s]; ic[k2 + 1] = syA[s];
      ia1[k2] = a1xA[s]; ia1[k2 + 1] = a1yA[s];
      ia2[k2] = a2xA[s]; ia2[k2 + 1] = a2yA[s];
      icon[k3] = caA[s]; icon[k3 + 1] = cbA[s]; icon[k3 + 2] = ccA[s];
      icol[k3] = rgb[s3]; icol[k3 + 1] = rgb[s3 + 1]; icol[k3 + 2] = rgb[s3 + 2];
      iop[k] = opa[s];
    }
    this._aCenter.needsUpdate = true;
    this._aAxis1.needsUpdate = true;
    this._aAxis2.needsUpdate = true;
    this._aConic.needsUpdate = true;
    this._aColor.needsUpdate = true;
    this._aOpacity.needsUpdate = true;

    this._stats.visible = visible;
    this._stats.prepareMs = _now() - t0;
  }

  /** 稳定 LSD radix（uint32 升序）；idx 被就地重排，key 同步重排 */
  _radix(idx, key, n) {
    const tmpIdx = this._tmpIdx, tmpKey = this._tmpKey, cnt = this._cnt;
    for (let shift = 0; shift < 32; shift += 8) {
      cnt.fill(0);
      for (let i = 0; i < n; i++) cnt[(key[i] >>> shift) & 255]++;
      let sum = 0;
      for (let b = 0; b < 256; b++) { const c = cnt[b]; cnt[b] = sum; sum += c; }
      for (let i = 0; i < n; i++) {
        const b = (key[i] >>> shift) & 255;
        const p = cnt[b]++;
        tmpIdx[p] = idx[i];
        tmpKey[p] = key[i];
      }
      idx.set(tmpIdx.subarray(0, n));
      key.set(tmpKey.subarray(0, n));
    }
  }

  _cacheCamera() {
    this._camCache.set(this.camera.matrixWorld.elements);
    this._projCache.set(this.camera.projectionMatrix.elements);
    this._dirty = false;
  }

  /* ═══════════════════════ 渲染 ═══════════════════════ */

  render() {
    if (this._disposed) return;
    const t0 = _now();
    const cam = this.camera;
    cam.updateMatrixWorld();

    if (this._data && this._geom) {
      const mw = cam.matrixWorld.elements;
      const pj = cam.projectionMatrix.elements;
      let changed = this._dirty;
      if (!changed) {
        for (let i = 0; i < 16; i++) if (mw[i] !== this._camCache[i]) { changed = true; break; }
        if (!changed) for (let i = 0; i < 16; i++) if (pj[i] !== this._projCache[i]) { changed = true; break; }
      }
      if (changed) {
        this._prepare();
        this._cacheCamera();
      }
    }

    this.renderer.render(this.scene, this.camera);
    const t1 = _now();
    this._stats.renderMs = t1 - t0;
    if (this._lastFrame) {
      const dt = t1 - this._lastFrame;
      const fps = dt > 0 ? 1000 / dt : 0;
      this._stats.fps = this._stats.fps ? this._stats.fps * 0.9 + fps * 0.1 : fps;
    }
    this._lastFrame = t1;
  }

  resize() {
    const host = this.host;
    const w = Math.max(1, Math.round(host.clientWidth || 1));
    const h = Math.max(1, Math.round(host.clientHeight || 1));
    this._cssW = w;
    this._cssH = h;
    this.renderer.setPixelRatio(this._pixelRatio);
    this.renderer.setSize(w, h, false);   // 不覆盖 canvas 的 width/height:100% 样式
    this._bufW = Math.max(1, Math.floor(w * this._pixelRatio));
    this._bufH = Math.max(1, Math.floor(h * this._pixelRatio));
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._uniforms.uResolution.value.set(this._bufW, this._bufH);
    this._dirty = true;
  }

  /* ═══════════════════════ 契约 API ═══════════════════════ */

  /** 点云质心（世界系） */
  center() { return this._center.clone(); }

  /** 包围球半径（取景用） */
  frameRadius() { return this._radius; }

  setBackground(hex) {
    // 显式声明源为 LinearSRGB，避免 ColorManagement 把「显示色」再线性化一次
    const c = new THREE.Color();
    c.setHex(hex >>> 0, THREE.LinearSRGBColorSpace);
    this.scene.background = c;
    this._backgroundHex = hex;
  }

  /** clamp 到 [1, 2] */
  setPixelRatio(r) {
    const v = clampNum(Number(r) || 1, 1, 2);
    if (v !== this._pixelRatio) {
      this._pixelRatio = v;
      this.resize();
    }
  }

  /** 全局尺度乘子，默认 1.0，clamp [0.25, 3] */
  setSplatScale(k) {
    const v = clampNum(Number(k) || 0, 0.25, 3);
    if (v !== this._splatScale) {
      this._splatScale = v;
      this._dirty = true;
    }
  }

  /** 片元 alpha 丢弃阈值，默认 0.0035 */
  setOpacityCutoff(a) {
    const v = clampNum(Number(a) || 0, 0, 1);
    this._cutoff = v;
    this._uniforms.uCutoff.value = v;
    this._dirty = true;
  }

  setCameraPose({ position, quaternion, fovY, near, far } = {}) {
    if (position) this.camera.position.set(position[0], position[1], position[2]);
    if (quaternion) this.camera.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    if (fovY != null) this.camera.fov = fovY;
    if (near != null) this.camera.near = near;
    if (far != null) this.camera.far = far;
    this.camera.updateProjectionMatrix();
    this._dirty = true;
  }

  /**
   * 由 ref_c2w / ref_intr / ref_size 推得参考相机（契约 5.2）。
   * ref_c2w 行主序、3DGS 约定（列 = right, down, forward）；
   * Three.js 约定列 = (right, up, backward)，backward = -forward。
   */
  refCamera() {
    const d = this._data;
    const pos = [0, 0, 2], quat = [0, 0, 0, 1];
    let fovY = 38;
    if (!d) return { position: pos, quaternion: quat, fovY };
    const m = d.refC2W;
    const xAxis = new THREE.Vector3(m[0], m[4], m[8]);
    const yAxis = new THREE.Vector3(-m[1], -m[5], -m[9]);
    const zAxis = new THREE.Vector3(-m[2], -m[6], -m[10]);
    pos[0] = m[3]; pos[1] = m[7]; pos[2] = m[11];
    if (xAxis.lengthSq() > 1e-12 && yAxis.lengthSq() > 1e-12 && zAxis.lengthSq() > 1e-12) {
      xAxis.normalize(); yAxis.normalize(); zAxis.normalize();
      const rot = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
      const q = new THREE.Quaternion().setFromRotationMatrix(rot);
      quat[0] = q.x; quat[1] = q.y; quat[2] = q.z; quat[3] = q.w;
    }
    const size = d.refSize, intr = d.refIntr;
    if (size && intr && intr[1] > 0) {
      fovY = 2 * Math.atan(size[1] / (2 * intr[1])) * 180 / Math.PI;
    }
    return { position: pos, quaternion: quat, fovY };
  }

  /** 绕 center() 环绕取景（front/left/right/three/reset） */
  orbitPreset(name) {
    const c = this.center();
    const r = Math.max(this.frameRadius(), 1e-6);
    const dist = r / Math.tan(19 * Math.PI / 180) * 1.35;
    const front = this._front.clone();
    let dir;
    if (name === 'left' || name === 'right') {
      const ang = (name === 'left' ? -1 : 1) * Math.PI / 3;   // 绕世界 +Y 轴 ∓60°
      dir = front.applyAxisAngle(new THREE.Vector3(0, 1, 0), ang);
    } else if (name === 'three') {
      // 前右 3/4：+40° yaw、+12° pitch（球坐标直接给，避免 roll 干扰）
      const yaw0 = Math.atan2(front.x, front.z);
      const pitch0 = Math.asin(clampNum(front.y, -1, 1));
      const yaw = yaw0 + 40 * Math.PI / 180;
      const pitch = clampNum(pitch0 + 12 * Math.PI / 180, -1.2, 1.2);
      dir = new THREE.Vector3(
        Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw));
    } else {
      dir = front;                                            // front / reset
    }
    dir.normalize();
    this.camera.position.copy(c).addScaledVector(dir, dist);
    this.camera.lookAt(c);
    this.camera.near = Math.max(dist / 100, 1e-5);
    this.camera.far = dist * 50;
    this.camera.updateProjectionMatrix();
    this._dirty = true;
  }

  /** 暂停自驱 rAF（此时由调用方手动 render()） */
  setPaused(b) {
    const p = !!b;
    if (p === this._paused) return;
    this._paused = p;
    if (p) {
      if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    } else if (!this._disposed && !this._raf) {
      this._raf = requestAnimationFrame(this._loop);
    }
  }

  stats() { return Object.assign({}, this._stats); }

  dispose() {
    this._disposed = true;
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    this._disposeGeometry();
    this.material.dispose();
    this.renderer.dispose();
    if (this.canvas && this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
    this._data = null;
    this._stats.count = 0;
  }
}
