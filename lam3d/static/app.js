/* ═══════════════════════════════════════════════════════════════
   LAM 单图生成 3D 数字人 · 前端
   - 左栏：选图 → 生成 → 重建结果（face.png）
   - 右栏：实时进度 → SplatView 高斯点云 360° 预览 + ARKit 52 参数表情
   ═══════════════════════════════════════════════════════════════ */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SplatView, parseLSPL } from './splat.js';

const $ = (id) => document.getElementById(id);
const API = '';

const state = {
  file: null,
  sample: null,
  jobId: null,
  polling: null,
  result: null,
};

/* ────────────────────────────── 启动 ────────────────────────────── */

async function init() {
  bindDropZone();
  bindButtons();
  initArkitPanel();
  initCovPanel();
  try {
    const cfg = await fetchJSON(`${API}/api/config`);
    renderSamples(cfg.samples);
    renderFacts(cfg);
  } catch (e) {
    setEngineHint(`无法连接后端：${e.message}`, true);
  }
  watchEngine();
}

function renderFacts(cfg) {
  const facts = [
    cfg.gpu ? `GPU · ${cfg.gpu}` : 'CPU 模式',
    '单图 3DGS 重建',
    '高斯点云 360° 预览',
    'ARKit 52 参数表情',
  ];
  $('facts').innerHTML = facts.map((f) => `<li>${esc(f)}</li>`).join('');
}

/* ────────────────────────────── 引擎状态 ────────────────────────────── */
async function watchEngine() {
  const tick = async () => {
    try {
      const s = await fetchJSON(`${API}/api/status`);
      if (s.error) {
        setEngineHint(`推理引擎载入失败：${s.error}`, true);
        return;
      }
      if (s.ready) {
        setEngineHint('推理引擎已就绪（模型常驻显存）');
        return;
      }
      setEngineHint(s.queue > 0 ? `引擎预热中…（队列 ${s.queue}）` : '引擎预热中，首次生成会稍慢…');
      setTimeout(tick, 2500);
    } catch {
      setTimeout(tick, 4000);
    }
  };
  tick();
}

function setEngineHint(text, isErr = false) {
  const el = $('engineHint');
  el.textContent = text;
  el.classList.toggle('is-err', isErr);
}

/* ────────────────────────────── 选图 ────────────────────────────── */
function bindDropZone() {
  const drop = $('drop');
  const file = $('file');

  drop.addEventListener('click', () => file.click());
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); }
  });
  file.addEventListener('change', () => file.files[0] && pickFile(file.files[0]));

  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));

  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    if (!/^image\//.test(f.type)) { showError('拖入的不是图片文件'); return; }
    pickFile(f);
  });

  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) pickFile(item.getAsFile());
  });
}

function pickFile(f) {
  state.file = f;
  state.sample = null;
  document.querySelectorAll('.samples img').forEach((i) => i.classList.remove('is-sel'));
  const url = URL.createObjectURL(f);
  const img = $('preview');
  img.src = url;
  img.hidden = false;
  $('dropHint').hidden = true;
  refreshGenerate();
}

function pickSample(name, el) {
  state.sample = name;
  state.file = null;
  document.querySelectorAll('.samples img').forEach((i) => i.classList.remove('is-sel'));
  el?.classList.add('is-sel');
  const img = $('preview');
  img.src = `${API}/api/samples/${encodeURIComponent(name)}`;
  img.hidden = false;
  $('dropHint').hidden = true;
  refreshGenerate();
}

function renderSamples(samples) {
  const box = $('samples');
  box.innerHTML = '';
  (samples || []).slice(0, 12).forEach((s) => {
    const img = document.createElement('img');
    img.src = `${API}/api/samples/${encodeURIComponent(s.name)}`;
    img.alt = s.name;
    img.title = s.name;
    img.loading = 'lazy';
    img.onclick = (e) => { e.stopPropagation(); pickSample(s.name, img); };
    box.appendChild(img);
  });
}

/* ────────────────────────────── 生成 ────────────────────────────── */
function bindButtons() {
  $('generate').onclick = startGenerate;
}

function refreshGenerate() {
  $('generate').disabled = !(state.file || state.sample) || !!state.jobId;
}

async function startGenerate() {
  if (!state.file && !state.sample) return;
  clearError();
  $('result').hidden = true;
  $('empty').hidden = true;
  $('faceCard').hidden = true;
  $('progress').hidden = false;
  setProgress('排队中', 0, '正在提交任务…');
  $('generate').disabled = true;

  const fd = new FormData();
  if (state.file) fd.append('image', state.file);
  else fd.append('sample', state.sample);

  try {
    const job = await fetchJSON(`${API}/api/generate`, { method: 'POST', body: fd });
    state.jobId = job.id;
    pollJob(job.id);
  } catch (e) {
    $('progress').hidden = true;
    $('empty').hidden = false;
    showError(e.message);
    state.jobId = null;
    refreshGenerate();
  }
}

function pollJob(id) {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    let job;
    try {
      job = await fetchJSON(`${API}/api/jobs/${id}`);
    } catch {
      return; // 网络抖动，下一轮再试
    }
    setProgress(stageLabel(job.stage), job.progress, job.detail, job);

    if (job.state === 'succeeded') {
      clearInterval(state.polling);
      state.polling = null;
      state.jobId = null;
      setProgress('完成', 100, '');
      setTimeout(() => { $('progress').hidden = true; }, 900);
      showResult(job);
      refreshGenerate();
    } else if (job.state === 'failed' || job.state === 'canceled') {
      clearInterval(state.polling);
      state.polling = null;
      state.jobId = null;
      $('progress').hidden = true;
      $('empty').hidden = false;
      showError(job.error || '任务已取消');
      refreshGenerate();
    }
  }, 700);
}

const STAGE_LABEL = {
  queued: '排队中',
  prepare: '准备模型',
  tracking: '人脸跟踪',
  reference: '参考图处理',
  infer: 'LAM 前馈推理',
  render: '渲染重建帧',
  mesh: '导出 3D 模型',
  done: '完成',
  canceled: '已取消',
};
const stageLabel = (s) => STAGE_LABEL[s] || s;

function setProgress(stage, pct, detail, job) {
  $('stageText').textContent = stage;
  $('pctText').textContent = `${Math.round(pct)}%`;
  $('bar').style.width = `${Math.min(100, pct)}%`;
  let d = detail || '';
  if (job?.elapsed) d += `${d ? ' · ' : ''}已用时 ${job.elapsed}s`;
  $('detailText').textContent = d;
}

/* ────────────────────────────── 结果 ────────────────────────────── */
function showResult(job) {
  const r = job.result;
  state.result = r;
  resetArkitForNewJob(job);
  $('empty').hidden = true;
  $('result').hidden = false;
  $('progress').hidden = true;

  const fileURL = (key, dl = false) =>
    `${API}/api/jobs/${job.id}/file/${key}${dl ? '?download=true' : ''}`;

  // 重建结果：LAM 单帧渲染出来的人物脸图
  const face = $('faceCard');
  const faceImg = $('faceImg');
  if (r && r.face !== false) {
    faceImg.src = fileURL('face');
    face.hidden = false;
  } else {
    face.hidden = true;
  }

  // 导出模型（默认折叠）
  const map = { dlGlb: ['glb', r.glb], dlObj: ['obj', r.obj], dlPly: ['gs_ply', r.gs_ply] };
  for (const [id, [key, rel]] of Object.entries(map)) {
    const a = $(id);
    if (!a) continue;
    a.classList.toggle('is-off', !rel);
    if (rel) a.href = fileURL(key, true);
  }

  // 统计
  const s = r.stats;
  const t = s.timings || {};
  $('stats').innerHTML = [
    ['高斯点数', s.gaussians?.toLocaleString()],
    ['网格面数', s.faces?.toLocaleString()],
    ['总耗时', `${s.seconds}s`],
    ['人脸跟踪', t.tracking != null ? `${t.tracking}s` : '-'],
    ['推理+渲染', t.infer_render != null ? `${t.infer_render}s` : '-'],
    ['模型预热', s.model_load_seconds != null ? `${s.model_load_seconds}s` : '-'],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v ?? '-'))}</dd>`).join('');

  /* 渲染器首次初始化需要元素可见（clientWidth/Height 才非零） */
  requestAnimationFrame(() => {
    if ($('result').hidden) return;
    initViewer();
    loadInitialSplat(job.id);
  });
}

/* ═══════════════════════════════════════════════════════════════
   高斯点云查看器（SplatView 宿主）
   - OrbitControls 交互参数与上一轮完全一致（rotateSpeed 0.85 /
     minDistance 0.2 / maxDistance 4.0 / dampingFactor 0.07 / damping on）
   - SplatView 的自驱 rAF 关闭，由本文件的单一渲染循环驱动
   ═══════════════════════════════════════════════════════════════ */
const viewer = {
  splat: null, controls: null, ro: null,
  raf: 0, spinning: false, points: false, hq: true, ready: false,
};

function initViewer() {
  const host = $('viewer');
  if (!host) return null;
  if (viewer.splat) { viewer.splat.resize(); return viewer.splat; }

  const splat = new SplatView(host, { background: 0x0f131c, maxPixelRatio: 2 });
  splat.setPaused(true);                       // 渲染由下面的统一循环驱动
  viewer.splat = splat;

  const controls = new OrbitControls(splat.camera, splat.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.85;
  controls.minDistance = 0.2;
  controls.maxDistance = 4.0;
  viewer.controls = controls;

  applySplatQuality();

  viewer.ro = new ResizeObserver(() => { splat.resize(); });
  viewer.ro.observe(host);

  bindViewerTools();

  const loop = () => {
    viewer.raf = requestAnimationFrame(loop);
    if (viewer.spinning && viewer.ready) spinStep();
    controls.update();
    if (arkitState.open) updateArkitOrbit();
    splat.render();
  };
  loop();
  viewer.ready = true;
  return splat;
}

/**
 * 设备像素比：恒为 clamp(devicePixelRatio, 1, 2)。
 * 契约 §5.1 / 验收 A14 要求 `canvas.width === cssW * dpr`，因此这里绝不能
 * 为了「高清」把倍率抬到 dpr 之上（那会让 canvas.width ≠ cssW*dpr）。
 */
function targetPixelRatio() {
  return Math.min(2, Math.max(1, globalThis.devicePixelRatio || 1));
}

/**
 * 清晰度切换：
 * - pixelRatio 恒定 = clamp(dpr, 1, 2) ⇒ canvas.width === cssW * dpr 恒成立（A14 前半）
 * - 「高清」= 把 .viewer-wrap 的 CSS 尺寸抬到 ≥640×640，
 *   于是绘制缓冲 = css × dpr ≥ 640×640（A14 后半）。
 */
function applySplatQuality() {
  const wrap = document.querySelector('.viewer-wrap');
  if (wrap) wrap.classList.toggle('is-hq', viewer.hq);
  if (!viewer.splat) return;
  viewer.splat.setPixelRatio(targetPixelRatio());
  requestAnimationFrame(() => { if (viewer.splat) viewer.splat.resize(); });
}

/**
 * #wire：从「线框」改为「仅显示点中心」的可视化开关（契约 6.3 允许）。
 * 点模式 = 关闭 alpha 截断 + 全局尺度收到最小 0.25，于是每个高斯只留下中心的小点。
 */
function applyPointMode() {
  const splat = viewer.splat;
  if (!splat) return;
  if (viewer.points) { splat.setOpacityCutoff(0); splat.setSplatScale(0.25); }
  else { splat.setOpacityCutoff(0.0035); splat.setSplatScale(1); }
}

function bindViewerTools() {
  document.querySelectorAll('.tool[data-view]').forEach((b) => {
    b.onclick = () => setView(b.dataset.view);
  });
  const spin = $('spin');
  if (spin) spin.onclick = (e) => {
    viewer.spinning = !viewer.spinning;
    e.currentTarget.classList.toggle('is-on', viewer.spinning);
  };
  const wire = $('wire');
  if (wire) wire.onclick = (e) => {
    viewer.points = !viewer.points;
    e.currentTarget.classList.toggle('is-on', viewer.points);
    applyPointMode();
  };
  const q = $('splatQuality');
  if (q) {
    q.value = viewer.hq ? 'hq' : 'standard';
    q.onchange = () => { viewer.hq = q.value === 'hq'; applySplatQuality(); };
  }
}

/** 4 个绝对视角预设 → SplatView.orbitPreset（不改 OrbitControls 的 min/max） */
function setView(which) {
  const splat = viewer.splat;
  if (!splat || !viewer.ready) return;
  splat.orbitPreset(which === 'side' ? 'right' : (which || 'front'));
  if (viewer.controls) {
    viewer.controls.target.copy(splat.center());
    viewer.controls.update();
  }
  arkitState.orbitYaw = 0;
  arkitState.orbitPitch = 0;
  arkitState.orbitZoom = 1;
  updateArkitOrbit();
}

/** 自转：绕 center() 的世界 +Y 轴缓慢环绕 */
function spinStep() {
  const splat = viewer.splat;
  if (!splat) return;
  const c = splat.center();
  const v = splat.camera.position.clone().sub(c);
  const a = 0.006;
  const ca = Math.cos(a), sa = Math.sin(a);
  const nx = v.x * ca + v.z * sa;
  const nz = -v.x * sa + v.z * ca;
  splat.camera.position.set(c.x + nx, splat.camera.position.y, c.z + nz);
  splat.camera.lookAt(c);
}

/* ────────────────────────────── 初始点云 ────────────────────────────── */

/** 生成成功后先拉一帧「ARKit 全 0」的点云铺上画布 */
async function loadInitialSplat(jobId) {
  const splat = viewer.splat;
  const badge = $('viewerBadge');
  if (!splat) return;
  badge.textContent = '点云加载中…';
  badge.className = 'viewer__badge';
  try {
    const data = await postArkitSplat(jobId, ZERO_ARKIT());
    if (!data || !data.buffer) throw new Error('空响应');
    const parsed = parseLSPL(data.buffer);
    splat.setData(parsed);
    splat.orbitPreset('front');
    viewer.controls?.target.copy(splat.center());
    viewer.controls?.update();
    arkitState.gaussians = parsed.count;
    const g = $('arkitGaussians');
    if (g) g.textContent = `高斯 ${parsed.count.toLocaleString()}`;
    const lat = $('arkitLatency');
    if (lat) lat.textContent = `延迟 ${data.ms || 0} ms`;
    badge.textContent = `已加载 · ${parsed.count.toLocaleString()} 高斯 · 可 360° 拖拽`;
    badge.className = 'viewer__badge is-ok';
  } catch (e) {
    badge.textContent = `点云加载失败：${e.message}`;
    badge.className = 'viewer__badge is-err';
  }
}

/* ═══════════════════════════════════════════════════════════════
   ARKit 实时表情面板（改造：POST arkit_splat → parseLSPL → setData）
   - 数据源 GET  /api/arkit/meta
   - 会话就绪 GET  /api/jobs/{id}/arkit_session
   - 点云     POST /api/jobs/{id}/arkit_splat -> LSPL v1
   ═══════════════════════════════════════════════════════════════ */
const arkitState = {
  open: false,
  values: new Float32Array(52).fill(0),
  orbitYaw: 0, orbitPitch: 0, orbitZoom: 1,
  busy: false, lastMs: 0, raf: 0, pending: false,
  jobId: null, meta: null, gaussians: null, session: null,
};
let arkitSliderRefs = [];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ZERO_ARKIT = () => new Array(arkitState.values.length).fill(0);

/* 拉一次 meta 并构建面板；失败则表情按钮保持 inert，其它一切不变 */
async function initArkitPanel() {
  try {
    arkitState.meta = await fetchJSON(`${API}/api/arkit/meta`);
  } catch {
    arkitState.meta = null;
    bindArkitToggle();                           // 开关仍要绑定，点击时给出提示
    return;
  }
  const n = (arkitState.meta.names || []).length;
  if (n && n !== arkitState.values.length) arkitState.values = new Float32Array(n).fill(0);
  buildArkitPresets();
  buildArkitSliders();
  bindArkitOrbit();
  bindArkitToggle();
  updateArkitOrbit();
}

const prettyArkitPreset = (name) => ({ neutral: '中性' }[name] || name);

function buildArkitPresets() {
  const host = $('arkitPresets');
  if (!host) return;
  host.innerHTML = '';
  const presets = arkitState.meta.presets || {};
  Object.keys(presets).forEach((name) => {
    if (name === 'neutral') return;            // 由下面的「重置」按钮承担
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'arkit-preset';
    b.textContent = prettyArkitPreset(name);
    b.title = name;
    b.onclick = () => applyArkitPreset(presets[name]);
    host.appendChild(b);
  });
  const r = document.createElement('button');
  r.type = 'button';
  r.className = 'arkit-preset arkit-preset--reset';
  r.textContent = '重置';
  r.onclick = () => applyArkitPreset({});
  host.appendChild(r);
}

const ARKIT_GROUPS = [
  { label: '眉毛', test: (n) => n.startsWith('brow') },
  { label: '眼睛', test: (n) => n.startsWith('eye') },
  { label: '鼻 / 颊', test: (n) => n.startsWith('nose') || n.startsWith('cheek') },
  { label: '嘴', test: (n) => n.startsWith('mouth') },
  { label: '下颌', test: (n) => n.startsWith('jaw') },
];

function buildArkitSliders() {
  const host = $('arkitSliders');
  if (!host) return;
  host.innerHTML = '';
  arkitSliderRefs = [];
  const names = arkitState.meta.names || [];
  const used = new Set();
  ARKIT_GROUPS.forEach((g, gi) => {
    const idxs = [];
    names.forEach((n, i) => { if (g.test(n)) { idxs.push(i); used.add(i); } });
    if (!idxs.length) return;
    const det = document.createElement('details');
    det.className = 'arkit-group';
    if (gi === 0) det.open = true;
    const sum = document.createElement('summary');
    sum.textContent = `${g.label}（${idxs.length}）`;
    det.appendChild(sum);
    idxs.forEach((i) => det.appendChild(makeArkitSlider(names[i], i)));
    host.appendChild(det);
  });
  const rest = [];                              // 未归组的通道（如 tongueOut）兜底
  names.forEach((n, i) => { if (!used.has(i)) rest.push(i); });
  if (rest.length) {
    const det = document.createElement('details');
    det.className = 'arkit-group';
    det.open = true;
    const sum = document.createElement('summary');
    sum.textContent = `其他（${rest.length}）`;
    det.appendChild(sum);
    rest.forEach((i) => det.appendChild(makeArkitSlider(names[i], i)));
    host.appendChild(det);
  }
}

function makeArkitSlider(name, i) {
  const row = document.createElement('label');
  row.className = 'arkit-slider';
  const nm = document.createElement('span');
  nm.className = 'arkit-slider__name';
  nm.textContent = name;
  nm.title = name;
  const range = document.createElement('input');
  range.type = 'range';
  range.min = '0';
  range.max = '1';
  range.step = '0.01';
  range.value = '0';
  const val = document.createElement('span');
  val.className = 'arkit-slider__val';
  val.textContent = '0.00';
  range.addEventListener('input', () => {
    arkitState.values[i] = parseFloat(range.value) || 0;
    val.textContent = arkitState.values[i].toFixed(2);
    scheduleArkitRender();
  });
  row.append(nm, range, val);
  arkitSliderRefs[i] = { range, val };
  return row;
}

function syncArkitSliders() {
  for (let i = 0; i < arkitState.values.length; i++) {
    const ref = arkitSliderRefs[i];
    if (!ref) continue;
    const v = arkitState.values[i];
    ref.range.value = String(v);
    ref.val.textContent = v.toFixed(2);
  }
}

function applyArkitPreset(preset) {
  const names = (arkitState.meta && arkitState.meta.names) || [];
  arkitState.values.fill(0);
  for (const [k, v] of Object.entries(preset || {})) {
    const idx = names.indexOf(k);
    if (idx >= 0) arkitState.values[idx] = clamp(Number(v) || 0, 0, 1);
  }
  syncArkitSliders();
  scheduleArkitRender();
}

/* ── 面板内视角按钮：驱动「客户端」相机绕 center() 环绕（契约 6.2） ── */
function bindArkitOrbit() {
  document.querySelectorAll('#arkitPanel [data-orbit]').forEach((b) => {
    b.onclick = () => {
      const a = b.dataset.orbit;
      if (a === 'yawL') arkitState.orbitYaw -= 15 * Math.PI / 180;
      else if (a === 'yawR') arkitState.orbitYaw += 15 * Math.PI / 180;
      else if (a === 'pitchU') arkitState.orbitPitch = clamp(arkitState.orbitPitch + 10 * Math.PI / 180, -1.2, 1.2);
      else if (a === 'pitchD') arkitState.orbitPitch = clamp(arkitState.orbitPitch - 10 * Math.PI / 180, -1.2, 1.2);
      else if (a === 'zoomIn') arkitState.orbitZoom = clamp(arkitState.orbitZoom * 1.15, 0.3, 3);
      else if (a === 'zoomOut') arkitState.orbitZoom = clamp(arkitState.orbitZoom / 1.15, 0.3, 3);
      else if (a === 'reset') {
        arkitState.orbitYaw = 0; arkitState.orbitPitch = 0; arkitState.orbitZoom = 1;
        viewer.splat?.orbitPreset('front');
      }
      arkitState.orbitYaw = clamp(arkitState.orbitYaw, -Math.PI, Math.PI);
      if (viewer.splat) applyClientOrbit();
      updateArkitOrbit();
    };
  });
}

/** 把 (orbitYaw, orbitPitch, orbitZoom) 相对 front 参考方向作用到客户端相机 */
function applyClientOrbit() {
  const splat = viewer.splat;
  if (!splat) return;
  const c = splat.center();
  const v = splat.camera.position.clone().sub(c);
  const dist = v.length() || 1e-6;

  // front 参考方向（由 ref_c2w 推得）
  const rc = splat.refCamera();
  const d0 = new THREE.Vector3(rc.position[0], rc.position[1], rc.position[2]).sub(c);
  if (d0.lengthSq() < 1e-12) d0.set(0, 0, 1);
  d0.normalize();
  const th0 = Math.atan2(d0.x, d0.z);
  const ph0 = Math.asin(clamp(d0.y, -1, 1));

  const th = th0 + arkitState.orbitYaw;
  const ph = clamp(ph0 + arkitState.orbitPitch, -1.35, 1.35);
  const r = dist / arkitState.orbitZoom;
  const pos = new THREE.Vector3(
    c.x + r * Math.cos(ph) * Math.sin(th),
    c.y + r * Math.sin(ph),
    c.z + r * Math.cos(ph) * Math.cos(th),
  );
  splat.camera.position.copy(pos);
  splat.camera.lookAt(c);
}

function bindArkitToggle() {
  const t = $('arkitToggle');
  if (t) {
    t.onclick = () => {
      if (arkitState.open) { closeArkitPanel(); return; }
      if (!arkitState.meta) { setEngineHint('表情服务暂不可用', true); return; }
      if (!arkitState.jobId) { setEngineHint('请先生成一次 3D 数字人，再打开表情面板', true); return; }
      openArkitPanel(arkitState.jobId);
    };
  }
  const c = $('arkitClose');
  if (c) c.onclick = closeArkitPanel;
}

async function openArkitPanel(jobId) {
  setArkitStatus('正在检查实时会话…');
  let data = { ready: false };
  try {
    const res = await fetch(`${API}/api/jobs/${jobId}/arkit_session`);
    if (res.ok) data = await res.json();
  } catch (e) {
    setArkitStatus(`无法连接表情服务：${e.message}`);
    return;
  }
  if (!data || !data.ready) {
    setArkitStatus('该任务的实时会话不可用（服务可能已重启），请重新生成');
    return;
  }
  arkitState.jobId = jobId;
  arkitState.session = data;
  arkitState.gaussians = data.n_gaussians;
  const g = $('arkitGaussians');
  if (g) g.textContent = `高斯 ${Number(data.n_gaussians).toLocaleString()}`;
  $('arkitPanel').hidden = false;
  arkitState.open = true;
  /* 契约 6.2：不再禁用 OrbitControls —— 面板在左栏，预览区始终可拖动旋转 */
  /* 面板位于左栏输入区（非画布覆盖层），展开后滚动到可见位置 */
  const panel = $('arkitPanel');
  if (panel && typeof panel.scrollIntoView === 'function') panel.scrollIntoView({ block: 'nearest' });
  setArkitStatus('');
  updateArkitOrbit();
  scheduleArkitRender();
}

function closeArkitPanel() {
  const p = $('arkitPanel');
  if (p) p.hidden = true;
  arkitState.open = false;
  if (arkitState.raf) { cancelAnimationFrame(arkitState.raf); arkitState.raf = 0; }
  setArkitStatus('');
}

/* rAF 合并：一次在途 POST，丢弃中间状态，只在最新状态上重渲 */
function scheduleArkitRender() {
  if (!arkitState.open || !arkitState.jobId) return;
  if (arkitState.raf) return;
  arkitState.raf = requestAnimationFrame(() => {
    arkitState.raf = 0;
    if (arkitState.busy) { arkitState.pending = true; return; }
    if (!arkitState.open) return;                // 帧到位前面板已关闭
    sendArkitSplat();
  });
}

/** POST arkit_splat，返回 {buffer, count, ms}；非 2xx 抛错 */
async function postArkitSplat(jobId, arkit) {
  const t0 = performance.now();
  const res = await fetch(`${API}/api/jobs/${jobId}/arkit_splat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arkit }),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const t = await res.text();
      if (t) {
        msg = t;
        try { msg = (JSON.parse(t) || {}).detail || t; } catch { /* 非 JSON 报错，原样展示 */ }
      }
    } catch { /* 忽略 */ }
    throw new Error(msg);
  }
  const buffer = await res.arrayBuffer();
  const count = Number(res.headers.get('X-LSPL-Count')) || 0;
  const serverMs = Number(res.headers.get('X-LSPL-Ms')) || 0;
  return { buffer, count, serverMs, ms: Math.round(performance.now() - t0) };
}

async function sendArkitSplat() {
  const splat = viewer.splat;
  if (!arkitState.open || !arkitState.jobId || !splat) return;
  arkitState.busy = true;
  arkitState.pending = false;
  try {
    const data = await postArkitSplat(arkitState.jobId, Array.from(arkitState.values));
    if (!arkitState.open) return;                // 帧到达前面板已关闭 → 丢弃
    splat.setData(parseLSPL(data.buffer));
    if (!viewer.controls) return;
    viewer.controls.target.copy(splat.center());
    viewer.controls.update();
    arkitState.lastMs = data.ms;
    const lat = $('arkitLatency');
    if (lat) lat.textContent = `延迟 ${data.ms} ms · 服务端 ${data.serverMs} ms`;
    const cms = $('covMs');                     // 当前协方差方法的服务端耗时
    if (cms) cms.textContent = `服务端 ${data.serverMs.toFixed(1)} ms`;
    setArkitStatus('');
  } catch (e) {
    setArkitStatus(`点云失败：${e.message}`);
  } finally {
    arkitState.busy = false;
    if (arkitState.pending && arkitState.open) sendArkitSplat();
  }
}

function setArkitStatus(msg) {
  const el = $('arkitStatus');
  if (el) el.textContent = msg;
}

/** 读数：当前相机相对 center() 与 front 参考方向的球坐标 */
function updateArkitOrbit() {
  const el = $('arkitOrbitReadout');
  if (!el) return;
  const splat = viewer.splat;
  if (!splat) { el.textContent = 'yaw 0.0° · pitch 0.0° · zoom 1.00'; return; }
  const c = splat.center();
  const v = splat.camera.position.clone().sub(c);
  const dist = v.length() || 1e-6;
  const rc = splat.refCamera();
  const d0 = new THREE.Vector3(rc.position[0], rc.position[1], rc.position[2]).sub(c);
  if (d0.lengthSq() < 1e-12) d0.set(0, 0, 1);
  d0.normalize();

  const th = Math.atan2(v.x, v.z);
  const th0 = Math.atan2(d0.x, d0.z);
  let dyaw = th - th0;
  while (dyaw > Math.PI) dyaw -= 2 * Math.PI;
  while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
  const dpitch = Math.asin(clamp(v.y / dist, -1, 1)) - Math.asin(clamp(d0.y, -1, 1));
  const dist0 = Math.max(splat.frameRadius(), 1e-6) / Math.tan(19 * Math.PI / 180) * 1.35;
  const zoom = dist0 / dist;

  el.textContent =
    `yaw ${(dyaw * 180 / Math.PI).toFixed(1)}° · pitch ${(dpitch * 180 / Math.PI).toFixed(1)}° · zoom ${zoom.toFixed(2)}`;
}

/* 新任务成功 → 复位/关闭表情面板，避免旧会话泄漏到新结果 */
function resetArkitForNewJob(job) {
  arkitState.pending = false;                    // 丢弃在途的重排请求
  if (arkitState.raf) { cancelAnimationFrame(arkitState.raf); arkitState.raf = 0; }
  if (arkitState.open) closeArkitPanel();
  arkitState.jobId = job && job.id ? job.id : null;
  arkitState.session = null;
  arkitState.orbitYaw = 0;
  arkitState.orbitPitch = 0;
  arkitState.orbitZoom = 1;
  if (arkitState.values) arkitState.values.fill(0);
  syncArkitSliders();
  updateArkitOrbit();
  setArkitStatus('');
  const g = $('arkitGaussians');
  if (g) g.textContent = '';
  const lat = $('arkitLatency');
  if (lat) lat.textContent = '延迟 --';
}

/* ═══════════════════════════════════════════════════════════════
   协方差形变方法切换（验收用）
   - GET  /api/cov/methods -> {method, params, methods:[{id,label,params}]}
   - POST /api/cov/method  {method, params} -> 同上 + class/session_rebound
   切换后后端复用现有 live 会话，这里只需重拉一帧当前表情即可看到差别。
   ═══════════════════════════════════════════════════════════════ */
const covState = {
  methods: [], method: null, params: {}, meta: {}, busy: false, loaded: false,
};

async function initCovPanel() {
  const sel = $('covMethod');
  if (!sel) return;
  try {
    const data = await fetchJSON(`${API}/api/cov/methods`);
    covState.methods = data.methods || [];
    covState.method = data.method || null;
    covState.params = data.params || {};
    covState.loaded = true;
    buildCovSelect();
    bindCovCompare();
  } catch (e) {
    sel.innerHTML = '<option>协方差接口不可用</option>';
    sel.disabled = true;
    const h = $('covHint');
    if (h) h.textContent = `方法列表读取失败：${e.message}`;
  }
}

function buildCovSelect() {
  const sel = $('covMethod');
  if (!sel) return;
  sel.innerHTML = '';
  covState.methods.forEach((m) => {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = m.label;
    o.title = m.desc || m.label;
    if (m.id === covState.method) o.selected = true;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    covState.method = sel.value;
    covState.params = defaultParamsOf(sel.value);
    buildCovParams();
    applyCovMethod();
  };
  buildCovParams();
}

function methodMeta(id) {
  return covState.methods.find((m) => m.id === id) || { params: [] };
}

function defaultParamsOf(id) {
  const out = {};
  (methodMeta(id).params || []).forEach((p) => { out[p.key] = p.default; });
  return out;
}

function buildCovParams() {
  const host = $('covParams');
  if (!host) return;
  host.innerHTML = '';
  const specs = methodMeta(covState.method).params || [];
  if (!specs.length) {
    const p = document.createElement('div');
    p.className = 'arkit-cov-empty';
    p.textContent = '该方法没有可调参数';
    host.appendChild(p);
    return;
  }
  specs.forEach((sp) => {
    const row = document.createElement('label');
    row.className = 'arkit-cov-param';
    const nm = document.createElement('span');
    nm.className = 'arkit-cov-param__name';
    nm.textContent = sp.label;
    nm.title = `${sp.key} · ${sp.label}`;
    const range = document.createElement('input');
    range.type = 'range';
    range.min = String(sp.min);
    range.max = String(sp.max);
    range.step = String(sp.step);
    const cur = covState.params[sp.key] != null ? covState.params[sp.key] : sp.default;
    range.value = String(cur);
    const val = document.createElement('span');
    val.className = 'arkit-cov-param__val';
    val.textContent = fmtCov(cur, sp.step);
    range.addEventListener('input', () => {
      const v = parseFloat(range.value);
      covState.params[sp.key] = v;
      val.textContent = fmtCov(v, sp.step);
    });
    range.addEventListener('change', applyCovMethod);
    row.append(nm, range, val);
    host.appendChild(row);
  });
}

function fmtCov(v, step) {
  const dec = String(step).includes('.') ? String(step).split('.')[1].length : 0;
  return Number(v).toFixed(Math.min(2, dec));
}

async function applyCovMethod() {
  if (!covState.loaded || covState.busy) return;
  covState.busy = true;
  setCovHint('切换中…');
  try {
    const data = await fetchJSON(`${API}/api/cov/method`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: covState.method, params: covState.params }),
    });
    covState.params = data.params || covState.params;
    const meta = covState.methods.find((m) => m.id === covState.method) || {};
    setCovHint(`${meta.desc || data.class || covState.method}${data.session_rebound ? '' : ' · 当前无实时会话'}`);
    if (arkitState.open) scheduleArkitRender();
    $('covMethod').title = meta.desc || meta.label || covState.method;
  } catch (e) {
    setCovHint(`切换失败：${e.message}`);
  } finally {
    covState.busy = false;
  }
}

function setCovHint(msg) {
  const el = $('covHint');
  if (el) el.textContent = msg;
}


/* ── 四法对比：把当前表情依次喂给四种调制器，四宫格并排看差别 ── */
const COV_COMPARE_ORDER = ['lam', 'tga', 'rot_sync', 'vertex_then_polar'];

function bindCovCompare() {
  const btn = $('covCompare');
  if (!btn) return;
  btn.onclick = covCompare;
}

async function covCompare() {
  const btn = $('covCompare');
  const hint = $('covCompareHint');
  if (!arkitState.open || !arkitState.jobId) {
    if (hint) hint.textContent = '先在右上角打开表情面板';
    return;
  }
  const keep = covState.method;
  const keepParams = { ...covState.params };
  const arkit = Array.from(arkitState.values);
  btn.disabled = true;
  const shots = [];
  try {
    for (const m of COV_COMPARE_ORDER) {
      if (hint) hint.textContent = `渲染 ${m} …`;
      await fetchJSON(`${API}/api/cov/method`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: m, params: {} }),
      });
      const res = await fetch(`${API}/api/jobs/${arkitState.jobId}/arkit_frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arkit, yaw: 0, size: 512 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      shots.push({ method: m, url: URL.createObjectURL(blob) });
    }
    showCovCompare(shots);
    if (hint) hint.textContent = '差异集中在嘴唇/下巴/鼻翼等薄片高斯区';
  } catch (e) {
    if (hint) hint.textContent = `对比失败：${e.message}`;
  } finally {
    // 还原用户原来的选择
    try {
      await fetchJSON(`${API}/api/cov/method`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: keep, params: keepParams }),
      });
    } catch { /* 还原失败不阻断 */ }
    btn.disabled = false;
  }
}

function showCovCompare(shots) {
  let box = $('covCompareBox');
  if (!box) {
    box = document.createElement('div');
    box.id = 'covCompareBox';
    box.className = 'covcmp';
    box.innerHTML = '<div class="covcmp__inner"><div class="covcmp__head">'
      + '<span>四法对比（同一表情 · 同一视角）</span>'
      + '<button class="covcmp__close" type="button">✕</button></div>'
      + '<div class="covcmp__grid" id="covcmpGrid"></div>'
      + '<p class="covcmp__foot">四格依次为 LAM 原版 / TGA 雅可比 / 旋转同步 / 先平均再极分解；'
      + '差异越大说明该方法对协方差朝向的修正越强。</p></div>';
    document.body.appendChild(box);
    box.addEventListener('click', (e) => {
      if (e.target === box || e.target.classList.contains('covcmp__close')) box.hidden = true;
    });
  }
  box.hidden = false;
  renderCovCompare(shots);
}

const COV_LABEL = {
  lam: 'LAM 原版', tga: 'TGA 雅可比', rot_sync: '旋转同步', vertex_then_polar: '先平均再极分解',
};

/* 把一张 blob URL 画到 canvas 取像素，用于算差图 */
function covPixels(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement('canvas');
      c.width = im.naturalWidth; c.height = im.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(im, 0, 0);
      resolve({ data: ctx.getImageData(0, 0, c.width, c.height), w: c.width, h: c.height, img: im });
    };
    im.onerror = () => reject(new Error('图片解码失败'));
    im.src = url;
  });
}

/* 与基准（LAM 原版）逐像素比较，输出差图 dataURL 与统计量 */
async function covDiff(baseUrl, url) {
  const a = await covPixels(baseUrl);
  const b = await covPixels(url);
  const n = a.w * a.h;
  const out = new ImageData(a.w, a.h);
  let sum = 0, changed = 0, mx = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const dr = Math.abs(a.data.data[o] - b.data.data[o]);
    const dg = Math.abs(a.data.data[o + 1] - b.data.data[o + 1]);
    const db = Math.abs(a.data.data[o + 2] - b.data.data[o + 2]);
    const d = Math.max(dr, dg, db);
    sum += (dr + dg + db) / 3;
    if (d > 2) changed++;
    if (d > mx) mx = d;
    const g = Math.min(255, d * 8);            // 差图增益 8 倍，便于肉眼看
    out.data[o] = g; out.data[o + 1] = g * 0.55; out.data[o + 2] = g * 0.15;
    out.data[o + 3] = 255;
  }
  const c = document.createElement('canvas');
  c.width = a.w; c.height = a.h;
  c.getContext('2d').putImageData(out, 0, 0);
  return {
    url: c.toDataURL('image/png'),
    mean: sum / n,
    changedPct: (100 * changed) / n,
    max: mx,
  };
}

async function renderCovCompare(shots) {
  const grid = $('covcmpGrid');
  grid.innerHTML = '';
  const base = shots.find((s) => s.method === 'lam') || shots[0];
  for (const s of shots) {
    const cell = document.createElement('figure');
    cell.className = 'covcmp__cell';
    const img = document.createElement('img');
    img.src = s.url;
    img.alt = s.method;
    const cap = document.createElement('figcaption');
    cap.className = 'covcmp__cap';
    const name = document.createElement('b');
    name.textContent = COV_LABEL[s.method] || s.method;
    cap.appendChild(name);
    cell.append(img, cap);
    grid.appendChild(cell);
    if (s.method === base.method) {
      const t = document.createElement('span');
      t.className = 'covcmp__metric';
      t.textContent = '基准';
      cap.appendChild(t);
      continue;
    }
    try {
      const d = await covDiff(base.url, s.url);
      const t = document.createElement('span');
      t.className = 'covcmp__metric';
      t.textContent = `均差 ${d.mean.toFixed(3)} · 改动 ${d.changedPct.toFixed(2)}% · 峰差 ${d.max}`;
      cap.appendChild(t);
      const dimg = document.createElement('img');
      dimg.src = d.url;
      dimg.className = 'covcmp__diff';
      dimg.alt = `${s.method} 差图`;
      dimg.title = '与 LAM 原版的像素差，橙色增益 8 倍';
      cell.appendChild(dimg);
    } catch (e) {
      const t = document.createElement('span');
      t.className = 'covcmp__metric';
      t.textContent = `差图失败：${e.message}`;
      cap.appendChild(t);
    }
  }
}
/* ────────────────────────────── 小工具 ────────────────────────────── */
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function showError(msg) {
  $('errorText').textContent = msg;
  $('errorBox').hidden = false;
}
function clearError() { $('errorBox').hidden = true; }
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

init();
