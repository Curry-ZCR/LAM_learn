#!/usr/bin/env node
// LSPL v1 解析器契约测试 (task-3 / CONTRACT_v2.md §4 + §5.1) —— Node 侧独立解析器验证
//
// 目标: 用 Python 造的合法 / 畸形 LSPL buffer 驱动 static/splat.js 的 parseLSPL，
//       断言 5 段 payload 的字节偏移与 Python 侧写出的完全一致、畸形必抛含 'LSPL' 的 Error。
//
// 用法:
//   # 1) 先造 fixture (不需要后端)
//   /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_splat_test.py --make-fixture /tmp/lspl_fixture
//   # 2) 跑本脚本
//   node scripts/splat_js_test.mjs --fixture /tmp/lspl_fixture
//
// 退出码: 0 全 PASS | 1 有 FAIL | 2 splat.js 尚未就绪 (NOT_READY，不算产品缺陷)
//
// 说明: static/splat.js 只允许 `import * as THREE from 'three'`，Node 无法解析该裸说明符，
//       因此本脚本把它复制到 /tmp 并把 'three' 重写为本地 vendor 路径后再 import。
//       这一重写只作用于测试副本，不改生产文件。

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};

const FIXTURE = path.resolve(opt('--fixture', '/tmp/lspl_fixture'));
const SPLAT_JS = path.resolve(opt('--splat', '/workspace/lam3d/static/splat.js'));
const THREE = path.resolve(opt('--three', '/workspace/lam3d/static/vendor/three/build/three.module.js'));
const HEADER_SIZE = 144;
const FLOATS_PER_GAUSSIAN = 56;
const SEGMENTS = ['xyz', 'rot', 'scale', 'opacity', 'rgb'];
const WIDTH = { xyz: 3, rot: 4, scale: 3, opacity: 1, rgb: 3 };

let pass = 0, fail = 0;
const rows = [];
function check(id, name, ok, detail = '') {
  rows.push([id, name, ok ? 'PASS' : 'FAIL', detail]);
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${id} ${name}${detail ? '  — ' + detail : ''}`);
  ok ? pass++ : fail++;
  return ok;
}
function hard(id, name, detail) { check(id, name, false, detail); }

function toArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

// ---------------------------------------------------------------- 载入被测模块
if (!fs.existsSync(SPLAT_JS)) {
  console.log(`[NOT_READY] 生产文件不存在: ${SPLAT_JS}`);
  console.log('  -> 前端 (task-2) 尚未交付 static/splat.js，无可测对象。');
  process.exit(2);
}
if (!fs.existsSync(THREE)) {
  console.log(`[NOT_READY] 找不到 three vendored 模块: ${THREE}`);
  process.exit(2);
}
const src = fs.readFileSync(SPLAT_JS, 'utf8');
if (!/from\s+['"]three['"]/.test(src) && !/from\s+['"]three\//.test(src)) {
  console.log('[WARN] splat.js 未见 `from \'three\'` 说明符（契约要求），继续尝试 import');
}
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lspl_js_'));
const patched = path.join(tmpDir, 'splat_under_test.mjs');
// vendor 的 three.module.js 旁边没有 package.json，Node 会按 CJS 解析而报
// "Unexpected token 'export'"。复制成 .mjs 强制 ESM（只动临时副本）。
const threeMjs = path.join(tmpDir, 'three_under_test.mjs');
fs.copyFileSync(THREE, threeMjs);
const threeUrl = pathToFileURL(threeMjs).href;
const threeDir = path.dirname(THREE);
fs.writeFileSync(
  patched,
  src.replace(/from\s+['"]three['"]/g, `from '${threeUrl}'`)
     .replace(/from\s+['"]three\/addons\/([^'"]*)['"]/g,
              (_m, sub) => `from '${pathToFileURL(path.join(threeDir, 'examples/jsm', sub)).href}'`),
  'utf8'
);
console.log(`[setup] 被测副本: ${patched}`);
console.log(`[setup] three: ${threeUrl}`);

let mod;
try {
  mod = await import(pathToFileURL(patched).href);
} catch (err) {
  console.log(`[NOT_READY] import splat.js 失败: ${err && err.stack ? err.stack : err}`);
  console.log('  -> 视为前端尚未就绪（NOT_READY），不算 QA FAIL。');
  process.exit(2);
}
const parseLSPL = mod.parseLSPL;
if (typeof parseLSPL !== 'function') {
  console.log('[FATAL] splat.js 未导出 parseLSPL 函数（契约 §5 要求 export function parseLSPL）');
  process.exit(1);
}
console.log('[setup] parseLSPL 已导出');

// ---------------------------------------------------------------- fixture
if (!fs.existsSync(path.join(FIXTURE, 'expected.json'))) {
  console.log(`[NOT_READY] 缺 fixture: ${FIXTURE}/expected.json`);
  console.log('  -> 先运行: python scripts/arkit_splat_test.py --make-fixture ' + FIXTURE);
  process.exit(2);
}
const exp = JSON.parse(fs.readFileSync(path.join(FIXTURE, 'expected.json'), 'utf8'));
const n = exp.n;
console.log(`[fixture] N=${n} total=${exp.totalBytes}B header=${exp.headerSize} md5=${exp.md5}`);

// ---------------------------------------------------------------- 1. 合法 buffer
const validBuf = toArrayBuffer(fs.readFileSync(path.join(FIXTURE, 'valid.lspl')));
let r = null;
try {
  r = parseLSPL(validBuf);
} catch (err) {
  hard('J1', 'parseLSPL(合法 buffer) 不抛错', `抛错: ${err && err.message}`);
}
if (r) {
  const isF32 = (v, len) => v instanceof Float32Array && v.length === len;
  check('J1', 'parseLSPL 返回 count/flags',
    r.count === n && r.flags === exp.flags,
    `count=${r.count}（期望 ${n}）flags=${r.flags}（期望 ${exp.flags}）`);
  check('J2', '头部数组长度: bboxMin/Max/center=3, refC2W=16, refIntr=4, refSize=2',
    isF32(r.bboxMin, 3) && isF32(r.bboxMax, 3) && isF32(r.center, 3) &&
    isF32(r.refC2W, 16) && isF32(r.refIntr, 4) && isF32(r.refSize, 2),
    `bboxMin=${r.bboxMin?.length} bboxMax=${r.bboxMax?.length} center=${r.center?.length} ` +
    `refC2W=${r.refC2W?.length} refIntr=${r.refIntr?.length} refSize=${r.refSize?.length}`);
  const segOk = SEGMENTS.every((k) => isF32(r[k], n * WIDTH[k]));
  check('J3', `payload 数组长度 = N*width (N=${n})`, segOk,
    SEGMENTS.map((k) => `${k}=${r[k]?.length}/${n * WIDTH[k]}`).join(' '));

  // 逐字段数值比对（与 Python 写入端完全一致）
  const cmp = (name, got, want, tol = 0) => {
    if (!got || got.length !== want.length) {
      check('J4.' + name, `header ${name} 数值与 Python 写入一致`, false,
        `长度 ${got?.length} vs ${want.length}`);
      return false;
    }
    let maxd = 0;
    for (let i = 0; i < want.length; i++) maxd = Math.max(maxd, Math.abs(got[i] - want[i]));
    return check('J4.' + name, `header ${name} 数值与 Python 写入一致`, maxd <= tol,
      `max|Δ|=${maxd}`);
  };
  cmp('bboxMin', r.bboxMin, exp.bboxMin);
  cmp('bboxMax', r.bboxMax, exp.bboxMax);
  cmp('center', r.center, exp.center);
  cmp('refC2W', r.refC2W, exp.refC2W);
  cmp('refIntr', r.refIntr, exp.refIntr);
  cmp('refSize', r.refSize, exp.refSize);

  // 字节偏移：与 Python 端 dump 的完全一致（按段比对原始字节）
  for (const k of SEGMENTS) {
    const want = Buffer.from(exp.segments[k], 'base64');
    const got = Buffer.from(r[k].buffer, r[k].byteOffset, r[k].byteLength);
    const equal = got.length === want.length && got.equals(want);
    const off = exp.offsets[k];
    check(`J5.${k}`, `seg[${k}] 位于偏移 ${off}（144 + 前序段长），字节与 Python 写入逐字节相同`,
      equal, equal ? `${want.length}B @${off}` :
        `不一致 got=${got.length}B want=${want.length}B @${off}`);
  }
  // 偏移自洽性（独立于 Python dump，纯按契约推算）
  let off = HEADER_SIZE, drift = [];
  for (const k of SEGMENTS) {
    if (off !== exp.offsets[k]) drift.push(`${k}:契约 ${off} vs python ${exp.offsets[k]}`);
    off += n * WIDTH[k] * 4;
  }
  check('J6', '5 段偏移按 144 + Σ(w*4) 推算与 fixture 一致', drift.length === 0, drift.join('; '));
  check('J7', `总长 = 144 + N*56 = ${exp.totalBytes}`, exp.totalBytes === HEADER_SIZE + n * FLOATS_PER_GAUSSIAN,
    `total=${exp.totalBytes}`);
}

// ---------------------------------------------------------------- 2. 畸形容错
const badCases = [
  ['bad_magic.lspl', 'magic 错误'],
  ['bad_version.lspl', 'version 错误'],
  ['truncated.lspl', '长度不足 (payload 截断)'],
  ['short.lspl', '长度不足 (< 144 头)'],
];
for (const [file, desc] of badCases) {
  const p = path.join(FIXTURE, file);
  if (!fs.existsSync(p)) { hard(`J8.${file}`, `${desc} 必抛错`, 'fixture 缺失'); continue; }
  const buf = toArrayBuffer(fs.readFileSync(p));
  let threw = null;
  try { parseLSPL(buf); } catch (e) { threw = e; }
  const ok = threw instanceof Error && typeof threw.message === 'string' && threw.message.includes('LSPL');
  check(`J8.${file}`, `${desc} -> 抛 Error 且 message 含 'LSPL'`, ok,
    threw ? `Error("${threw.message}")` : '未抛错（契约 §5.1 要求抛 Error）');
}
// 空 buffer / 非 ArrayBuffer
for (const [label, mk] of [['空 buffer(0B)', () => new ArrayBuffer(0)],
                           ['3 字节 buffer', () => new ArrayBuffer(3)]]) {
  let threw = null;
  try { parseLSPL(mk()); } catch (e) { threw = e; }
  check(`J9.${label}`, `${label} -> 抛 Error 且含 'LSPL'`,
    threw instanceof Error && threw.message.includes('LSPL'),
    threw ? `Error("${threw.message}")` : '未抛错');
}

// ---------------------------------------------------------------- 汇总
console.log('\n================ splat.js parseLSPL 验收汇总 ================');
for (const [id, name, st, detail] of rows) {
  console.log(`  ${st} ${id}  ${name}${detail && st === 'FAIL' ? '   [' + detail + ']' : ''}`);
}
console.log(`-------- PASS=${pass} FAIL=${fail} --------`);
process.exit(fail ? 1 : 0);
