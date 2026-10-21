# T6 QA 报告 — ARKit 面板迁入左栏后的全量 e2e 回归 + 布局不遮挡验收

**验证方**：qa-browser（独立 QA，真实无头 Chromium）
**被测代码**：task-5 交付的 `lam3d/static/index.html` / `app.css` / `app.js`（面板迁入左栏 `aside.panel--input`）
**脚本**：`/workspace/lam3d/scripts/e2e_splat_panel.py`（`--tag t6`，我独占）
**报告 JSON**：`/workspace/lam3d/workspace/e2e/report_t6.json`
**环境**：Playwright Python + chromium headless（`--no-sandbox --disable-dev-shm-usage --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`），viewport 1600×950，`device_scale_factor=2`，SwiftShader 软件渲染

---

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 全量 e2e | **54 passed / 0 failed / 0 skipped**（exit=0） |
| console error | **0 条** |
| pageerror | **0 条** |
| 弹窗 / 新标签页 | 0 / 0 |
| 4 条新增布局断言 | **全部 PASS** |
| 旧断言回归 | 全部 PASS |
| 产品缺陷 | **未发现**（详见 §4 观察项） |

原文输出（截尾）：

```
==== 54 passed / 0 failed / 0 skipped → /workspace/lam3d/workspace/e2e/report_t6.json ====
```

---

## 1. 新增的 4 条布局断言（T6 核心）

### 1.1 矩形交集面积 = 0（面板不遮挡预览区）

用 `getBoundingClientRect()` 取两个矩形的**视口坐标**再求交：

| 元素 | x | y | width | height | 右边界 | 下边界 |
|---|---|---|---|---|---|---|
| `#arkitPanel`（左栏卡片） | 39.0 | 296.3 | 350.0 | 633.4 | 389.0 | 929.7 |
| `.viewer-wrap`（预览区） | 447.0 | 37.0 | 756.0 | 640.0 | 1203.0 | 677.0 |

- 水平方向：panel 右边界 389.0 < viewer-wrap 左边界 447.0 ⇒ 无重叠（间距 58 px）
- **交集面积 = 0.0 px²** ⇒ PASS

> 视觉证据：`t6_panel_vs_viewer.png`（面板红框、viewer-wrap 绿框、`#viewer` 蓝框、canvas 黄框叠加，肉眼可见红框与绿框完全不相交）。
> 补充说明：我上一轮（迁移前）实测面板是**绝对定位覆盖层**，与 viewer-wrap 大面积极相交；本轮交集归零，证明 overlay 语义已真正消除。

### 1.2 `#arkitPanel` 位于左栏 `aside.panel--input` DOM 子树内

```
closest('aside.panel--input') = True
getComputedStyle(position)   = static      （不再是 absolute/fixed）
flex 收缩比例                 = 0 / 0        （不被压扁）
```

⇒ PASS。面板已成为左栏的**流式卡片**，而非查看器内的覆盖层。

### 1.3 面板打开前后 canvas 尺寸不变，且仍满足 A14 两条

| 时刻 | canvas.width | canvas.height |
|---|---|---|
| 面板打开**前**（生成完成、面板仍 hidden） | 1508 | 1276 |
| 面板打开**后** | 1508 | 1276 |

- 尺寸完全不变 ⇒ 面板迁位／打开**没有**触发布局回流把画布挤小
- A14 两条同时成立：
  - (a) `canvas.width === cssW * dpr`：`1508 == 754 × 2`，`eff = 1508 / 754 = 2.0`，dpr 被 clamp 在 ≤2 ✔
  - (b) 绘制缓冲区 ≥ 640×640：`1508 × 1276` ✔（CSS 尺寸 `754 × 638`，靠 `.viewer-wrap.is-hq{min-height:640px}` 撑出）
- 另据 Lead 澄清，**未**通过抬高 dpr 倍率来满足 (b)；倍率恒为 `clamp(dpr,1,2)`。

⇒ PASS。

### 1.4 `#arkitSliders` 内部滚动生效

分两种状态测量（契约允许「内部滚动」**或**「面板完整可见」两种形态）：

| 状态 | scrollHeight | clientHeight | 是否溢出 | scrollTop |
|---|---|---|---|---|
| 默认态（仅第 1 组「眉毛」展开 + 「其他」组） | 363 | 363 | 否 | 0（恒 0，属正常） |
| **展开全部 6 组后** | 1484 | 414 | **是** | **1070** |

- 默认态：无溢出 ⇒ 内容完整可见，无需内部滚动（契约允许的另一形态）✔
- 展开全部组后：`1484 > 414` ⇒ **内部滚动确实生效**，且真的能滚到底（`scrollTop=1070 = 1484 − 414`）✔

⇒ PASS（动画/滚动区域受限，不会把左栏或整页撑爆）。

> 视觉证据：`t6_panel_scrolled.png`（展开全部组并滚到底；可见 mouth*/jaw*/tongueOut 等滑块，面板仍在左栏 350 px 宽内）。

---

## 2. 旧断言回归

| 断言 | 实测 | 结果 |
|---|---|---|
| `#motions` / `#skipVideo` / `#historyWrap` / `#video` / `#dlMp4` 全部不存在 | `{"motions": false, "skipVideo": false, "historyWrap": false, "video": false, "dlMp4": false}` | PASS |
| `static/` 下 html 只有 1 个 | `htmls=['index.html']` | PASS |
| 全站 JS/HTML 无 `alert(` / `confirm(` / `window.open(` | `js={} html=[]` | PASS |
| OrbitControls damping / min / max / enableDamping 未变 | `damping=0.07 min=0.2 max=4 enableDamping=True` | PASS |
| OrbitControls rotateSpeed 未变 | `rotateSpeed=0.85` | PASS |
| 页面结构（9）上传/生成/画布/`#faceCard`/`#exportCard` 折叠/`#statsCard`/`#splatQuality` | 全部存在 | PASS |
| 驱动视频卡片、下载 MP4 卡片已消失 | `side cards=['导出模型', '本次统计']` | PASS |

---

## 3. 与 T6 同批复核的既有验收（11–16）

| 条目 | 实测 | 结果 |
|---|---|---|
| 11 渲染色数 | `colors=28115`（≥200） | PASS |
| 11 非背景覆盖率 | `coverage=0.0903`（>5%） | PASS |
| 12 拖拽后像素变化 | `mean|Δpixel|=7.4803` | PASS |
| 12 拖拽后相机位置改变 | 有改变（pos 前后不同） | PASS |
| 12 left/right/three 三画面互异 | 两两 mean\|Δ\|>0.5 | PASS |
| 13 预设 → 像素变化（前景掩码口径） | 最强 `surprise` 前景 `4.7378`；**12/12 个预设 >0.5**（最弱 `cheekPuff` 1.0712） | PASS |
| 13 每个预设高于噪声本底 | 噪声地板 `0.0`，最弱 1.0712 | PASS |
| 13 滑块路径 → 像素变化 | `jawOpen` 2.7163 / `browInnerUp` 1.1533 / `mouthSmileLeft` 0.5180 | PASS |
| 13 `#arkitLatency` 出现数值 | `'延迟 26 ms · 服务端 11.75 ms'` | PASS |
| 13 两次不同预设 → 服务端几何不同 | `xyz_mean_abs_delta=0.000167` | PASS |
| 14 canvas 尺寸与 dpr | `width=1508 cssW=754 dpr=2 eff=2.0` | PASS |
| 14 高清绘制缓冲区 ≥640×640 | `1508x1276` | PASS |
| 14 `#splatQuality` 默认高清 | `value='hq' options=['标准','高清']` | PASS |
| 15 `#wire` / `#spin` 点击无异常 | 均 `is-on=True` | PASS |
| 15 console error / pageerror | 0 / 0 | PASS |
| 16 OrbitControls 实例存在 | 1 个（`dom=CANVAS`） | PASS |

---

## 4. 产品缺陷与观察项

**产品缺陷：未发现。** 面板迁位后所有既有功能与本次新增布局要求均通过。

诚实登记的**观察项**（不构成缺陷，供 Lead 判读）：

1. **默认态 `#arkitSliders` 无内部溢出**（`scrollHeight=363 == clientHeight=363`）。契约 §6.2 允许「内部滚动生效」或「面板完整可见」两种形态，此处属后者：默认只展开第 1 组（眉毛）+「其他」组，内容未超出 414 px 的滚动区。**只有当用户手动展开全部 6 组时才会溢出**（1484 > 414），此时滚动正常工作。
2. **`#arkitSliders` 的 clientHeight=363 而滚动区 clientHeight=414**：说明外层容器有 51 px 的 padding/border/间隙；不影响功能，仅记录。
3. 我上一轮（T4）曾把 item 13 误判为 FAIL，根因是**测试脚本**取了 DOM 中第 0 个预设 = `smile`（最温和表情），且用了**全画布均值**口径（前景只占约 11.6%，被稀释 ≈8.6×）。本轮改用 Lead 已写入契约的**前景掩码口径**后，12/12 预设均 >0.5。此前的失败不是产品缺陷。
4. 一次并发运行中，我的第一版滚动断言（`scrollTop > 0`）在**默认态**必然为 FAIL（无溢出 ⇒ scrollTop 恒 0）。这是**我的断言缺陷**，已修为「默认态断言无溢出 / 展开全组后断言可滚到底」，判据未放宽。

---

## 5. 截图清单（`/workspace/lam3d/workspace/e2e/`）

T6 指定 3 张：

| 文件 | 尺寸 | 说明 |
|---|---|---|
| `t6_panel_in_left_col.png` | 3200×1900 | 面板打开的整体页，可见面板在左栏卡片内 |
| `t6_panel_vs_viewer.png` | 3200×1900 | 四矩形描边叠加（panel 红 / viewer-wrap 绿 / `#viewer` 蓝 / canvas 黄），红绿不相交 |
| `t6_panel_scrolled.png` | 3200×1900 | 展开全部 6 组并滚到底（滑块区内部滚动生效） |

同批产出的全流程截图（16 张，尺寸均为实际值）：

```
final_00_baseline_page.png   3200x1900
final_01_initial.png         3200x1900
final_02_ready.png           3200x1900
final_03_generating.png      3200x1900
final_04_result_facecard.png 3200x1900
final_05_splat_render.png    1508x1276   （canvas 原生绘制缓冲区位图）
final_06_after_drag.png      3200x1900
final_07_view_left.png       1508x1276
final_07_view_right.png      1508x1276
final_07_view_three.png      1508x1276
final_08_arkit_panel.png     3200x1900
final_09_arkit_preset.png    3200x1900   （重放最强预设 surprise）
final_10_arkit_slider.png    3200x1900
final_11_final_state.png     3200x1900
t6_panel_in_left_col.png     3200x1900
t6_panel_vs_viewer.png       3200x1900
t6_panel_scrolled.png        3200x1900
```

> 尺寸口径：`page.screenshot(full_page=True)` 在 `device_scale_factor=2` 下输出 `1600×950 × 2 = 3200×1900`；canvas 原生截图输出其绘制缓冲区真实尺寸 `1508×1276`。

---

## 6. 复现性确认（第二次独立全量运行）

为确认 `report_t6.json` 的截图元数据（`size`/`bytes`）与真实文件一致，脚本中新增 `mark_shot()`（尺寸/字节数一律从落盘文件读取，不再手写），随后**完整重跑一次**：

```
==== 54 passed / 0 failed / 0 skipped → /workspace/lam3d/workspace/e2e/report_t6.json ====
```

- 两次运行的 failed 均为 0，条目数一致（54），逐条 extra 数值一致（如交集 `0.0px²`、canvas `1508x1276`、`scrollTop=1070`、`surprise fg=4.7378`）。
- `report_t6.json` 中已无 `bytes == 0` 的截图条目；三张 T6 截图实测为 `3200x1900`（793962 / 798211 / 822359 字节）。

## 7. 取证环境（本轮）

| 项 | 值 |
|---|---|
| 服务进程 | pid 49848，启动于 `Fri Oct 16 10:41:04 2026` |
| served == disk | `static/index.html`、`static/app.js`、`static/app.css`、`static/splat.js` 四者 md5 全部相同 ✔ |
| 生产代码是否被我改动 | **否**。我只写了 `scripts/e2e_splat_panel.py`、`workspace/e2e/T6_REPORT.md`、`workspace/e2e/report_t6.json` 与 `workspace/e2e/*.png` |
| 本次 e2e 生成的 job | `8eb78424cf5a`（另一次运行 `76f4c61a519d`），均 succeeded |
