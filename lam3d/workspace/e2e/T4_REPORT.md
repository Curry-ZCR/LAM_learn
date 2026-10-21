# T4 独立验收报告 — 真实无头 Chromium 端到端与视觉验收

- 任务：task-4（revision 3，owner qa-browser）
- 契约：/workspace/lam3d/CONTRACT_v2.md（§8 B 组 9-16 + C 组 17-18 交叉参考）
- e2e 脚本：lam3d/scripts/e2e_splat_panel.py（我独占，47644 B）
- 报告 JSON：workspace/e2e/report_final3.json（权威）/ report_baseline.json（阶段 1 旧页基线）
- item 13 前景口径专项：workspace/e2e/report_item13_foreground.json
- 浏览器：playwright chromium headless，`--no-sandbox --disable-dev-shm-usage --use-gl=angle
  --use-angle=swiftshader --enable-unsafe-swiftshader`，viewport 1600x950，device_scale_factor=2
- 本轮生成 job：76020c15eada（probe13c）/ 352daca95083（final）
- 服务：cd /workspace/lam3d && bash run.sh --daemon → http://127.0.0.1:8000

## 总结论

**44 passed / 0 failed / 0 skipped**（report_final3.json，exit=0）。
契约 §8 B 组 9-16 全部 PASS，无产品缺陷。

## 逐条结果（9-16）

| 条 | 结论 | 关键实测证据 |
|---|---|---|
| 9 | PASS | #drop/#file/#preview/#samples、#generate/#engineHint、#viewer/#viewerBadge、#faceCard+#faceImg、#exportCard（含 3 下载链接且 details 默认**未 open**）、#statsCard、#splatQuality 全部存在；side cards=['导出模型','本次统计'] |
| 10 | PASS | #motions/#skipVideo/#historyWrap/#video/#dlMp4 全部不存在；static/ 下 html 仅 index.html；全站 JS/HTML 无 alert/confirm/window.open |
| 11 | PASS | colors=28115（≥200）；coverage=0.0903（>5%），bg=[15,19,28] |
| 12 | PASS | 拖拽 mean|Δpixel|=7.4695（>0.5）；相机 pos [-0.00766,0.00274,1.30261] → [1.12791,0.62890,-0.28987]（位置真实改变）；三预设 max|Δpos| left~right=2.286 left~three=1.970 right~three=0.327；三图互异 left↔right=7.08 left↔three=7.42 right↔three=3.97 |
| 13 | PASS | 见下节（前景口径 12/12 预设 + 5/5 滑块通道 > 0.5） |
| 14 | PASS | canvas.width=1508 == cssW(754) × dpr(2)，eff=2.0 且 clamp ≤2；高清绘制缓冲 1508×1276 ≥640×640 等效；#splatQuality 默认 hq，选项 ['标准','高清'] |
| 15 | PASS | console error 0；pageerror 0；dialogs=[] popups=[]；#wire 与 #spin 点击均无异常（is-on=True） |
| 16 | PASS | OrbitControls 1 个实例（dom=CANVAS）；damping=0.07 min=0.2 max=4.0 enableDamping=True rotateSpeed=0.85 |

附：B0/B1/B2 全流程 PASS —— 选样例 barbara.jpg → POST /api/generate 200
{"id":"352daca95083",...} → 长轮询至 state=succeeded → #result 显示、
#viewerBadge 页面原文 `已加载 · 20,018 高斯 · 可 360° 拖拽`。

## item 13（修正口径后）

Lead 裁决：契约 8.13 的 mean|Δpixel| 只应在**前景 = 非背景像素**上计算；全图口径下
背景约占 80.6%，会把信号稀释 ≈5.2×，温和表情在数学上难以越过 0.5。

- 前景掩码：与画布背景色（出现最多的颜色，实测 bg=(15,19,28)）任一通道差 > 6 的像素
- 实测 fg=372972/1924208（19.38%），全图稀释 ≈5.2×
- 噪声地板：同一状态连拍两帧 mean|Δ|=0.0（捕获完全确定性），前景内亦 0.0

逐预设（canonical front；fg=前景 mean|Δ|，full=全图 mean|Δ|，括号=前景变化像素占比）：

| 预设 | 前景 mean | 全图 mean | 前景变化占比 | #arkitLatency |
|---|---|---|---|---|
| smile | 1.004 | 0.207 | 5.00% | 延迟 35 ms · 服务端 16.87 ms |
| bigSmile | 1.364 | 0.287 | 6.44% | 延迟 36 ms · 服务端 14.71 ms |
| laugh | 2.382 | 0.476 | 9.63% | 延迟 30 ms · 服务端 13.96 ms |
| surprise | **4.936** | 1.033 | 13.47% | 延迟 34 ms · 服务端 16.19 ms |
| angry | 3.430 | 0.701 | 10.42% | 延迟 30 ms · 服务端 14.36 ms |
| sad | 2.438 | 0.496 | 8.60% | 延迟 27 ms · 服务端 14 ms |
| kiss | 1.244 | 0.243 | 5.42% | 延迟 39 ms · 服务端 21.91 ms |
| mouthOpen | 2.238 | 0.425 | 6.99% | 延迟 27 ms · 服务端 14.1 ms |
| blink | 1.140 | 0.215 | 2.74% | 延迟 32 ms · 服务端 14.17 ms |
| winkLeft | 1.125 | 0.212 | 4.00% | 延迟 30 ms · 服务端 14.42 ms |
| winkRight | 1.198 | 0.227 | 4.23% | 延迟 28 ms · 服务端 14.11 ms |
| cheekPuff | 1.020 | 0.198 | 4.31% | 延迟 27 ms · 服务端 13.89 ms |

**12/12 预设 > 0.5，最弱 cheekPuff=1.020**；全部预设均显著高于噪声地板（≥1.0 ≫ 0）。

逐滑块（推满 1.0，契约 8.13 明写「预设/滑块」两路径）：

| 通道 | 前景 mean | 全图 mean | 前景变化占比 | #arkitLatency |
|---|---|---|---|---|
| jawOpen | 1.837 | 0.568 | 6.84% | 延迟 21 ms · 服务端 14.12 ms |
| browInnerUp | 1.679 | 0.558 | 5.19% | 延迟 21 ms · 服务端 6.27 ms |
| mouthSmileLeft | 0.811 | 0.537 | 3.56% | 延迟 20 ms · 服务端 5.58 ms |
| eyeBlinkLeft | 1.812 | 1.275 | 4.41% | 延迟 21 ms · 服务端 5.62 ms |
| cheekPuff | 1.810 | 1.174 | 6.85% | 延迟 20 ms · 服务端 5.89 ms |

**5/5 滑块通道 > 0.5**。

独立几何交叉验证（前端两次不同预设的 POST 请求体回放真实后端）：
n=20018，xyz_mean_abs_delta=1.67e-4（>1e-5），scale_mean_abs_delta=1.7e-5，
bytes=1121152==144+20018*56，说明「不同预设 ⇒ 服务端几何确实不同」，非前端伪影。

目视确认（已下载原图检查）：neutral 为平静人脸；surprise 为睁大眼睛、张嘴、抬眉
（滑块 0.80/0.60/0.60），确为真实渲染像素变化。

## 阶段 1 harness 基线（旧页面，report_baseline.json）

harness 自检全绿：H1 page.screenshot（3200×1900 = viewport×dsf）✓；H2 WebGL2 合成
canvas（dpr=2，640×480）✓；H3 像素统计（colors=65536 coverage=0.9954）✓；
H4 拖拽工具（pointermove=13）✓；H5 dpr clamp（devicePixelRatio=2）✓。
旧页面预期 FAIL：#faceCard/#exportCard/#splatQuality 缺失、5 个 legacy id 仍在、
§16 无 OrbitControls 实例（旧页未加载 #result 故 initViewer() 未跑）。
OrbitControls 观察器机制另行单元验证（/tmp/verify_oc_tracker.py）：
list_len=1 ctor=__OC_Tracked damping=0.07 min=0.2 max=4 —— 证明 FAIL 是
「尚未加载结果」而非埋点缺陷。

## 截图清单（/workspace/lam3d/workspace/e2e/）

| 文件 | 尺寸 | 字节 |
|---|---|---|
| final_00_baseline_page.png | 3200x1900 | 432331 |
| final_01_initial.png | 3200x1900 | 427475 |
| final_02_ready.png | 3200x1900 | 630065 |
| final_03_generating.png | 3200x1900 | 576546 |
| final_04_result_facecard.png | 3200x1900 | 1123804 |
| final_05_splat_render.png | 1508x1276 | 287317 |
| final_06_after_drag.png | 3200x1900 | 1076868 |
| final_07_view_left.png | 1508x1276 | 280896 |
| final_07_view_right.png | 1508x1276 | 272528 |
| final_07_view_three.png | 1508x1276 | 279991 |
| final_08_arkit_panel.png | 3200x1900 | 1167129 |
| final_09_arkit_preset.png | 3200x1900 | 1169116 |
| final_10_arkit_slider.png | 3200x1900 | 1173384 |
| final_11_final_state.png | 3200x1900 | 1176389 |
| final_13_neutral_front.png（item13 基线） | 1508x1276 | 340842 |
| final_13_preset_{smile,bigSmile,laugh,surprise,angry,sad,kiss,mouthOpen,blink,winkLeft,winkRight,cheekPuff}.png | 1508x1276 | 各 ~340 KB |
| final_13_slider_{jawOpen,browInnerUp,mouthSmileLeft,eyeBlinkLeft,cheekPuff}.png | 1508x1276 | 各 ~342-357 KB |

## 产品缺陷

**无。** 唯一一条曾 FAIL 的 [13] 经 Lead 裁决确认为**契约判据度量口径歧义**
（全图 vs 前景），已由 Lead 更新 CONTRACT_v2.md:295-297；按新口径复测 12/12 预设与
5/5 滑块通道全部 > 0.5。未发现任何真实产品缺陷。

## 角色纪律

仅写：lam3d/scripts/e2e_splat_panel.py（我独占）、lam3d/workspace/e2e/、/tmp/。
未修改 lam3d/lam3d/ 或 lam3d/static/ 任何生产代码。
