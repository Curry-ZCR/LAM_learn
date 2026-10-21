# CONTRACT v2 — 单图 3DGS 重建 + ARKit 表情点云预览（**冻结接口，禁止单方面更改**）

> 需求来源：用户 m01987。
> 保留：上传图片 → **单图 3DGS 重建** → 重建后人物脸图 → **ARKit 52 参数调节** → **浏览器内 360° 鼠标拖拽的高斯点云预览（高清晰度）**。
> 删除：动作序列 / 驱动视频 / MP4 编码 / 历史记录 / 视频下载。
> 任何需要修改本文件的场景，**必须回报 Lead**，由 Lead 改，其他人不得自行改契约。

---

## 1. 硬约束（沿用上一轮，全部有效）

1. 形变只作用于**解耦**的旋转四元数 `q`（必须 `F.normalize`）与缩放 `s`（无 NaN/Inf）；**严禁直接修改合成后的协方差 Σ**。
2. **禁止改动 `/workspace/LAM/**` 的任何文件**。上一轮已完成的 `covariance_modulation.py`、`gs_renderer.py`（`animate_gs_model` / `forward_animate_gs` 钩子）、`flame.py`（`dynamic_offset`）本轮**零改动**，直接调用。
   - 严禁修改 `diff-gaussian-rasterization` 的 CUDA C++ 源码。
3. 后端禁止在 N 个高斯点维度上写 Python `for` 循环；只用广播 / `bmm` / `matmul` / `scatter_add` / `einsum`。
4. `ARKit 参数全 0` ⇒ `Δq` 严格 `[1,0,0,0]`、`Δs` 严格 `0`（静态重建与改动前像素级一致）；推到 `1.0` 时 `Δs` 有 `tanh` 软截断。
5. **只在原有 lam3d 页面上增量改造**，严禁新建 HTML 页面、严禁 `alert/confirm/window.open`、严禁 `cv2.imshow`。
6. 前后端数据契约（字段名 / shape / dtype / 字节布局）100% 按本文件执行。

**远端工作目录 = `/workspace`（主机 117.50.181.29）。所有文件读写一律用 `rw_*` 工具操作远端路径**，不要用本地镜像是权威副本。

---

## 2. 数据流（最终形态）

```
用户上传图片
  → POST /api/generate (multipart: image | sample)
  → JobManager 单工作线程 → SingleImage3DPipeline.run()
       1. FLAME 跟踪（tracker）                          [保留]
       2. 参考图预处理 → shape_param                     [保留]
       3. prepare_motion_seqs(指向 tracking/export/input) ← 只有 1 帧！
       4. forward_latent_points → forward_gs → gs_model_list（canonical 高斯，用中性 flame_data）
       5. 渲染唯一 1 帧（拟合位姿 + 参考相机）→ 重建后人物脸图 face.png
       6. _export_meshes → GLB / OBJ / PLY               [保留]
       7. arkit_live.save_session(...)（中性 flame_data + 参考相机）
  → 前端轮询 GET /api/jobs/{id}
  → 前端 POST /api/jobs/{id}/arkit_splat {arkit:[52]} → LSPL v1 二进制
  → static/splat.js 解析 → Three.js InstancedBufferGeometry + 深度排序 → 画布
```

---

## 3. HTTP API（冻结）

### 3.1 保留不变
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | `{ready, warming, error, queue}` |
| GET | `/api/samples/{name}` | 样例图 |
| GET | `/api/jobs` | 列表 |
| POST | `/api/jobs/{id}/cancel` | 仅 queued 可取消 |
| GET | `/api/arkit/meta` | `{names[52], presets, count}` |
| POST | `/api/jobs/{id}/arkit_frame` | PNG 渲染；body `{arkit, yaw, pitch, zoom, size}` |

### 3.2 变更
- `GET /api/config` → **删除** `motions`、`frames_hint`；保留 `{samples, gpu, device, max_upload_mb}`。
- `GET /api/jobs/{id}` → **删除** `motion` 字段。
- `GET /api/jobs/{id}/file/{name}`：
  | name | 内容 |
  |---|---|
  | `face` | **新增**：重建后人物脸图 PNG（= 第 5 步渲染帧）|
  | `preview` | `face` 的别名（同一文件，字节相同）|
  | `thumb` | 缩略图 JPG |
  | `glb` / `obj` / `gs_ply` | 不变 |
  | `video` / `video_noaudio` | **删除** → 404 |

- `POST /api/generate`：multipart，字段 `image`(UploadFile) 或 `sample`(str) **二选一**，都没有 → 400。
  **兼容性要求**：若请求里带了 `motion` / `skip_video` 字段，**接受并忽略**（不得返回 422）。
  返回 `{job_id, state}`。

### 3.3 `GET /api/jobs/{id}/arkit_session`
```json
{"ready": true, "job_id": "abc", "n_gaussians": 20018, "arkit_count": 52,
 "render_hw": [512, 512],
 "center": [x, y, z], "bbox_min": [x,y,z], "bbox_max": [x,y,z],
 "ref_c2w": [16 floats], "ref_intr": [fx, fy, cx, cy], "ref_size": [w, h]}
```
未就绪 → `{"ready": false}`。`ref_c2w` 行主序、**3DGS/NeRF 相机约定**（见 5.3）。

### 3.4 `POST /api/jobs/{id}/arkit_splat`（新增）
- 请求：`application/json`，body `{"arkit": [52 个 0..1 浮点]}`（多余字段忽略）
- 200 → `application/octet-stream`，响应体 = **LSPL v1**（第 4 节）
  - 响应头：`X-LSPL-Count: <N>`、`X-LSPL-Ms: <服务端耗时毫秒, 保留 2 位>`、`Cache-Control: no-store`
- 400 arkit 长度/数值非法（`{"detail": "中文说明"}`）｜404 job 不存在｜409 会话未就绪｜500 渲染失败
- **幂等 / 确定性**：同一 arkit 向量两次请求，响应体**逐字节相同**

---

## 4. LSPL v1 二进制格式（冻结，little-endian）

### 4.1 头部（固定 **144** 字节）
| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | char[4] | magic | `"LSPL"` |
| 4 | uint32 | version | `1` |
| 8 | uint32 | count | N（高斯数）|
| 12 | uint32 | flags | bit0 = has_rgb（=1）|
| 16 | float32[3] | bbox_min | B=0 位姿下 xyz 最小值 |
| 28 | float32[3] | bbox_max | B=0 位姿下 xyz 最大值 |
| 40 | float32[3] | center | 不透明度加权质心 |
| 52 | float32[16] | ref_c2w | **行主序** 4×4，3DGS 约定（列 = right, down, forward）|
| 116 | float32[4] | ref_intr | `[fx, fy, cx, cy]`（像素，与 ref_size 同一坐标系）|
| 132 | float32[2] | ref_size | `[w, h]`（**宽度在前**）|
| 140 | uint8[4] | reserved | `0` |
| **144** | — | payload | 见 4.2 |

### 4.2 payload（全部 float32，C 序连续，长度均 = N，顺序不可变）
| # | 名称 | 形状 | 语义 |
|---|---|---|---|
| 1 | `xyz` | N×3 | LBS 蒙皮后的世界坐标（= `animate_gs_model` 返回的 `gs.xyz`）|
| 2 | `rot` | N×4 | 四元数 **(w, x, y, z)**，已 `F.normalize` |
| 3 | `scale` | N×3 | **已激活的线性尺度**（= `gs.scaling`，满足 `Σ = R·diag(s²)·Rᵀ`）|
| 4 | `opacity` | N×1 | 已激活 alpha ∈ (0,1) |
| 5 | `rgb` | N×3 | 直接显示色 [0,1]（= `gs.shs.squeeze(1)`，**不做任何 gamma 变换**）|

- 总字节 = `144 + N*56`。N=20018 → **1,121,152** 字节。
- 无任何隐式对齐填充。客户端按同一偏移切片。
- `ref_size == [render_w, render_h]`（注意与 pipeline 里的 `render_hw=(h,w)` **顺序相反**，写错会导致画面拉伸）。

---

## 5. `static/splat.js` 对外 API（冻结）

纯 ES module，**只允许** `import * as THREE from 'three'`（由 index.html 既有 importmap 解析）。不得引入其它依赖，不得访问 DOM 全局（除自己创建的元素），不得用 `alert/confirm/window.open`。

```js
export function parseLSPL(buffer: ArrayBuffer): {
  count: number, flags: number,
  bboxMin: Float32Array,   // 长度 3
  bboxMax: Float32Array,   // 长度 3
  center:  Float32Array,   // 长度 3
  refC2W:  Float32Array,   // 长度 16，行主序
  refIntr: Float32Array,   // [fx, fy, cx, cy]
  refSize: Float32Array,   // [w, h]
  xyz: Float32Array, rot: Float32Array, scale: Float32Array,
  opacity: Float32Array, rgb: Float32Array,
}

export class SplatView {
  constructor(host: HTMLElement, opts?: {background?: number, maxPixelRatio?: number})
  canvas: HTMLCanvasElement            // 内部创建，style: display:block; width:100%; height:100%
  camera: THREE.PerspectiveCamera      // fov 38°，交给 app.js 的 OrbitControls
  center(): THREE.Vector3              // 点云质心（世界系）
  frameRadius(): number                // 包围球半径（用于取景）
  setData(parsed): void                // 装载新一帧高斯（内部重建 GPU 缓冲，不重建 renderer/scene）
  setBackground(hex: number): void
  setPixelRatio(r: number): void       // clamp 到 [1, 2]
  setSplatScale(k: number): void       // 全局尺度乘子，默认 1.0，clamp [0.25, 3]
  setOpacityCutoff(a: number): void    // 默认 0.0035
  setCameraPose({position:[3], quaternion:[4], fovY:number, near?:number, far?:number}): void
  refCamera(): {position:number[3], quaternion:number[4], fovY:number}   // 由 ref_c2w/ref_intr 推得
  orbitPreset(name: 'front'|'left'|'right'|'three'|'reset'): void        // 绕中心环绕
  setPaused(b: boolean): void
  resize(): void
  render(): void                       // 自驱 rAF 时不必手动调
  dispose(): void
  stats(): {count, visible, prepareMs, renderMs, fps}
}
```

### 5.1 强制行为
- `parseLSPL` 对 magic/version/长度非法**抛 `Error`**，message 含 `LSPL`。
- **深度排序**：按相机空间深度**由远到近**（back-to-front）排序，配合 `NormalBlending` 实现与 CUDA 光栅器一致的 over 合成。N=20018 时 CPU 侧「投影 + 排序 + 属性写入」≤ 16 ms（`stats().prepareMs`）。
- **2D 协方差投影**（相机为 Three.js 约定：看向自身 -Z，+X 右，+Y 上）：
  ```
  p = 相机空间坐标 (view * x)
  zv = -p.z                       // >0 在前方
  fx = cam.projectionMatrix.elements[0] * 0.5 * canvasW
  fy = cam.projectionMatrix.elements[5] * 0.5 * canvasH
  J = [[ fx/zv,      0,     -fx*p.x/zv² ],
       [   0,    -fy/zv,    fy*p.y/zv² ]]      // y 取负：屏幕 y 向下
  Σ2 = J · Rw · Σ · Rwᵀ · Jᵀ + ε·I ,  ε = 0.3 (px²), Σ = R·diag(s²)·Rᵀ
  ```
- **片元 alpha**：`d = (dx, dy)` 为该片元相对高斯中心的屏幕像素偏移（y 向下），
  `alpha = opacity * exp(-0.5 * dᵀ Σ2⁻¹ d)`；`alpha < cutoff` 时 `discard`。
  输出 `gl_FragColor = vec4(rgb, alpha)`，`NormalBlending`，关闭 `depthTest/depthWrite`。
- **色彩空间**：renderer 必须设 `outputColorSpace = THREE.LinearSRGBColorSpace`（即不做 linear→sRGB 变换），因为 `rgb` 已是显示色。
- **四边形范围**：Σ2 的特征分解，半轴 = `3·sqrt(λ)`（≈3σ），逐实例计算。
- 3 个「几何量」必须经受住 QA 的 `refCamera()` 对位校验（见第 7 节 A7）。

### 5.2 `refCamera()` 的推导（易错，务必照抄）
```
// ref_c2w 为 3DGS 约定：列 = (right, down, forward)
// Three.js 相机：列 = (right, up, backward)，backward = -forward
xAxis =  col0(R)          yAxis_up = -col1(R)      zAxis_back = -col2(R)
quaternion = 由基 (xAxis, yAxis_up, zAxis_back) 构造
position   = col3(R)
fovY       = 2 * atan(refSize[1] / (2 * refIntr[1])) * 180/π
```
`THREE.Matrix4.fromArray` 是**列主序**，而头部里 `ref_c2w` 是**行主序**，必须先转置。

### 5.3 其余
- `orbitPreset('front')`：把相机放到 `refCamera()` 的**方向**上、距离 = `frameRadius()/tan(19°)*1.35`，看向 `center()`（保持 fov 38°）。
- `left` / `right`：绕世界 +Y 轴 ∓60°；`three`：前右 3/4（+40° yaw, +12° pitch）；`reset` == `front`。
- `setData` 必须能在不重建 WebGL 上下文的前提下连续调用（≥ 60 次）。

---

## 6. 前端页面结构（冻结）

### 6.1 `static/index.html`
- 左栏 `aside.panel--input`：
  - 步骤 1「上传图片」：**保留** `#drop` / `#file` / `#preview` / `#samples`
  - 步骤 2「生成 3D 数字人」：仅 `#generate` + `#engineHint`
  - **删除**：`#motions`（动作选择）、`#skipVideo`
  - **新增** `#faceCard`：`<img id="faceImg">` 标题「重建结果」，任务成功后显示
  - **新增** `#arkitPanel`（T5 起）：ARKit 表情调节**卡片**，位于 `#faceCard` 之后、`</aside>` 之前；与「① 上传、② 生成」同栏，形成「上传 → 生成 → 表情调节 / 右侧预览」工作流
- 右栏 `main.panel--result`：
  - **保留** `#progress` / `#empty` / `#errorBox`
  - `#result` → `div.viewer-wrap`：
    - `div#viewer`（`SplatView` 宿主）+ `#viewerBadge`（显示高斯数）
    - `div.viewer__tools`：保留 4 个 `.tool[data-view]`（front/left/right/three）、`#spin`、`#wire`、`#arkitToggle`；**新增** `#splatQuality`（`<select>`：标准 / 高清，默认高清）
    - `#arkitPanel` 位于**左栏输入区**（非画布覆盖层），`#arkitToggle` 仍是唯一开关；结构与语义调整见 6.2
    - **删除**：驱动视频卡片、`#historyWrap`、MP4 下载
  - `div.side`：仅保留
    - `<details id="exportCard">`「导出模型」内含 `#dlGlb` / `#dlObj` / `#dlPly` 三个下载链接
      （**保留理由**：这正是用户要的「生成 3D 人物」产物；默认折叠，不占视觉空间）
    - `#statsCard`（高斯数 / 耗时）
- 静态目录**只允许**存在 `index.html` 一个 html 文件。

### 6.2 `#arkitPanel` 语义变更
- **位置（T5 起）**：面板在**左栏** `aside.panel--input` 内（`position:static;width:100%;flex:none`），**不得覆盖** `.viewer-wrap`：两者矩形**交集面积必须为 0**，且面板高度 ≥ 220px、`#arkitSliders` 有真实高度（防止父级 flex 列容器把面板压扁成 24px 时「交集为 0」的假阳性）；生成完成后面板必须保持 `hidden`（`#arkitToggle` 是唯一入口，自动打开会使 e2e 的「点开关 → 断言可见」变成关闭）；打开时仅额外调一次 `scrollIntoView({block:'nearest'})`。
- 结构保留：`#arkitToggle` / `#arkitPanel` / `#arkitClose` / `#arkitStatus` / `#arkitPresets` / `[data-orbit]` / `#arkitOrbitReadout` / `#arkitSliders` / `#arkitLatency` / `#arkitGaussians`
- `[data-orbit]` 按钮与 `#arkitOrbitReadout` 现在驱动**客户端**相机：调用 `SplatView.orbitPreset(name)`；读数文本 `yaw 0.0° · pitch 0.0° · zoom 1.00` 反映当前相机相对 `center()` 的球坐标。
- ARKit 滑块 / 预设 → `POST /api/jobs/{id}/arkit_splat` → `SplatView.setData(parseLSPL(buf))`。
  **不再使用 `<img id="arkitFrame">`**（可保留 `arkit_frame` PNG 端点供 QA，但 UI 不用）。
- **不再**在打开面板时设置 `viewer.controls.enabled = false`：面板在左栏、不遮挡画布，预览区始终可拖动旋转（用户要求同时调表情 + 360° 拖拽）。

### 6.3 相机交互（保持原样）
- `OrbitControls(view.camera, view.canvas)`：`rotateSpeed 0.85`、`minDistance 0.2`、`maxDistance 4.0`、`dampingFactor 0.07`、`enableDamping true`。
- 仍保留 `#spin`（自转）与 `#wire`（点云边框/尺寸可视化开关，wire 模式可改为仅显示点中心）。

### 6.4 `app.js` 数据契约
- `GET /api/config` → `{samples, gpu, device, max_upload_mb}`（**不再有 motions**）
- `POST /api/generate` → FormData：`image`(File) 或 `sample`(str)
- 轮询 `GET /api/jobs/{id}` → `{id, image_name, state, stage, progress, detail, error, result:{...}}`（**无 motion**）
- `result` 里的下载文件名继续用 `GET /api/jobs/{id}/file/{name}`：`glb` / `obj` / `gs_ply` / `face` / `thumb`
- `GET /api/arkit/meta` → `{names[52], presets, count}`
- `GET /api/jobs/{id}/arkit_session` → 见 3.3
- `POST /api/jobs/{id}/arkit_splat` → body `{arkit: number[52]}`，`res.arrayBuffer()`

---

## 7. 后端文件级改动清单（冻结）

| 文件 | 动作 |
|---|---|
| `lam3d/lam3d/core/splat.py` | **新建**：`encode_lspl(arrays: dict, meta: dict) -> bytes`；布局常量 `LSPL_MAGIC/LSPL_VERSION/LSPL_HEADER_SIZE/LSPL_FLOATS_PER_GAUSSIAN`；header 打包。必须与 `static/splat.js` 的 `parseLSPL` 一一对应。|
| `lam3d/lam3d/core/pipeline.py` | `run(self, image_path, job_dir, on_progress=None, job_id="")`（删 `motion_name`/`skip_video`）；删 `_resolve_motion` / `_encode_video`；第 3 步把 `prepare_motion_seqs` 指向 `job_dir/tracking/export/input`（**1 帧**）；`_infer_and_render` 产出 `frames[1,H,W,3]` + `flame_params`(拟合) + `flame_params_neutral`；写 `face.png` + `thumb.jpg`；`_export_meshes` 保留。|
| `lam3d/lam3d/core/arkit_live.py` | `save_session` 签名不变，内部改用**中性** `flame_data`，额外缓存 `center/bbox/ref_c2w/ref_intr/ref_size`；**新增** `compute_splat(session, arkit) -> dict`（只调 `animate_gs_model`，不光栅化）与 `splat_bytes(session, arkit) -> (bytes, ms)`；`render_frame` 保留。|
| `lam3d/lam3d/core/jobs.py` | 删 `Job.motion`；`submit(image_path, image_name)`；`_execute` 调新 `run`。|
| `lam3d/lam3d/server/app.py` | `/api/config` 去 motions/frames_hint；`POST /api/generate` 新签名（兼容忽略 motion/skip_video）；`file/{name}` 映射更新；`arkit_session` 增字段；**新增** `POST /api/jobs/{job_id}/arkit_splat`。|
| `lam3d/lam3d/core/paths.py` | 删除已无引用的 `MOTION_SEQ_ROOT` / `RENDER_FPS`。|
| `lam3d/README.md` | 同步更新（HTTP 表、模块树、启动方式、ARKit 点云预览说明）。|

**`prepare_motion_seqs` 的复用（禁止自造相机/参数加载逻辑）**：
```python
export_dir = job_dir / "tracking" / "export" / "input"
motion_seq = prepare_motion_seqs(
    str(export_dir).rstrip("/") + "/",      # 末尾斜杠不能省
    None, save_root=str(job_dir / "tmp"), fps=paths.RENDER_FPS,
    bg_color=1.0, aspect_standard=paths.ASPECT_STANDARD,
    enlarge_ratio=[1.0, 1.0], render_image_res=paths.RENDER_SIZE, multiply=16,
    need_mask=False, vis_motion=False, shape_param=shape_param,
    test_sample=False, cross_id=False, src_driven=[image_path.stem, "reference"],
)
motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
n_frames = 1     # 跟踪导出目录只有 1 帧
```

---

## 8. 验收标准（QA 必须全绿才可交付）

**A. 后端**
1. `POST /api/generate`(sample) → succeeded；产物含 `face.png`(512×512 PNG, 非全白, 非背景像素 > 5%)、`thumb.jpg`、`input_head.glb`、`input_gaussians.ply`
2. `file/face` → 200 `image/png`；`file/preview` 与 `face` **字节相同**；`file/video` → 404
3. `GET /api/config` 无 `motions` 键；`GET /api/jobs/{id}` 无 `motion` 键
4. `GET /api/arkit/meta` → count=52、`names[24]==='jawOpen'`
5. `arkit_session` ready=true 且 `center`(3)/`ref_c2w`(16)/`ref_intr`(4)/`ref_size`(2) 齐全
6. `POST arkit_splat`：
   - B=0 → 200，`X-LSPL-Count`=20018，`len(body)` = `144 + 20018*56`
   - header magic/version/count/flags 正确；payload 全 `isfinite`
   - **B=0 两次请求逐字节相同**（幂等）；**B=0 的 xyz/rot/scale 必须等于「不传 arkit」路径**（退化一致性）
   - `jawOpen=1` 与 B=0 的 xyz `mean|Δ| > 1e-5`；`|q|` ∈ [1-1e-5, 1+1e-5]；`s > 0`
   - 全 1 → 无 NaN/Inf，`s ≤ 1.35·s0`
   - 非法长度 → 400
7. **交叉一致性**：用 LSPL 的 `ref_c2w/ref_intr` 在 Python 里把 `xyz` 投影到 `ref_size` 得到不透明度加权包围盒，与 `POST arkit_frame(B=0, size=512)` 的非白像素包围盒 **IoU > 0.5**
8. CUDA 零改动：`find /workspace/LAM/thirdparty_src/diff-gaussian-rasterization -newermt "<本轮开始时刻>"` 为空；`git status` 中 `/workspace/LAM` 仅既有 4 项

**B. 前端（真实无头 Chromium）**
9. 页面只有：上传 / 生成 / ARKit 面板 / 点云画布 / `#faceCard` / `#exportCard`(折叠) / `#statsCard`
10. `#motions`、`#skipVideo`、`#historyWrap`、视频卡片**不存在**；`static/` 下 html 只有 1 个；无 `alert/confirm/window.open`
11. `SplatView` 真实渲染：canvas 像素 ≥ 200 种颜色，覆盖率 > 5%
12. **360° 拖拽**：拖动后 canvas 像素变化且相机位置改变；切到 `left`/`right`/`three` 预设后画面不同
13. ARKit 预设/滑块 → 点云像素变化（**在前景 = 非背景像素上** mean|Δpixel| > 0.5；背景像素按定义不变，纳入全图均值会把信号稀释 ~10x），`#arkitLatency` 出现数值
    - 判定口径（Lead 裁决，本轮）：先取「neutral 帧中与画布背景色差异 > 6 的像素」作为前景掩码，再在掩码内算 mean|Δpixel| 与变化像素占比。
    - 参考量级（512² 参考相机、前景 118030 px）：smile 0.988 / bigSmile 1.673 / jawOpen 5.221 / surprise 7.360 / 全 1 9.975（均 > 0.5），变化前景像素占比 5.4% ~ 44.6%。
14. **清晰度**：`canvas.width === cssW * dpr`（dpr clamp ≤ 2）；高清模式下绘制像素 ≥ 640×640 等效
    - 判定口径（澄清，T5 后补充）：「绘制像素」指 `canvas.width`/`canvas.height`（即绘制缓冲区尺寸），**不是** CSS 盒尺寸。CSS 盒高由 `.viewer-wrap.is-hq{min-height:640px}` 保证，其 `clientHeight` 会因上下各 1px 边框而比 640 少 2px（实测 638），这是正常的。断言请用绘制缓冲区与 `getBoundingClientRect().height`，不要用 `clientHeight` 卡 640。
15. `console` error 0 条、`pageerror` 0 条
16. OrbitControls 保留且参数未被改（damping 0.07、min 0.2、max 4.0）

**C. 性能**
17. `stats().prepareMs ≤ 16`（N=20018）
18. 后端 `X-LSPL-Ms ≤ 30`

---

## 9. 环境与运行

```bash
# 启动（后端改完后必须重启）
cd /workspace/lam3d && bash run.sh --daemon      # http://127.0.0.1:8000
bash run.sh --stop
tail -f /workspace/lam3d/logs/server.log

# Python
/usr/local/miniconda3/envs/lam/bin/python
PYTHONPATH=/workspace/lam3d:/workspace/LAM      # run.sh 已导出

# 无头浏览器（QA）
node v20.5.0；playwright chromium 已安装
--no-sandbox --disable-dev-shm-usage --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader
```

- 生成一张图约 **20~30 s**（去掉视频与多帧渲染后更快）。
- 服务是**单 GPU 串行**队列；QA 跑 e2e 时不要并发跑多个生成任务。
- `rw_exec` 单次调用约 20 s 会超时：长任务必须 `nohup ... &` 后台跑 + 写 flag 文件轮询。

## 10. 上一轮遗留的既有事实（可直接依赖，勿重复探索）
- `arkit_live.sanitize_arkit(raw, count)` 已做长度/NaN/clip 校验；`arkit_meta()` 返回 52 名 + 13 预设。
- `DynamicCovarianceModulator` / `animate_gs_model(..., arkit_params=..., cov_modulation=True)` 已通过 76/76 dry-run（含 B=0 位级一致、B=1 无 NaN、s≤1.35·s0、极分解 T9/T10 与 SVD 对齐）。
- `forward_single_batch(gs_list, c2ws[Nv,4,4], intrs[Nv,4,4], h, w, bg[Nv,3])`；`forward_single_view` 返回 `comp_rgb` 为 `[H,W,3]`。
- `gs.shs` 形状 `[N,1,3]`（`gs_use_rgb: true`）→ `gs.shs.squeeze(1)` 即 RGB。
- 前端既有 importmap：`three` → `/static/vendor/three/build/three.module.js`（r160），`three/addons/` → `examples/jsm`。
- `app.js` 里 ARKit 面板代码（`initArkitPanel` 等 17 个函数）已存在且经 34/34 浏览器 e2e 验证，本轮**改造**而非重写。
