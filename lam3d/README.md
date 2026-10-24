# LAM 单图生成 3D 数字人 · 独立前端

从 [aigc3d/LAM](https://github.com/aigc3d/LAM) 里把**「单图生成 3D 人物」**这条链路单独抽出来，
套了一个自研的 Web 前端：上传一张人脸照片 → 得到**可旋转的 3D 人头模型**（GLB/OBJ/高斯点云）
+ **一张重建后的正面人脸图** + **基于 ARKit 52 通道的实时表情预览**（`LSPL v1` 二进制流）。

LAM 原仓库是一个完整的训练/推理框架，入口是 `lam.launch infer.lam` 命令行或一个 Gradio demo。
本项目把其中的单图分支提取成一个可复用的 Python 包（`lam3d`）+ FastAPI 服务 + Three.js 前端，
**不改动 LAM 的任何文件**。

```
┌──────────────┐   ┌──────────────────────── lam3d 包 ────────────────────────┐   ┌─────────────┐
│  浏览器前端   │   │                                                          │   │  /workspace │
│  Three.js    │◄─►│  FastAPI  ──►  JobManager（单工作线程串行 GPU 队列）        │◄─►│   /LAM      │
│  3D 查看器    │   │                    │                                      │   │  源码+权重   │
└──────────────┘   │                    ▼                                      │   └─────────────┘
                   │          SingleImage3DPipeline                          │
                   │      跟踪 → 参考图 → LAM → 渲染 → 网格 → 人脸图            │
                   │                    │                                      │
                   │                    ▼                                      │
                   │   arkit_live（中性基准缓存）──► LSPL v1 二进制 ──► SplatView │
                   └──────────────────────────────────────────────────────────┘
```

---

## 快速开始

```bash
cd /workspace/lam3d

bash run.sh --daemon        # 后台启动，监听 0.0.0.0:8000
bash run.sh                 # 前台启动
bash run.sh --stop          # 停止

tail -f logs/server.log     # 看日志
```

浏览器打开 `http://<服务器IP>:8000`。

> 启动后模型会**在后台预热**（约 60 秒，2GB 权重常驻显存）。
> 页面左下角会显示「推理引擎已就绪」；预热期间提交的任务会自动排队，不会失败。

---

## 前端长什么样

| 区域 | 功能 |
|---|---|
| 左栏 · 选图 | 点击/拖拽上传、**Ctrl+V 粘贴**、或直接点内置示例缩略图 |
| 左栏 · 生成 | 单次点击即可（**无动作选择、无视频渲染**，全程约 30~50 秒） |
| 右栏 · 进度 | 阶段名 + 百分比 + 明细（跟踪 → 参考图 → 推理渲染 → 网格导出） |
| 右栏 · 3D 查看器 | **拖拽旋转 / 滚轮缩放**，预设视角（正面/左侧/右侧/3-4）、自转、线框；「表情」按钮在**左栏**展开 ARKit 实时表情卡片（见下，不遮挡预览区） |
| 右栏 · 重建人脸 | 第 5 步渲染得到的正面人脸图（`file/face`，`file/preview` 是其别名） |
| 右栏 · 下载 | GLB / OBJ / 高斯点云 PLY / 人脸图 PNG |
| 右栏 · 历史 | 最近 12 次生成，点击即回到结果 |

Three.js 与 OrbitControls / GLTFLoader 都**已内置在 `static/vendor/`**，无需联网。

### ARKit 实时表情（动态协方差调制）

生成完成后点查看器工具条的「表情」按钮，会在**左栏输入区展开一块表情调节卡片**（不新开页面、不覆盖 3D 预览区、不破坏原有相机控制）：

- 52 个 ARKit 通道滑块（0~1，按眉/眼/鼻颊/嘴/下颌分组折叠）+ 13 个预设表情（微笑/大笑/惊讶/生气/伤心/亲吻/张嘴/眨眼/单眼眨眼/鼓腮…）；
- 左转/右转/抬头/低头/放大/缩小/复位 7 个视角按钮（在渲染相机上叠加轨道变换，零增量时与原渲染完全一致）；
- 每次拖动滑块只做**一次前向，不跑光栅化**：后端把 20018 个高斯的 `xyz/rot/scale/opacity/rgb` 打成
  `LSPL v1` 二进制（144 字节头 + 5 段 float32，见 `CONTRACT_v2.md` §4），前端 `SplatView` 在浏览器里
  用三.js 自定义 shader 做各向异性溅射（`Σ₂ = J·Rw·Σ·Rwᵀ·Jᵀ + 0.3px²I`），帧延迟实时显示在面板底部。

会话的基准是**完全中性的 FLAME**（rotation / neck / eyes / jaw / translation / expr 全 0，只保留 `betas`），
所以「ARKit 全 0」= 中性脸，与上传照片的姿态无关；`arkit_live.save_session` 同时缓存
`center / bbox_min / bbox_max / ref_c2w / ref_intr / ref_size` 供 `/api/jobs/{id}/arkit_session` 返回。

背后是 LAM 侧新增的 `lam/models/rendering/covariance_modulation.py`：表情形变不再只搬动高斯均值，而是按「局部形变梯度」解出每个高斯的旋转增量 Δq 与缩放增量 Δs，与预测属性复合成 `q' = normalize(Δq ⊗ q0)`、`s' = s0 + Δs`：

- Δq：对每个三角形取静止/形变两套边坐标，解平面形变梯度 `A = QP⁻¹`，用闭式 2-D 极分解拆出旋转（绕面法线的角度）与面积 Jacobian，再按面积权重做一环平均；
- Δs：`ds = s0 · k · tanh(g · u)`，`u` 为面积 Jacobian 的对数——拉伸区适度放大补空、挤压区适度收缩防叠，`tanh` 软截断保证极值下不出刺状伪影；
- ARKit→FLAME 复用 LAM 自带的 `flame_arkit_bs.npy`（[52,5023,3] 逐顶点偏移基），用 FLAME 自己的 Loop 细分器升采样到 20018 个高斯点，再走 `animation_forward(dynamic_offset=…)` 与表情 blendshape 同一套蒙皮——没有手写映射矩阵，也没改 CUDA 光栅化；
- **全向量化**：只有 `matmul` / `scatter_add` / 广播，没有沿高斯点维度的 Python 循环；
- **退化一致**：ARKit 全 0 时渲染器直接短路，输出与改动前逐比特一致（静态单图重建 100% 不变）。

自检脚本：`scripts/arkit_dryrun.py`（全 0 退化一致性 / 全 1 无 NaN 且缩放有界 / 旋转归一 / 渲染契约 / 耗时 / 向量化审计 / CUDA 未被改动）、
`scripts/arkit_splat_test.py`（LSPL v1 契约验收，`CONTRACT_v2.md` 第 8 节 A1~A8 全项）。

```bash
cd /workspace/lam3d && PYTHONPATH=/workspace/lam3d:/workspace/LAM \
  /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_dryrun.py

# LSPL 契约验收（需服务已启动；C17 prepareMs 需真实浏览器，--no-browser 可跳过）
/usr/local/miniconda3/envs/lam/bin/python scripts/arkit_splat_test.py \
    --base http://127.0.0.1:8000 --sample status.png
```

### TGA：雅可比协方差形变（PGT，NeurIPS'25）

上面那套调制只做「面内旋转 + 各向同性面积缩放」，从**不构造协方差矩阵**。论文
*TGA: True-to-Geometry Avatar Dynamic Reconstruction* 的主张是：按父三角形的形变梯度
`J = Eb E^-1` 直接形变协方差 —— `Sigma' = J Sigma J^T`（Perspective-Aware Gaussian Transformation）。
它替代「按父三角形面积做各向同性缩放」，因此不会在**剪切/旋转但不改面积**的区域把高斯糊成一片
（over-blending），也不会在**只改一个方向**的区域欠融合（under-blending）。

**先踩到的一个坑（本轮已用实验证伪）**：直觉做法是把形变后的 `Sigma'` 直接喂给光栅器。
但官方渲染路径 `lam/models/rendering/gs_renderer.py:526-559` 的 `forward_single_view` 里
`cov3D_precomp` 被写死成 `None`，随后又总是把 `gs.scaling / gs.rotation` 传给
`diff_gaussian_rasterization` —— 光栅器只会用 `(s, R)` 现算 `Sigma = R diag(s^2) R^T`。
所以**外部预计算的协方差会被静默丢弃**，这就是「加了协方差却没有效果」的原因。

实证：`scripts/diag_cov_path.py` 在真实 checkpoint 上做同坐标的 A/B/C 对照 ——
把 `4*Sigma` 挂成 `GaussianModel.cov3D_precomp` 走官方 `forward_single_batch`，输出与
不挂时**逐位相同**（mean/max diff = 0）；而同一条 `Sigma` 用原始 rasterizer 直接喂
`cov3D_precomp`，CUDA 路径确实生效（mean|diff| = 0.067）。⇒ 数学没错，是渲染路径丢了协方差。

**本仓库的做法（不碰 LAM、不碰 CUDA）**：`lam3d/core/tga_covariance.py` 在 lam3d 侧实现
`TgaCovarianceModulator`，通过预置 `renderer._cov_modulator` 注入
（`gs_renderer.py:571-578` 的惰性构造正好留了这个口子）。它把 `Sigma' = J Sigma J^T` 做
**对称特征分解**后精确回写成 `(q', s')` 再交给光栅器 —— 于是 `R' diag(s'^2) R'^T == Sigma'`，
光栅器内部现算出来的就是 TGA 的协方差。契约（`CONTRACT_v2.md` §1/§4）因此完全不变：
从不向渲染器塞 Sigma、LSPL 仍然只传 `xyz/rot/scale/opacity/rgb` 且语义仍是 `Sigma = R diag(s^2) R^T`。

- `J`：每面取静止/形变两套基 `E=[e1,e2,n0]`、`Eb=[d1,d2,n1]`，`J_f = Eb E^-1`；再按**面积加权**
  `scatter_add_` 平均到顶点（sliver 面 `E^-1` 病态，等权会被少数退化面主导）。孤立顶点回落恒等阵。
- 回写：`Sigma'` 在 float64 下 `eigh`；特征值与**升序**的 `s0` 一一配对、逐轴夹在
  `[min_scale_ratio, max_scale_ratio]`（默认 `0.25 / 1.35`，正好覆盖契约 §8 A6 的 `s' <= 1.35*s0`），
  再 scatter 回原下标；特征向量逐列对齐 `R0` 的符号（否则 `R'` 可能差 180 度，`Sigma` 却完全一样），
  兜底 `det<0` 翻最后一列；四元数在 float64 解出再降精度。
- 全向量化：只有 `matmul / eigh / scatter_add_ / 广播`，**没有沿高斯维度的 Python 循环**；
  ARKit 全 0 时短路，输出与改动前逐位一致。

开关（都是环境变量，默认开启）：

```bash
LAM3D_TGA_COV=0          # 退回 LAM 原调制器（对照/回滚）
LAM3D_TGA_STRENGTH=1.0   # J 相对恒等的插值强度（2.0 = 两倍形变）
LAM3D_TGA_MAX_RATIO=1.35 # 逐轴尺度上限（契约 A6）
LAM3D_TGA_MIN_RATIO=0.25 # 逐轴尺度下限
```

验收脚本（真实 checkpoint）：

```bash
cd /workspace/lam3d && PYTHONPATH=/workspace/lam3d:/workspace/LAM \
  /usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_covariance.py  # M1~M6 纯数学，不占 GPU

# 走官方 forward_single_batch 的渲染对照（R1~R7）
/usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_render.py
# 走真实部署链路 save_session -> compute_splat -> LSPL 打包（L1~L10）
/usr/local/miniconda3/envs/lam/bin/python scripts/verify_tga_live.py
# 出对照图（B=0 / LAM / TGA / 差分）
/usr/local/miniconda3/envs/lam/bin/python scripts/make_tga_compare.py
```

`scripts/arkit_dryrun.py` 的 T11 段专门覆盖这套实现：回写保真（`R' diag(s'^2) R'^T == J Sigma J^T`，
相对误差 < 1e-5）、`J = I` 还原、极端 `J` 有界、退化逐位一致、向量化审计（且断言
`tga_covariance.py` 里不出现 `cov3D_precomp`，即绝不回头去注入协方差）。

### 旋转同步（RotSync）与 RSB（法向贴合）

协方差方法在前端「表情」卡片里可切换（`GET /api/cov/methods`），当前共五种：
`lam`（原版面内 2×2 极分解）/ `tga` / `rot_sync` / `vertex_then_polar` / `rsb`。

- **RotSync**（`lam3d/core/rot_sync.py`）：对每面构造完整 3D 形变梯度
  `J_f = [d1 d2 n1][e1 e2 n0]⁻¹`（第三列取**单位**法向，不是 J·e3），
  Newton 极分解出旋转 `R_f`，转四元数后按面积权重做 **Markley 平均**聚合到顶点
  （外积矩阵主特征向量，免半球对齐，副产物 coherence = 1−λ₂/λ₁），
  再过 恒等吸附 → 死区 → 增益 → 45° 限幅+coherence 门控（合并成一次 slerp）。
  `s'` 与 `s0` 逐位相等（只转不拉）。端到端 ~7.2 ms/帧（4090，B=1）。
- **RSB**（`lam3d/core/rsb.py`，方法四）：在 RotSync 之上加一级「顶点法向贴合」——
  从形变网格边直接叉积出**形变后曲面法向** `n_v`（=面法向面积加权和，无需求逆），
  对 `u_v = Rot(δq)·n_{0,v}` 与 `n_v` 的夹角做 `θ_cap·tanh(θ/θ_cap)` 软限幅的最短弧修正
  `δq_fix`，门控 `a = g^1.5` 与 RotSync 共用同一 coherence 门；总旋转 `q' = δq_fix ⊗ δq ⊗ q0`。
  增量 +0.8 ms；法向修正 mean 0.9° / p99 13.7° / 上限 25°（软限幅生效）。

开关（环境变量；也走 `POST /api/cov/method` 动态切换）：

```bash
LAM3D_ROT_SYNC=0            # 关 RotSync 回退 LAM
LAM3D_ROT_SYNC_MAX_ANGLE=45 # 旋转限幅（度）
LAM3D_RSB=1                 # 启用 RSB（需先装 RotSync）
LAM3D_RSB_NORMAL_CAP=25     # 法向修正软限幅（度）
LAM3D_RSB_GATE_P=1.5        # 门控幂次
```

自检：`scripts/check_rot_sync_mesh.py`（网格级 20 项）、`/tmp/w8_assert.py`（RSB 12 项逐位断言，
恒等帧/刚体/门控回退/尺度透传/限幅全覆盖）。文档：`docs/rot_sync_plan.md`（设计）、
`docs/method4_rsb_design.md`（RSB v2.0 + 实测判决）、`docs/rsb_reverify.md`（slerp 半球修复后重验）。

---

## 抽取出来的模块

```
lam3d/
├── lam3d/
│   ├── core/
│   │   ├── paths.py        所有路径常量（改一处即可搬迁）
│   │   ├── lam_env.py      把 LAM 接进 sys.path / cwd，并校验权重
│   │   ├── pipeline.py     ★ 单图 → 3D 人物的完整管线
│   │   ├── arkit_live.py   ★ ARKit 实时表情会话（缓存 canonical 高斯 + LSPL 打包 / 逐帧渲染）
│   │   ├── splat.py        ★ LSPL v1 二进制编码器（144B 头 + 5 段 float32）
│   │   └── jobs.py         单工作线程任务队列 + 进度广播
│   └── server/app.py       FastAPI 路由
├── static/                 前端（index.html / app.css / app.js + vendor）；splat.js = LSPL 解析 + 泼溅渲染
├── scripts/
│   ├── selftest.py         不占 GPU 的自检（仍是动作序列时代的旧版，见「自检」一节）
│   ├── arkit_dryrun.py     ★ ARKit 协方差调用的数值/性能自检
│   ├── arkit_splat_test.py ★ LSPL v1 契约验收（CONTRACT_v2.md §8 A1~A8）
│   ├── e2e_frontend.py     ★ 真实浏览器端到端测试（Playwright）
│   └── verify_glb.mjs      ★ 用 three.js 官方 GLTFLoader 校验 GLB
├── experiments/            开发期的验证脚本（保留作证据，非运行时依赖）
├── run.sh                  启动脚本
└── workspace/jobs/<id>/    每次任务的产物
```

### 当成库用

```python
from lam3d.core.pipeline import SingleImage3DPipeline

pipe = SingleImage3DPipeline(device="cuda")
result = pipe.run(
    "face.jpg", "out_dir",
    on_progress=lambda stage, pct, detail: print(f"{pct:5.1f}% {stage} {detail}"),
)
print(result.glb, result.face)      # 3D 模型 / 重建人脸图
print(result.stats)                 # 高斯点数、耗时…
```

`on_progress` 是**真实进度**（见下文），不是估算。

### 自己写 GLB 而不复用 trimesh

`lam3d/core/glb.py` 是一个约 130 行的 glTF 2.0 二进制作者。不用 `trimesh.export` 的原因
是它的三个行为都会**静默**损坏产物（都是实测踩到的）：

| 问题 | 后果 |
|---|---|
| `process=True` 时合并重复顶点 | `ColorVisuals` 不再匹配，COLOR_0 整块丢失 → 前端全灰 |
| COLOR_0 固定写 `uint8 + normalized` | glTF 的 COLOR_0 是**线性**空间，线性暗部在 8 bit 下精度极差 → 深色区域出现色带 |
| 不自动生成 NORMAL | 缺法线的 glTF 在 three.js 里光照失效 → 模型死黑 |

自研写出器直接写 `float32` 颜色、显式写 `uint32` 索引与 NORMAL，并在导出后回读校验
（属性齐全、顶点数与颜色数一致、索引为 3 的倍数且不越界）。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/config` | 示例图列表、GPU 名称、上传上限 |
| GET | `/api/status` | 引擎是否就绪 / 预热中 / 出错 |
| POST | `/api/generate` | `multipart`：`image`(文件) 或 `sample`(示例名) **二选一**；兼容旧字段 `motion`/`skip_video` 但忽略 |
| GET | `/api/jobs/{id}` | 任务状态与进度（无 `motion` 字段） |
| GET | `/api/jobs` | 历史任务 |
| POST | `/api/jobs/{id}/cancel` | 取消排队中的任务 |
| GET | `/api/jobs/{id}/file/{name}` | 取产物；`name` ∈ `glb` `obj` `gs_ply` `face` `thumb`（`preview` 是 `face` 的别名，字节相同），加 `?download=true` 强制下载 |
| GET | `/api/arkit/meta` | 52 个 ARKit 通道名 + 预设表情（前端滑块的数据源） |
| GET | `/api/jobs/{id}/arkit_session` | 该任务的实时表情会话：`center`/`bbox_min`/`bbox_max`/`ref_c2w`/`ref_intr`/`ref_size`（服务重启后失效） |
| POST | `/api/jobs/{id}/arkit_splat` | JSON `{"arkit":[52 个 0~1]}` → **LSPL v1 二进制**（`application/octet-stream`，头 `X-LSPL-Count` / `X-LSPL-Ms`） |
| POST | `/api/jobs/{id}/arkit_frame` | JSON `{"arkit":[52 个 0~1], "yaw":0, "pitch":0, "zoom":1, "size":320}` → `image/png` 单帧（服务端光栅化，调试用） |

### 上传格式

前端 `static/index.html` 的 `accept="image/png,image/jpeg,image/webp,image/bmp"` 与
上限 **20 MB**；服务端 `ALLOWED_SUFFIX` 与之一致（PNG/JPG/JPEG/WEBP/BMP）。

LAM 内部的 `torchvision.io.read_image` **只认 JPEG/PNG**（webp/bmp/tiff/gif 会抛
`RuntimeError: Unsupported image file...`），所以上传入口按**内容真实格式**（`PIL.Image.format`，
不是后缀）分流：

- 内容确实是 **JPEG / PNG** → **逐字节原样落盘**，不重编码、无画质损失；
- 其他格式（**WEBP/BMP/TIFF/GIF**，或「后缀写 `.png` 但内容是 webp」这类错配）→
  在**上传时**用 `ImageOps.exif_transpose(...).convert("RGB")` 拍正 EXIF 方向后
  **自动转为 PNG**，后缀一并为 `.png`，再进入重建流程。

也就是说：前端可以照常宣传 WEBP，用户在 HTTP 层拿到立刻的明确 400（而不是任务异步
failed）；归一化后的 `input.png` 才是真正喂给 LAM 的那张图。

### LSPL v1（前端泼溅用的二进制格式）

`POST /arkit_splat` 返回的 buffer 就是完整的 `LSPL v1`（小端）：固定 **144 字节头**
（magic `"LSPL"` / version / count / flags / bbox / center / `ref_c2w` / `ref_intr` / `ref_size`）
+ **5 段连续 float32 载荷**（`xyz[N,3]` `rot[N,4]` `scale[N,3]` `opacity[N,1]` `rgb[N,3]`），
总长 `144 + N*56`（N=20018 → 1,121,152 字节），无隐式填充。完整字段表见 `CONTRACT_v2.md` §4。

前端 `static/splat.js` 的 `parseLSPL(buf)` 解析出这些字段，`SplatView` 用自定义 shader
在浏览器里做各向异性泼溅：投影协方差 `Σ₂ = J·Rw·Σ·Rwᵀ·Jᵀ + εI`（ε = 0.3px²），
`alpha = opacity · exp(-0.5 dᵀΣ₂⁻¹d)`，按远到近排序后混合。

同一份 arkit 向量重复请求，返回体**逐字节相同**（幂等）；ARKit 全 0 与「不传 arkit」
的静态路径逐位一致。

`sanitize_arkit` 的入参校验是**长度/有限性报错、值域裁剪**（契约 §3.4 / §10）：
长度 ≠ 52 或含 NaN/Inf → `400`；数值越界（如 `[2.0]*52`）**不报错**，按
`np.clip(arr, 0.0, 1.0)` **裁剪到 1.0** 后正常返回 `200`——前端滑块本就限定 0~1，
裁剪是对越界/浮点误差的容错，不是拒绝。

```bash
# 命令行跑一次
JOB=$(curl -s -X POST http://127.0.0.1:8000/api/generate \
      -F sample=status.png | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
watch -n2 "curl -s http://127.0.0.1:8000/api/jobs/$JOB"
curl -O -J "http://127.0.0.1:8000/api/jobs/$JOB/file/glb?download=true"

# 取一份 LSPL（B=0）
curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"arkit":'"$(python3 -c 'print([0.0]*52)')"'}' \
     -D /tmp/lspl.headers -o /tmp/lspl.bin \
     "http://127.0.0.1:8000/api/jobs/$JOB/arkit_splat"
grep -i x-lspl /tmp/lspl.headers      # X-LSPL-Count / X-LSPL-Ms
```

---

## 抽取时发现并修掉的问题

抽取过程复读了 LAM 的整条推理路径，发现并修掉了六个问题（都在本模块里修正，
**没有改动 LAM 源码**）：

### 1. 带色网格的面索引错配 —— 上游真 Bug

`lam/runners/infer/lam.py:404-413` 导出 `*_textured_mesh.obj` 时：

```python
vtxs  = res['cano_gs_lst'][0].xyz                    # 20018 个点（细分后的空间）
faces = self.model.renderer.flame_model.faces        # 9976 个面（原始的 5023 顶点空间）
mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")
```

实测产物 `exps/cano_gs/status_textured_mesh.obj`：

```
顶点 20018 个，面 9976 个，面的最大顶点索引 = 5023
→ 只有前 5024 个顶点被引用，另外 15000 个细分顶点是死数据
```

结果是**导出的模型丢失了绝大部分形变细节**。本模块改用 `flame_model.faces_up`
（**39904 面 / 20018 顶点**），与高斯点严格一一对应。

### 2. 逐高斯的 `f_dc` 不能直接当顶点色

`gs.shs` 的直流分量是为**体积渲染**优化的逐点颜色：渲染时每个像素是几十个高斯
按不透明度加权混合的结果，所以低不透明度的点即使颜色完全乱掉，画面也看不出来——
网络从来没有被约束去把它们学对。

直接把它当顶点色贴到网格上，脸上会出现明显的**彩色噪点**（实测额头、脸颊一片
红紫斑点）。本模块改为**把顶点投影回 LAM 自己渲染的帧采样**：做 FLAME 动画 →
投影 → 只取正面朝向相机的顶点 → 回读像素色 → 平均。实测单帧覆盖 66.6% 的顶点
（13325/20018），其余用 `f_dc` 兜底。这样拿到的是干净、稳定的反照率。

### 3. 色彩空间：sRGB 被当线性用

glTF 规范规定 `COLOR_0` 必须是**线性**空间，而 LAM 渲染输出的是 sRGB 值。
直接写入会让 three.js / Blender 把 sRGB 当线性用，表现为整体发灰、暗部被压死
（实测近黑顶点占比从 7% 涨到 40%）。本模块在导出前做 sRGB→linear 转换。

### 4. 高斯网格要清理才能看

高斯的 offset 是为体积渲染优化的，相邻点偏移互不约束。把它们用 FLAME 拓扑连成
三角面后有两类问题，渲染时都被高斯混合掩盖，导出成网格才暴露：

- **颈部以下**：LAM 的 FLAME 是 `add_shoulder=False` 构建的，只在「头 + 颈」上
  训练过；颈部以下是外推区，会长出大跨接三角面（实测两头各一片"翅膀"）。
- **全身细刺**：相邻点偏移不连续导致表面自交起刺。

本模块分别用**颈部裁剪**（以 FLAME 模板最低点为基准上抬 0.03）与**拉普拉斯平滑**
（3 次，λ=0.5）处理。实测长边面比例从 5.8% 降到接近 0，且面部细节保留完好。
两者都可用环境变量关闭。

### 5. 进度是黑盒

上游 `infer_single_view` 把渲染循环藏在模型内部，调用方拿不到任何中间状态。
本模块把该循环展开（`_infer_and_render`），逐帧回调并上报阶段与百分比：

```
[render]  50.0%  生成 20018 个 3D 高斯
[render]  88.0%  1/1 帧
```

### 6. `os.path.dirname` 对末尾斜杠的依赖

`prepare_motion_seqs` 用 `os.path.dirname(motion_seqs_dir)` 定位 `transforms.json`，
所以传进去的路径**末尾必须有斜杠**，少了就静默失败（退出码仍为 0）。现在跟踪结果
就落在 `job_dir/tracking/export/input/`，本模块在第 3 步统一补上斜杠后传入，
并复用 LAM 自带的 `_load_pose`（含 OpenGL→3DGS 相机约定转换与 512 尺度内参缩放）。


---

## 性能（RTX 4090 24GB 实测）

| 项目 | 数值 |
|---|---|
| 模型预热（一次性，常驻显存） | 18~22 s（进程冷启动含磁盘读取约 60 s） |
| 人脸跟踪（FLAME 拟合） | 12~15 s |
| LAM 前馈 + 单帧渲染 | ≈36 s（含 20018 高斯生成与大模型前馈） |
| 顶点色烘焙（单帧投影采样） | ≈0.3 s |
| 3D 网格 + 高斯点云导出 | ≈0.1 s |
| **端到端（1 帧，模型已预热）** | **≈50 s** |
| `POST /arkit_splat`（20018 高斯，无光栅化） | **2~7 ms**（冷启首次 ≤ 40 ms，会话创建时已预热） |
| 显存占用 | ≈2.2 GB |
| 输出 | 512×512 人脸图 PNG + GLB + OBJ + 高斯 PLY + LSPL 泼溅流 |

---

## 验证

三层测试，都在本机跑通：

```bash
# 1. ARKit 调制自检（随机初始化，不载权重、不占 GPU 队列）
PYTHONPATH=/workspace/lam3d:/workspace/LAM \
  /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_dryrun.py

# 2. LSPL v1 契约验收（CONTRACT_v2.md §8 A1~A8；需服务已启动）
python scripts/arkit_splat_test.py --base http://127.0.0.1:8000 --sample status.png

# 3. 用 three.js 官方 GLTFLoader 校验产出的 GLB
cd /tmp/glbtest && npm i three@0.160.1
node /workspace/lam3d/scripts/verify_glb.mjs <job>/input_head.glb

# 4. 真实浏览器端到端（上传 → 生成 → 3D 渲染 → 交互）
export LAM3D_CHROME=/opt/chrome/chrome-headless-shell-linux64/chrome-headless-shell
python scripts/e2e_frontend.py http://127.0.0.1:8000
```

第 4 项会真的开一个 Chrome，走完「选图 → 点生成 → 等进度 → 3D 模型加载 → 截图数像素
确认画布真的有东西 → 拖拽/切视角/线框 → 检查历史记录 → 断言无 JS 报错」。

截图落在 `workspace/e2e/`。

> 注意：`gl.readPixels` 在未开 `preserveDrawingBuffer` 时读回来恒为 0
> （合成之后 drawing buffer 已清空），所以画布内容检查走**截图 + PIL 数像素**，
> 而不是 `readPixels`。

---

## 环境要求

沿用 LAM 自己的环境（`/usr/local/miniconda3/envs/lam`），额外只需：

```bash
pip install fastapi uvicorn python-multipart      # LAM 装 Web 栈时已带入
```

本模块用到的其余依赖（`trimesh` / `Pillow` / `omegaconf`）LAM 环境里已有。
视频编码已随动作序列一并移除，**不再依赖 ffmpeg**。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `LAM_ROOT` | `/workspace/LAM` | LAM 检出位置 |
| `LAM3D_PORT` | `8000` | 监听端口 |
| `LAM3D_HOST` | `0.0.0.0` | 监听地址 |
| `LAM3D_DEVICE` | `cuda` | `cuda` / `cpu` |
| `LAM3D_WORK` | `<repo>/workspace` | 产物根目录 |
| `LAM3D_NECK_TRIM` | `0.03` | 颈部裁剪量；设 `0` 关闭 |
| `LAM3D_SMOOTH_ITERS` | `3` | 拉普拉斯平滑次数；设 `0` 关闭 |
| `LAM3D_SMOOTH_LAMBDA` | `0.5` | 平滑强度 |

---

## 自检

```bash
# 不载权重、不占 GPU
PYTHONPATH=/workspace/lam3d:/workspace/LAM \
  /usr/local/miniconda3/envs/lam/bin/python scripts/arkit_dryrun.py
```

校验：ARKit 全 0 与「不传 arkit」的逐位一致性、极值输入无 NaN 且缩放有界、
四元数归一、渲染契约、单次 `arkit_splat` 耗时、向量化审计、CUDA 零改动。

> `paths.RENDER_FPS = 30` 现在只剩两个调用点：`pipeline.py` 的
> `cfg.motion_video_read_fps` 与第 3 步 `prepare_motion_seqs(fps=paths.RENDER_FPS)`
> ——后者是 `CONTRACT_v2.md` §7 要求**逐字照抄**的片段，因此该常量**保留**不删
> （契约表格与片段冲突时以片段为准）。

> `scripts/selftest.py` 仍是**上一版动作序列时代**的脚本（引用已删除的
> `list_motions` / `MOTION_SEQ_ROOT`），本轮后端改动不涉及 `scripts/`，尚未同步。
> 服务侧自检请用上面的 `arkit_dryrun.py` 与 `arkit_splat_test.py`。

## 已知限制

- **只做人头**。LAM 单图分支输出的是 FLAME 头部（含颈部），不含躯干与手。上游的
  Open Avatar Chat 导出（`generateARKITGLBWithBlender.py`）需要 FBX SDK + Blender，
  且强依赖一段固定动作，**未包含在本项目中**——本项目的 GLB 是**静态规范网格 + 顶点色**，
  不含骨骼动画；口型/表情由 ARKit 实时泼溅预览承担。
- **不做骨骼动画导出**。GLB 里没有 skin/joint，所以不能在其他引擎里驱动口型。
  需要离线 ARKit 口型驱动的话，需补齐 FBX SDK + Blender，可直接复用 LAM 的
  `tools/generateARKITGLBWithBlender.py`；在线预览走 `/api/jobs/{id}/arkit_splat`。
- **单工作线程**。GPU 推理串行执行，多人同时用会排队（前端会显示队列长度）。
  `arkit_splat` 只做一次前向、不跑光栅化，与会话共用同一把渲染锁。
- 静态网格**不带贴图 UV**，颜色是逐顶点色。想要贴图可改用上游的
  `texture_type='surface'` 分支（需额外生成 UV 与贴图图集）。
- **模型侧脸比正面弱**。LAM 是单图方法，输入没拍到的角度靠先验补，侧后脑与耳后
  会偏平滑；正面与 3/4 视角效果最好（`assets/sample_input` 里的正脸照都是这个规律）。
- **表情会话不跨重启**。会话缓存的是那次推理的显存对象，服务重启后
  `/api/jobs/{id}/arkit_session` 返回 `{"ready": false}`，需要重新生成。

---

## 演示截图

`workspace/e2e/` 下是端到端测试自动产出的截图：

| 文件 | 内容 |
|---|---|
| `0_hero.png` | 真人照片（Messi）→ 3D 模型，正面视角的完整界面 |
| `0_hero_side.png` | 同一模型的 3/4 / 侧视角 |
| `demo_对比.png` | 输入图 ↔ 第 5 步重建人脸图 ↔ 3D 查看器 三方对比 |
| `demo_多视角.png` | 3D 网格的四视角渲染（正面/侧/3-4/俯） |
| `demo_顶点投影校验.png` | 顶点色烘焙的几何校验：顶点是否准确投到脸上 |
| `2_result.png` / `3_interact.png` | 测试过程中自动截取的界面状态 |
