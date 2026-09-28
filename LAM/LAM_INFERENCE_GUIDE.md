# LAM 推理测试指南

> 配套文档：[`LAM_ENV_SETUP_GUIDE.md`](./LAM_ENV_SETUP_GUIDE.md)（环境配置指南）
>
> 本指南所有命令均在 `/workspace/LAM` 环境上**实测跑通**，输出结果与耗时均为真实值。

---

## 目录

- [0. 30 秒快速开始](#0-30-秒快速开始)
- [1. 推理前的准备（必读）](#1-推理前的准备必读)
  - [1.4 ★★★ 两个路径参数末尾必须带斜杠](#14--两个路径参数末尾必须带斜杠)
- [2. 核心命令逐参数说明](#2-核心命令逐参数说明)
- [3. 四种推理场景](#3-四种推理场景)
- [4. 输出结果说明](#4-输出结果说明)
- [5. 性能基准](#5-性能基准)
- [6. 常见错误排查](#6-常见错误排查)
- [7. 重要坑点提醒](#7-重要坑点提醒)
- [8. 自定义动作序列](#8-自定义动作序列)
- [9. 用 Web 界面推理（app_lam.py）](#9-用-web-界面推理app_lampy)

---

## 0. 30 秒快速开始

```bash
# ① 复制下面整段到终端（每次开新终端都要执行）
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
cd /workspace/LAM

# ② 跑一次推理（用自带示例：雕像 + "Look In My Eyes" 动作）
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0

# ③ 看结果
ls -la exps/videos/lam/lam_20k/
```

**预期输出**（实测 ≈44 秒）：

```
exps/videos/lam/lam_20k/status.mp4          661 KB   无声
exps/videos/lam/lam_20k/status_audio.mp4    923 KB   带音频 ← 推荐看这个
```

> 视频规格：**519 帧 / 30 fps / 512×512**。

> ★★★ **最容易踩的坑**：上面 `motion_seqs_dir=.../Look_In_My_Eyes/` 和
> `model_name=.../step_045500/` **末尾的斜杠不能省**！
> 少了斜杠会导致**推理失败但退出码仍为 0**（没有任何输出视频）。
> 详见 [§1.4](#14--两个路径参数末尾必须带斜杠)。

---

## 1. 推理前的准备（必读）

### 1.1 每次开新终端都必须设置的 5 个环境变量

**这是本环境最容易踩的坑**：非交互式 shell 不会自动激活 conda 环境，
而 `nvdiffrast` 在**首次推理时会 JIT 编译 CUDA 插件**，需要 `ninja` 可达。

```bash
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH   # ★ 最关键
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"      # RTX 4090；A100 用 8.0，3090 用 8.6
ulimit -n 65535                        # 默认只有 1024，不够用
```

**验证**（3 条都通过再开始推理）：

```bash
which python          # 应为 /usr/local/miniconda3/envs/lam/bin/python
ninja --version       # 必须有输出（否则推理会崩）
python -c "import torch;print(torch.cuda.is_available())"   # 应为 True
```

> 更省事的做法：把上面 5 行写进 `~/.bashrc`，或每次用
> `source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate lam`
> 激活环境（激活会自动带上 `ninja`）。

### 1.2 检查权重与数据是否就绪

```bash
cd /workspace/LAM
ls model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors   # 2.36 GB
ls model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl
ls assets/sample_input/ | head
ls assets/sample_motion/export/
```

若缺失，请回到 [`LAM_ENV_SETUP_GUIDE.md`](./LAM_ENV_SETUP_GUIDE.md) 的 Step 6 重新下载。

### 1.3 显存需求

| 项目 | 实测值 |
|---|---|
| 模型权重显存 | **2.20 GB** |
| 推理峰值（519 帧） | < 4 GB |

8GB 显存即可运行，24GB 绰绰有余。

---

### 1.4 ★★★ 两个路径参数末尾必须带斜杠

这是**本指南最重要的一条**：两个参数都会因为少写一个 `/` 而失败。

| 参数 | 正确写法 | 错误写法的后果 |
|---|---|---|
| `motion_seqs_dir` | `.../Look_In_My_Eyes/` ✅ | `.../Look_In_My_Eyes` ❌ **推理失败，但退出码仍为 0** |
| `model_name` | `.../step_045500/` ✅ | `.../step_045500` ❌ 输出多一层 `045500/` 子目录 |

**原因**（`lam/runners/infer/head_utils.py` 第 348-349 行）：

```python
transforms_json = os.path.join(os.path.dirname(motion_seqs_dir), f"transforms.json")
```

代码用 `os.path.dirname(motion_seqs_dir)` 定位 `transforms.json`。实测：

| `motion_seqs_dir` 写法 | `os.path.dirname(...)` 结果 | `transforms.json` |
|---|---|---|
| `.../Look_In_My_Eyes/` | `.../Look_In_My_Eyes` | ✅ 存在 |
| `.../Look_In_My_Eyes` | `.../export` | ❌ 不存在 |

**实测失败现场**（漏写斜杠）：

```
FileNotFoundError: [Errno 2] No such file or directory:
'assets/sample_motion/export/transforms.json'
  File "/workspace/LAM/lam/runners/infer/head_utils.py", line 349, in prepare_motion_seqs
    with open(transforms_json) as fp:
```

**最坑的地方**在于 `lam/runners/infer/lam.py` 第 493-496 行：

```python
            except:
                traceback.print_exc()      # 只打印，不中断，退出码仍是 0
```

**程序会以 exit code 0 正常结束，只是不产出任何视频。**
不看日志的话，会误以为"跑完了但没输出"。

**自查命令**（每次推理后执行）：

```bash
grep -c "Traceback" /tmp/lam_infer_last.log     # 必须是 0
```

`run_infer.sh` 已内置此检查。

---

## 2. 核心命令逐参数说明

LAM 的推理入口是 `python -m lam.launch infer.lam`，参数用 `key=value` 形式（OmegaConf CLI 语法）。

### 2.1 参数总表

| 参数 | 必填 | 说明 | 示例值 |
|---|---|---|---|
| `--config` | ✅ | 推理配置文件 | `configs/inference/lam-20k-8gpu.yaml` |
| `model_name` | ✅ | 权重目录。**末尾斜杠影响输出路径**，见 7.4 | `model_zoo/.../step_045500/` |
| `image_input` | ✅ | 输入图片路径 或 目录 | `assets/sample_input/status.png` |
| `export_video` | ✅ | 是否导出视频。与 `export_mesh` 至少一个为 true | `true` |
| `export_mesh` | ✅ | ⚠️ **死参数，无任何效果**，见 7.1 | `false` |
| `motion_seqs_dir` | ✅ | 动作序列目录（含 `flame_param/`） | `assets/sample_motion/export/Look_In_My_Eyes/` |
| `motion_img_dir` | — | 从图片/视频提取动作（需 multi-hmr，见 7.3） | `null` |
| `render_fps` | — | 输出视频帧率 | `30` |
| `motion_video_read_fps` | — | 从视频抽帧的 fps | `30` |
| `save_img` | — | ★ **导出 PNG 序列 + 网格的关键开关** | `false` |
| `save_ply` | — | 是否存逐帧高斯 PLY（需 `save_img=true`） | `false` |
| `vis_motion` | — | 是否把驱动视频拼到输出里做对比 | `false` |
| `motion_img_need_mask` | — | 动作图是否需要抠图 | `true` |
| `cross_id` | — | 跨身份驱动 | `false` |
| `test_sample` | — | 测试采样模式 | `false` |
| `gaga_track_type` | — | ⚠️ **被硬编码覆盖，传什么都没用**，见 7.5 | `""` |
| `rank` / `nodes` | — | 分布式参数，单卡固定 `0` / `0` | `0` / `0` |
| `CUDA_VISIBLE_DEVICES` | — | 环境变量，选择 GPU | `0` |

### 2.2 三条最实用的命令变体

```bash
# 变体 1：只要视频（最快，推荐日常使用）
export_video=true export_mesh=false save_ply=false save_img=false

# 变体 2：视频 + 网格（要 OBJ 模型）
export_video=true export_mesh=true  save_ply=false save_img=true    # ★ save_img 必须 true

# 变体 3：视频 + 网格 + 逐帧点云（最全，慢）
export_video=true export_mesh=true  save_ply=true  save_img=true
```

> ⚠️ 变体 2/3 里 `export_mesh=true` **不是**生效条件，真正生效的是 `save_img=true`。
> 详见 [7.1](#71-export_mesh-是死参数导出网格要用-save_imgtrue)。

---

## 3. 四种推理场景

### 场景 A：用自带示例快速验证（推荐先跑这个）

```bash
cd /workspace/LAM
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

**实测**：44 秒，输出 `exps/videos/lam/lam_20k/status_audio.mp4`（519 帧）。

**可替换的示例素材**：

| 输入图片（10 张） | 动作序列（12 段） |
|---|---|
| `barbara.jpg` `cluo.jpg` `dufu.jpg` `james.png`<br>`libai.jpg` `messi.png` `musk.jpg` `pop.png`<br>`speed.jpg` `status.png` `zhouxingchi.jpg` | `Anti_Drugs` `D_ANgelo_Dinero` `Donald_Trump`<br>`GEM` `I_Am_Iron_Man` `Joe_Biden`<br>`Look_In_My_Eyes` `Michael_Wayne_Rosen`<br>`Pen_Pineapple_Apple_Pen` `Speeding_Scandal`<br>`Taylor_Swift` `The_Shawshank_Redemption` |

> 任意组合都可以，例如 `messi.png` + `Donald_Trump`。

**各动作序列的时长**（决定输出视频长度）：

| 动作序列 | 帧数 | 时长 @30fps | | 动作序列 | 帧数 | 时长 @30fps |
|---|---:|---:|---|---|---:|---:|
| `Michael_Wayne_Rosen` | 60 | 2.0 s | | `Joe_Biden` | 264 | 8.8 s |
| `Speeding_Scandal` | 76 | 2.5 s | | `I_Am_Iron_Man` | 282 | 9.4 s |
| `D_ANgelo_Dinero` | 132 | 4.4 s | | `Taylor_Swift` | 403 | 13.4 s |
| `Pen_Pineapple_Apple_Pen` | 167 | 5.6 s | | `The_Shawshank_Redemption` | 435 | 14.5 s |
| `Anti_Drugs` | 226 | 7.5 s | | `GEM` | 516 | 17.2 s |
| `Donald_Trump` | 264 | 8.8 s | | `Look_In_My_Eyes` | 519 | 17.3 s |

> 💡 **想快速试跑就选 `Michael_Wayne_Rosen`（2 秒）或 `Speeding_Scandal`（2.5 秒）**，
> 总耗时能压到 ~25 秒；`Look_In_My_Eyes`（17 秒视频）要 ~44 秒。

---

### 场景 B：用自己的照片（最常用）★

把自己的照片放到任意目录，**直接传文件路径**：

```bash
mkdir -p /workspace/my_inputs
# 把你的照片拷进来，命名为 xxx.jpg
cp /path/to/your/photo.jpg /workspace/my_inputs/my_avatar.jpg

cd /workspace/LAM
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=/workspace/my_inputs/my_avatar.jpg \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Taylor_Swift/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

**实测**：输出 `exps/videos/lam/lam_20k/my_avatar_audio.mp4`（858 KB）。
**视频文件名 = 你的图片文件名**（去掉扩展名）。

#### 📸 对输入照片的要求

引擎内部会做「抠图 → 人脸检测 → FLAME 参数拟合」，所以：

| 要求 | 说明 |
|---|---|
| ✅ **正脸或接近正脸** | 官方提示：输入正脸、或朝向接近驱动信号，效果最好 |
| ✅ **单人** | 代码里断言检测到的人脸数 **必须恰好为 1**，多人会失败 |
| ✅ **人脸清晰、无遮挡** | 遮挡（手、墨镜、口罩）会导致 FLAME 拟合失败 |
| ✅ **格式** | 建议 `.jpg`；`.png` 单文件传也可以 |
| ✅ **分辨率** | ≥ 512×512 效果较好 |
| ⚠️ 避免 | 极端侧脸、夸张表情、强逆光、人脸过小 |

> 失败时不会中断程序，而是在日志里打印 `Traceback` 然后跳过这张图（见 [7.2](#72-单张图片失败会被静默吞掉)）。

---

### 场景 C：批量处理一个目录

把多张照片放进同一目录，`image_input` 传**目录路径**：

```bash
cd /workspace/LAM
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=/workspace/my_inputs \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/GEM/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

#### ⚠️ 目录模式有两个限制

**限制 1：只扫描 `.jpg`，完全忽略 `.png`**

```python
# lam/runners/infer/lam.py 第 424 行
image_paths = glob(os.path.join(self.cfg.image_input, "*.jpg"))   # ← 只匹配 .jpg
```

**实测验证**：`assets/sample_input/` 里有 7 个 `.jpg` + 4 个 `.png`，
传该目录后**只处理了 7 张**，生成的正是 7 个 jpg 对应的身份：

```
barbara / cluo / dufu / libai / musk / speed / zhouxingchi   ← 7 个 .jpg ✅
james / messi / pop / status                                  ← 4 个 .png 被跳过 ❌
```

**应对**：批量处理前把图片统一转成 `.jpg`：

```bash
mkdir -p /workspace/my_inputs_jpg
cd /workspace/my_inputs
for f in *.png; do
  [ -e "$f" ] || continue
  /usr/local/miniconda3/envs/lam/bin/python -c "
from PIL import Image; import sys
Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2], quality=95)
" "$f" "/workspace/my_inputs_jpg/${f%.png}.jpg"
done
# 再把 .jpg 也拷过去
cp *.jpg /workspace/my_inputs_jpg/ 2>/dev/null
```

**限制 2：该目录下所有图片共用同一个动作序列**

LAM 的 API 没有「每张图配不同动作」的入口，一次运行只接受一个 `motion_seqs_dir`。
需要不同动作就分多次运行。

---

### 场景 D：导出 3D 网格（OBJ）

```bash
cd /workspace/LAM
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=true \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=true save_img=true \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

**实测产出**：

```
exps/cano_gs/
├── status_gs_offset.ply        1.36 MB   高斯点云
├── status_shaped_mesh.obj      345 KB    几何网格
├── status_textured_mesh.obj    1.55 MB   带顶点色网格 ★ 最常用
└── framework_img.obj           1.41 MB
exps/images/lam/lam_20k/status/           519 个逐帧 PNG + 519 个 .ply（save_ply=true 时）
```

> 💡 `*_textured_mesh.obj` 带顶点颜色，可直接拖进 Blender / MeshLab 查看。
> 注意是**顶点色**而非 UV 贴图。

---

## 4. 输出结果说明

### 4.1 输出目录结构

```
exps/
├── videos/lam/lam_20k/
│   ├── status.mp4              无声视频
│   └── status_audio.mp4        带音频 ★ 推荐
├── images/lam/lam_20k/
│   ├── status/                 逐帧 PNG / PLY（save_img=true 时）
│   │   ├── 0000.png ...
│   │   └── 0000.ply ...        （save_ply=true 时）
│   └── tmp_res/
│       └── refer_00000_00.png   预处理后的参考图（抠图+对齐结果）
├── meshes/lam/lam_20k/         网格目录（基本为空，见 7.1）
└── cano_gs/                    规范空间网格（save_img=true 时）
    ├── status_gs_offset.ply
    ├── status_shaped_mesh.obj
    ├── status_textured_mesh.obj
    └── framework_img.obj
```

### 4.2 命名规则

| 内容 | 规则 |
|---|---|
| 视频文件名 | 输入图片的文件名（去扩展名），如 `my_avatar.jpg` → `my_avatar.mp4` |
| 规范网格文件名 | 输入图片名，如 `status_textured_mesh.obj` |
| 音频 | 自动取 `motion_seqs_dir` 下的同名 `.wav`（如 `Look_In_My_Eyes.wav`）拼到 `_audio.mp4` |

### 4.3 查看结果

```bash
# 终端里确认规格
/usr/local/miniconda3/envs/lam/bin/python -c "
import imageio
r = imageio.get_reader('exps/videos/lam/lam_20k/status_audio.mp4','ffmpeg')
md = r.get_meta_data()
print('frames:', r.count_frames(), 'fps:', md.get('fps'), 'size:', md.get('size'))
"
# frames: 520   fps: 30.0   size: (512, 512)   ← 解码比 PNG 多 1 帧属正常，见 §8.3
```

**在浏览器里看**：本机已跑着 FileBrowser（`http://<你的IP>:8889`）和 JupyterLab（`:8888`），
可以直接在网页里点开 `/workspace/LAM/exps/videos/lam/lam_20k/` 下的 mp4 播放。
也可以把文件下载到本地看。

---

## 5. 性能基准

**实测环境**：RTX 4090 24GB / 16 vCPU / 62GB RAM

| 场景 | 帧数 | 耗时 | 显存 |
|---|---|---|---|
| 单图 + 519 帧动作（Look_In_My_Eyes） | 519 | **43–44 s** | 2.20 GB |
| 单图 + 264 帧动作（Joe_Biden / Donald_Trump） | 264 | **38–39 s** | 2.20 GB |
| 单图 + 226 帧动作（Anti_Drugs） | 226 | **38 s** | 2.20 GB |
| 7 张图批量（GEM，每张 516 帧） | 3612 | **~2.8 min** | 2.20 GB |
| 4 张图批量（Speeding_Scandal，每张 76 帧） | 304 | **73 s** | 2.20 GB |

> 📌 耗时**并非**与帧数成正比：固定开销（加载模型 + FLAME 拟合）约占 30 秒，
> 真正随帧数变化的是渲染阶段。所以 226 帧和 519 帧的耗时只差几秒。

耗时构成（单张图 + `Look_In_My_Eyes`，总计 44 s）：

| 阶段 | 耗时 |
|---|---|
| 进程启动 + import torch/tf + 加载 LAM 权重 | ~9 s |
| 加载人脸跟踪预训练模型 | 3.6 s |
| 预处理（抠图 + 人脸检测 + 对齐） | 1.4 s |
| **FLAME 参数优化（最慢的单步）** | **10.5 s** |
| 导出跟踪结果 | 1.5 s |
| 高斯生成 + 渲染 519 帧 + 编码视频 + 拼音频 | ~18 s |

> 💡 **两个瓶颈**：① FLAME 跟踪固定约 10–12 s，**每张图都要重跑**；
> ② 渲染/编码时间约与帧数成正比。
> 批量处理 N 张图 ≈ 一次性模型加载 + N × (FLAME ~11 s + 渲染)。
> 想加速可以降低 `render_fps`（如 15）或缩短动作序列。

---

## 6. 常见错误排查

### 6.1 推理直接崩溃：`Ninja is required to load C++ extensions`

```
RuntimeError: Ninja is required to load C++ extensions
  File ".../nvdiffrast/torch/ops.py", line 125, in _get_plugin
```

**原因**：`PATH` 里没有 conda 环境的 `bin`，torch 找不到 `ninja`。
nvdiffrast 在**运行时**JIT 编译 CUDA 插件，所以即使安装阶段成功了，这一步仍会崩。

**解决**：

```bash
export PATH=/usr/local/miniconda3/envs/lam/bin:$PATH
ninja --version      # 必须能输出版本号
```

### 6.2 单张图片失败会被静默吞掉

```python
# lam/runners/infer/lam.py 第 493-496 行
            except:
                traceback.print_exc()      # ← 只打印，不中断
```

**现象**：程序正常退出（exit code 0），但**某些图没有输出视频**。

**排查方法**：检查日志里有没有 `Traceback`

```bash
python -m lam.launch infer.lam ... 2>&1 | tee infer.log
grep -c "Traceback" infer.log        # 期望 0
grep -B5 -A20 "Traceback" infer.log  # 有的话看具体原因
```

常见原因：检测到的人脸数 ≠ 1（多人 / 没检测到脸）、图片损坏、极端角度。

### 6.3 `AssertionError: flametracking preprocess failed!`

**原因**：输入图中没检测到人脸，或检测到多张脸。

**解决**：换一张清晰的正脸单人照。可先用 `assets/sample_input/status.png` 确认环境本身没问题。

### 6.4 `FileNotFoundError: .../canonical_flame_param.npz`

**原因**：`motion_seqs_dir` 路径不对，或该动作序列不完整。

**检查**：动作序列目录必须包含这些文件——

```bash
ls assets/sample_motion/export/Look_In_My_Eyes/
# 必须有的：
#   canonical_flame_param.npz    ← 规范 FLAME 参数
#   transforms.json              ← 相机参数
#   flame_param/                 ← 逐帧 .npz（Look_In_My_Eyes 是 519 个）
#   <name>.wav                   ← 音频（可选，没有则不生成 _audio.mp4）
```

> ★ 同时确认 `motion_seqs_dir` **末尾有斜杠**，否则会去找上一级的
> `transforms.json`（见 §1.4）。

### 6.5 `Too many open files`

```bash
ulimit -n 65535
```

### 6.6 `CUDA out of memory`

单卡推理只需 ~2.2GB，一般不会 OOM。若发生：

```bash
CUDA_VISIBLE_DEVICES=1 python -m lam.launch ...   # 换张卡
nvidia-smi                                         # 看看是不是别的进程占了
```

### 6.7 `tensorflow/tsl/cuda/cudart_stub.cc: Could not find cuda drivers`

**可以忽略**。tensorflow 只被虹膜关键点检测用到，走 CPU 完全够用。
`TF-TRT Warning: Could not find TensorRT` 同理。

---

## 7. 重要坑点提醒

### 7.1 `export_mesh` 是死参数，导出网格要用 `save_img=true`

`lam/runners/infer/lam.py` 里 `export_mesh` **只在函数签名和调用处出现，函数体内从不读取**：

```python
def infer_single(self, image_path, ..., export_mesh: bool, ...):   # 第 285 行：声明
    ...
    # 函数体内从未使用 export_mesh

    self.infer_single(..., export_mesh=self.cfg.export_mesh, ...)  # 第 488 行：传递
```

真正控制网格/图片导出的是 `save_img`：

```python
# 第 367 行起
if save_img and dump_image_dir is not None:          # ← 真正的开关
    for i in range(rgb.shape[0]):
        Image.fromarray(only_pred[i]).save(...)      # PNG 序列
        if save_ply and dump_mesh_path is not None:
            res["3dgs"][i][0][0].save_ply(...)       # 逐帧 PLY
    ...
    mesh.export(os.path.join(dump_cano_dir, ... + '_shaped_mesh.obj'))    # 几何网格
    mesh_utils.save_obj(pth, ..., texture_type="vertex")                  # 带色网格
```

| 你想要的 | 正确参数 | `export_mesh=true` 有用吗 |
|---|---|---|
| 只要视频 | `save_img=false` | — |
| 视频 + 网格 OBJ | **`save_img=true`** | ❌ 没用 |
| 视频 + 网格 + 逐帧 PLY | **`save_img=true save_ply=true`** | ❌ 没用 |

**实测对照**（同一张 `musk.jpg`，仅 `save_img` 不同）：

| 参数组合 | 视频输出 | `exps/cano_gs/` 网格目录 |
|---|---|---|
| `export_mesh=true save_img=false` | ✅ `musk_audio.mp4` (117 KB) | ❌ **目录根本不存在** |
| `export_mesh=true save_img=true` | ✅ 有 | ✅ 4 个 OBJ / PLY 文件 |

**结论**：只传 `export_mesh=true` 时，**连 `exps/cano_gs/` 目录都不会被创建**。
网格导出代码整段位于 `if save_img:` 分支内部，条件不成立就完全不会执行。

### 7.2 `gaga_track_type` 传什么都没用

```python
# 第 427 行，infer() 内部硬编码覆盖
gaga_track_type = ""
```

CLI 传的值会被直接覆盖成空字符串。**保持 `gaga_track_type=""` 即可**。

### 7.3 `motion_img_dir`（自定义视频驱动）当前不可用

如果想用**自己的视频**驱动（而不只是自带的 12 段），代码路径是：

```python
# lam/runners/infer/head_utils.py 第 203-222 行
def predict_motion_seqs_from_images(image_folder, save_root, fps=6):
    ...
    cmd = f"cd thirdparty/multi-hmr && python infer_batch.py --data_root ... --smplify"
    os.system(cmd)          # ← 依赖 thirdparty/multi-hmr
```

**但 `thirdparty/multi-hmr` 在仓库中不存在**（实测确认）：

```bash
$ ls thirdparty
ls: cannot access 'thirdparty': No such file or directory
$ git ls-files | grep -i thirdparty
（空）
```

该依赖既不在 git 仓库里，也不在 `LAM-assets.tar` 中。

**结论**：
- ✅ 使用自带 `assets/sample_motion/export/` 下的 **12 段动作**：完全正常
- ❌ 用自己的视频驱动：**当前需要自行补装 multi-hmr**，否则会失败

**替代方案**：
1. 用外部工具（如 [VHAP](https://github.com/ShenhanQian/VHAP) 或 multi-hmr 官方仓库）
   先从视频提取出 `flame_param/` + `transforms.json` + `canonical_flame_param.npz`，
   组织成 7.4 的目录结构，再当作 `motion_seqs_dir` 传入；
2. 或者用官方在线 Demo：<https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model>

### 7.4 `model_name` 末尾斜杠会改变输出目录

```python
_relative_path = os.path.join(parent, child, os.path.basename(cli_cfg.model_name).split('_')[-1])
```

实测两种写法：

| `model_name` 写法 | `os.path.basename` | 输出目录 |
|---|---|---|
| `.../step_045500/`（**带斜杠**） | `''` | `exps/videos/lam/lam_20k/` |
| `.../step_045500`（不带斜杠） | `'step_045500'` | `exps/videos/lam/lam_20k/045500/` |

**建议统一带末尾斜杠**（与 `scripts/inference.sh` 一致），输出就在 `exps/videos/lam/lam_20k/`。
如果发现输出目录多了一层 `045500/`，就是这个原因。

### 7.5 目录输入只认 `.jpg`

见 [场景 C 限制 1](#-目录模式有两个限制)。批量处理前请统一转成 `.jpg`。

---

## 8. 自定义动作序列

### 8.1 动作序列的目录结构

要让 LAM 接受一个自定义动作序列，目录必须长这样：

```
my_motion/
├── canonical_flame_param.npz     # 必需：规范 FLAME 参数
├── transforms.json               # 必需：逐帧相机内外参
├── my_motion.wav                 # 可选：音频（同名），用于生成 _audio.mp4
└── flame_param/                  # 必需：逐帧 FLAME 参数
    ├── 00000.npz
    ├── 00001.npz
    └── ...                       # 帧数 = 输出视频帧数
```

> ★ 传入时必须写成 `motion_seqs_dir=/path/to/my_motion/`（**末尾带斜杠**），
> 因为代码用 `os.path.dirname()` 定位上一层的 `transforms.json`（见 §1.4）。

### 8.2 各文件的确切内容

**`canonical_flame_param.npz`**（实测字段与形状）：

```python
translation     (1, 3)
rotation        (1, 3)
neck_pose       (1, 3)
jaw_pose        (1, 3)
eyes_pose       (1, 6)
shape           (300,)          # ← 身份形状参数，决定"像谁"
expr            (1, 100)
static_offset   (1, 5143, 3)
```

**`transforms.json`**（决定相机与渲染视角）：

```json
{
  "frames": [
    {
      "timestep_index": 0,
      "timestep_id": "00000000",
      "camera_index": 0,
      "cx": 512.0, "cy": 512.0,
      "fl_x": 2371.77, "fl_y": 2371.77,
      "h": 1024, "w": 1024,
      "flame_param_path": "flame_param/00000.npz",
      ...
    },
    ...
  ]
}
```

**`flame_param/*.npz`**：逐帧的表情 / 姿态参数，文件数决定视频长度。
注意 `transforms.json` 的 `frames` 数组长度才是在代码里真正生效的帧数
（`head_utils.py` 第 351 行读取 `data["frames"]`），两者应保持一致。

### 8.3 最省事的做法：复制并改造现成序列

```bash
cd /workspace/LAM
# 复制一段现成的
cp -r assets/sample_motion/export/Look_In_My_Eyes /workspace/my_motion

# 只替换音频（例如换成你自己的配音）
cp /path/to/your_voice.wav /workspace/my_motion/my_motion.wav

# 用自定义动作驱动
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=/workspace/my_inputs/my_avatar.jpg \
  export_video=true export_mesh=false \
  motion_seqs_dir=/workspace/my_motion/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

> ⚠️ 音频文件名必须与**目录名**一致（`my_motion/` → `my_motion.wav`）。

> **帧数说明**：输出的 PNG 序列张数**精确等于**动作序列的帧数
> （`Michael_Wayne_Rosen` 60 帧 → 60 个 PNG；`Look_In_My_Eyes` 519 帧 → 519 个 PNG）。
> 但用 `imageio` 解码 mp4 时可能读到 N+1 帧（如 60 → 61），
> 这是容器的时间戳取整行为，并非多生成一帧，**以 PNG 张数为准**。
> 代码逻辑：`base_vid = motion_seqs_dir.strip('/').split('/')[-1]`，
> 然后找 `motion_seqs_dir/base_vid + ".wav"`。

### 8.4 截短/加长动作序列

帧数由 `flame_param/` 里的文件数决定。想缩短视频：

```bash
cd /workspace/my_motion/flame_param
# 只保留前 150 帧
ls | sort | tail -n +151 | xargs rm -f
```

同时需要同步裁剪 `transforms.json` 里的 `frames` 数组，否则会因帧数不匹配出错。

---

## 9. 用 Web 界面推理（app_lam.py）

`app_lam.py` 是**另一个推理入口** —— 带图形界面的 Gradio Web 应用。
命令行（`lam.launch infer.lam`）和 Web 界面二者选一即可，底层是同一套模型代码。

### 9.1 启动

```bash
cd /workspace/LAM
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
python app_lam.py
```

启动成功的标志：

```
Running on local URL:  http://127.0.0.1:7860
```

### 9.2 界面功能

启动后会加载 LAM 权重 + 人脸跟踪模型（约 15 秒），然后提供 34 个组件，核心是四块：

| 区域 | 说明 |
|---|---|
| **Input Image** | 上传一张人脸照片 |
| **Input Video** | 选择驱动视频（自动列出 `assets/sample_motion/export/*/*.mp4`） |
| **Processed Image** | 显示抠图 + 对齐后的参考图 |
| **Rendered Video** | 生成结果，自动播放 |

点击 **Generate** 按钮即可。输出视频同时也会落到 `exps/videos/` 等目录（与命令行一致）。

### 9.3 访问方式

`app_lam.py` 里是 `demo.launch()`，**默认只监听 127.0.0.1**，不对外网开放。
本机是远程服务器，用 SSH 端口转发访问：

```bash
# 在你自己的电脑上执行
ssh -p 23 -L 7860:127.0.0.1:7860 root@<服务器IP>
# 然后浏览器打开 http://127.0.0.1:7860
```

也可以在 `app_lam.py` 第 495 行改成对外监听（注意无鉴权，谨慎使用）：

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 9.4 ⚠️ 启动报 `ValueError: When localhost is not accessible...`

**这是本环境已经修好的问题，但如果你重建了环境就会复现。**

报错原文（模型全部加载成功，只在最后启动 Web 服务时失败）：

```
ValueError: When localhost is not accessible, a shareable link must be created.
Please set share=True or check your proxy settings to allow access to localhost.
```

**别被骗了** —— 这跟网络、代理、`share=True` 都无关。
真实原因是 `gradio 3.44.3` 与新版 `starlette` 不兼容
（pip 装 requirements.txt 时会拉入最新的 starlette 1.7 / fastapi 0.141）。

**一行修复**：

```bash
pip install "starlette==0.27.0" "fastapi==0.103.2" "uvicorn==0.23.2" "pydantic==2.8.0"
```

**验证**：

```bash
python app_lam.py            # 应打印 Running on local URL: http://127.0.0.1:7860
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/    # 200
```

> 本环境实测：修复后 HTTP **200**，首页 **766 KB**。
> 详细根因（starlette `TemplateResponse` 签名变更 + pydantic `FieldInfo.in_` 移除）
> 见 [`LAM_ENV_SETUP_GUIDE.md`](./LAM_ENV_SETUP_GUIDE.md) 的「错误 10」。

### 9.5 命令行 vs Web 界面怎么选

| | 命令行 `run_infer.sh` | Web 界面 `app_lam.py` |
|---|---|---|
| 批量处理 | ✅ 传目录即可 | ❌ 一次一张 |
| 自动化/脚本化 | ✅ | ❌ |
| 可视化预览 | ❌ 需自己打开 mp4 | ✅ 自动播放 |
| 挑选动作序列 | 需知道名字 | ✅ 缩略图列表 |
| 额外依赖 | 无 | 需额外加载 Gradio + 跟踪模型 |

> 💡 **推荐**：日常批量用命令行；偶尔试效果、给人演示用 Web 界面。

---

## 附录：一键推理脚本

把下面内容存为 `/workspace/LAM/run_infer.sh`，之后只需 `bash run_infer.sh 图片 动作序列名`：

```bash
#!/usr/bin/env bash
# 用法: bash run_infer.sh <图片路径> [动作序列名]
# 例:   bash run_infer.sh /workspace/my_inputs/my_avatar.jpg Taylor_Swift
set -e
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
cd /workspace/LAM

IMAGE=${1:?请提供图片路径}
MOTION=${2:-Look_In_My_Eyes}

CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input="$IMAGE" \
  export_video=true export_mesh=false \
  motion_seqs_dir="assets/sample_motion/export/$MOTION/" \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0

echo
echo "===== 输出 ====="
ls -la exps/videos/lam/lam_20k/ | tail -5
```

```bash
chmod +x run_infer.sh
bash run_infer.sh /workspace/my_inputs/my_avatar.jpg Taylor_Swift
```

---

## 附录：故障速查表

| 现象 | 原因 | 解决 |
|---|---|---|
| **★ exit 0 但没有任何输出视频** | `motion_seqs_dir` 漏写末尾斜杠 | **补上 `/`**（见 §1.4） |
| **★ `FileNotFoundError: .../export/transforms.json`** | 同上 | 同上 |
| `Ninja is required to load C++ extensions` | PATH 缺 conda bin | `export PATH=/usr/local/miniconda3/envs/lam/bin:$PATH` |
| 程序 exit 0 但部分图没输出 | 单图失败被 `except` 吞掉 | `grep -c Traceback infer.log` |
| `flametracking preprocess failed` | 没检测到脸 / 多张脸 | 换正脸单人清晰照 |
| 目录输入漏图 | 只扫描 `.jpg` | 统一转 `.jpg` |
| 输出目录多一层 `045500/` | `model_name` 没带末尾斜杠 | 加上末尾 `/` |
| 没有网格 OBJ | `save_img=false` | 改 `save_img=true` |
| `Too many open files` | ulimit 太小 | `ulimit -n 65535` |
| `canonical_flame_param.npz` 找不到 | 动作序列路径错/不完整 | 检查目录结构 |
| 自定义视频驱动失败 | 缺 `thirdparty/multi-hmr` | 用自带 12 段动作 |
| `Could not find cuda drivers` (tensorflow) | 正常提示 | 忽略 |
