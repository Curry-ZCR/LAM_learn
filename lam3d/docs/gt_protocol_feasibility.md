# ARKit 协方差调制：真实照片级 GT 可行性侦察报告

> 侦察人：gt-scout｜目标：为四种 ARKit 协方差调制方法找**逐像素真值**，支撑 PSNR/SSIM/LPIPS 定量评测。
> 纪律：全程只读 + CPU 轻量探测，未加载 LAM 模型，未跑 GPU，:8000 服务未动（`curl -o /dev/null -w %{http_code}` → `200`）。
> 环境：`/usr/local/miniconda3/envs/lam/bin/python`（3.10.21，torch 2.3.0+cu121，numpy 1.23.0）。

## 0. 结论先行

**30 分钟内拿不到真实照片级 GT。** 四条候选路径中 P1 只解决了「驱动」没解决「真值」，P2 的视频是**别人身份**，P3 的单图素材**没有同人第二张**，P4/P5 只解决「指标能算」。

---

## 1. P1：LAM 视频驱动管线

### 1.1 入口确实存在，但只能「驱动」不能「重建」

- 唯一入口 `/workspace/LAM/run_infer.sh`（95 行），核心命令：
  `python -m lam.launch infer.lam --config configs/inference/lam-20k-8gpu.yaml model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ image_input=$IMAGE export_video=true motion_seqs_dir=$MOTION_DIR motion_img_dir=null render_fps=30 motion_video_read_fps=30`
- 注意 `motion_seqs_dir` **必须带尾斜杠**（代码用 `os.path.dirname(motion_seqs_dir)` 定位 `transforms.json`），否则静默失败且退出码仍为 0。
- 真正的 motion 加载器是 `/workspace/LAM/lam/runners/infer/head_utils.py:329-422` `prepare_motion_seqs(...)`；
  `lam/runners/infer/lam.py:28` 确实 import 的是 `head_utils` 的版本，`lam/runners/infer/utils.py` 里同名旧函数是**死代码**（实测喂 `flame_params.json` 会抛 `TypeError: must be real number, not dict`）。

### 1.2 视频→FLAME 的自建路径在本机不可用

- `head_utils.py:194-200` `extract_imgs_from_video` 用 `decord.VideoReader`；**decord 未安装**。
- `head_utils.py:203-221` `predict_motion_seqs_from_images` 走 `cd thirdparty/multi-hmr && python infer_batch.py --data_root ... --crop_head --crop_hand --pad_ratio 0.2 --smplify`；
  但 `ls /workspace/LAM/thirdparty` → **不存在**（`thirdparty_src` 只有 diff-gaussian-rasterization / nvdiffrast / pytorch3d / simple-knn，无 `.gitmodules`）。
- ⇒ 想「从一段视频重建一个人」需要额外拉 Multi-HMR 代码 + 权重（数 GB）+ 装 decord，**远超 30 分钟**。

### 1.3 但「用已有 FLAME 序列驱动我方单图重建」是通的（有产物为证）

- 历史日志 `/tmp/lam_infer_last.log`：`motion_seqs_dir: /workspace/LAM/assets/sample_motion/export/Look_In_My_Eyes/` → `len motion_seq:519` → `start to inference...` → `time elapsed: 3.196784019470215` → 写出 `exps/videos/lam/lam_20k/status.mp4`（520 帧）。
- 现有渲染产物：`/workspace/LAM/exps/videos/lam/lam_20k/` 下 11 条（barbara/cluo/dufu/libai/messi/musk/my_avatar/pop/speed/status/zhouxingchi），`ffmpeg -i barbara.mp4` → `512x512, 30 fps, 17.23s`。
- 逐帧 PNG：`/workspace/LAM/exps/images/lam/lam_20k/status/`，实测 `(512,512) RGB`。

### 1.4 关键否决点：渲染帧与原始视频帧**不可直接逐像素比**

1. **输出是 512×512 的白底合成图**，不是原始构图：
   `head_utils.py:155` `rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])`，`bg_color=1.0`（`lam.py:291-299` `rendered_bg=1.0`）。
2. **做过按 mask 包围盒的方形裁剪**：`head_utils.py:61-118` `center_crop_according_to_mask`，包围盒中心取图像中心，`half_w/half_h` 由 mask 极值决定，`aspect_standard=1.0`。
3. **裁剪框没有落盘**：`vhap/export_as_nerf_dataset.py:56-150` 写的 `transforms.json` 只有 `fl_x/fl_y/cx/cy/h/w/transform_matrix/file_path/fg_mask_path/flame_param_path`，无 crop/offset 键；6 个变体顶层键完全一致。
4. **crop 里带随机分支**：`head_utils.py:99-104`
   `if abs(enlarge_ratio[0]-1)>0.01 or abs(enlarge_ratio[1]-1)>0.01: ... enlarge_ratio_cur = np.random.rand()*(max-min)+min`。
   - 源图调用 `lam.py:314` 传 `enlarge_ratio=[1.0, 1.0]` → **不触发随机**，源图参考可复现。
   - motion 调用 `lam.py:327` 传 `enlarge_ratio=[1.0, 1,0]`（源码里的字面 typo，即 `[1.0, 0.0]`）→ 若该分支执行则每次运行裁框不同。
   - 注意 `run_infer.sh` 传 `motion_img_dir=null`，且 `head_utils.py:420` 的 `motion_seqs['rgbs']` 那行是**注释掉的** → 渲染循环从不读取 motion 图像像素。
5. **渲染循环本身不做任何比对**：`lam.py:340-364` 只吃 `render_c2ws/render_intrs/render_bg_colors/flame_params`，`lam.py:350-352` `res['comp_rgb']` → `(np.clip(rgb,0,1)*255).astype(uint8)` 直接编码 mp4。**仓库里没有任何 GT 对齐代码。**

> 小结：P1 能产出「同一张脸 + 别人表情」的视频，但**没有可信的逐帧参考图**；若强行比，必须自己反解 mask 包围盒→crop→白底合成→resize 全链路，且 motion 侧随机裁框使结果不可复现。

---

## 2. P2：`assets/sample_motion/export/` 的 12 条序列（不是 13 条）

实测目录：Anti_Drugs、D_ANgelo_Dinero、Donald_Trump、GEM、I_Am_Iron_Man、Joe_Biden、Look_In_My_Eyes、Michael_Wayne_Rosen、Pen_Pineapple_Apple_Pen、Speeding_Scandal、Taylor_Swift、The_Shawshank_Redemption。

每条内容：`<name>.mp4` + `<name>.wav` + `canonical_flame_param.npz` + `flame_params.json` + `flame_param/`（403~519 个 `00000.npz`）+ 6 个 `transforms*.json`。
**没有 `images/` 目录，也没有 `fg_masks/` 目录** —— 导出时裁好的 GT 帧根本没随仓库发布，只剩原始 mp4。

### 2.1 帧参数到底是什么

- `flame_param/00000.npz`（Taylor_Swift）键与形状：`translation (1,3)`, `rotation (1,3)`, `neck_pose (1,3)`, `jaw_pose (1,3)`, `eyes_pose (1,6)`, `shape (300,)`, `expr (1,100)`, `static_offset (1,5143,3)`。
- `canonical_flame_param.npz` 同结构（去掉 static_offset 也在）。
- `flame_params.json` 顶层键 `['translation','rotation','neck_pose','jaw_pose','eyes_pose','shape','expr']`，由 `/workspace/LAM/assets/sample_motion/export_expression_json.py` 生成（expr 字典化，其它 `tolist()`）。
  - Taylor_Swift 实测：`translation (403,3)`、`rotation (403,3)`、`jaw_pose (403,3)` min -0.0766 max 0.2781、`eyes_pose (403,6)`、`shape (300,)`、`expr` 长度 **402**（比 rotation 少 1 帧，既有缺陷）。
- **这是 FLAME 参数，不是 ARKit 52 维。** 若评测走 ARKit 路径，必须另行抽取 52 维 blendshape。

### 2.2 mp4 能不能直接当 GT —— 不能

`ffmpeg -i` 实测（bundled：`/usr/local/miniconda3/envs/lam/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`）：

| 序列 | 分辨率 | 时长 | 帧数(transforms.json) |
|---|---|---|---|
| Anti_Drugs | 1280x720 | 7.53s | 226 |
| D_ANgelo_Dinero | 852x480 | 4.40s | 132 |
| Donald_Trump | 480x480 | 8.80s | 264 |
| GEM | 270x480 | 17.20s | 516 |
| I_Am_Iron_Man | 640x360 | 9.40s | 282 |
| Joe_Biden | 480x480 | 8.80s | 264 |
| Look_In_My_Eyes | 1278x720 | 17.30s | 519 |
| Michael_Wayne_Rosen | 3840x2160 | 2.00s | 60 |
| Pen_Pineapple_Apple_Pen | 1024x1024 | 5.57s | 167 |
| Speeding_Scandal | 1280x720 | 2.53s | 76 |
| Taylor_Swift | 360x640 | 13.43s | 403 |
| The_Shawshank_Redemption | 1080x1920 | 14.50s | 435 |

- 全部 30 fps + aac。`transforms.json` 内参一律 1024x1024、`cx=cy=512`、`fl_x=fl_y` ∈ [2248.53, 2604.94]（Anti_Drugs 2604.938720703125 / Taylor_Swift 2435.701171875）。
- **motion 视频与单图上传者不是同一人**（Taylor_Swift vs barbara/messi/…），只能当「动作来源」，**不能当 GT**。

### 2.3 MediaPipe 可装可用（已实测，CPU-only）

- 清华镜像 py3-none `mediapipe-1.1.1` 41.5MB 可解析；但 `--dry-run` 显示它会拉 `numpy-2.2.6` + `opencv-contrib-python-5.0.0.93`，**与现有 numpy 1.23.0 / opencv 4.11 冲突**。
- 改用 `mediapipe==0.10.14`（cp310 wheel，35.7MB，14.7MB/s）+ `--no-deps` 安装成功：`import mediapipe` → `0.10.14`，**numpy 仍是 1.23.0**。依赖 `absl/flatbuffers/attrs/jax/jaxlib` 本机已满足（仅 `sounddevice` 缺失，图像路径不需要）。
- 模型 `face_landmarker.task`（3.7MB，storage.googleapis.com 可下）实测：
  `FaceLandmarker(... output_face_blendshapes=True)` 对 `sample_input/barbara.jpg` → `faces:1 n_bs:52`（`_neutral, browDownLeft, ...`），耗时 **0.04s/张**，messi.png 0.02s。⇒ **52 维 ARKit 抽取完全可行且极廉价**。

---

## 3. P3：单图路径有没有「同身份不同表情」的真实照片

- `/workspace/LAM/assets/sample_input/` 共 **11 张**：barbara.jpg、cluo.jpg、dufu.jpg、james.png、libai.jpg、messi.png、musk.jpg、pop.png、speed.jpg、status.png、zhouxingchi.jpg（jpg 约 170-570KB，png 0.5-3MB）。**11 个不同的人/角色，无同人多表情。**
- `/workspace/lam3d/workspace/uploads/` 共 **133 个文件，去重后仅 16 张**（最大重复组 58 份同一 md5）。其中只有 5 张能在 `sample_input` 找到原文（james/status/messi/barbara/cluo 的副本），其余 11 张分辨率 384²~1650×1458、彼此不同人。
- `/workspace/lam3d/workspace/jobs/` 共 133 个任务，有 `input.png` 的 94 个，**去重后只有 10 张不同输入**；单任务产物 `face.png` 为 512×512。
- ⇒ **不存在同身份的第二张真实照片**，P3 判负。

---

## 4. P4：仓库自带评测脚本/指标

- `grep -rn "psnr|PSNR|lpips|LPIPS|ssim" --include=*.py /workspace/LAM | grep -v thirdparty_src` 仅命中 `lam/losses/perceptual.py:20-61`（`class LPIPSLoss`，训练用 loss，支持 alex/vgg 两种 backbone，`_get_model` 里 `import lpips` + `torch.compile`）。
- `/workspace/LAM/scripts/` 只有 convert_hf.py、exp、inference.sh、install、upload_hub.py；`/workspace/LAM/tools/` 只有 GLB/FBX 导出脚本。**无 eval 脚本、无 PSNR/SSIM 实现**。
- `/workspace/lam3d/scripts/`、`/workspace/lam3d/experiments/` 下 `grep psnr|ssim|lpips` **零命中**。
- ⇒ 指标必须自己写。现成可用：`skimage.metrics.structural_similarity` / `peak_signal_noise_ratio`（已 `from skimage.metrics import ...` 验证可导入）。

---

## 5. P5：LPIPS 权重与可用性

- `import lpips` 可用：`/usr/local/miniconda3/envs/lam/lib/python3.10/site-packages/lpips/__init__.py`。
- 线性层权重已随包：`lpips/weights/v0.0/{alex 5455B, vgg 6735B, squeeze 10057B}`、`v0.1/{alex 6009B, vgg 7289B, squeeze 10811B}`。
- 骨干权重缓存 `/root/.cache/torch/hub/checkpoints/` 仅 2 个：`alexnet-owt-7be5be79.pth (244408911 B)`、`dinov2_vitl14_reg4_pretrain.pth (1217607321 B)`。**无 VGG16 权重** ⇒ 用 `net='vgg'` 会触发联网下载。
- CPU smoke test（barbara 256² vs 平移 2px 副本）：SSIM 0.688938 / PSNR 18.7259 dB / LPIPS(alex) 0.059161 ⇒ 三指标链路已验证可用（该测试顺带下载了 alexnet 权重）。

---

## 6. 已验证 / 未验证清单

| 项 | 状态 | 证据 |
|---|---|---|
| motion 驱动单图重建可跑通 | **已验证** | `/tmp/lam_infer_last.log` + `exps/videos/lam/lam_20k/*.mp4` |
| 渲染输出 512×512 白底 | **已验证** | `lam.py:291-299,314,327`；PNG 实测 (512,512) |
| 渲染帧 vs 原始视频帧逐像素比 | **不可行** | `head_utils.py:61-118,155`；crop 未落盘 |
| 视频→FLAME 自建 | **已验证不可行** | 无 `thirdparty/`、无 decord |
| 52 维 ARKit 抽取 | **已验证（CPU）** | mediapipe 0.10.14 + face_landmarker，0.04s/张 |
| 同身份多表情真实照片 | **已验证不存在** | 11 张全不同人；uploads 去重 16 张 |
| 仓库自带 eval | **已验证不存在** | grep 仅命中训练 loss |
| LPIPS 三指标可算 | **已验证** | smoke test 数值 |
| LAM 自身 ARKit 基 | **已验证存在** | `flame_arkit_bs.npy (52,5023,3)` float64，min -0.0443 max 0.0302 |
| ARKit 参数接进推理器 | **未验证/未接线** | `grep arkit_params lam/runners/` 零命中；仅 `gs_renderer.py:580-660`、`flame_arkit.py:102-130` 有钩子 |

---

## 7. 推荐的最小可行评测协议（替代参考，不自欺）

**核心原则：把「质量」拆成「可验证的几何正确性」与「相对改动量」，不要用任何一个方法当 GT。**

1. **P0 恒等性（零成本，先跑）**：ARKit 全 0 时输出必须与 neutral 逐像素一致（`covariance_modulation.py:327-331 is_neutral` + `gs_renderer.py:624-625` 已保证静态直通）。
   指标：`max|I_0 - I_neutral|`，期望 `0`；任何非零即为 bug。**这是唯一真正无争议的 GT。**
2. **P1 非循环几何真值（推荐主指标）**：用 `flame_arkit_bs.npy (52,5023,3)` 直接驱动 FLAME 网格得顶点位移 `Δx_gt`，再取变形网格的面法向 `n_gt`。
   方法输出的是高斯 `Δq/Δs`；把 `Δs` 的三主轴与 `n_gt` 比：`ang_err = arccos|cos|`（度），以及 thin-plate 各向异性比 `λ_min/λ_max` 与曲率的一致性。
   **为什么非循环**：`Δx_gt` 来自仓库自带的外部基，不经过被测方法的协方差路径。
3. **P2 薄片朝向正确性**：对每个高斯，其最短轴应与局部表面法向平行（薄片贴合表面）。用 `n_gt` 做监督，统计 5°/10°/20° 内占比。同样是几何真值，无需图像。
4. **P3 相对改动量（明确标注非质量）**：以 neutral 渲染为锚，报 `ΔPSNR = PSNR(I_m, I_neutral)`、`1-SSIM`、`LPIPS(alex)`。
   用途：衡量「表情强度→视觉变化」是否单调、是否过冲；**不得**写成"质量优于 X"。
5. **P4 跨方法一致性（弱指标）**：四种方法两两 LPIPS 距离矩阵；只说明「分歧大小」，不排序优劣。
6. **P5 真实照片弱监督（可选，非 GT）**：mediapipe 从原图抽 52 维 ARKit 作为**输入信号**（已验证可用），检查「输入强度 vs 输出形变量」的相关性；因 FLAME expr 与 ARKit 不同空间，只能做 Spearman 秩相关，**不能**做像素级。

> 若必须做像素级：唯一合法路线是自己合成参考——固定 `enlarge_ratio=[1.0,1.0]` 重放 `preprocess_image` 得到 **source 参考图**（与上传图同身份同表情），再以 neutral 渲染对它算 PSNR/SSIM。但它只度量"重建保真度"，**与 ARKit 调制无关**，不能用于四种方法的比较。

---

## 8. 最终裁决

**30 分钟内无法获得真实照片级 GT。** 阻断原因（按强度排序）：
1. 单图路径只有 1 帧、无同身份第二张真实照片（P3 去重后仅 16 张，11 张官方样例全不同人）；
2. 视频驱动路径只能「用别人的 FLAME 序列驱动」，且缺 Multi-HMR + decord，无法从 mp4 反建同一人（P1.2）；
3. `sample_motion` 的 mp4 是**他人身份**，且导出裁好的 GT 帧（images/fg_masks）未随仓库发布（P2）；
4. 即便有视频，渲染图经过白底合成 + mask 方形裁剪 + resize 三步，crop 未落盘且 motion 侧 `enlarge_ratio=[1.0, 0,0]` 触发随机，逐像素对齐不可复现（P1.4）。

**不自欺的替代定义**：以 `flame_arkit_bs.npy` 驱动的变形网格法向为**几何真值**（非循环，外部给定），度量「高斯薄片朝向/各向异性是否与真实表面形变一致」（协议 P1/P2，角度误差、各向异性比、阈值内占比）；像素类指标只以 neutral 为锚报**改动量**并显式标注「非质量」（协议 P3）。P0 恒等性作为硬性回归门禁。
