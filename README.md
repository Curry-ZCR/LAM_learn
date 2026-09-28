# LAM 学习 / 部署工作区

本仓库记录在 **RTX 4090 服务器**上从零配置 [aigc3d/LAM](https://github.com/aigc3d/LAM)
（SIGGRAPH 2025, Large Avatar Model for One-shot Animatable Gaussian Head）环境的完整过程，
包含可复现的安装脚本、推理脚本、踩坑记录与验收日志。

> ⚠️ 本仓库**只包含源码、脚本与文档**（约 4 MB）。
> 模型权重（3.1 GB）、数据集（300 MB）、推理产物、第三方编译源码均已被 `.gitignore` 排除，
> 克隆后需按下方步骤重新下载与编译。

---

## 目录结构

```
.
├── .gitignore                     # 排除模型/数据/产物，只保留源码
├── LAM/                           # LAM 项目本体（上游 aigc3d/LAM）
│   ├── LAM_ENV_SETUP_GUIDE.md     # ★ 环境配置指南（含 10 个错误的完整记录）
│   ├── LAM_INFERENCE_GUIDE.md     # ★ 推理测试指南（4 种场景 + 9 类坑点）
│   ├── install_lam.sh             # ★ 一键环境配置脚本（已内置全部修复）
│   ├── run_infer.sh               # ★ 一键推理脚本
│   ├── requirements-lock.txt      # pip freeze 导出（157 个包）
│   ├── environment-lam.yml        # conda env 导出
│   ├── app_lam.py                 # Gradio Web 界面入口
│   ├── lam/  vhap/  external/     # 核心源码
│   ├── configs/  scripts/  tools/ # 配置与工具
│   └── assets/images/             # README 引用的示意图
└── lam_setup_logs/                # 安装全过程日志（56 个文件）
    ├── 00_conda_create.sh/.log    # 创建 conda 环境
    ├── 01_torch.sh/.log           # PyTorch 2.3.0 + cu121
    ├── 03*_requirements*          # 依赖安装与 chumpy/numpy 冲突修复
    ├── 04*_build_ext*             # 4 个 CUDA 扩展编译（含 ninja 修复）
    ├── 05_faceboxes*              # FaceBoxesV2 Cython 扩展
    ├── 06*_download*              # 权重下载（modelscope CLI 修复）
    ├── 07*_dinov2/load_model*     # DINOv2 主干镜像加速
    ├── 08*_inference*             # 端到端推理验证
    └── 09-15_*                    # 回归测试与最终验收
```

---

## 快速开始

### 1. 克隆并配置环境

```bash
git clone git@github.com:Curry-ZCR/LAM_learn.git
cd LAM_learn

# 一键配置（自动处理 g++、chumpy、ninja、CUDA 扩展、Web 栈降级等所有已知问题）
bash LAM/install_lam.sh 2>&1 | tee install.log

# 若 GPU 不是 RTX 4090，指定计算能力：A100=8.0，3090=8.6
TORCH_ARCH=8.0 bash LAM/install_lam.sh
```

### 2. 推理

```bash
# 命令行（推荐批量）
bash LAM/run_infer.sh assets/sample_input/status.png Look_In_My_Eyes

# 导出 3D 网格
bash LAM/run_infer.sh assets/sample_input/status.png Look_In_My_Eyes mesh

# Web 界面
cd LAM && python app_lam.py    # http://127.0.0.1:7860
```

---

## 环境规格（实测验证）

| 组件 | 版本 | 备注 |
|---|---|---|
| OS | Ubuntu 22.04.5 | 容器 uhost-1vu6accbztchu |
| GPU | RTX 4090 24GB (sm_89) | 驱动 595.80，系统 CUDA 12.8 |
| Python | 3.10.21 | conda env `lam` |
| PyTorch | 2.3.0+cu121 | 配 xformers 0.0.26.post1 |
| numpy / scipy | **1.23.0** / **1.10.1** | ⚠️ 必须锁 1.x，受 tensorflow 2.12 约束 |
| pytorch3d | 0.7.6 | 源码编译 |
| Web 栈 | fastapi 0.103.2 / starlette 0.27.0<br>uvicorn 0.23.2 / pydantic 2.8.0 | ⚠️ 必须降级，否则 Gradio 起不来 |

推理性能：单图 519 帧动作 **≈44 秒**，显存 **2.20 GB**，输出 512×512 @30fps。

---

## 配置过程中踩过的坑（详见指南）

环境配置指南记录了 **10 个错误** 的完整报错、根因与修复：

| # | 问题 | 一句话解决 |
|---|---|---|
| 1 | `g++: command not found` | `apt-get install -y g++` |
| 2 | chumpy 编译 `No module named 'pip'` | `--no-build-isolation --no-deps` |
| 2b | numpy 被静默升到 2.x | 锁 `numpy==1.23.0 scipy==1.10.1` |
| 3 | `Ninja is required to load C++ extensions` | `export PATH=$ENV_PREFIX/bin:$PATH` |
| 4 | pytorch3d 编译 >1 小时（串行） | 同 3 + `rm -rf build` 重编 |
| 5 | `No module named modelscope.__main__` | 用 `modelscope` CLI 而非 `python -m` |
| 6 | `FileNotFoundError: ./pretrained_models/...` | 走 `lam.launch infer.lam`，别用 `from_pretrained()` |
| 7 | DINOv2 下载 300KB/s | 从 HF 镜像预置到 torch 缓存 |
| 8 | `FileNotFoundError: .../export/transforms.json` | `motion_seqs_dir` 末尾要加 `/` |
| 9 | assets 混入 macOS `._*` 垃圾文件 | `find assets -name "._*" -delete` |
| 10 | `ValueError: When localhost is not accessible` | 降级 starlette/fastapi/uvicorn/pydantic |

推理指南另记录了 3 个官方脚本 Bug 与 5 类使用陷阱（`export_mesh` 是死参数、
目录输入只认 `.jpg`、`gaga_track_type` 被硬编码覆盖、自定义视频驱动缺 `multi-hmr` 等）。

---

## 上游来源

- 项目：[aigc3d/LAM](https://github.com/aigc3d/LAM)
- 配置时的上游 commit：`339573649dd93df4cba8093a964e85a80d1b61f3`（2026-06-10, Create LICENSE_WEIGHT）
- 本仓库为**全新历史**（单次初始提交），未保留上游 git 历史

## 许可

LAM 上游代码遵循 Apache-2.0（见 `LAM/LICENSE`），模型权重有单独的 `LICENSE_WEIGHT`。
