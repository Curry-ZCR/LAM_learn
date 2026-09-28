# LAM 环境配置指南（Linux / CUDA）

> 面向 [aigc3d/LAM](https://github.com/aigc3d/LAM)（SIGGRAPH 2025, Large Avatar Model for One-shot Animatable Gaussian Head）
>
> 本指南基于**一次真实的全新配置过程**编写：从空 conda 环境开始，直到跑通端到端推理。
> 过程中遇到的全部 7 个错误都记录了**完整报错原文 + 根因 + 解决方法 + 验证方式**，
> 并标注了哪些是「官方文档没写」的坑。
>
> **实测验证环境**：Ubuntu 22.04.5 / RTX 4090 24GB (sm_89) / 驱动 595.80 / 系统 CUDA Toolkit 12.8 / 16 vCPU / 62GB RAM

---

## 目录

- [0. 快速开始（一键脚本）](#0-快速开始一键脚本)
- [1. 环境要求](#1-环境要求)
- [2. 版本矩阵（已验证可用）](#2-版本矩阵已验证可用)
- [3. 分步配置流程](#3-分步配置流程)
- [4. 错误记录与解决方法（10 个）](#4-错误记录与解决方法10-个)
- [5. 官方脚本的 3 个隐藏 Bug](#5-官方脚本的-3-个隐藏-bug)
- [6. 验证清单](#6-验证清单)
- [7. 常见问题 FAQ](#7-常见问题-faq)
- [8. 性能与资源参考](#8-性能与资源参考)

---

## 0. 快速开始（一键脚本）

本仓库附带一键配置脚本 **`install_lam.sh`**，已包含下面所有修复：

```bash
bash install_lam.sh 2>&1 | tee install_lam.log
```

默认参数（可通过环境变量覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PROJECT_DIR` | `/workspace/LAM` | LAM 项目路径 |
| `CONDA_ROOT` | `/usr/local/miniconda3` | conda 安装根目录 |
| `ENV_NAME` | `lam` | conda 环境名 |
| `TORCH_ARCH` | `8.9` | `8.9`=4090, `8.6`=3090, `8.0`=A100 |
| `MAX_JOBS` | `16` | 并行编译线程数 |
| `MODEL_SOURCE` | `modelscope` | 权重下载源：`modelscope` / `huggingface` |

```bash
# 例：A100 机器
TORCH_ARCH=8.0 bash install_lam.sh
```

---

## 1. 环境要求

### 1.1 硬件

| 项目 | 最低 | 本指南实测 |
|---|---|---|
| GPU | 支持 sm_70+，显存 ≥ 8GB | RTX 4090 24GB |
| 内存 | ≥ 16GB | 62GB |
| 磁盘 | ≥ 25GB 可用 | 79GB 盘，实占约 20GB |

磁盘占用明细：

| 内容 | 大小 |
|---|---|
| conda 环境 | 8.4 GB |
| `model_zoo/`（权重 + FLAME + 人脸跟踪） | 3.1 GB |
| `assets/`（示例输入/动作） | 305 MB |
| `thirdparty_src/`（扩展源码与编译产物） | 256 MB |
| DINOv2 主干缓存 | 1.13 GB |
| pip 缓存（可清理） | 7.7 GB |

### 1.2 系统依赖（**最容易漏掉的一步**）

```bash
# g++ 必需。很多精简 CUDA 镜像只有 gcc 没有 g++，
# 会导致所有 CUDA 扩展编译失败。
apt-get update && apt-get install -y g++ git curl

g++ --version    # 必须能输出版本号
```

> ⚠️ **这是本指南遇到的第 1 个坑**，详见 [错误 1](#错误-1g-command-not-found)。

### 1.3 CUDA Toolkit

LAM 需要 nvcc 来编译 4 个 CUDA 扩展。系统 CUDA Toolkit 版本**不必**与 PyTorch 的 CUDA 版本完全一致：

- PyTorch 2.3.0 编译目标是 **CUDA 12.1**
- 系统 nvcc 是 **CUDA 12.8**
- `torch.utils.cpp_extension` 只在 **major 版本不同**时硬报错（如 11.x vs 12.x），minor 不同仅告警

```bash
ls -d /usr/local/cuda*          # 确认 CUDA 安装位置
/usr/local/cuda/bin/nvcc --version
```

> 本机 `nvcc` 不在 PATH 中，需要显式 `export CUDA_HOME=/usr/local/cuda` 并把它加进 PATH。

---

## 2. 版本矩阵（已验证可用）

### 2.1 核心框架

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | **3.10.21** | conda |
| PyTorch | **2.3.0+cu121** | pytorch 官方源 |
| torchvision | 0.18.0+cu121 | pytorch 官方源 |
| torchaudio | 2.3.0+cu121 | pytorch 官方源 |
| xformers | 0.0.26.post1 | pytorch 官方源 |
| CUDA (torch) | 12.1 | — |
| CUDA (nvcc) | 12.8 | 系统 |
| cuDNN | 8.9.2.26 | 随 torch |
| **numpy** | **1.23.0** ⚠️ | 必须锁 1.x |
| **scipy** | **1.10.1** ⚠️ | 必须兼容 numpy 1.23 |

### 2.2 需编译的扩展

| 包 | 版本 | 源码 |
|---|---|---|
| pytorch3d | 0.7.6 | `facebookresearch/pytorch3d` @ v0.7.6 |
| diff_gaussian_rasterization | 0.0.0 | `ashawkey/diff-gaussian-rasterization` |
| simple_knn | 1.0.0 | `camenduru/simple-knn` |
| nvdiffrast | 0.3.3 | `ShenhanQian/nvdiffrast` @ **backface-culling** |
| FaceBoxesV2 cpu_nms | — | 项目内置 Cython 扩展 |

### 2.3 其他关键依赖

| 包 | 版本 | 备注 |
|---|---|---|
| tensorflow | 2.12.0 | 🚨 **强制 numpy<1.24 的元凶** |
| face-detection-tflite | 0.6.0 | 依赖 tensorflow |
| transformers | 4.41.2 | |
| diffusers | 0.32.2 | |
| accelerate | 1.15.0 | |
| gradio | 3.44.3 | Gradio Web UI |
| fastapi | **0.103.2** | ⚠️ 必须降级，见错误 10 |
| starlette | **0.27.0** | ⚠️ 必须降级，见错误 10 |
| uvicorn | **0.23.2** | ⚠️ 必须降级，见错误 10 |
| pydantic | **2.8.0** | ⚠️ 必须降级，见错误 10 |
| chumpy | 0.70 | ⚠️ 需特殊处理 |
| opencv-python-headless | 4.11.0.86 | |
| modelscope | 1.40.1 | 仅用于下载权重 |

### 2.4 两个必须锁定的版本约束

```
tensorflow 2.12  →  numpy < 1.24   (硬约束，pip 会硬失败)
numpy 1.23.0     →  scipy 必须 ≤ 1.10.1
                     (scipy 1.11+ 要求 numpy>=1.23.5；
                      scipy 1.15 要求 numpy>=1.23.5 且会拉入 numpy 2.x)
```

**这是整个配置过程的核心矛盾**：`requirements.txt` 里 `chumpy` 声明 `scipy>=0.13.0` 无上限 → pip 解析出 scipy 1.15.3 → 连带把 numpy 从 1.23.0 升到 2.2.6 → tensorflow 2.12 以及所有按 numpy 1.x ABI 编译的扩展全部崩溃。

---

## 3. 分步配置流程

> 全程使用**绝对路径**调用环境内的可执行文件。本机 PATH 里既没有 conda env 的 `bin`，
> 也没有 CUDA 的 `bin` —— 这是后面好几个 bug 的共同根源。

### Step 0 — 系统依赖

```bash
apt-get update && apt-get install -y g++ git curl
g++ --version
```

### Step 1 — 创建 conda 环境

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda create -y -n lam python=3.10

ENV_PREFIX=/usr/local/miniconda3/envs/lam
PY=$ENV_PREFIX/bin/python
PIP=$ENV_PREFIX/bin/pip

# ★ 关键：把 conda env 的 bin 和 CUDA 的 bin 放进 PATH
#   同时解决 nvcc 查找和 ninja 查找两个问题
export PATH="$ENV_PREFIX/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"      # 按你的 GPU 改
export MAX_JOBS=16
```

### Step 2 — PyTorch

```bash
$PIP install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
     --index-url https://download.pytorch.org/whl/cu121
$PIP install -U xformers==0.0.26.post1 \
     --index-url https://download.pytorch.org/whl/cu121

$PY -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

> `xformers` 需要逐版本匹配 torch：`0.0.26.post1` ↔ `torch 2.3.0`。

### Step 3 — 基础依赖（含 chumpy 特殊处理）

```bash
cd /workspace/LAM

# 3.1 numpy 必须最先锁定为 1.x
$PIP install "numpy==1.23.0"

# 3.2 chumpy：必须 --no-build-isolation --no-deps
$PIP install "chumpy==0.70" --no-build-isolation --no-deps

# 3.3 scipy 锁到兼容 numpy 1.23 的版本
$PIP install "scipy==1.10.1" "six"

# 3.4 其余依赖（剔除 git+ 行、chumpy、numpy、scipy）
grep -vE '^git\+|@git\+|^chumpy|^numpy|^scipy' requirements.txt > /tmp/lam_req.txt
$PIP install -r /tmp/lam_req.txt "numpy==1.23.0" "scipy==1.10.1"

$PY -c "import numpy,scipy;print(numpy.__version__, scipy.__version__)"
```

**为什么不用 `pip install -r requirements.txt` 一把梭？** 因为 `requirements.txt` 里有 4 条 `git+` 依赖需要编译，pip 会把它们克隆到临时目录，无法传递 `--no-build-isolation` 和 `TORCH_CUDA_ARCH_LIST`，失败率极高。这 4 个包在 Step 4 单独从源码构建。

### Step 4 — 编译 4 个 CUDA 扩展

```bash
mkdir -p thirdparty_src && cd thirdparty_src

git clone --recursive --depth 1 https://github.com/ashawkey/diff-gaussian-rasterization/
git clone --recursive --depth 1 https://github.com/camenduru/simple-knn/
git clone --recursive --branch backface-culling --depth 1 https://github.com/ShenhanQian/nvdiffrast
git clone --recursive --branch v0.7.6 --depth 1 https://github.com/facebookresearch/pytorch3d.git

$PIP install setuptools wheel ninja
ninja --version || echo "ERROR: ninja 不在 PATH 上！"
```

> ⚠️ `ninja --version` 这步**千万不要跳过**，见 [错误 4](#错误-4ninja-is-required-to-load-c-extensions--静默串行编译)。

```bash
for ext in diff-gaussian-rasterization simple-knn nvdiffrast; do
    ( cd $ext && $PIP install . --no-build-isolation )
done

( cd pytorch3d && FORCE_CUDA=1 $PIP install . --no-build-isolation )
```

各扩展编译耗时（16 线程，sm_89）：

| 扩展 | 耗时 |
|---|---|
| diff-gaussian-rasterization | ~1.5 min |
| simple-knn | ~1 min |
| nvdiffrast | ~1 min |
| pytorch3d（67 个编译单元） | ~3 min（启用 ninja 后） |

### Step 5 — FaceBoxesV2 Cython 扩展

```bash
cd /workspace/LAM/external/landmark_detection/FaceBoxesV2/utils

# ★ 不要用 sh make.sh（它调用 python3 → 系统 python，无 Cython/numpy）
$PY build.py build_ext --inplace

ls -la nms/*.so   # 应生成 cpu_nms.cpython-310-x86_64-linux-gnu.so
```

### Step 6 — 下载权重与资源

**方式 A：ModelScope（本机实测更快，约 15–18 MB/s）**

```bash
$PIP install modelscope
MSC=$ENV_PREFIX/bin/modelscope

# ★ README 里的 `python -m modelscope download` 已失效，见 错误 5
mkdir -p ./tmp
$MSC download --repo-type model Damo_XR_Lab/LAM-assets --local-dir ./tmp/
tar -xf ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar
$MSC download --repo-type model Damo_XR_Lab/LAM-20K \
    --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
rm -rf ./tmp/
```

**方式 B：HuggingFace（约 7 MB/s）**

```bash
$PIP install "huggingface_hub[cli]"
huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
huggingface-cli download 3DAIGC/LAM-20K \
    --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

下载后的目录结构：

```
LAM/
├── assets/
│   ├── sample_input/          10 张示例人脸
│   └── sample_motion/export/  11 段驱动视频（含 flame_param/）
└── model_zoo/
    ├── flame_tracking_models/      # 人脸跟踪（68 关键点 / vgghead / matting / FaceBoxesV2）
    ├── human_parametric_models/    # FLAME 模型（flame_assets / flame_vhap）
    └── lam_models/releases/lam/lam-20k/step_045500/
        ├── config.json
        └── model.safetensors       # 2.36 GB
```

> 📌 `thirdparty_models.tar` 解包出来的内容就是 `model_zoo/`。
> 权重目录名 `step_045500` 在 `README.md`、`scripts/inference.sh`、`app_lam.py` 中是一致的，直接照抄即可。

> ⚠️ **务必清理 macOS 垃圾文件**：`LAM_assets.tar` 是在 macOS 上打包的，
> 解包后会带入大量 AppleDouble 文件（`._*` 与 `.DS_Store`）。本机实测有 **3506 个**，
> 它们会让 `flame_param/` 的帧数**翻倍**（如 519 帧变成 1038 个文件），
> 干扰对动作序列长度的判断。建议解包后立即执行：
>
> ```bash
> find /workspace/LAM/assets \( -name "._*" -o -name ".DS_Store" \) -delete
> ```

### Step 7 — 预置 DINOv2 主干权重

LAM 的 image encoder 会通过 `torch.hub` 下载 DINOv2 ViT-L/14-reg4 主干（1.13GB）。
官方源 `dl.fbaipublicfiles.com` 从国内主机只有 **~300 KB/s（约需 1 小时）**，
且 torch.hub 会写出两个竞争的 `.partial` 文件。

**推荐做法：从 HuggingFace 镜像预置到 torch 缓存目录**（实测 ~15 MB/s，4 倍以上提速）：

```bash
DEST=/root/.cache/torch/hub/checkpoints
FILE=dinov2_vitl14_reg4_pretrain.pth
mkdir -p $DEST
curl -L --retry 3 -o $DEST/$FILE.tmp \
  "https://huggingface.co/showstarpro/dinov2_vitl14_reg4_pretrain/resolve/main/$FILE"
mv $DEST/$FILE.tmp $DEST/$FILE
```

> ⚠️ 文件名必须**完全一致**，`torch.hub.load_state_dict_from_url` 靠文件名命中缓存，
> 命中后就不会再发起下载。

---

## 4. 错误记录与解决方法（10 个）

### 错误 1：`g++: command not found`

**现象**：编译任意 CUDA 扩展时失败。

```
error: command 'g++' failed: No such file or directory
```

**根因**：精简 CUDA 镜像只装了 `gcc`，没有 `g++`。
本机初始状态下：

```
gcc  → /usr/bin/gcc        ✅
g++  → MISSING             ❌
cc   → /usr/bin/cc         ✅
c++  → MISSING             ❌
```

nvcc 编译 `.cu` 文件时需要 C++ 前端；`torch.utils.cpp_extension` 的链接阶段也需要 `g++`。

**解决**：

```bash
apt-get update && apt-get install -y g++
g++ --version    # g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
```

**验证**：

```bash
ls -l /usr/bin/g++     # 应存在
/usr/bin/g++ --version
```

> 💡 顺带一提，`apt-get` 有 dpkg 前端锁。若并发执行两个 apt 会看到
> `Could not get lock /var/lib/dpkg/lock-frontend`，串行执行即可。

---

### 错误 2：`ModuleNotFoundError: No module named 'pip'`（chumpy 编译失败）

**完整报错**：

```
Collecting chumpy (from -r requirements.txt (line 23))
  Downloading chumpy-0.70.tar.gz (50 kB)
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: finished with status 'error'
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [23 lines of output]
      Traceback (most recent call last):
        File "<string>", line 9, in <module>
      ModuleNotFoundError: No module named 'pip'

      During handling of the above exception, another exception occurred:
      ...
        File "/tmp/pip-build-env-1e9gkc4y/overlay/lib/python3.10/site-packages/setuptools/build_meta.py",
        line 317, in run_setup
          exec(code, locals())  # noqa: S102 # exec is intentional here
        File "<string>", line 11, in <module>
      ModuleNotFoundError: No module named 'pip'
      [end of output]

ERROR: Failed to build 'chumpy' when getting requirements to build wheel
```

**根因**：`chumpy-0.70/setup.py` 第 8–11 行：

```python
from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements   # ← 这里
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
```

pip 在**构建隔离沙箱**（`/tmp/pip-build-env-*/overlay/`）里构建 wheel 时，
沙箱内只安装了 `setuptools`/`wheel`，**没有 pip 自己**，所以 `import pip` 失败。
注意：`chumpy-0.70` 目录里根本没有 `pyproject.toml`，但新版 pip 仍会走 PEP 517 流程并创建隔离环境。

**解决**：关闭构建隔离，让它使用当前环境（环境里装有 pip）。

```bash
$PIP install "chumpy==0.70" --no-build-isolation --no-deps
```

**⚠️ 连带陷阱（非常重要）**：如果只加 `--no-build-isolation` 而不加 `--no-deps`，pip 会继续解析 chumpy 的依赖：

```
chumpy 0.70  →  scipy>=0.13.0（无上限）
             →  pip 解析出 scipy 1.15.3
             →  scipy 1.15.3 要求 numpy>=1.23.5，于是安装 numpy 2.2.6
             →  numpy 1.23.0 被卸载！
```

实机观察到的输出：

```
Collecting scipy>=0.13.0 (from chumpy==0.70)
  Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Collecting numpy<2.5,>=1.23.5 (from scipy>=0.13.0->chumpy==0.70)
  Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  Attempting uninstall: numpy
    Found existing installation: numpy 1.23.0
    Uninstalling numpy-1.23.0:
Successfully installed chumpy-0.70 numpy-2.2.6 scipy-1.15.3
```

numpy 1.23.0 被静默升级到 2.2.6 后，**tensorflow 2.12 会直接不可用**，所有按 numpy 1.x ABI 编译的扩展也都要重编。

**所以必须三步走**：

```bash
$PIP install "chumpy==0.70" --no-build-isolation --no-deps
$PIP install "scipy==1.10.1" "six"
$PY -c "import numpy;print(numpy.__version__)"   # 必须仍是 1.23.0
```

> 📎 chumpy 在这里的唯一用途是**反序列化 FLAME 的 `.pkl` 文件**
> （`lam/models/rendering/flame_model/flame.py:123` 的 `pickle.load(f, encoding="latin1")`
> 会还原 pickled 的 `chumpy.array` 对象）。
> chumpy 的运行时真实需求只有 `scipy.sparse` / `scipy.optimize`，
> **不需要任何编译扩展**，所以 `--no-deps` 完全安全。

---

### 错误 3：`RuntimeError: Ninja is required to load C++ extensions`

**完整报错**：

```
  File "/usr/local/miniconda3/envs/lam/lib/python3.10/site-packages/nvdiffrast/torch/ops.py",
  line 184, in __init__
    self.cpp_wrapper = _get_plugin().RasterizeCRStateWrapper(cuda_device_idx)
  File ".../nvdiffrast/torch/ops.py", line 125, in _get_plugin
    torch.utils.cpp_extension.load(name=plugin_name, sources=source_paths, ...)
  File ".../torch/utils/cpp_extension.py", line 1802, in _write_ninja_file_and_build_library
    verify_ninja_availability()
  File ".../torch/utils/cpp_extension.py", line 1851, in verify_ninja_availability
    raise RuntimeError("Ninja is required to load C++ extensions")
RuntimeError: Ninja is required to load C++ extensions
```

**根因**：**conda 环境的 `bin` 目录不在 PATH 上**。

因为脚本用绝对路径 `.../envs/lam/bin/python` 调用解释器，python 并不需要 conda 环境"激活"，
但 `torch.utils.cpp_extension` 是**通过 PATH 查找 `ninja` 可执行文件**的：

```python
# torch/utils/cpp_extension.py
def verify_ninja_availability():
    with open(os.devnull, "wb") as devnull:
        if not _is_ninja_available():
            raise RuntimeError("Ninja is required to load C++ extensions")
```

本机实测：

```
$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
                                                                          ↑ 没有 conda env 的 bin

$ python -c "import subprocess;subprocess.check_output(['ninja','--version'])"
ninja NOT found on PATH -> [Errno 2] No such file or directory: 'ninja'
```

尽管 `ninja` 已经装在 `$ENV_PREFIX/bin/ninja`。

**关键点**：`nvdiffrast` 不只是安装时编译一次 —— 它**在首次运行时会 JIT 编译 CUDA 光栅化插件**
（`RasterizeCRStateWrapper`），所以即使前面 4 个扩展都装成功了，**推理时仍会因为这个错误崩溃**。

**解决**：把 conda 环境的 `bin` 加入 PATH。

```bash
export PATH=/usr/local/miniconda3/envs/lam/bin:$PATH
```

或正式激活环境：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lam
```

**验证**：

```bash
$ENV_PREFIX/bin/python -c "import subprocess;print(subprocess.check_output(['ninja','--version']).decode())"
# 1.13.2.git.kitware.jobserver-pipe-1
```

---

### 错误 4：`ninja is required to load C++ extensions` / 静默串行编译

> 这是**错误 3 的编译期版本**，也是本次配置中**危害最大、最隐蔽**的问题。

**现象**：pytorch3d 编译极其缓慢 —— **8 分钟只编译出 10 个目标文件**，
而它总共有 **31 个 `.cu` + 36 个 `.cpp` = 67 个编译单元**，照此速度需要 **1 小时以上**。

```
$ find pytorch3d/build -name "*.o" | wc -l
10        # 8 分钟后

$ ls pytorch3d/build/
lib.linux-x86_64-cpython-310  temp.linux-x86_64-cpython-310
#   ↑ 完全没有 build.ninja / .ninja_log
```

**根因**：pytorch3d 的 `setup.py` 使用了 `torch.utils.cpp_extension.BuildExtension`：

```python
# pytorch3d/setup.py
if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":
    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)
else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension
```

而 `BuildExtension` 内部调用 `verify_ninja_availability()`；
**当 ninja 不在 PATH 上时，它不会报错**（因为 `PYTORCH3D_NO_NINJA` 未设置，
但 torch 的 `load()` 路径与普通 `setup.py build_ext` 路径行为不同），
而是**静默退化为 setuptools 串行编译** —— 一次只跑一个 `cc1plus`。

对比实测：

| 配置 | 8 分钟产出 | 预计总耗时 |
|---|---|---|
| ninja 不在 PATH | 10 / 67 个 .o | **> 60 分钟** |
| ninja 在 PATH | 18 个 .o / 30 秒 | **~3 分钟** |

**解决**：同错误 3，把 `$ENV_PREFIX/bin` 加进 PATH，然后**清掉 build 目录重编**：

```bash
export PATH=/usr/local/miniconda3/envs/lam/bin:$PATH
ninja --version          # 必须先确认能找到
cd thirdparty_src/pytorch3d
rm -rf build             # 清掉串行编译的残留
FORCE_CUDA=1 $PIP install . --no-build-isolation
```

**验证**：编译期间观察并行度

```bash
ps -eo comm | grep -cE "cc1plus|cicc"   # 应接近 MAX_JOBS(如 16)，而不是 1
```

> ⚠️ **通用教训**：任何用 `torch.utils.cpp_extension` 的项目，
> 编译前请务必执行 `python -c "import subprocess;subprocess.check_output(['ninja','--version'])"`
> 确认 ninja 可达，否则会浪费大量时间在串行编译上，而且**没有任何报错提示**。

---

### 错误 5：`No module named modelscope.__main__`

**完整报错**：

```
$ python -m modelscope download --model "Damo_XR_Lab/LAM-assets" --local_dir "./tmp/"
/usr/local/miniconda3/envs/lam/bin/python: No module named modelscope.__main__;
'modelscope' is a package and cannot be directly executed
```

连带失败：

```
$ tar -xf ./tmp/LAM_assets.tar
tar: ./tmp/LAM_assets.tar: Cannot open: No such file or directory
tar: Error is not recoverable: exiting now
```

**根因**：LAM 的 README（以及 ModelScope 官方旧文档）使用
`python -m modelscope download ...` 调用方式，
但 **modelscope ≥ 1.40 不再提供 `__main__.py`**，改为提供 console-script 入口。

实测安装的是 `modelscope 1.40.1`：

```bash
$ ls $ENV_PREFIX/bin | grep modelscope
modelscope          # ← 可执行文件，这才是正确入口
modelscope-hub
```

**解决**：直接调用 `modelscope` 可执行文件（参数名也改为 `--repo-type` / `--local-dir`）：

```bash
$ENV_PREFIX/bin/modelscope download \
    --repo-type model Damo_XR_Lab/LAM-assets --local-dir ./tmp/
```

下载 LAM-20K 权重（注意 README 写的是位置参数形式，新版同样支持）：

```bash
$ENV_PREFIX/bin/modelscope download \
    --repo-type model Damo_XR_Lab/LAM-20K \
    --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

**验证**：

```bash
$ENV_PREFIX/bin/modelscope download --help    # 能打印 help
ls -la /workspace/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500/
# model.safetensors  2356556212 字节
```

---

### 错误 6：`FileNotFoundError: ./pretrained_models/human_model_files/...`

**完整报错**（使用 `PyTorchModelHubMixin.from_pretrained()` 加载权重时）：

```
  File "/workspace/LAM/./lam/models/rendering/flame_model/flame.py", line 699, in __init__
    super().__init__(shape_params=shape_params,
  File "/workspace/LAM/./lam/models/rendering/flame_model/flame.py", line 122, in __init__
    with open(flame_model_path, "rb") as f:
FileNotFoundError: [Errno 2] No such file or directory:
'./pretrained_models/human_model_files/flame_assets/flame/flame2023.pkl'
```

**根因**：权重目录里的 `config.json` 带有 **HuggingFace Space 的绝对布局路径**：

```json
"human_model_path": "./pretrained_models/human_model_files"
```

但 ModelScope/HuggingFace 下载脚本把资源放到了 `./model_zoo/human_parametric_models`。
这个 `config.json` 是给 HF Space 用的，**与仓库自身的目录约定不一致**。

**解决（推荐）**：**不要**用 `from_pretrained()`，走项目自己的推理入口
`lam.launch infer.lam` —— 它用 `configs/inference/lam-20k-8gpu.yaml` 里的
`human_model_path`（正确值 `./model_zoo/human_parametric_models`）构造模型：

```yaml
# configs/inference/lam-20k-8gpu.yaml
model:
    human_model_path: "./model_zoo/human_parametric_models"
```

即 `ModelLAM(**cfg.model)` 而非 `from_pretrained(...)`。

**替代方案**：如果你确实需要 `from_pretrained()`，可以建软链接：

```bash
cd /workspace/LAM
mkdir -p pretrained_models
ln -sfn /workspace/LAM/model_zoo/human_parametric_models \
        pretrained_models/human_model_files
```

**验证**（正确的加载方式，835 个张量全部匹配）：

```python
import sys; sys.path.insert(0, '.')
from omegaconf import OmegaConf
from lam.models import ModelLAM
from safetensors.torch import load_file

cfg = OmegaConf.load('configs/inference/lam-20k-8gpu.yaml')
model = ModelLAM(**cfg.model)
ckpt = load_file('model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors')
sd = model.state_dict()
loaded = sum(1 for k, v in ckpt.items() if k in sd and sd[k].shape == v.shape)
print(loaded, len(ckpt))
```

实测输出：

```
human_model_path = ./model_zoo/human_parametric_models
#########scale sphere:False, add_teeth:False
face_upsampled:(39904, 3), face_ori:torch.Size([9976, 3]), vertex_num_pixels...
ModelLAM constructed OK in 7.1s
checkpoint tensors: 835 loaded, 0 unexpected, 0 shape-mismatch
MODEL OK on cuda, VRAM 2.20 GB
```

---

### 错误 7：DINOv2 主干下载极慢 + `.partial` 文件冲突

**现象**：

```
Downloading: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
to /root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
  0%|          | 2.38M/1.13G [00:09<1:08:05, 297kB/s]
```

实测速度仅 **~300–350 KB/s**，1.13GB 需要约 **1 小时**。
并且缓存目录里出现了两个竞争的 partial 文件：

```
dinov2_vitl14_reg4_pretrain.pth.4b9cd2b3ac524cf5be9091d7686ebfef.partial   335 MB
dinov2_vitl14_reg4_pretrain.pth.5d387df93b7d40b8b256e8c44c71b4cb.partial     5 MB
```

**根因**：`lam/models/encoders/dinov2/hub/utils.py` 硬编码了 Meta 的源：

```python
_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
```

该域名从国内主机速度很差。若同时启动两个加载进程，会各自写一份 `.partial`。

**解决**：从 HuggingFace 镜像**预置到 torch 的缓存目录**，让
`torch.hub.load_state_dict_from_url` 直接命中缓存、完全跳过下载。

```bash
DEST=/root/.cache/torch/hub/checkpoints
FILE=dinov2_vitl14_reg4_pretrain.pth
mkdir -p $DEST
rm -f $DEST/*.partial                    # 清掉残留
curl -L --retry 3 -o $DEST/$FILE.tmp \
  "https://huggingface.co/showstarpro/dinov2_vitl14_reg4_pretrain/resolve/main/$FILE"
mv $DEST/$FILE.tmp $DEST/$FILE
ls -la $DEST/$FILE                       # 1217607321 字节
```

实测速度对比：

| 源 | 速度 | 1.13GB 耗时 |
|---|---|---|
| `dl.fbaipublicfiles.com`（默认） | ~300 KB/s | ~60 min |
| `huggingface.co` 镜像 | **~15 MB/s** | **~1.3 min** |

> ⚠️ **两个关键细节**：
> 1. 文件名必须**逐字符一致** —— torch.hub 用 URL 的文件名部分做缓存键。
> 2. 预置前请**确认没有正在运行的加载进程**，否则它会继续写自己的 `.partial`
>    并在结束时覆盖你的文件。

---

### 错误 10：`python app_lam.py` 启动 Gradio 报 `ValueError: When localhost is not accessible...`

**完整报错**（模型、权重、FLAME 全部加载成功之后，最后一步启动 Web 服务时才失败）：

```
  File "/workspace/LAM/app_lam.py", line 552, in <module>
    launch_gradio_app()
  File "/workspace/LAM/app_lam.py", line 495, in demo_lam
    demo.launch()
  File ".../gradio/blocks.py", line 1982, in launch
    raise ValueError(
ValueError: When localhost is not accessible, a shareable link must be created.
Please set share=True or check your proxy settings to allow access to localhost.
```

**这个报错极具误导性** —— 它提示你去设 `share=True` 或检查代理，
但真实原因跟网络/代理**完全无关**。

**真正根因**：`pip install -r requirements.txt` 时，`gradio==3.44.3` 声明的
`fastapi` 与 `uvicorn` **没有版本上限**，于是 pip 装上了最新的
`fastapi 0.141.1` + `starlette 1.7.0`。而 gradio 3.44.3 的代码是按**旧版 starlette API** 写的：

```python
# gradio/routes.py 第 313 行（旧写法：位置参数）
return templates.TemplateResponse(
    template,                                    # ← 旧 API：第 1 个参数是模板名
    {"request": request, "config": config},
)
```

新版 starlette 的签名已改成 `TemplateResponse(request, name, context)`：

```python
# starlette/templating.py（新版）
def TemplateResponse(self, request, name, context=None, ...)
```

参数错位后，`request` 对象被当成字典 key 去查 Jinja2 缓存，抛：

```
TypeError: unhashable type: 'dict'
  File ".../jinja2/utils.py", line 515, in __getitem__
    rv = self._mapping[key]
```

**连锁反应**：首页返回 HTTP 500 → gradio 的 `networking.url_ok()` 探测本地 URL 失败
（它只接受 200/401/302）→ 于是抛出那句"localhost 不可访问"。

**验证根因**（能复现出真正错误）：

```bash
python -c "
import requests
print(requests.get('http://127.0.0.1:7860/', timeout=5).status_code)   # 500，而不是连接失败
"
```

**解决**：把 web 栈降级到 gradio 3.44.3 兼容的版本。

```bash
$PIP install "starlette==0.27.0" "fastapi==0.103.2" "uvicorn==0.23.2"
```

**另外还要注意 pydantic**：降级 fastapi 后若 pydantic 仍是 2.13+，会报另一个错：

```
AttributeError: 'FieldInfo' object has no attribute 'in_'
  File ".../fastapi/dependencies/utils.py", line 462, in add_param_to_fields
    if field_info.in_ == params.ParamTypes.path:
```

因为 pydantic 2.13 移除了 `FieldInfo.in_`。LAM 的 HF Space 明确把 pydantic 固定为 2.8.0：

```bash
$PIP install "pydantic==2.8.0"
```

**完整修复（三条一起装）**：

```bash
$PIP install "starlette==0.27.0" "fastapi==0.103.2" "uvicorn==0.23.2" "pydantic==2.8.0"
```

**验证**：

```bash
cd /workspace/LAM
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
python app_lam.py
# 期望看到: Running on local URL:  http://127.0.0.1:7860

# 另开一个终端
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/     # 200
```

实测修复后：HTTP **200**，页面 **766 KB**，含 34 个组件（Input Image / Input Video /
Processed Image / Rendered Video）。

**影响范围**：此修复只涉及 Web 层，**不影响命令行推理**。
修复后已回归验证 `run_infer.sh` 仍正常出片（0 Traceback）。

---

## 5. 官方脚本的 3 个隐藏 Bug

这三个问题不影响"能否跑通"，但会让你困惑或得到非预期结果。

### Bug 1：`scripts/inference.sh` 使用未定义的 `$EXPORT_MESH`

```bash
# scripts/inference.sh
SAVE_PLY=false
SAVE_IMG=false
VIS_MOTION=false
EXPORT_VIDEO=true
CROSS_ID=false
# ... 通篇没有定义 EXPORT_MESH ...

CUDA_VISIBLE_DEVICES=$device python -m lam.launch infer.lam --config $TRAIN_CONFIG \
        ...
        export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH \   # ← 展开成空字符串
```

shell 会把空变量展开为 `export_mesh=`，tyro/OmegaConf 解析后 `export_mesh` 变成 `None`：

```python
>>> parse_configs()  # 传入 'export_mesh='
PARSED export_mesh = None
```

因为 `parse_configs()` 里有 `assert cfg.export_video or cfg.export_mesh`，
只要 `EXPORT_VIDEO=true` 就不会触发断言，所以**这个 bug 平时不会报错，只是静默把 mesh 导出关掉了**。

**解决**：显式传入 `EXPORT_MESH`。

```bash
export EXPORT_MESH=false        # 或 true
sh scripts/inference.sh
```

### Bug 2：`export_mesh` 参数是"死参数"

`lam/runners/infer/lam.py` 中，`export_mesh` 只在两处出现：

- 第 285 行：函数签名 `export_mesh: bool,`
- 第 488 行：调用处 `export_mesh=self.cfg.export_mesh,`

**函数体内从未读取它**。真正控制网格导出的是 `save_img`：

```python
# lam/runners/infer/lam.py 第 366 行起
if save_img and dump_image_dir is not None:
    for i in range(rgb.shape[0]):
        save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
        Image.fromarray(only_pred[i]).save(save_file)
        if save_ply and dump_mesh_path is not None:
            res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

    dump_cano_dir = "./exps/cano_gs/"
    ...
    res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)
    mesh.export(os.path.join(dump_cano_dir, ..., '_shaped_mesh.obj'))
    # → *_textured_mesh.obj
```

**结论**：想导出网格，**必须传 `save_img=true`**（想要每帧 PLY 再加 `save_ply=true`），
传 `export_mesh=true` 是没有任何效果的。

```bash
# 正确：导出网格
... export_mesh=true save_img=true save_ply=true
```

实测产出：

```
exps/cano_gs/
├── status_gs_offset.ply        1.36 MB   # 高斯点云
├── status_shaped_mesh.obj      345 KB    # 几何网格
├── status_textured_mesh.obj    1.55 MB   # 带贴图网格
└── framework_img.obj           1.41 MB
exps/images/lam/lam_20k/status/  # 515 个逐帧 .ply
```

### Bug 3：`make.sh` 调用系统 `python3`

```bash
# external/landmark_detection/FaceBoxesV2/utils/make.sh
#!/usr/bin/env bash
python3 build.py build_ext --inplace      # ← python3 = /usr/bin/python3
```

`python3` 解析到系统解释器（本机为 `/usr/bin/python3`，没有 numpy/Cython），
而**非交互式 shell 不会自动激活 conda 环境**，所以 `sh make.sh` 会失败。

**解决**：直接用环境内的 python 执行 `build.py`：

```bash
$ENV_PREFIX/bin/python build.py build_ext --inplace
```

`app_lam.py` / `app_hf_space.py` 里的 `compile_module()` 也有同样问题（它调 `sh make.sh`），
所以若通过 Gradio 启动，请先手动编译好 FaceBoxesV2 扩展。

---

## 6. 验证清单

逐项执行，全部通过即为配置成功。

### 6.1 依赖完整性

```bash
PY=/usr/local/miniconda3/envs/lam/bin/python

$PY -c "
import torch, numpy, scipy
print('torch      ', torch.__version__, '| cuda avail', torch.cuda.is_available())
print('numpy/scipy', numpy.__version__, scipy.__version__)
assert numpy.__version__.startswith('1.23'), 'numpy 版本错误！'
import diff_gaussian_rasterization, simple_knn._C, nvdiffrast, pytorch3d
print('DGR / simple_knn / nvdiffrast', nvdiffrast.__version__, '/ pytorch3d', pytorch3d.__version__)
import tensorflow as tf; print('tensorflow', tf.__version__)
print('全部依赖 OK')
"
```

### 6.2 ninja / nvcc 可达性（编译前必查）

```bash
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
ninja --version
nvcc --version | tail -2
$PY -c "import subprocess;subprocess.check_output(['ninja','--version']);print('ninja 对 torch 可见')"
```

### 6.3 FaceBoxesV2 扩展

```bash
$PY -c "
import sys; sys.path.insert(0,'/workspace/LAM/external/landmark_detection/FaceBoxesV2/utils')
from nms.cpu_nms import cpu_nms; print('FaceBoxesV2 cpu_nms OK')"
```

### 6.4 FLAME 模型 + pytorch3d 网格运算

```bash
cd /workspace/LAM && $PY -c "
import sys; sys.path.insert(0,'.')
from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
h='./model_zoo/human_parametric_models'
f = FlameHeadSubdivided(300,100,add_teeth=False,add_shoulder=False,
  flame_model_path=f'{h}/flame_assets/flame/flame2023.pkl',
  flame_lmk_embedding_path=f'{h}/flame_assets/flame/landmark_embedding_with_eyes.npy',
  flame_template_mesh_path=f'{h}/flame_assets/flame/head_template_mesh.obj',
  flame_parts_path=f'{h}/flame_assets/flame/FLAME_masks.pkl',
  subdivide_num=1, teeth_bs_flag=False, oral_mesh_flag=False)
print('FLAME OK, v_template', tuple(f.v_template.shape))
"
```

### 6.5 端到端推理（最终验收）

```bash
cd /workspace/LAM
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=8.9
ulimit -n 65535

$PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0
```

**预期输出**：

```
exps/videos/lam/lam_20k/status.mp4          661 KB   # 无声视频
exps/videos/lam/lam_20k/status_audio.mp4    923 KB   # 带音频
```

**实测结果**：

```
frames: 520   fps: 30.0   size: (512, 512)
EXIT_INFER=0                 # 零报错
```

### 6.6 Gradio 应用（可选）

```bash
cd /workspace/LAM
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
python app_lam.py
# demo.launch() 默认监听 0.0.0.0:7860
```

> ⚠️ 启动前请确保 FaceBoxesV2 扩展**已手动编译**（Step 5）。
> `app_lam.py` 内部的 `compile_module()` 会调用 `sh make.sh`，
> 而 `make.sh` 用的是系统 `python3`，在非交互式环境下会失败（见 [Bug 3](#bug-3makeh-调用系统-python3)）。

---

## 7. 常见问题 FAQ

### Q1：`python` / `python3` 找不到 torch？

非交互式 shell **不会**自动激活 conda 环境。请用绝对路径，或显式激活：

```bash
PY=/usr/local/miniconda3/envs/lam/bin/python
# 或
source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate lam
```

### Q2：`RuntimeError: The detected CUDA version (12.8) mismatches ... (12.1)`？

**这不是错误，只是警告**。torch 只在 **major** 版本不同时抛异常：

```python
# torch/utils/cpp_extension.py
if cuda_ver.major != torch_cuda_version.major:
    raise RuntimeError(CUDA_MISMATCH_MESSAGE...)     # 11.x vs 12.x → 报错
warnings.warn(CUDA_MISMATCH_WARN...)                 # 12.8 vs 12.1 → 仅告警
```

12.1 与 12.8 同属 CUDA 12，可以正常编译。

### Q3：`ulimit -n` 太小导致 DataLoader 报 `Too many open files`

本机默认 `ulimit -n` 只有 **1024**，训练时多 worker DataLoader 很容易耗尽。
建议在脚本开头加上：

```bash
ulimit -n 65535
```

### Q4：编译扩展时内存不足 / 卡死？

限制并行度：

```bash
export MAX_JOBS=4        # 降低并行编译数
```

### Q5：`/model` 数据盘只读，能不能放权重？

不能写。本项目所有产出（权重、`exps/`）都放在容器盘 `/workspace/LAM` 下。
若需长期保存，请在容器销毁前复制出去。

### Q6：`tensorflow/tsl/cuda/cudart_stub.cc: Could not find cuda drivers`

**可忽略的提示**。tensorflow 2.12 是 CPU 包且探测不到 CUDA，
本项目中 tensorflow 只被 `face-detection-tflite`（虹膜关键点检测）使用，
实际推理走 CPU 完全够用。`TF-TRT Warning: Could not find TensorRT` 同理。

### Q7：能否升级到 torch 2.4 / 2.5？

可以，但要同步换掉三个东西：

1. `xformers` 需匹配新 torch 版本
2. `pytorch3d` 需换预编译 wheel 源：
   `https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html`
   （官方 HF Space 用的就是 `torch 2.4.0 + pytorch3d 0.7.8`）
3. `numpy` 仍需锁 1.x（受 tensorflow 2.12 约束）

本指南选择 torch 2.3.0 是因为它是**项目自带 `install_cu121.sh` 的官方版本**，兼容性最有保障。

### Q8：为什么不用 pip 直接装 `git+https://.../pytorch3d.git`？

因为 pip 会把仓库克隆到临时目录后构建，你**无法**传递：

- `--no-build-isolation`（构建隔离环境里没有 torch，必然失败）
- `TORCH_CUDA_ARCH_LIST`（否则会为所有架构编译，耗时暴增数倍）
- `FORCE_CUDA=1`

所以务必"先克隆，再本地 `pip install .`"。

---

## 8. 性能与资源参考

### 8.1 各阶段耗时（16 vCPU / RTX 4090）

| 阶段 | 耗时 |
|---|---|
| apt 装 g++ | < 1 min |
| conda 创建环境 | ~1 min |
| PyTorch + nvidia 依赖下载 | ~6 min（~3.5 GB） |
| requirements.txt 依赖 | ~4 min |
| chumpy 修复 | < 1 min |
| diff-gaussian-rasterization 编译 | ~1.5 min |
| simple-knn 编译 | ~1 min |
| nvdiffrast 编译 | ~1 min |
| pytorch3d 编译（ninja 启用） | ~3 min |
| FaceBoxesV2 Cython | < 1 min |
| 权重下载（~2.7 GB） | ~3 min |
| DINOv2 主干（1.13 GB，HF 镜像） | ~1.5 min |
| **合计** | **约 25–30 分钟** |

> 若不修 ninja 问题，仅 pytorch3d 一步就会多花 **1 小时以上**。

### 8.2 推理性能

| 指标 | 实测值 |
|---|---|
| 模型显存占用 | 2.20 GB |
| 单张图 + 520 帧动作序列 | ~41 s（含 FLAME 跟踪优化） |
| 单张图 + 264 帧动作序列 | ~19 s |
| 输出规格 | 512×512 @ 30fps |

### 8.3 最终产物

| 类型 | 路径 |
|---|---|
| 视频（带音频） | `exps/videos/lam/lam_20k/<name>_audio.mp4` |
| 视频（无声） | `exps/videos/lam/lam_20k/<name>.mp4` |
| 高斯点云 | `exps/cano_gs/<name>_gs_offset.ply` |
| 几何网格 | `exps/cano_gs/<name>_shaped_mesh.obj` |
| 带贴图网格 | `exps/cano_gs/<name>_textured_mesh.obj` |
| 逐帧 PLY（需 `save_ply=true`） | `exps/images/lam/lam_20k/<name>/*.ply` |

---

## 附录 A：完整环境导出

```bash
$ENV_PREFIX/bin/pip freeze > requirements-lock.txt
conda env export -n lam > environment-lam.yml
```

## 附录 B：错误速查表

| # | 报错关键词 | 一句话解决 |
|---|---|---|
| 1 | `g++: command not found` | `apt-get install -y g++` |
| 2 | `No module named 'pip'` (chumpy) | `pip install chumpy==0.70 --no-build-isolation --no-deps` |
| 2b | numpy 被升到 2.x | `pip install numpy==1.23.0 scipy==1.10.1` |
| 3 | `Ninja is required to load C++ extensions` | `export PATH=$ENV_PREFIX/bin:$PATH` |
| 4 | pytorch3d 编译极慢（串行） | 同上 + `rm -rf build` 后重编 |
| 5 | `No module named modelscope.__main__` | 用 `$ENV_PREFIX/bin/modelscope` 而非 `python -m modelscope` |
| 6 | `FileNotFoundError: ./pretrained_models/...` | 用 `lam.launch infer.lam`（走 cfg.model），别用 `from_pretrained()` |
| 7 | DINOv2 下载 300KB/s | 从 HF 镜像预置到 `~/.cache/torch/hub/checkpoints/` |
| — | Bug 1 | `inference.sh` 需显式 `export EXPORT_MESH=false` |
| — | Bug 2 | 导出网格要用 `save_img=true`，`export_mesh` 是死参数 |
| — | Bug 3 | `make.sh` 里的 `python3` 需换成环境内的 python |
| 8 | `FileNotFoundError: .../export/transforms.json` | `motion_seqs_dir` 末尾漏了斜杠 |
| 9 | assets 里混入 macOS `._*` 垃圾文件 | `find assets -name "._*" -delete` |
| 10 | `ValueError: When localhost is not accessible` | 降级 `starlette==0.27.0 fastapi==0.103.2 uvicorn==0.23.2 pydantic==2.8.0` |
| 10b | `'FieldInfo' object has no attribute 'in_'` | 同上（pydantic 需降到 2.8.0） |

> 推理相关的更多坑点（斜杠、`save_img`、帧数等）见
> [`LAM_INFERENCE_GUIDE.md`](./LAM_INFERENCE_GUIDE.md)。

---

## 附录 C：目录结构总览

```
/workspace/LAM/
├── install_lam.sh                    # ★ 本指南附带的一键安装脚本
├── LAM_ENV_SETUP_GUIDE.md            # ★ 本指南（环境配置）
├── LAM_INFERENCE_GUIDE.md            # ★ 推理测试指南
├── run_infer.sh                      # ★ 一键推理脚本
├── app_lam.py                        # Gradio 应用入口
├── requirements.txt                  # 官方依赖（含 4 个 git+ 编译依赖）
├── configs/inference/lam-20k-8gpu.yaml
├── scripts/
│   ├── inference.sh                  # 推理入口（有 EXPORT_MESH bug）
│   └── install/install_cu121.sh      # 官方安装脚本
├── lam/                              # 核心代码
├── vhap/                             # FLAME 跟踪（third party）
├── external/landmark_detection/      # 人脸关键点检测
│   └── FaceBoxesV2/utils/nms/*.so    # ★ 需手动编译
├── thirdparty_src/                   # ★ 4 个扩展的源码与编译产物
│   ├── diff-gaussian-rasterization/
│   ├── simple-knn/
│   ├── nvdiffrast/
│   └── pytorch3d/
├── model_zoo/                        # ★ 权重（Step 6 下载）
├── assets/                           # ★ 示例输入与动作（Step 6 下载）
└── exps/                             # 推理输出
```
