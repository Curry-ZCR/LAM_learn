#!/usr/bin/env bash
# =============================================================================
#  LAM (Large Avatar Model) 一键环境配置脚本
#  https://github.com/aigc3d/LAM
#
#  适用环境（本脚本已在此环境完整验证）:
#    - Ubuntu 22.04 / NVIDIA RTX 4090 (sm_89) / 驱动 595.80 / CUDA Toolkit 12.8
#    - 系统 nvcc 版本与 torch 的 cu121 不同也没关系（仅 major 不同才会报错）
#    - /usr/local/miniconda3 已安装 conda
#
#  用法:  bash install_lam.sh 2>&1 | tee install_lam.log
# =============================================================================
set -euo pipefail

# ------------------------------ 可调参数 -------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/workspace/LAM}"
CONDA_ROOT="${CONDA_ROOT:-/usr/local/miniconda3}"
ENV_NAME="${ENV_NAME:-lam}"
TORCH_ARCH="${TORCH_ARCH:-8.9}"        # sm_89 = RTX 4090；A100 用 8.0，3090 用 8.6
MAX_JOBS="${MAX_JOBS:-16}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"   # modelscope | huggingface

ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

log() { echo -e "\n\033[1;36m===== $* =====\033[0m"; }

# =============================================================================
log "0. 系统依赖 (g++ / ninja 必需，否则 CUDA 扩展无法编译)"
# 关键: g++ 必须存在。纯 gcc 的镜像（本机初始状态）缺少 g++，所有 CUDA 扩展都会
# 因 "g++: command not found" 编译失败。
if ! command -v g++ >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y g++ curl git
fi
g++ --version | head -1

# =============================================================================
log "1. 创建 conda 虚拟环境 ($ENV_NAME, Python 3.10)"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
$PY -V

# >>> 贯穿全程的最重要一条 <<<
# 绝不要依赖 PATH 里的 python/ninja，始终用绝对路径调用环境内的可执行文件。
# 本机 PATH 里既没有 conda env 的 bin，也没有 CUDA 的 bin。下面这行同时解决:
#   - nvcc 需要被 torch cpp_extension 找到
#   - ninja 需要被 torch.utils.cpp_extension 找到（编译期 + nvdiffrast 运行期 JIT）
export PATH="$ENV_PREFIX/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"
export MAX_JOBS="$MAX_JOBS"

# =============================================================================
log "2. PyTorch 2.3.0 + cu121 + xformers"
$PIP install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
     --index-url https://download.pytorch.org/whl/cu121
$PIP install -U xformers==0.0.26.post1 \
     --index-url https://download.pytorch.org/whl/cu121
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# =============================================================================
log "3. 基础依赖 (requirements.txt，剔除 4 个需编译的 git 依赖)"
cd "$PROJECT_DIR"

# 3.1 numpy 必须最先固定为 1.x。tensorflow 2.12 要求 numpy<1.24，
#     而所有 CUDA 扩展也都按 numpy 1.x 的 C ABI 编译。先装 numpy 2.x 会连锁崩。
$PIP install "numpy==1.23.0"

# 3.2 chumpy: setup.py 里 `from pip._internal.req import parse_requirements` 在
#     构建隔离沙箱中找不到 pip —— 必须 --no-build-isolation；
#     同时它声明 `scipy>=0.13.0` 无上限，会拉入 scipy 1.15 + numpy 2.x，
#     所以还要 --no-deps，再手动把 scipy 固定到兼容 numpy 1.23 的版本。
$PIP install "chumpy==0.70" --no-build-isolation --no-deps
$PIP install "scipy==1.10.1" "six"

# 3.3 其余依赖（去掉 git+ 行、chumpy、numpy、scipy）
grep -vE '^git\+|@git\+|^chumpy|^numpy|^scipy' requirements.txt > /tmp/lam_req.txt
$PIP install -r /tmp/lam_req.txt "numpy==1.23.0" "scipy==1.10.1"
$PY -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

# 3.4 ★ 降级 Web 栈（否则 python app_lam.py 无法启动 Gradio）
# gradio 3.44.3 依赖的 fastapi/uvicorn 没有版本上限，pip 会装上最新的
# fastapi 0.141 + starlette 1.7，其 TemplateResponse 签名已变，导致首页 500，
# gradio 随后误报 "When localhost is not accessible"。
# pydantic 2.13 也移除了 FieldInfo.in_，需一并降到 2.8.0。
$PIP install "starlette==0.27.0" "fastapi==0.103.2" "uvicorn==0.23.2" "pydantic==2.8.0"

# =============================================================================
log "4. 编译 4 个 CUDA/C++ 扩展"
# 不从 pip 的 git+ URL 安装：那样 pip 会克隆到临时目录，无法传
# --no-build-isolation / TORCH_CUDA_ARCH_LIST，编译极易失败。改为先克隆再本地装。
mkdir -p "$PROJECT_DIR/thirdparty_src"
cd "$PROJECT_DIR/thirdparty_src"
[ -d diff-gaussian-rasterization ] || \
    git clone --recursive --depth 1 https://github.com/ashawkey/diff-gaussian-rasterization/
[ -d simple-knn ] || \
    git clone --recursive --depth 1 https://github.com/camenduru/simple-knn/
[ -d nvdiffrast ] || \
    git clone --recursive --branch backface-culling --depth 1 https://github.com/ShenhanQian/nvdiffrast
[ -d pytorch3d ] || \
    git clone --recursive --branch v0.7.6 --depth 1 https://github.com/facebookresearch/pytorch3d.git

$PIP install setuptools wheel ninja

# 必须确认 ninja 真的在 PATH 上。若不在，setuptools 会静默退化为串行编译，
# pytorch3d 的 67 个编译单元会从 ~3 分钟变成 >1 小时。
ninja --version || { echo "ERROR: ninja 不在 PATH，请检查 $ENV_PREFIX/bin"; exit 1; }

for ext in diff-gaussian-rasterization simple-knn nvdiffrast; do
    log "编译 $ext"
    ( cd "$ext" && $PIP install . --no-build-isolation )
done

log "编译 pytorch3d (最慢, 约 2-4 分钟)"
( cd pytorch3d && FORCE_CUDA=1 $PIP install . --no-build-isolation )

# =============================================================================
log "5. 编译 FaceBoxesV2 Cython 扩展"
# 项目自带的 make.sh 调用 `python3`，会命中系统 python（无 Cython/numpy）。
# 必须用 conda 环境的 python 显式执行 build.py。
cd "$PROJECT_DIR/external/landmark_detection/FaceBoxesV2/utils"
$PY build.py build_ext --inplace
ls -la nms/*.so

# =============================================================================
log "6. 下载 LAM-assets 与 LAM-20K 权重"
cd "$PROJECT_DIR"
$PIP install modelscope
MSC="$ENV_PREFIX/bin/modelscope"

if [ "$MODEL_SOURCE" = "modelscope" ]; then
    # 注意: README 写的 `python -m modelscope download` 在 modelscope>=1.40 已失效
    # （包内不再有 __main__），必须直接调用 modelscope 可执行文件。
    mkdir -p ./tmp
    $MSC download --repo-type model Damo_XR_Lab/LAM-assets --local-dir ./tmp/
    tar -xf ./tmp/LAM_assets.tar
    tar -xf ./tmp/thirdparty_models.tar
    $MSC download --repo-type model Damo_XR_Lab/LAM-20K \
        --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
    rm -rf ./tmp/
else
    $PIP install "huggingface_hub[cli]"
    huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
    tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
    tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
    huggingface-cli download 3DAIGC/LAM-20K \
        --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
fi

# =============================================================================
log "7. 预置 DINOv2 主干权重"
# LAM 的 encoder 会通过 torch.hub 下载 dinov2_vitl14_reg4_pretrain.pth。
# 官方源 dl.fbaipublicfiles.com 从本机只有 ~300KB/s（1.13GB 需约 1 小时），
# 这里改从 HuggingFace 镜像预置到 torch 的缓存目录（实测 ~15MB/s）。
DEST=/root/.cache/torch/hub/checkpoints
FILE=dinov2_vitl14_reg4_pretrain.pth
mkdir -p "$DEST"
if [ ! -f "$DEST/$FILE" ]; then
    curl -L --retry 3 -o "$DEST/$FILE.tmp" \
      "https://huggingface.co/showstarpro/dinov2_vitl14_reg4_pretrain/resolve/main/$FILE"
    mv "$DEST/$FILE.tmp" "$DEST/$FILE"
fi
ls -la "$DEST/$FILE"

# =============================================================================
log "8. 验证安装"
$PY - <<'PYEOF'
import torch, numpy, scipy
print('torch      ', torch.__version__, '| cuda', torch.cuda.is_available())
print('numpy/scipy', numpy.__version__, scipy.__version__)
import diff_gaussian_rasterization, simple_knn._C, nvdiffrast, pytorch3d
print('CUDA 扩展    OK: dgr / simple_knn / nvdiffrast', nvdiffrast.__version__, '/ pytorch3d', pytorch3d.__version__)
import fastapi, starlette, pydantic
print('Web 栈       fastapi', fastapi.__version__, '| starlette', starlette.__version__, '| pydantic', pydantic.VERSION)
PYEOF

log "9. 端到端推理自检"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
ulimit -n 65535
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0

log "完成！输出视频: exps/videos/lam/lam_20k/status_audio.mp4"
