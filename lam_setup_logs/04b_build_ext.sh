#!/usr/bin/env bash
# Step 4b: build + install the 4 CUDA/C++ extension dependencies.
#
# Key env choices for THIS machine:
#   - System nvcc is CUDA 12.8 but torch is built against CUDA 12.1. torch's
#     cpp_extension only hard-fails on a MAJOR version mismatch, so 12.8 is
#     accepted with a warning. That is why CUDA_HOME must point at /usr/local/cuda.
#   - TORCH_CUDA_ARCH_LIST="8.9" -> build only for RTX 4090 (sm_89). Avoids
#     compiling for a dozen architectures and keeps build time sane.
#   - --no-build-isolation so the extension sees the torch we installed.
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
PIP=/usr/local/miniconda3/envs/lam/bin/pip
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=16
cd /workspace/LAM/thirdparty_src

$PIP install setuptools wheel ninja 2>&1 | tail -3

echo "############ 1/4 diff-gaussian-rasterization ############"
cd diff-gaussian-rasterization && $PIP install . --no-build-isolation 2>&1 | tail -30
echo "EXIT_DGR=${PIPESTATUS[0]}"
cd ..

echo "############ 2/4 simple-knn ############"
cd simple-knn && $PIP install . --no-build-isolation 2>&1 | tail -30
echo "EXIT_KNN=${PIPESTATUS[0]}"
cd ..

echo "############ 3/4 nvdiffrast ############"
cd nvdiffrast && $PIP install . --no-build-isolation 2>&1 | tail -30
echo "EXIT_NDR=${PIPESTATUS[0]}"
cd ..

# The git nvdiffrast is version 0.3.3 while requirements.txt pins 0.3.2 semantics
# via a git URL; the fork @backface-culling is what the project expects.
echo "############ 4/4 pytorch3d ############"
cd pytorch3d && FORCE_CUDA=1 $PIP install . --no-build-isolation 2>&1 | tail -40
echo "EXIT_P3D=${PIPESTATUS[0]}"
cd ..
