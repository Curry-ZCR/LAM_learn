#!/usr/bin/env bash
# Step 4c: rebuild pytorch3d with ninja ENABLED.
#
# Problem found: torch.utils.cpp_extension looks up the `ninja` executable on PATH.
# In this container the conda env's bin dir is NOT on PATH (we call the env's
# python by absolute path), so ninja was invisible and BuildExtension silently
# fell back to serial setuptools compilation -> ~1 object file per minute for 67
# translation units (est. >1 hour). Putting $CONDA_PREFIX/bin on PATH enables
# ninja and parallelises the build across MAX_JOBS workers.
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
PIP=/usr/local/miniconda3/envs/lam/bin/pip
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=16
export FORCE_CUDA=1
cd /workspace/LAM/thirdparty_src/pytorch3d

ninja --version
echo "NINJA_ON_PATH=$?"

rm -rf build
$PIP install . --no-build-isolation 2>&1 | tail -40
echo "EXIT_P3D=${PIPESTATUS[0]}"
