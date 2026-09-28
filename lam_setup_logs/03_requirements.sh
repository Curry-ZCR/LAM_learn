#!/usr/bin/env bash
# Step 3: official dependency install from requirements.txt.
# The 4 VCS (git+) packages are stripped out here and built in step 4 from the
# pre-cloned sources, because `pip install git+...` re-clones into a temp dir and
# has no way to pass --no-build-isolation / TORCH_CUDA_ARCH_LIST to the compiler.
set -x
PIP=/usr/local/miniconda3/envs/lam/bin/pip
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

grep -vE '^git\+' requirements.txt > /workspace/lam_setup_logs/requirements_local.txt
echo "--- filtered requirements ---"
cat /workspace/lam_setup_logs/requirements_local.txt

# numpy 1.23.0 must land first: tensorflow 2.12 requires numpy<1.24 and every
# downstream C-extension in this project is compiled against the numpy 1.x C ABI.
$PIP install "numpy==1.23.0" 2>&1 | tail -5
echo "EXIT_NUMPY=$?"

$PIP install -r /workspace/lam_setup_logs/requirements_local.txt 2>&1 | tail -60
echo "EXIT_REQ=$?"
