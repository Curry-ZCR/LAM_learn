#!/usr/bin/env bash
# Step 1: install PyTorch 2.3.0 + cu121 exactly as scripts/install/install_cu121.sh specifies
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
PIP=/usr/local/miniconda3/envs/lam/bin/pip

$PIP install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
echo "EXIT_TORCH=$?"

$PIP install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
echo "EXIT_XFORMERS=$?"

$PY -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('avail',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
echo "EXIT_CHECK=$?"
