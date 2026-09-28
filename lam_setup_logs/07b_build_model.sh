#!/usr/bin/env bash
# Step 7b: build the model exactly the way scripts/inference.sh -> lam.launch does.
# The inferrer uses ModelLAM(**cfg.model) from configs/inference/lam-20k-8gpu.yaml
# (human_model_path = ./model_zoo/human_parametric_models), NOT the checkpoint's own
# config.json (which still points at the HF-Space path ./pretrained_models/human_model_files).
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM
$PY - <<'EOF' 2>&1
import sys, os, torch, time
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from lam.models import ModelLAM
from safetensors.torch import load_file

cfg = OmegaConf.load('configs/inference/lam-20k-8gpu.yaml')
print('human_model_path =', cfg.model.human_model_path)

t = time.time()
model = ModelLAM(**cfg.model)
print('ModelLAM constructed OK in %.1fs' % (time.time()-t))

resume = 'model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors'
ckpt = load_file(resume, device='cpu')
sd = model.state_dict()
miss = [k for k in ckpt if k not in sd]
shape = [k for k in ckpt if k in sd and sd[k].shape != ckpt[k].shape]
loaded = 0
for k, v in ckpt.items():
    if k in sd and sd[k].shape == v.shape:
        sd[k].copy_(v); loaded += 1
print('checkpoint tensors: %d loaded, %d unexpected, %d shape-mismatch' % (loaded, len(miss), len(shape)))
if miss[:5]:   print('  unexpected sample:', miss[:5])
if shape[:5]:  print('  mismatch sample:', shape[:5])

model = model.to('cuda').eval()
torch.cuda.synchronize()
print('MODEL OK on cuda, VRAM %.2f GB' % (torch.cuda.memory_allocated()/2**30))
EOF
echo "EXIT_BUILD=$?"
