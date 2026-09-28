#!/usr/bin/env bash
# Step 7: load the LAM-20K checkpoint the way app_lam.py / the inference runner does.
# This exercises: huggingface_hub PyTorchModelHubMixin from_pretrained -> safetensors
# load -> FLAME head construction -> pytorch3d mesh ops.
set -x
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM
$PY - <<'EOF' 2>&1
import sys, torch, time
sys.path.insert(0, '.')
from lam.models import ModelLAM
from lam.utils.hf_hub import wrap_model_hub

t = time.time()
cls = wrap_model_hub(ModelLAM)
m = cls.from_pretrained('./model_zoo/lam_models/releases/lam/lam-20k/step_045500/')
print('MODEL LOADED OK in %.1fs' % (time.time()-t))
n = sum(p.numel() for p in m.parameters())
print('params: %.1fM' % (n/1e6))
m = m.to('cuda').eval()
print('moved to cuda OK, VRAM %.2f GB' % (torch.cuda.memory_allocated()/2**30))
EOF
echo "EXIT_LOAD=$?"
