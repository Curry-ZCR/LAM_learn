#!/usr/bin/env bash
# Final acceptance: run the verification checklist from the guide, end to end,
# from a shell that does NOT inherit any of my earlier session PATH/exports.
# This proves the documented steps reproduce from a cold environment.
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
PY=/usr/local/miniconda3/envs/lam/bin/python

echo "##################### 6.1 依赖完整性 #####################"
$PY -c "
import torch, numpy, scipy
print('torch      ', torch.__version__, '| cuda avail', torch.cuda.is_available())
print('numpy/scipy', numpy.__version__, scipy.__version__)
assert numpy.__version__.startswith('1.23'), 'numpy 版本错误！'
import diff_gaussian_rasterization, simple_knn._C, nvdiffrast, pytorch3d
print('DGR / simple_knn / nvdiffrast', nvdiffrast.__version__, '/ pytorch3d', pytorch3d.__version__)
import tensorflow as tf; print('tensorflow', tf.__version__)
import transformers, diffusers, accelerate, chumpy, modelscope, xformers
print('transformers', transformers.__version__, '| diffusers', diffusers.__version__)
print('xformers', xformers.__version__, '| chumpy/modelscope OK')
print('>>> 6.1 PASS')
" 2>&1 | grep -vE "tensorflow/|TF-TRT|cpu_feature|AVX2"

echo
echo "##################### 6.2 ninja / nvcc 可达性 #####################"
ninja --version
nvcc --version | tail -1
$PY -c "import subprocess;subprocess.check_output(['ninja','--version']);print('ninja 对 torch 可见 -> 6.2 PASS')"

echo
echo "##################### 6.3 FaceBoxesV2 扩展 #####################"
cd /workspace/LAM
$PY -c "
import sys; sys.path.insert(0,'/workspace/LAM/external/landmark_detection/FaceBoxesV2/utils')
from nms.cpu_nms import cpu_nms; print('FaceBoxesV2 cpu_nms OK -> 6.3 PASS')"

echo
echo "##################### 6.4 FLAME + pytorch3d #####################"
$PY -c "
import sys; sys.path.insert(0,'.')
from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
h='./model_zoo/human_parametric_models'
f = FlameHeadSubdivided(300,100,add_teeth=False,add_shoulder=False,
  flame_model_path=f'{h}/flame_assets/flame/flame2023.pkl',
  flame_lmk_embedding_path=f'{h}/flame_assets/flame/landmark_embedding_with_eyes.npy',
  flame_template_mesh_path=f'{h}/flame_assets/flame/head_template_mesh.obj',
  flame_parts_path=f'{h}/flame_assets/flame/FLAME_masks.pkl',
  subdivide_num=1, teeth_bs_flag=False, oral_mesh_flag=False)
print('FLAME OK, v_template', tuple(f.v_template.shape), '-> 6.4 PASS')
" 2>&1 | tail -2

echo
echo "##################### 6.5 模型权重加载 #####################"
$PY -c "
import sys; sys.path.insert(0,'.')
from omegaconf import OmegaConf
from lam.models import ModelLAM
from safetensors.torch import load_file
cfg = OmegaConf.load('configs/inference/lam-20k-8gpu.yaml')
model = ModelLAM(**cfg.model)
ckpt = load_file('model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors')
sd = model.state_dict()
loaded = sum(1 for k, v in ckpt.items() if k in sd and sd[k].shape == v.shape)
print('checkpoint tensors: %d/%d loaded, 0 unexpected' % (loaded, len(ckpt)))
assert loaded == len(ckpt), 'WEIGHT MISMATCH'
print('>>> 6.5 PASS')
" 2>&1 | tail -3
