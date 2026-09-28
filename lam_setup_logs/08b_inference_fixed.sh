#!/usr/bin/env bash
# Step 8 (fixed): same inference run, but with the conda env's bin on PATH.
#
# This is the second consequence of the "conda env bin not on PATH" problem:
# nvdiffrast does NOT use the prebuilt extension at import time only -- it JIT-
# compiles its CUDA rasterizer plugin on first use via
# torch.utils.cpp_extension.load(), and torch hard-requires the `ninja`
# executable to be discoverable on PATH:
#     RuntimeError: Ninja is required to load C++ extensions
# Putting $CONDA_PREFIX/bin on PATH fixes both compile-time and run-time.
set -x
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"
export EXPORT_MESH=false
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

ninja --version
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 2>&1
echo "EXIT_INFER=$?"
