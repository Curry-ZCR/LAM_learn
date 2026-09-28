#!/usr/bin/env bash
# Step 9: second inference run on a different identity + motion sequence, to confirm
# the environment is reproducibly working (not a one-off).
set -x
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/messi.png \
  export_video=true export_mesh=true motion_seqs_dir=assets/sample_motion/export/Donald_Trump/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 2>&1 | tail -40
echo "EXIT_INFER2=${PIPESTATUS[0]}"
