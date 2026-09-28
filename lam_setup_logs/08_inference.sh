#!/usr/bin/env bash
# Step 8: run the project's own inference entry point.
#
# scripts/inference.sh has a latent bug: it references $EXPORT_MESH on line 40 but
# never defines it (EXPORT_MESH is unset), so the CLI receives a bare `export_mesh=`
# which tyro cannot parse. Defining EXPORT_MESH=false reproduces the intent of the
# script's other flags (SAVE_PLY=false / SAVE_IMG=false) and keeps the run light.
set -x
export PYTHONPATH=/workspace/LAM:$PYTHONPATH
export EXPORT_MESH=false
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

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
