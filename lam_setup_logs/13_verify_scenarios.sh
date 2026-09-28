#!/usr/bin/env bash
# Verify inference scenario B: a custom image (not from sample_input) + custom output naming.
# Also verify scenario C: directory input (to document the .jpg-only gotcha).
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

echo "##################### B. 自定义图片 #####################"
mkdir -p /workspace/my_inputs
cp assets/sample_input/libai.jpg /workspace/my_inputs/my_avatar.jpg
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=/workspace/my_inputs/my_avatar.jpg \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Taylor_Swift/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 > /tmp/b.log 2>&1
echo "EXIT_B=$?  (0 且无 Traceback 才算成功)"
grep -cE "Traceback|RuntimeError" /tmp/b.log
ls -la exps/videos/lam/lam_20k/ | grep -i my_avatar

echo
echo "##################### C. 目录输入 (.jpg 过滤验证) #####################"
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/GEM/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 > /tmp/c.log 2>&1
echo "EXIT_C=$?"
echo "生成的视频数量（应为 7，即 .jpg 数量）:"
ls exps/videos/lam/lam_20k/*.mp4 2>/dev/null | wc -l
ls exps/videos/lam/lam_20k/ | head -20
