#!/usr/bin/env bash
# Step 10: produce an actual mesh. Source reading shows `export_mesh` is a DEAD
# parameter in infer_single(): it is declared (line 285) and passed (line 488) but
# never read. The PLY/OBJ exports are actually gated by `save_img`:
#     if save_img and dump_image_dir is not None:
#         for i in ...: if save_ply ...: save_ply(...)
#         ... exps/cano_gs/*.ply / *_shaped_mesh.obj
# So to get meshes you must pass save_img=true (and save_ply=true for per-frame PLY).
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
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=true \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=true save_img=true \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 2>&1 | tail -25
echo "EXIT_MESH=${PIPESTATUS[0]}"
echo "===== MESH ARTIFACTS ====="
find /workspace/LAM/exps -name "*.ply" -o -name "*.obj" 2>/dev/null | head -10
du -sh /workspace/LAM/exps/cano_gs 2>/dev/null
