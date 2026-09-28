#!/usr/bin/env bash
# 6.6 Final end-to-end acceptance: fresh inference into a CLEAN output dir,
# from a cold shell with no inherited session state.
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
PY=/usr/local/miniconda3/envs/lam/bin/python
cd /workspace/LAM

# move previous outputs aside so we prove fresh generation, not stale files
rm -rf /workspace/accept_out && mkdir -p /workspace/accept_out
mv exps /workspace/accept_out/exps_prev 2>/dev/null || true

START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $PY -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 > /tmp/accept_infer.log 2>&1
RC=$?
END=$(date +%s)
echo "EXIT_CODE=$RC   耗时=$((END-START))s"
echo "===== 报错统计 ====="
grep -cE "Traceback|RuntimeError|ModuleNotFoundError" /tmp/accept_infer.log
echo "===== 新生成的文件 ====="
ls -la exps/videos/lam/lam_20k/ 2>/dev/null
echo "===== 视频规格 ====="
$PY -c "
import imageio
r=imageio.get_reader('exps/videos/lam/lam_20k/status_audio.mp4','ffmpeg')
md=r.get_meta_data(); n=r.count_frames()
print('frames:',n,'fps:',md.get('fps'),'size:',md.get('size'))
assert n>0 and md.get('size')==(512,512), 'VIDEO BAD'
print('>>> 6.6 PASS')
"
