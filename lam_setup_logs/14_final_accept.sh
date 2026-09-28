#!/usr/bin/env bash
# 最终验收：完全按《推理测试指南》§0 的命令执行，从冷 shell 开始，
# 并在最后核对输出与 §1.4 的斜杠坑点。
set -o pipefail
export PATH=/usr/local/miniconda3/envs/lam/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/workspace/LAM
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535
cd /workspace/LAM

echo "########## 前置检查 (§1.1) ##########"
command -v python && python -V
command -v ninja && ninja --version
python -c "import torch;print('cuda:',torch.cuda.is_available())"

echo
echo "########## 记录运行前的输出状态 ##########"
BEFORE=$(md5sum exps/videos/lam/lam_20k/status_audio.mp4 2>/dev/null | awk '{print $1}')
echo "旧 status_audio.mp4 md5: ${BEFORE:-<不存在>}"
rm -f exps/videos/lam/lam_20k/status_audio.mp4 exps/videos/lam/lam_20k/status.mp4

echo
echo "########## §0 快速开始命令（逐字复制）##########"
START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input=assets/sample_input/status.png \
  export_video=true export_mesh=false \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=false save_img=false \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 > /tmp/final_accept.log 2>&1
RC=$?
END=$(date +%s)
echo "退出码=$RC   耗时=$((END-START))s"

echo
echo "########## 输出核对 ##########"
ls -la exps/videos/lam/lam_20k/status*.mp4
echo "Traceback 数量 (必须为 0): $(grep -c Traceback /tmp/final_accept.log)"
python -c "
import imageio
r=imageio.get_reader('exps/videos/lam/lam_20k/status_audio.mp4','ffmpeg')
md=r.get_meta_data()
print('规格: fps=%s size=%s' % (md.get('fps'), md.get('size')))
assert md.get('size')==(512,512), 'SIZE WRONG'
print('>>> 最终验收 PASS')
"
