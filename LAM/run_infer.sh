#!/usr/bin/env bash
# =============================================================================
#  LAM 一键推理脚本
#
#  用法:
#    bash run_infer.sh <图片路径> [动作序列名] [选项]
#
#  示例:
#    bash run_infer.sh /workspace/my_inputs/my_avatar.jpg
#    bash run_infer.sh /workspace/my_inputs/my_avatar.jpg Taylor_Swift
#    bash run_infer.sh assets/sample_input/status.png Look_In_My_Eyes mesh
#
#  第三个参数（可选）:
#    (空)  只出视频（最快，推荐）
#    mesh  额外导出 OBJ 网格和高斯点云
#
#  可用动作序列（assets/sample_motion/export/ 下）:
#    Anti_Drugs  D_ANgelo_Dinero  Donald_Trump  GEM  I_Am_Iron_Man
#    Joe_Biden  Look_In_My_Eyes  Michael_Wayne_Rosen  Pen_Pineapple_Apple_Pen
#    Speeding_Scandal  Taylor_Swift  The_Shawshank_Redemption
# =============================================================================
set -e

IMAGE=${1:?用法: bash run_infer.sh <图片路径> [动作序列名] [mesh]}
MOTION=${2:-Look_In_My_Eyes}
MODE=${3:-}

LAM_ROOT=/workspace/LAM
ENV_PREFIX=/usr/local/miniconda3/envs/lam

# 动作序列：支持两种写法
#   1) 只给名字（如 Taylor_Swift）—— 从自带示例里找
#   2) 给绝对/相对路径（如 /workspace/my_motion）—— 用自定义序列
case "$MOTION" in
    /*|.*/*) MOTION_DIR=$MOTION ;;                        # 路径形式
    *)       MOTION_DIR=$LAM_ROOT/assets/sample_motion/export/$MOTION ;;
esac
# ★ 末尾斜杠绝对不能省：代码用 os.path.dirname(motion_seqs_dir) 定位
#   transforms.json，少了斜杠会去找上一级目录，导致推理静默失败（退出码仍为 0）
MOTION_DIR="${MOTION_DIR%/}/"

# --- 环境变量（每次开新终端都必须设置）---
export PATH=$ENV_PREFIX/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$LAM_ROOT
export TORCH_CUDA_ARCH_LIST="8.9"
ulimit -n 65535

# --- 前置检查 ---
command -v ninja >/dev/null || { echo "ERROR: ninja 不在 PATH，推理会崩溃"; exit 1; }
[ -e "$IMAGE" ] || { echo "ERROR: 图片不存在: $IMAGE"; exit 1; }
[ -d "$MOTION_DIR" ] || {
    echo "ERROR: 动作序列不存在: $MOTION_DIR"
    echo "可用序列:"; ls "$LAM_ROOT/assets/sample_motion/export/"
    exit 1
}

# --- 导出网格需要 save_img=true（export_mesh 是死参数）---
if [ "$MODE" = "mesh" ]; then
    SAVE_IMG=true; SAVE_PLY=true; EXPORT_MESH=true
else
    SAVE_IMG=false; SAVE_PLY=false; EXPORT_MESH=false
fi

cd "$LAM_ROOT"
echo "图片     : $IMAGE"
echo "动作序列 : $MOTION"
echo "导出网格 : $MODE"
echo "开始推理..."

START=$(date +%s)
CUDA_VISIBLE_DEVICES=${GPU:-0} $ENV_PREFIX/bin/python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input="$IMAGE" \
  export_video=true export_mesh=$EXPORT_MESH \
  motion_seqs_dir="$MOTION_DIR" \
  motion_img_dir=null vis_motion=false motion_img_need_mask=true \
  render_fps=30 motion_video_read_fps=30 \
  save_ply=$SAVE_PLY save_img=$SAVE_IMG \
  gaga_track_type="" cross_id=false test_sample=false rank=0 nodes=0 2>&1 | tee /tmp/lam_infer_last.log

END=$(date +%s)
echo
echo "===== 耗时 $((END-START))s ====="
ERRS=$(grep -c "Traceback" /tmp/lam_infer_last.log || true)
if [ "$ERRS" != "0" ]; then
    echo "⚠️  检测到 $ERRS 个 Traceback（该图可能被跳过），请查看 /tmp/lam_infer_last.log"
fi
echo "===== 本次输出视频（按时间倒序）====="
ls -lat exps/videos/lam/lam_20k/*.mp4 2>/dev/null | head -4
if [ "$MODE" = "mesh" ]; then
    echo "===== 网格产物 ====="
    ls -la exps/cano_gs/ 2>/dev/null
fi
