#!/usr/bin/env bash
# =============================================================================
#  LAM 单图生成 3D 数字人 · 启动脚本
#
#  用法:
#    bash run.sh                  # 前台启动（Ctrl-C 停止）
#    bash run.sh --daemon         # 后台启动，日志写入 logs/server.log
#    bash run.sh --stop           # 停止后台服务
#
#  环境变量:
#    LAM3D_PORT   监听端口（默认 8000）
#    LAM3D_HOST   监听地址（默认 0.0.0.0）
#    LAM3D_DEVICE cuda / cpu（默认 cuda）
#    LAM_ROOT     LAM 检出位置（默认 /workspace/LAM）
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX=${ENV_PREFIX:-/usr/local/miniconda3/envs/lam}
PY="$ENV_PREFIX/bin/python"
PORT=${LAM3D_PORT:-8000}
HOST=${LAM3D_HOST:-0.0.0.0}
LOG_DIR="$HERE/logs"
PID_FILE="$LOG_DIR/server.pid"

[ -x "$PY" ] || { echo "ERROR: Python 环境不存在：$PY"; exit 1; }

export PATH="$ENV_PREFIX/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LAM_ROOT=${LAM_ROOT:-/workspace/LAM}
export PYTHONPATH="$HERE:$LAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export NUMBA_THREADING_LAYER=omp
ulimit -n 65535 2>/dev/null || true

# ninja 必须在 PATH 里：LAM 的 4 个 CUDA 扩展在 import 时按需加载
command -v ninja >/dev/null 2>&1 || { echo "ERROR: ninja 不在 PATH，LAM 的 CUDA 扩展会崩"; exit 1; }

mkdir -p "$LOG_DIR"

if [ "${1:-}" = "--stop" ]; then
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        kill "$(cat "$PID_FILE")" && echo "已停止（pid $(cat "$PID_FILE")）"
        rm -f "$PID_FILE"
    else
        echo "没有正在运行的服务"
    fi
    exit 0
fi

if [ "${1:-}" = "--daemon" ]; then
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "服务已在运行（pid $(cat "$PID_FILE")）： http://$HOST:$PORT"
        exit 0
    fi
    nohup "$PY" -m lam3d.server.app > "$LOG_DIR/server.log" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    echo "服务已启动（pid $(cat "$PID_FILE")）"
    echo "  地址： http://<服务器IP>:$PORT"
    echo "  日志： tail -f $LOG_DIR/server.log"
    exit 0
fi

echo "启动 LAM 3D 数字人服务： http://$HOST:$PORT"
exec "$PY" -m lam3d.server.app
