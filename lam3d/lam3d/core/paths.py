"""路径与运行时常量。

所有路径集中在这里，方便把本模块搬到别的机器上时只改一处。
默认指向 /workspace 下的 LAM 检出；可用环境变量覆盖。
"""
from __future__ import annotations

import os
from pathlib import Path

# --- LAM 检出位置 -----------------------------------------------------------
LAM_ROOT = Path(os.environ.get("LAM_ROOT", "/workspace/LAM")).resolve()

# --- 本模块自己的位置（与 LAM 解耦，便于整体拷贝） --------------------------
LAM3D_ROOT = Path(__file__).resolve().parents[2]

# --- 由 LAM 提供的资源 ------------------------------------------------------
MODEL_ZOO = LAM_ROOT / "model_zoo"
LAM_CKPT = MODEL_ZOO / "lam_models/releases/lam/lam-20k/step_045500"
LAM_INFER_CFG = LAM_ROOT / "configs/inference/lam-20k-8gpu.yaml"
TRACKING_MODELS = MODEL_ZOO / "flame_tracking_models"
HUMAN_PARAMETRIC_MODELS = MODEL_ZOO / "human_parametric_models"
FLAME_ASSETS = HUMAN_PARAMETRIC_MODELS / "flame_assets/flame"
SAMPLE_INPUT_ROOT = LAM_ROOT / "assets/sample_input"

# --- 本模块自己的产物 -------------------------------------------------------
WORK_ROOT = Path(os.environ.get("LAM3D_WORK", LAM3D_ROOT / "workspace")).resolve()
STATIC_DIR = LAM3D_ROOT / "static"
VENDOR_DIR = STATIC_DIR / "vendor"

# --- 推理用到的固定超参（与 run_infer.sh / app_lam.py 保持一致） -------------
RENDER_FPS = 30
SOURCE_SIZE = 512
RENDER_SIZE = 512
ASPECT_STANDARD = 1.0 / 1.0

# FLAME 头部：20018 顶点 / 39904 面的细分网格，与 LAM 的高斯点数严格一一对应
FLAME_SHAPE_PARAMS = 300
FLAME_EXPR_PARAMS = 100
FLAME_SUBDIVIDE_NUM = 1

# 导出网格时的颈部裁剪量（单位与 FLAME 一致，头高约 0.32）。
# LAM 的 FLAME 以 add_shoulder=False 构建，只见过「头 + 颈」；颈部以下
# （肩、胸像底座）是外推区，实测会长出跨接三角面。以此值上抬裁剪面。
# 设为 0 可完全关闭裁剪（保留上游的原始行为）。
NECK_TRIM_MARGIN = float(os.environ.get("LAM3D_NECK_TRIM", "0.03"))

# 导出网格前的拉普拉斯平滑次数与强度。
# 高斯 offset 无邻域约束，直接连面会自交起刺；轻度平滑后表面才连续。
# 设为 0 可关闭（保留上游原始几何）。
SMOOTH_ITERS = int(os.environ.get("LAM3D_SMOOTH_ITERS", "3"))
SMOOTH_LAMBDA = float(os.environ.get("LAM3D_SMOOTH_LAMBDA", "0.5"))

__all__ = [
    "LAM_ROOT",
    "LAM3D_ROOT",
    "MODEL_ZOO",
    "LAM_CKPT",
    "LAM_INFER_CFG",
    "TRACKING_MODELS",
    "HUMAN_PARAMETRIC_MODELS",
    "FLAME_ASSETS",
    "SAMPLE_INPUT_ROOT",
    "WORK_ROOT",
    "STATIC_DIR",
    "VENDOR_DIR",
    "RENDER_FPS",
    "SOURCE_SIZE",
    "RENDER_SIZE",
    "ASPECT_STANDARD",
    "FLAME_SHAPE_PARAMS",
    "FLAME_EXPR_PARAMS",
    "FLAME_SUBDIVIDE_NUM",
    "NECK_TRIM_MARGIN",
    "SMOOTH_ITERS",
    "SMOOTH_LAMBDA",
]
