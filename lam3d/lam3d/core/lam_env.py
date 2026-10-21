"""把 /workspace/LAM 的源码与权重接进当前解释器。

LAM 不是 pip 包：推理脚本靠 shell 里设置 ``PYTHONPATH=$LAM_ROOT`` 才能
``import lam`` / ``import vhap`` / ``import external``。本模块用等价的
``sys.path`` 注入 + 工作目录切换来替代，让管线可以在任意 cwd 下被调用。

同时校验权重是否齐备，缺失时直接给出可读的中文报错，而不是等到
torch.load 阶段抛 FileNotFoundError。
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from . import paths

_LOCK = threading.RLock()
_PREPARED = False

# LAM 推理必需的权重文件：(相对 LAM_ROOT 的路径, 说明)
REQUIRED_ASSETS: list[tuple[str, str]] = [
    ("model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors", "LAM 主模型"),
    ("model_zoo/flame_tracking_models/68_keypoints_model.pkl", "关键点对齐模型"),
    ("model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd", "VGGHead 人脸检测"),
    ("model_zoo/flame_tracking_models/matting/stylematte_synth.pt", "人像抠图模型"),
    ("model_zoo/flame_tracking_models/FaceBoxesV2.pth", "FaceBoxesV2 检测器"),
    ("model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl", "FLAME 2023 模型"),
    (
        "model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy",
        "FLAME 关键点嵌入",
    ),
    (
        "model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj",
        "FLAME 模板网格",
    ),
    ("configs/inference/lam-20k-8gpu.yaml", "LAM 推理配置"),
    ("configs/vhap_tracking/base_tracking_config.yaml", "FLAME 跟踪配置"),
]


class MissingAssetsError(RuntimeError):
    """权重/配置缺失。"""


def missing_assets() -> list[tuple[str, str]]:
    """返回缺失的 (相对路径, 说明) 列表。"""
    return [(rel, desc) for rel, desc in REQUIRED_ASSETS if not (paths.LAM_ROOT / rel).exists()]


def ensure_assets() -> None:
    missing = missing_assets()
    if missing:
        lines = "\n".join(f"  - {rel}  ({desc})" for rel, desc in missing)
        raise MissingAssetsError(
            f"LAM 权重/配置缺失，共 {len(missing)} 项，请先执行 LAM/install_lam.sh：\n{lines}"
        )


def prepare() -> None:
    """幂等地把 LAM 接入 sys.path（必须在 import lam.* 之前调用）。"""
    global _PREPARED
    with _LOCK:
        if _PREPARED:
            return
        root = str(paths.LAM_ROOT)
        if not paths.LAM_ROOT.is_dir():
            raise MissingAssetsError(f"LAM_ROOT 不存在：{paths.LAM_ROOT}（可设置环境变量 LAM_ROOT 覆盖）")
        if root not in sys.path:
            sys.path.insert(0, root)
        # 若干第三方扩展（FaceBoxesV2 的 Cython 模块等）依赖 LAM 根作为模块根
        os.environ.setdefault("PYTHONPATH", root)
        _PREPARED = True


@contextmanager
def lam_cwd():
    """临时切到 LAM_ROOT 工作目录。

    LAM 内部有大量相对路径（``./model_zoo/...``、``configs/...``、``./output/...``），
    必须在 LAM 根目录下运行。
    """
    prepare()
    with _LOCK:
        prev = os.getcwd()
        os.chdir(paths.LAM_ROOT)
        try:
            yield paths.LAM_ROOT
        finally:
            os.chdir(prev)


__all__ = ["prepare", "ensure_assets", "missing_assets", "lam_cwd", "MissingAssetsError"]
