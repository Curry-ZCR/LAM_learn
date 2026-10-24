"""ARKit 实时表情驱动：把 52 个 ARKit 通道渲染成帧 / 编成 LSPL。

一次「单图 → 3D 人物」跑完后，规范高斯（``gs_model_list``）、查询点、
**中性** FLAME 参数与首帧相机都还在显存里。本模块把它们存成一份 *live
session*，之后每个 ARKit 向量有两条出口::

    compute_splat(...)  ->  animate_gs_model(..., arkit_params=B)  # 只算属性
    render_frame(...)   ->  同上 + forward_single_batch(...)       # 光栅化一帧

前者给 ``POST /arkit_splat``（LSPL 二进制，见 ``core/splat.py`` 与
CONTRACT_v2.md §4），只做 LBS + Δq/Δs 调制与一次内存打包，**不做光栅化**，
所以能稳定满足 X-LSPL-Ms <= 30ms；后者给 ``POST /arkit_frame``（PNG 预览）。

会话里的 ``flame_data`` 是**完全中性**的基准表情：rotation / neck_pose /
eyes_pose / jaw_pose / translation / expr 全部置零，只保留 betas（身份形状）。
这样 ARKit 全 0 时前后端看到的是「不带动画的本人脸」，且与「不传 arkit」
的静态路径逐位一致。

会话状态挂在 pipeline 对象上（``pipeline.arkit_session``），并在 job 目录写
``arkit_session.json`` 作为就绪标记；旧任务被新任务顶掉后自动失效。
"""
from __future__ import annotations

import importlib.util
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import cov_switch, paths, splat, tga_covariance

# 实时渲染串行锁：光栅化与 animate_gs_model 都不是线程安全的，
# 且 JobManager 本身也是单 GPU 串行队列。
_RENDER_LOCK = threading.RLock()

# 实时预览的默认分辨率（渲染成本约按平方增长；画布是 CSS 尺寸，512 过剩）
DEFAULT_LIVE_SIZE = 320
MAX_LIVE_SIZE = 768

_LAM_MODULE_PATH = (
    paths.LAM_ROOT / "lam" / "models" / "rendering" / "covariance_modulation.py"
)


def _load_lam_modulation_module():
    """按文件路径加载 LAM 的 covariance_modulation 模块。

    绕过 ``lam/__init__.py`` / ``lam/models/__init__.py``（会拖进整个
    ModelLAM 与 diff-gaussian-rasterization），只取静态元数据用。
    """
    if not _LAM_MODULE_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 LAM 协方差调制模块：{_LAM_MODULE_PATH}"
            "（请确认 LAM_ROOT 指向 LAM 检出）"
        )
    spec = importlib.util.spec_from_file_location(
        "lam_covariance_modulation_meta", _LAM_MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def arkit_meta() -> dict:
    """ARKit 通道名 + 预设表情（前端滑块/按钮的唯一数据源）。"""
    mod = _load_lam_modulation_module()
    names = list(mod.ARKIT_BLENDSHAPE_NAMES)
    presets = {k: dict(v) for k, v in mod.ARKIT_PRESETS.items()}
    return {"names": names, "presets": presets, "count": len(names)}


def sanitize_arkit(raw, count: int) -> np.ndarray:
    """把前端传来的 ARKit 向量校验/裁剪成 float32 [count]，值域 [0, 1]。"""
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape[0] != count:
        raise ValueError(f"ARKit 向量长度应为 {count}，收到 {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("ARKit 向量含有 NaN/Inf")
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------------- 会话
def neutral_flame_data(renderer, flame_params) -> dict:
    """把拟合出的 FLAME 参数清成**完全中性**基准，只保留身份形状 betas。

    拟合参数（``flame_param/00000.npz``）带着上传照片里的头部姿态与表情；
    ARKit 面板要的是一个中性基准，否则「全 0」滑块给出的仍是那张照片的
    表情，且不同照片的基准不可比。这里把 rotation / neck_pose / eyes_pose /
    jaw_pose / translation / expr 全部置零。
    """
    data = renderer.get_sing_batch_smpl_data(
        renderer.get_single_view_smpl_data(flame_params, 0), 0
    )
    zeroed = ("rotation", "neck_pose", "eyes_pose", "jaw_pose", "translation", "expr")
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            v = v.clone()
            if k in zeroed:
                v.zero_()
        out[k] = v
    return out


def _to_numpy(a) -> np.ndarray:
    """torch.Tensor / ndarray -> C 连续 ndarray（不拷贝语义由调用方决定）。"""
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    if hasattr(a, "numpy"):
        a = a.numpy()
    return np.asarray(a)


def _splat_header_from_arrays(arrays: dict, session: dict, n: int) -> dict:
    """由 **B=0** 的载荷实算 LSPL 头里的 bbox / 不透明度加权质心。

    契约（§8 A6n）要求头里的 bbox/center 与 B=0 载荷实算值一致，
    因此这里必须从真实的静态路径属性算出来，不能取别的中间量。
    """
    xyz = _to_numpy(arrays["xyz"]).astype(np.float32)
    op = _to_numpy(arrays["opacity"]).reshape(-1).astype(np.float64)
    gmin = xyz.min(axis=0).astype(np.float32)
    gmax = xyz.max(axis=0).astype(np.float32)
    gctr = ((xyz.astype(np.float64) * op[:, None]).sum(axis=0)
            / max(float(op.sum()), 1e-9)).astype(np.float32)
    return {
        "count": int(n),
        "bbox_min": gmin,
        "bbox_max": gmax,
        "center": gctr,
        "ref_c2w": np.asarray(session["c2w"], dtype=np.float32).reshape(16),
        "ref_intr": np.asarray(
            [session["intr"][0, 0], session["intr"][1, 1],
             session["intr"][0, 2], session["intr"][1, 2]], dtype=np.float32),
        "ref_size": np.asarray(
            [session["render_hw"][1], session["render_hw"][0]], dtype=np.float32),
    }


def save_session(pipeline, job_id: str, res: dict, job_dir: Path, bg_color) -> dict:
    """把一次推理的 canonical 高斯存成实时会话，并写就绪标记文件。

    会话里的 ``flame_data`` 必须是**中性**基准（``_infer_and_render`` 产出的
    ``flame_params_neutral``），否则 ARKit 全 0 会退化成上传照片的姿态与表情。
    animate_gs_model 期望「批维已去掉、视角维=1」的字典：
    ``get_single_view_smpl_data`` 先取第 0 帧 -> [B, 1, ...]，
    ``get_sing_batch_smpl_data`` 再去掉批维 -> [1, ...]（num_view == 1）。
    """
    renderer = res["renderer"]
    flame_data = res.get("flame_params_neutral")
    if flame_data is None:  # 兜底：调用方没给中性份就现算
        flame_data = neutral_flame_data(renderer, res["flame_params"])
    gs_model = res["gs_model_list"][0]
    query_points = res["query_points"][0]

    c2w = np.asarray(res["c2ws"][0], dtype=np.float64)
    intr = np.asarray(res["intrs"][0], dtype=np.float64)
    session = {
        "job_id": str(job_id),
        "renderer": renderer,
        "gs_model": gs_model,
        "query_points": query_points,
        "flame_data": flame_data,
        "c2w": c2w,
        "intr": intr,
        "bg": torch.as_tensor(bg_color, dtype=torch.float32).reshape(3),
        "render_hw": tuple(res["render_hw"]),
    }

    # ARKit 协方差形变：按 cov_switch 里**当前选中的方法**装进 renderer，替换
    # LAM 自带的面内(2x2)旋转+各向同性缩放调制器。注入点是
    # renderer._cov_modulator —— gs_renderer.py:571-578 的 _get_cov_modulator
    # 是惰性的，预置实例后 animate_gs_model 会直接用它，因此
    # /workspace/LAM 一个字节都不用改。
    # 支持的方法（前端下拉框可切）：lam / tga / rot_sync / vertex_then_polar。
    # 失败不阻断会话（cov_switch.register 内部会退回 LAM 原调制器，外观仍可用）。
    try:
        mod = cov_switch.register(renderer)
        if mod is not None:
            session["cov_modulator"] = mod.__class__.__name__
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("协方差调制器安装失败（回退 LAM 原调制器）")

    # 静态（B=0）一份属性：既是环绕中心 pivot 的来源，也是 LSPL 头的来源。
    arrays0 = compute_splat(session, None)
    op = _to_numpy(arrays0["opacity"]).reshape(-1).astype(np.float64)
    xyz0 = _to_numpy(arrays0["xyz"]).astype(np.float64)
    op = np.clip(op, 0.0, None)
    session["pivot"] = (xyz0 * op[:, None]).sum(axis=0) / max(float(op.sum()), 1e-9)
    session["splat_meta"] = _splat_header_from_arrays(
        arrays0, session, int(gs_model.xyz.shape[0])
    )
    # 预热「调制路径」：首次带 arkit 的调用会触发 torch 内核选择/编译，
    # 实测冷启首次可达 40ms+，会顶破契约 §8 C18（X-LSPL-Ms <= 30）。
    # 会话创建本身已经是重步骤，在这里付掉这笔一次性开销，
    # 保证此后每一次 /arkit_splat 都落在稳态耗时内。
    _warm_modulated_path(session)
    pipeline.arkit_session = session

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "arkit_session.json").write_text(
        json.dumps(
            {
                "ready": True,
                "job_id": str(job_id),
                "n_gaussians": int(res.get("n_gaussians", gs_model.xyz.shape[0])),
                "render_hw": list(session["render_hw"]),
                "arkit_count": arkit_meta()["count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return session


def get_session(pipeline, job_id: str) -> Optional[dict]:
    """取当前 live 会话；任务不匹配（已被新任务顶掉）时返回 None。"""
    sess = getattr(pipeline, "arkit_session", None)
    if not sess or sess.get("job_id") != str(job_id):
        return None
    return sess


# --------------------------------------------------------------------- 泼溅
def compute_splat(session: dict, arkit=None) -> dict:
    """只跑 LBS + Δq/Δs 调制，返回可直接打包的高斯属性（不做光栅化）。

    这是 ``POST /arkit_splat`` 的算力去处：一次 ``animate_gs_model``（≤30 ms）
    即可拿到全部 5 段属性，省掉整条光栅化流水线。

    Args:
        session: :func:`save_session` 的返回值
        arkit: float32 ``[52]`` 的 ARKit 向量；``None`` 或全 0 走静态路径
            （与「不传 arkit」逐位一致，见 CONTRACT_v2.md §8 A6）

    Returns:
        dict: xyz ``[N,3]`` / rot ``[N,4]``(w,x,y,z) / scale ``[N,3]`` /
        opacity ``[N,1]`` / rgb ``[N,3]``，均为 float32、已激活、可直接用的值。
    """
    renderer = session["renderer"]
    arkit_params = None
    if arkit is not None:
        arkit_params = torch.from_numpy(np.asarray(arkit, dtype=np.float32))
    with _RENDER_LOCK:
        with torch.no_grad():
            gs = renderer.animate_gs_model(
                session["gs_model"],
                session["query_points"],
                session["flame_data"],
                arkit_params=arkit_params,
            )[0]
            return {
                "xyz": gs.xyz,
                "rot": gs.rotation,
                "scale": gs.scaling,
                "opacity": gs.opacity.reshape(-1, 1),
                "rgb": gs.shs.reshape(-1, 3),
            }


def splat_bytes(session: dict, arkit) -> tuple[bytes, float]:
    """算出属性并编成 LSPL v1，返回 ``(body, 耗时ms)``（契约 §3.4）。"""
    t0 = time.perf_counter()
    arrays = compute_splat(session, arkit)
    body = splat.encode_lspl(arrays, session["splat_meta"])
    return body, (time.perf_counter() - t0) * 1000.0


def _warm_modulated_path(session: dict) -> None:
    """在会话创建时跑一次「全 1」ARKit，把惰性内核编译开销挪出响应路径。

    只影响耗时，不影响任何输出：结果直接丢弃，不写文件、不触碰
    ``splat_meta``（它仍由 B=0 静态路径算出）。
    """
    try:
        arkit = np.ones(arkit_meta()["count"], dtype=np.float32)
        compute_splat(session, arkit)
    except Exception:  # noqa: BLE001 — 预热失败不应影响主流程
        logging.getLogger(__name__).exception("ARKit 调制路径预热失败（忽略）")


# --------------------------------------------------------------------- 渲染
def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋转矩阵（angle 为弧度）。"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def orbit_c2w(c2w: np.ndarray, pivot: np.ndarray, yaw: float, pitch: float,
              zoom: float) -> np.ndarray:
    """在原有相机上叠加轨道旋转/推拉。

    零增量（yaw=pitch=0, zoom=1）时原样返回，保证与既有渲染逐像素一致。
    yaw 绕世界竖直轴、pitch 绕相机右轴、zoom 缩放到环绕中心的距离。
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    if abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and abs(zoom - 1.0) < 1e-9:
        return c2w.copy()
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    right = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-12)
    M = _rot_axis(np.array([0.0, 1.0, 0.0]), yaw) @ _rot_axis(right, pitch)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = M @ R
    out[:3, 3] = pivot + (M @ (t - pivot)) * float(zoom)
    return out


def render_frame(session: dict, arkit: np.ndarray, yaw: float = 0.0, pitch: float = 0.0,
                 zoom: float = 1.0, size: Optional[int] = None) -> np.ndarray:
    """渲染一帧，返回 uint8 RGB ``[H, W, 3]``（与管线渲染同一条代码路径）。"""
    renderer = session["renderer"]
    h0, w0 = session["render_hw"]
    if size is None:
        size = DEFAULT_LIVE_SIZE
    size = int(max(64, min(MAX_LIVE_SIZE, int(size))))
    scale = size / float(w0)
    h, w = int(round(h0 * scale)), int(round(w0 * scale))

    with _RENDER_LOCK:
        with torch.no_grad():
            gs_list = renderer.animate_gs_model(
                session["gs_model"],
                session["query_points"],
                session["flame_data"],
                arkit_params=torch.from_numpy(np.asarray(arkit, dtype=np.float32)),
            )
            c2w = orbit_c2w(session["c2w"], session["pivot"], yaw, pitch, zoom)
            intr = np.array(session["intr"], dtype=np.float64, copy=True)
            intr[0, 0] *= scale
            intr[1, 1] *= scale
            intr[0, 2] *= scale
            intr[1, 2] *= scale
            device = gs_list[0].xyz.device
            out = renderer.forward_single_batch(
                [gs_list[0]],
                torch.from_numpy(c2w).float().to(device)[None],
                torch.from_numpy(intr).float().to(device)[None],
                h,
                w,
                session["bg"].to(device)[None],
            )
            # forward_single_view 已给出 [H, W, 3]；forward_single_batch 只按
            # 视角堆叠 -> [Nv, H, W, 3]，因此这里取 [0] 而不再 permute。
            rgb = out["comp_rgb"][0].clamp(0.0, 1.0)
            return (rgb * 255.0).to(torch.uint8).cpu().numpy()
