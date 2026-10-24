# -*- coding: utf-8 -*-
"""ARKit 协方差调制器的后端切换层（前端「方法」下拉框的唯一后端）。

为什么需要这一层：
  - lam3d/core/arkit_live.py:184 原本写死 `tga_covariance.install(renderer)`，
    也就是建会话时一律装 TGA；rot_sync 只能靠环境变量在进程启动前选。
  - 用户要在一个已经跑起来的服务里来回切「LAM 原版 / TGA / 旋转同步 / 对照组」，
    并且要能实时改 rotation_gain / max_angle / coherence_gate 这类参数。
  - 契约 CONTRACT_v2.md：不改 /workspace/LAM/**，只从 renderer._cov_modulator
    这个惰性口子注入；A 组「ARKit 全 0 与改动前逐位一致」对四种方法都必须成立。

设计要点：
  1. 四种方法都实现 prepare_params / arkit_offsets / is_neutral 这组接口，
     所以 renderer.animate_gs_model 的调制分支对它们完全同构。
  2. 切换是「重建一个新的调制器实例再整体替换」，不做原地改参数——
     rot_sync 有 EMA 状态（_prev_delta），原地改 gain 会让历史状态残留。
  3. 切换后**复用**已有 live 会话（`rebind_session`）：renderer 是同一个对象，
     换掉 `_cov_modulator` 即可。B=0 静态基准不用重算——ARKit 全 0 时
     `is_neutral` 短路、调制器根本不被调用，四种方法的 B=0 载荷逐位相同。
     这样验收时切方法只要 3~12 ms，不必重新生成 60~90 s。

用法（服务端）：
    from lam3d.core import cov_switch
    cov_switch.set_method("rot_sync", {"rotation_gain": 0.8, "max_angle_deg": 30})
    info = cov_switch.describe()

用法（前端）：POST /api/cov/method  {"method": "rot_sync", "params": {...}}
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import torch

from . import rot_sync, rsb, tga_covariance

log = logging.getLogger(__name__)

# 下拉框里放短名（长名会被 select 截断），完整说明放 DESC 里做 title 提示
METHODS = {
    "lam": "LAM 原版",
    "tga": "TGA 雅可比",
    "rot_sync": "旋转同步",
    "rsb": "RSB 法向贴合",
    "vertex_then_polar": "要先平均再分解",
}

DESC = {
    "lam": "面内 2x2 极分解旋转 + 各向同性面积缩放（LAM 自带）",
    "tga": "Sigma Prime = J Sigma0 J^T，含旋转与拉伸",
    "rot_sync": "Sigma Prime = R Sigma0 R^T，只改朝向、严格保轴长（本方案方法一）",
    "rsb": "方法一 + 逐高斯法向贴合修正：q' = fix ⊗ (R q0)，fix 把薄片轴拉向形变后曲面法向（soft-cap）",
    "vertex_then_polar": "对照组：先把 J 平均到顶点再极分解，逐面方向相反时会跳变",
}

# 每个方法暴露给前端的可调参数：key -> (中文名, 默认值, 最小, 最大, 步长)
PARAMS = {
    "lam": {},
    "tga": {
        "jacobian_strength": ("形变强度", 1.0, 0.0, 2.0, 0.05),
        "max_scale_ratio": ("逐轴缩放上限", 1.35, 1.0, 2.0, 0.05),
        "min_scale_ratio": ("逐轴缩放下限", 0.25, 0.1, 1.0, 0.05),
    },
    "rot_sync": {
        "rotation_gain": ("旋转增益 cR", 1.0, 0.0, 1.0, 0.05),
        "max_angle_deg": ("逐高斯转角上限", 45.0, 0.0, 180.0, 5.0),
        "deadzone_deg": ("死区角度", 0.0, 0.0, 90.0, 1.0),
        "coherence_gate": ("置信度门控 tau", 0.0, 0.0, 0.95, 0.05),
    },
    "vertex_then_polar": {
        "rotation_gain": ("旋转增益 cR", 1.0, 0.0, 1.0, 0.05),
        "max_angle_deg": ("逐高斯转角上限", 45.0, 0.0, 180.0, 5.0),
    },
    "rsb": {
        "rotation_gain": ("旋转增益 cR", 1.0, 0.0, 1.0, 0.05),
        "max_angle_deg": ("逐高斯转角上限", 45.0, 0.0, 180.0, 5.0),
        "coherence_gate": ("置信度门控 tau", 0.3, 0.0, 0.95, 0.05),
        "normal_cap_deg": ("法向修正上限", 25.0, 0.0, 90.0, 1.0),
        "gate_power": ("门控幂次", 1.5, 0.5, 3.0, 0.1),
    },
}

_LOCK = threading.RLock()
_STATE = {"method": "rot_sync", "params": {}, "renderer": None}


def _make(mod_name, renderer, params):
    if mod_name == "lam":
        from lam.models.rendering.covariance_modulation import DynamicCovarianceModulator

        mod = DynamicCovarianceModulator(renderer.flame_model)
    elif mod_name == "tga":
        mod = tga_covariance.TgaCovarianceModulator(
            renderer.flame_model,
            jacobian_strength=float(params.get("jacobian_strength", 1.0)),
            max_scale_ratio=float(params.get("max_scale_ratio", 1.35)),
            min_scale_ratio=float(params.get("min_scale_ratio", 0.25)),
        )
    elif mod_name in ("rot_sync", "vertex_then_polar"):
        avg = "face" if mod_name == "rot_sync" else "vertex"
        mod = rot_sync.RotSyncModulator(
            renderer.flame_model,
            rotation_gain=float(params.get("rotation_gain", 1.0)),
            max_angle_deg=float(params.get("max_angle_deg", 45.0)),
            deadzone_deg=float(params.get("deadzone_deg", 0.0)),
            average=avg,
            coherence_gate=float(params.get("coherence_gate", 0.0)),
        )
    elif mod_name == "rsb":
        mod = rsb.RsbModulator(
            renderer.flame_model,
            rotation_gain=float(params.get("rotation_gain", 1.0)),
            max_angle_deg=float(params.get("max_angle_deg", 45.0)),
            coherence_gate=float(params.get("coherence_gate", 0.3)),
            normal_cap_deg=float(params.get("normal_cap_deg", 25.0)),
            gate_power=float(params.get("gate_power", 1.5)),
        )
    else:
        raise ValueError("unknown covariance method: %r" % (mod_name,))
    device = getattr(renderer, "device", None) or torch.device("cuda")
    mod.to(device)
    mod.eval()
    return mod


def current():
    return _STATE["method"]


def _param_spec(name):
    spec = PARAMS.get(name, {})
    out = []
    for k, v in spec.items():
        out.append({
            "key": k, "label": v[0], "default": v[1],
            "min": v[2], "max": v[3], "step": v[4],
        })
    return out


def describe():
    with _LOCK:
        return {
            "method": _STATE["method"],
            "params": dict(_STATE["params"]),
            "methods": [
                {
                    "id": k,
                    "label": v,
                    "desc": DESC.get(k, v),
                    "params": _param_spec(k),
                }
                for k, v in METHODS.items()
            ],
        }


def set_method(name, params=None, renderer=None):
    with _LOCK:
        if name not in METHODS:
            raise ValueError("unknown method %r; choose from %s" % (name, list(METHODS)))
        r = renderer if renderer is not None else _STATE.get("renderer")
        if r is None:
            raise RuntimeError("renderer 尚未登记：服务还没加载模型，先跑一次生成")
        merged = {}
        for k, v in PARAMS.get(name, {}).items():
            merged[k] = v[1]
        merged.update(params or {})
        mod = _make(name, r, merged)
        r._cov_modulator = mod          # 契约允许的唯一注入点
        _STATE["method"] = name
        _STATE["params"] = merged
        _STATE["renderer"] = r
        return {"method": name, "params": merged, "class": mod.__class__.__name__}


def register(renderer):
    """管线建好会话后登记 renderer，并按当前方法装一次调制器。"""
    with _LOCK:
        _STATE["renderer"] = renderer
        try:
            return set_method(_STATE["method"] or "rot_sync", _STATE["params"], renderer)
        except Exception:
            log.exception("切换协方差方法失败，退回 LAM 原调制器")
            from lam.models.rendering.covariance_modulation import DynamicCovarianceModulator

            renderer._cov_modulator = DynamicCovarianceModulator(renderer.flame_model)
            return None


def rebind_session(pipeline, job_id=None):
    """切方法后把**现有** live 会话接到新调制器上，免去重新生成。

    为什么不用「让会话失效再重新生成」那一套：一次生成要 60~90 s，而验收
    场景需要反复 A/B 同一张脸。会话里已经存了 gs_model / query_points /
    flame_data / 相机，renderer 也是同一个对象，所以只要把新的
    `renderer._cov_modulator` 接上，下一次 compute_splat 就自动走新方法。

    B=0 静态基准（splat_meta / pivot）无需重算：ARKit 全 0 时渲染器的
    `is_neutral` 短路，调制器根本没被调用，四种方法的 B=0 载荷逐位相同。

    Returns: 是否成功接到会话（没有会话时返回 False，不报错）。
    """
    sess = getattr(pipeline, "arkit_session", None)
    if not sess:
        return False
    if job_id is not None and sess.get("job_id") != str(job_id):
        return False
    mod = getattr(sess.get("renderer"), "_cov_modulator", None)
    if mod is not None:
        sess["cov_modulator"] = mod.__class__.__name__
        log.info("协方差方法已切到 %s（会话 %s 直接复用）",
                 mod.__class__.__name__, sess.get("job_id"))
    return True


def invalidate_session(pipeline):
    """切换方法后让旧会话失效（旧 LSPL 的 B=0 基准是旧方法算的属性）。"""
    sess = getattr(pipeline, "arkit_session", None)
    if not sess:
        return False
    log.info("协方差方法已切换，旧实时会话失效（job=%s）", sess.get("job_id"))
    pipeline.arkit_session = None
    return True
