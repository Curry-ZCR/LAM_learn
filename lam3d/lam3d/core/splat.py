"""LSPL v1 —— ARKit 驱动的实时高斯泼溅传输格式（见 CONTRACT_v2.md §4）。

一次 `POST /api/jobs/{id}/arkit_splat` 的响应体就是本格式：固定 144 字节头
+ 5 段连续 float32 载荷。前端 `static/splat.js` 只做一次 `Float32Array` 视图
（零拷贝），因此本模块必须保证：

* **字节序**：全部小端（`<f4` / `<u4`），与浏览器端 TypedArray 一致。
* **无隐式填充**：头 144 字节逐字段紧凑排布，载荷紧随其后。
* **C 连续**：每段都是 `[N, w]` 行优先展平，段间无对齐空隙。

布局（偏移单位：字节）::

    0   char[4]   magic = b"LSPL"
    4   uint32    version = 1
    8   uint32    count   = N（高斯数）
    12  uint32    flags   = 1（bit0 = has_rgb）
    16  float32[3] bbox_min
    28  float32[3] bbox_max
    40  float32[3] center   不透明度加权质心
    52  float32[16] ref_c2w 行主序 4x4（3DGS 约定：列 = 右 / 下 / 前）
    116 float32[4] ref_intr [fx, fy, cx, cy]
    132 float32[2] ref_size [w, h]  ★ 宽在前
    140 uint8[4]  reserved = 0
    144 载荷: xyz[N,3] rot[N,4](w,x,y,z) scale[N,3] opacity[N,1] rgb[N,3]

总长度 = 144 + N * 56。N = 20018 时为 1,121,152 字节。

载荷的数值语义：「已激活、可直接渲染」的高斯属性——xyz 是 LBS 之后的
世界坐标，rotation 已归一化，scale 是线性尺度（Σ = R·diag(s²)·Rᵀ），
opacity 是 (0,1) 的 alpha，rgb 是 [0,1] 的显示色（**不做** gamma 变换）。
"""
from __future__ import annotations

import struct
from typing import Any, Mapping

import numpy as np

# --- 契约常量（前端与 QA 脚本按字面值依赖，不要改） -------------------------
LSPL_MAGIC = b"LSPL"
LSPL_VERSION = 1
LSPL_HEADER_SIZE = 144
LSPL_FLOATS_PER_GAUSSIAN = 56

#: 载荷段顺序与每段的 float 宽度（总宽 3+4+3+1+3 = 14 float = 56 字节）
LSPL_SEGMENTS: tuple[tuple[str, int], ...] = (
    ("xyz", 3),
    ("rot", 4),
    ("scale", 3),
    ("opacity", 1),
    ("rgb", 3),
)

#: flags 的 bit0：载荷含 rgb 段
LSPL_FLAG_HAS_RGB = 1

__all__ = [
    "LSPL_MAGIC",
    "LSPL_VERSION",
    "LSPL_HEADER_SIZE",
    "LSPL_FLOATS_PER_GAUSSIAN",
    "LSPL_SEGMENTS",
    "LSPL_FLAG_HAS_RGB",
    "encode_lspl",
]


def _f32(a: Any, n: int | None = None) -> np.ndarray:
    """转成 C 连续的 float32 一维数组（detach 掉 torch 的计算图）。"""
    if hasattr(a, "detach"):  # torch.Tensor
        a = a.detach()
    if hasattr(a, "cpu"):  # torch.Tensor
        a = a.cpu()
    if hasattr(a, "numpy"):  # torch.Tensor
        a = a.numpy()
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f4")).reshape(-1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"LSPL 载荷段长度应为 {n}，收到 {arr.shape[0]}")
    return arr


def encode_lspl(arrays: Mapping[str, Any], meta: Mapping[str, Any]) -> bytes:
    """把高斯属性编成 LSPL v1 字节串（见模块 docstring 的布局表）。

    Args:
        arrays: 5 个键 xyz / rot / scale / opacity / rgb，可传 torch.Tensor 或
            np.ndarray；形状需分别可展平为 ``[N,3] [N,4] [N,3] [N,1] [N,3]``。
        meta: ``count``、``bbox_min``、``bbox_max``、``center``、``ref_c2w``、
            ``ref_intr``、``ref_size``；可选 ``version`` / ``flags``。

    Returns:
        长度为 ``LSPL_HEADER_SIZE + count * LSPL_FLOATS_PER_GAUSSIAN`` 的 bytes。
    """
    n = int(meta["count"])
    if n < 0:
        raise ValueError(f"LSPL count 不能为负：{n}")
    version = int(meta.get("version", LSPL_VERSION))
    flags = int(meta.get("flags", LSPL_FLAG_HAS_RGB))

    xyz = _f32(arrays["xyz"], n * 3)
    rot = _f32(arrays["rot"], n * 4)
    scale = _f32(arrays["scale"], n * 3)
    opacity = _f32(arrays["opacity"], n * 1)
    rgb = _f32(arrays["rgb"], n * 3)

    # 头 16 字节：magic/version/count/flags（小端，紧凑，无填充）
    parts: list[bytes] = [struct.pack("<4sIII", LSPL_MAGIC, version, n, flags)]
    parts.append(_f32(meta["bbox_min"], 3).tobytes())
    parts.append(_f32(meta["bbox_max"], 3).tobytes())
    parts.append(_f32(meta["center"], 3).tobytes())
    parts.append(_f32(meta["ref_c2w"], 16).tobytes())
    parts.append(_f32(meta["ref_intr"], 4).tobytes())
    parts.append(_f32(meta["ref_size"], 2).tobytes())
    parts.append(b"\x00" * 4)  # reserved
    head = b"".join(parts)
    if len(head) != LSPL_HEADER_SIZE:  # pragma: no cover — 防回归
        raise AssertionError(f"LSPL 头长度应为 {LSPL_HEADER_SIZE}，实际 {len(head)}")

    body = b"".join(a.tobytes() for a in (xyz, rot, scale, opacity, rgb))
    out = head + body
    expect = LSPL_HEADER_SIZE + n * LSPL_FLOATS_PER_GAUSSIAN
    if len(out) != expect:  # pragma: no cover — 防回归
        raise AssertionError(f"LSPL 总长度应为 {expect}，实际 {len(out)}")
    return out
