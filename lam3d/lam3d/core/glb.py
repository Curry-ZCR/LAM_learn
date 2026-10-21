"""极简 glTF 2.0 二进制（GLB）写出器。

为什么不直接用 ``trimesh`` 导出：实测有两个坑，而且都会静默损坏产物。

1. **顶点色被丢**：``Trimesh(..., process=True)`` 会合并重复顶点，合并后
   ``ColorVisuals`` 不再匹配，导出的 GLB 里没有 COLOR_0，前端只能显示成灰色。
2. **颜色精度**：trimesh 写 COLOR_0 时固定用 ``uint8 + normalized``
   （见 ``trimesh/exchange/gltf.py``）。glTF 规定 COLOR_0 是**线性**空间，
   而线性值的暗部在 8 bit 下精度极差（线性 0.01 处相邻档位相差近 40%），
   深色区域会出现明显色带。本模块直接写 ``float32``，规避该问题。

另外 trimesh 不会自动生成 NORMAL；缺法线的 glTF 在 three.js 里光照完全失效。

本模块只实现「一个带顶点色与法线的三角网格」这一种情形，够用且完全可控，
不引入额外依赖。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# glTF 常量
_COMPONENT_FLOAT = 5126
_COMPONENT_UINT = 5125          # indices 用 uint32，避免 >65535 顶点时溢出
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963
_MODE_TRIANGLES = 4


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    """补齐到 4 字节边界（glTF 的硬性要求）。"""
    rem = len(b) % 4
    return b if rem == 0 else b + fill * (4 - rem)


def write_glb(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    name: str = "head",
) -> Path:
    """写出单网格 GLB。

    Args:
        path: 输出路径
        vertices: [N, 3] float32 顶点坐标
        faces: [F, 3] 三角面索引
        colors: [N, 3] 或 [N, 4]，**线性**空间，0~1
        normals: [N, 3] 顶点法线（不给则按面法线面积加权现算）
        name: 网格名

    Returns:
        输出路径
    """
    path = Path(path)
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    idx = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    n = len(verts)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices 必须是 [N, 3]，收到 {verts.shape}")
    if not np.isfinite(verts).all():
        raise ValueError("vertices 含 NaN/Inf")
    if faces.size and (faces.max() >= n or faces.min() < 0):
        raise ValueError(f"面索引越界：faces ∈ [{faces.min()}, {faces.max()}]，但只有 {n} 个顶点")

    if normals is None:
        normals = _vertex_normals(verts, faces)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    if len(nrm) != n:
        raise ValueError(f"法线数量 {len(nrm)} 与顶点数 {n} 不一致")

    views: list[dict] = []
    accessors: list[dict] = []
    blob = bytearray()

    def add(buffer: bytes, target: int | None, accessor: dict) -> int:
        """把一段数据追加进 BIN chunk，返回 bufferView 索引。"""
        offset = len(blob)
        blob.extend(_pad4(buffer))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(buffer)}
        if target is not None:
            view["target"] = target
        views.append(view)
        accessor["bufferView"] = len(views) - 1
        accessors.append(accessor)
        return len(accessors) - 1

    # POSITION（规范要求必须带 min/max）
    a_pos = add(
        verts.tobytes(),
        _TARGET_ARRAY_BUFFER,
        {
            "componentType": _COMPONENT_FLOAT,
            "count": n,
            "type": "VEC3",
            "min": [float(v) for v in verts.min(axis=0)],
            "max": [float(v) for v in verts.max(axis=0)],
        },
    )
    attributes = {"POSITION": a_pos}

    a_nrm = add(
        nrm.tobytes(),
        _TARGET_ARRAY_BUFFER,
        {"componentType": _COMPONENT_FLOAT, "count": n, "type": "VEC3"},
    )
    attributes["NORMAL"] = a_nrm

    if colors is not None:
        col = np.asarray(colors, dtype=np.float32)
        if col.ndim != 2 or col.shape[0] != n or col.shape[1] not in (3, 4):
            raise ValueError(f"colors 必须是 [N, 3] 或 [N, 4]，收到 {col.shape}")
        if col.shape[1] == 3:
            col = np.concatenate([col, np.ones((n, 1), np.float32)], axis=1)
        col = np.ascontiguousarray(np.clip(col, 0.0, 1.0))
        attributes["COLOR_0"] = add(
            col.tobytes(),
            _TARGET_ARRAY_BUFFER,
            {"componentType": _COMPONENT_FLOAT, "count": n, "type": "VEC4"},
        )

    a_idx = add(
        idx.tobytes(),
        _TARGET_ELEMENT_ARRAY_BUFFER,
        {"componentType": _COMPONENT_UINT, "count": int(idx.size), "type": "SCALAR"},
    )

    gltf = {
        "asset": {"version": "2.0", "generator": "lam3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": a_idx,
                        "mode": _MODE_TRIANGLES,
                        "material": 0,
                    }
                ],
            }
        ],
        # 顶点色材质：baseColorFactor 留白，颜色全部来自 COLOR_0
        "materials": [
            {
                "name": "vertexColor",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,   # 颈部裁剪后是开放曲面，双面避免背面镂空
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad4(bytes(blob))

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))          # magic, version, length
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))    # JSON chunk
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))     # BIN chunk
        f.write(bin_bytes)
    return path


def _vertex_normals(verts, faces) -> np.ndarray:
    """按面法线面积（叉积模长）加权平均出顶点法线。"""
    v = np.asarray(verts, dtype=np.float64)
    tri = v[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.where(ln < 1e-12, 1.0, ln)).astype(np.float32)


def read_glb_info(path: str | Path) -> dict:
    """读回 GLB 的结构信息，用于测试与自检。"""
    raw = Path(path).read_bytes()
    magic, version, _ = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ValueError("不是合法 GLB 文件")
    jlen, jtype = struct.unpack("<II", raw[12:20])
    if jtype != 0x4E4F534A:
        raise ValueError("首个 chunk 不是 JSON")
    gltf = json.loads(raw[20 : 20 + jlen].decode("utf-8"))
    bin_off = 20 + jlen
    blen, btype = struct.unpack("<II", raw[bin_off : bin_off + 8])
    if btype != 0x004E4942:
        raise ValueError("第二个 chunk 不是 BIN")
    prim = gltf["meshes"][0]["primitives"][0]
    return {
        "version": version,
        "attributes": sorted(prim["attributes"].keys()),
        "accessors": gltf["accessors"],
        "bin_length": blen,
        "materials": gltf.get("materials", []),
    }


__all__ = ["write_glb", "read_glb_info"]
