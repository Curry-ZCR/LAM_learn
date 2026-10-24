"""FastAPI 服务：把「单图生成 3D 人物」管线暴露成 HTTP 接口 + 静态前端。"""
from __future__ import annotations

import io
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..core import paths
from ..core.jobs import JobManager

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

app = FastAPI(title="LAM 3D 数字人", version="1.0.0", docs_url="/api/docs")
manager = JobManager(max_jobs=50, device=os.environ.get("LAM3D_DEVICE", "cuda"))


# --------------------------------------------------------------------- 启动
@app.on_event("startup")
def _startup() -> None:
    paths.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    # 后台预热：把 2GB 权重提前读进显存，第一次生成就不用等了
    manager.preload()


# --------------------------------------------------------------------- 元信息
@app.get("/api/config")
def api_config() -> dict:
    import torch

    samples = []
    for name in _safe_samples():
        samples.append({"name": name, "url": f"/api/samples/{name}"})
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "samples": samples,
        "gpu": gpu,
        "device": manager._device,  # noqa: SLF001 — 内部只读
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/api/status")
def api_status() -> dict:
    err = manager.warmup_error
    loaded = manager._pipeline is not None and manager._pipeline._loaded  # noqa: SLF001
    return {
        "ready": bool(loaded),
        "warming": bool(not loaded and err is None),
        "error": err,
        "queue": manager.queue_depth(),
    }


def _safe_samples() -> list[str]:
    from ..core.pipeline import SingleImage3DPipeline

    try:
        return SingleImage3DPipeline.list_samples()
    except Exception:  # noqa: BLE001
        return []


@app.get("/api/samples/{name}")
def api_sample(name: str) -> FileResponse:
    if name not in _safe_samples():
        raise HTTPException(404, "示例图片不存在")
    p = paths.SAMPLE_INPUT_ROOT / name
    return FileResponse(p, media_type=_media_type(p))


def _media_type(p: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(p.suffix.lower(), "application/octet-stream")


# --------------------------------------------------------------------- 生成
@app.post("/api/generate")
async def api_generate(
    sample: str = Form(""),
    image: UploadFile | None = File(None),
    motion: str = Form(""),          # 兼容旧前端：接受并忽略
    skip_video: bool = Form(False),  # 兼容旧前端：接受并忽略
) -> JSONResponse:
    upload_dir = paths.WORK_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    if image is not None and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIX:
            raise HTTPException(400, f"不支持的图片格式 {suffix}，请用 PNG/JPG/WEBP")
        raw = await image.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"图片过大（>{MAX_UPLOAD_BYTES // 1024 // 1024} MB）")
        if not raw:
            raise HTTPException(400, "上传内容为空")
        # 真伪校验 + 按「内容真实格式」归一化。
        #
        # 为什么必须按内容判断：LAM 侧 torchvision.io.read_image 只认 JPEG/PNG，
        # webp/bmp/tiff/gif 会在推理进程里抛
        # "RuntimeError: Unsupported image file. Only jpeg and png are currently supported."
        # 所以这里按 im.format（而非后缀）分流：JPEG/PNG 原样落盘，其余转 PNG。
        try:
            from PIL import Image as _Image
            from PIL import ImageOps as _ImageOps

            with _Image.open(io.BytesIO(raw)) as im:
                im.verify()
            with _Image.open(io.BytesIO(raw)) as im:
                src_format = (im.format or "").upper()
                width, height = im.size
                if src_format not in ("JPEG", "PNG"):
                    # 非 JPEG/PNG（含「后缀是 .png 但内容是 webp」这类错配）→ 转 PNG。
                    # exif_transpose 先把手机照片的 EXIF 方向拍正，再统一 RGB 编码。
                    frame = _ImageOps.exif_transpose(im).convert("RGB")
                    buf = io.BytesIO()
                    frame.save(buf, format="PNG")
                    raw = buf.getvalue()
                    suffix = ".png"
                    width, height = frame.size
                # src_format 是 JPEG/PNG 时什么都不做：raw / suffix 一个字节都不改，
                # 落盘字节与上传字节逐字节相同（无重编码损失）。
        except Exception as exc:  # noqa: BLE001
            # 只给用户看得懂的文案，不要把 PIL 的 `<_io.BytesIO object at 0x...>`
            # 或 torchvision 的英文原句透出去。
            from PIL import UnidentifiedImageError as _Unidentified

            if isinstance(exc, _Unidentified):
                raise HTTPException(400, "无法解析为图片：文件内容不是有效图片，请改用 PNG/JPG/WEBP") from exc
            raise HTTPException(400, f"无法解析为图片（可尝试 PNG/JPG）：{exc}") from exc
        dest = upload_dir / f"{uuid.uuid4().hex[:12]}{suffix}"
        dest.write_bytes(raw)
        # image_name 的**后缀**决定 JobManager 在 job 目录里落盘的文件名
        # （jobs.py: suffix = Path(image_name).suffix → job_dir/input<suffix>）。
        # 归一化过就必须让它是 .png，保证 job 目录里是真 PNG 的 input.png。
        image_name = Path(image.filename).name
        if suffix != Path(image.filename).suffix.lower():
            image_name = str(Path(image_name).with_suffix(suffix))
    elif sample:
        if sample not in _safe_samples():
            raise HTTPException(400, "示例图片不存在")
        src = paths.SAMPLE_INPUT_ROOT / sample
        suffix = src.suffix.lower()
        dest = upload_dir / f"{uuid.uuid4().hex[:12]}{suffix}"
        shutil.copyfile(src, dest)
        image_name = sample
        width = height = 0
    else:
        raise HTTPException(400, "请上传图片或选择一张示例图片")

    try:
        job = manager.submit(dest, image_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    payload = job.as_dict()
    # 契约 §3.2 要求返回 {job_id, state}；前端 §6.4 沿用 id。两个 key 都给，互不冲突。
    payload["job_id"] = payload["id"]
    payload["input_size"] = [width, height]
    return JSONResponse(payload)


# --------------------------------------------------------------------- 任务
@app.get("/api/jobs")
def api_jobs(limit: int = 20) -> list[dict]:
    return [j.as_dict() for j in manager.list(limit=limit)]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str) -> dict:
    if not manager.cancel(job_id):
        raise HTTPException(409, "任务已在运行或已结束，无法取消")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/file/{name}")
def api_job_file(job_id: str, name: str, download: bool = False) -> FileResponse:
    p = manager.artifact(job_id, name)
    if p is None:
        raise HTTPException(404, "产物不存在或尚未生成")
    media = {
        "glb": "model/gltf-binary",
        "obj": "text/plain; charset=utf-8",
        "gs_ply": "application/octet-stream",
        "face": "image/png",
        "preview": "image/png",
        "thumb": "image/jpeg",
    }.get(name, "application/octet-stream")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{p.name}"'
    return FileResponse(p, media_type=media, headers=headers)


# --------------------------------------------------------------------- ARKit 实时表情
_ARKIT_META_CACHE: dict | None = None


def _arkit_meta() -> dict:
    """52 个 ARKit 通道名 + 预设表情（前端滑块的唯一数据源）。"""
    global _ARKIT_META_CACHE  # noqa: PLW0603
    if _ARKIT_META_CACHE is None:
        from ..core import arkit_live

        _ARKIT_META_CACHE = arkit_live.arkit_meta()
    return _ARKIT_META_CACHE


def _get_loaded_pipeline():
    p = manager._pipeline  # noqa: SLF001 — 只读访问
    if p is None or not getattr(p, "_loaded", False):
        raise HTTPException(409, "模型尚未加载，请稍候或先完成一次生成")
    return p


@app.get("/api/arkit/meta")
def api_arkit_meta() -> dict:
    try:
        return _arkit_meta()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ARKit 元数据不可用：{exc}") from exc


# ------------------------------------------------- 协方差方法切换（验收用）
@app.get("/api/cov/methods")
def api_cov_methods() -> dict:
    """当前协方差调制方法 + 每个方法的可调参数（前端下拉框/滑块的数据源）。"""
    from ..core import cov_switch

    try:
        return cov_switch.describe()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"协方差方法不可用：{exc}") from exc


@app.post("/api/cov/method")
def api_cov_method(payload: dict) -> dict:
    """切换协方差调制方法（lam / tga / rot_sync / vertex_then_polar）。

    切换后**复用当前 live 会话**：renderer 是同一个对象，把新的调制器接上即可，
    下一次 /arkit_splat 就按新方法算属性。B=0 静态基准无需重算——ARKit 全 0 时
    渲染器的 is_neutral 短路、调制器根本没被调用，四种方法的 B=0 载荷逐位相同。
    因此前端切完方法只要重拉一次首帧，**不需要重新生成**。
    """
    from ..core import cov_switch

    name = str(payload.get("method") or "").strip()
    if not name:
        raise HTTPException(400, "缺少 method 字段")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(400, "params 必须是对象")

    pipeline = _get_loaded_pipeline()
    renderer = getattr(getattr(pipeline, "lam", None), "renderer", None)
    try:
        info = cov_switch.set_method(name, params, renderer=renderer)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    rebound = cov_switch.rebind_session(pipeline)
    out = cov_switch.describe()
    out["class"] = info["class"]
    out["session_rebound"] = bool(rebound)
    return out


@app.get("/api/jobs/{job_id}/arkit_session")
def api_arkit_session(job_id: str) -> dict:
    if manager.get(job_id) is None:
        raise HTTPException(404, "任务不存在")
    from ..core import arkit_live

    pipeline = _get_loaded_pipeline()
    sess = arkit_live.get_session(pipeline, job_id)
    if sess is None:
        return {"ready": False}
    meta = sess["splat_meta"]
    return {
        "ready": True,
        "job_id": str(job_id),
        "n_gaussians": int(sess["gs_model"].xyz.shape[0]),
        "render_hw": [int(sess["render_hw"][0]), int(sess["render_hw"][1])],
        "arkit_count": _arkit_meta()["count"],
        "center": [float(v) for v in meta["center"]],
        "bbox_min": [float(v) for v in meta["bbox_min"]],
        "bbox_max": [float(v) for v in meta["bbox_max"]],
        "ref_c2w": [float(v) for v in meta["ref_c2w"]],
        "ref_intr": [float(v) for v in meta["ref_intr"]],
        "ref_size": [float(v) for v in meta["ref_size"]],
    }


@app.post("/api/jobs/{job_id}/arkit_splat")
def api_arkit_splat(job_id: str, payload: dict) -> Response:
    """按 ARKit 表情向量算一份 LSPL v1 二进制（见 CONTRACT_v2.md §4）。

    只做 LBS + Δq/Δs 调制与内存打包，不做光栅化，因此响应时间稳定在
    ``X-LSPL-Ms`` 内（验收要求 <= 30ms）。
    """
    if manager.get(job_id) is None:
        raise HTTPException(404, "任务不存在")
    from ..core import arkit_live

    pipeline = _get_loaded_pipeline()
    sess = arkit_live.get_session(pipeline, job_id)
    if sess is None:
        raise HTTPException(409, "实时表情会话不可用（服务可能已重启），请重新生成")

    raw = payload.get("arkit")
    if raw is None:
        raise HTTPException(400, "缺少 arkit 字段")
    try:
        count = _arkit_meta()["count"]
        arkit = arkit_live.sanitize_arkit(raw, count)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        body, ms = arkit_live.splat_bytes(sess, arkit)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception(
            "arkit_splat failed (job=%s, arkit_nonzero=%s)", job_id,
            bool(getattr(arkit, "sum", lambda: 0)())
        )
        raise HTTPException(500, f"渲染失败：{exc}") from exc

    headers = {
        "X-LSPL-Count": str(sess["gs_model"].xyz.shape[0]),
        "X-LSPL-Ms": f"{ms:.2f}",
        "Cache-Control": "no-store",
    }
    return Response(content=body, media_type="application/octet-stream", headers=headers)


@app.post("/api/jobs/{job_id}/arkit_frame")
def api_arkit_frame(job_id: str, payload: dict) -> Response:
    """按 ARKit 表情向量渲染一帧，返回 PNG（image/png）。

    同步端点 ⇒ 由 FastAPI 丢进线程池执行，事件循环不被阻塞；
    GPU 上的前向与光栅化在 arkit_live 内部用同一把锁串行化。
    """
    if manager.get(job_id) is None:
        raise HTTPException(404, "任务不存在")
    from ..core import arkit_live

    pipeline = _get_loaded_pipeline()
    sess = arkit_live.get_session(pipeline, job_id)
    if sess is None:
        raise HTTPException(409, "实时表情会话不可用（服务可能已重启），请重新生成")

    raw = payload.get("arkit")
    if raw is None:
        raise HTTPException(400, "缺少 arkit 字段")
    try:
        count = _arkit_meta()["count"]
        arkit = arkit_live.sanitize_arkit(raw, count)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        yaw = float(payload.get("yaw", 0.0) or 0.0)
        pitch = float(payload.get("pitch", 0.0) or 0.0)
        zoom = float(payload.get("zoom", 1.0) or 1.0)
        size = payload.get("size") or arkit_live.DEFAULT_LIVE_SIZE
        size = int(size)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"相机参数非法：{exc}") from exc
    yaw = max(-3.14159, min(3.14159, yaw))
    pitch = max(-1.2, min(1.2, pitch))
    zoom = max(0.3, min(3.0, zoom))

    try:
        from ..core import arkit_live

        arr = arkit_live.render_frame(sess, arkit, yaw=yaw, pitch=pitch,
                                      zoom=zoom, size=size)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception(
            "arkit_frame failed (job=%s, arkit_nonzero=%s)", job_id,
            bool(getattr(arkit, "sum", lambda: 0)())
        )
        raise HTTPException(500, f"渲染失败：{exc}") from exc

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# --------------------------------------------------------------------- 前端
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (paths.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if paths.STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(paths.STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("LAM3D_HOST", "0.0.0.0")
    port = int(os.environ.get("LAM3D_PORT", "8000"))
    print(f"[lam3d] 服务启动： http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
