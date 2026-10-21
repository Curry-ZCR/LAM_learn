"""后台任务队列：把 GPU 推理串行化，并把进度暴露给 HTTP 轮询。

GPU 推理天然不能并发（显存 + 全局 cuBLAS workspace 在 fork 后会崩），
所以这里用**单工作线程 + 队列**：所有任务排队执行，前端轮询 ``/api/jobs/{id}``。
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .pipeline import ProgressFn, SingleImage3DPipeline

TERMINAL_STATES = {"succeeded", "failed", "canceled"}


@dataclass
class Job:
    id: str
    image_name: str
    state: str = "queued"          # queued | preparing | running | succeeded | failed | canceled
    stage: str = "queued"
    progress: float = 0.0          # 0~100
    detail: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    log: list[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "image_name": self.image_name,
            "state": self.state,
            "stage": self.stage,
            "progress": round(self.progress, 1),
            "detail": self.detail,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 1),
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    """单进程内的任务管理器。"""

    def __init__(self, max_jobs: int = 50, device: str = "cuda") -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.RLock()
        self._max_jobs = max_jobs
        self._pipeline: SingleImage3DPipeline | None = None
        self._device = device
        self._warm_error: str | None = None
        self._worker = threading.Thread(target=self._loop, name="lam3d-worker", daemon=True)
        self._worker.start()
        paths.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        (paths.WORK_ROOT / "jobs").mkdir(parents=True, exist_ok=True)
        (paths.WORK_ROOT / "uploads").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 对外接口
    def submit(self, image_path: Path, image_name: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], image_name=image_name)
        job_dir = paths.WORK_ROOT / "jobs" / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        # 把上传的图片收进任务目录，避免上传区被清理后任务失败
        suffix = Path(image_name).suffix or ".png"
        target = job_dir / f"input{suffix}"
        shutil.copyfile(image_path, target)
        (job_dir / "job.json").write_text(
            json.dumps({"id": job.id, "image_name": image_name},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._gc_locked()
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[i] for i in ids]

    def cancel(self, job_id: str) -> bool:
        """只能取消还在排队的任务（GPU 推理无法安全中断）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "queued":
                return False
            job.state = "canceled"
            job.stage = "canceled"
            job.detail = "已取消"
            job.finished_at = time.time()
            return True

    def job_dir(self, job_id: str) -> Path:
        return paths.WORK_ROOT / "jobs" / job_id

    def artifact(self, job_id: str, name: str) -> Path | None:
        """安全地把结果文件解析到任务目录内（防目录穿越）。"""
        job = self.get(job_id)
        if job is None or job.result is None:
            return None
        rel = job.result.get(name)
        if not rel:
            return None
        p = (paths.WORK_ROOT / rel).resolve()
        root = self.job_dir(job_id).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return None
        return p if p.is_file() else None

    @property
    def warmup_error(self) -> str | None:
        return self._warm_error

    def preload(self) -> None:
        """提前把模型载入显存（由服务启动时的后台线程调用）。"""
        self._q.put("__preload__")

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state == "queued")

    # -------------------------------------------------------------- 内部实现
    def _gc_locked(self) -> None:
        if len(self._order) <= self._max_jobs:
            return
        drop = self._order[: -self._max_jobs]
        self._order = self._order[-self._max_jobs :]
        for jid in drop:
            job = self._jobs.pop(jid, None)
            if job is not None and job.is_terminal:
                shutil.rmtree(paths.WORK_ROOT / "jobs" / jid, ignore_errors=True)

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                if item == "__preload__":
                    self._ensure_pipeline(self._log_to(None))
                else:
                    self._execute(item)
            except Exception:  # noqa: BLE001 — 工作线程绝不能死
                traceback.print_exc()
            finally:
                self._q.task_done()

    def _log_to(self, job: Job | None) -> ProgressFn:
        def fn(stage: str, pct: float, detail: str = "") -> None:
            line = f"[{stage}] {pct:5.1f}%  {detail}"
            print(f"[lam3d] {line}", flush=True)
            if job is not None:
                with self._lock:
                    job.stage = stage
                    job.progress = max(job.progress, float(pct))
                    job.detail = detail
                    job.log.append(line)
                    if len(job.log) > 400:
                        job.log = job.log[-400:]
        return fn

    def _ensure_pipeline(self, report: ProgressFn) -> SingleImage3DPipeline:
        if self._pipeline is None:
            self._pipeline = SingleImage3DPipeline(device=self._device, lazy_load=True)
        try:
            self._pipeline.load(report)
            self._warm_error = None
        except Exception as e:  # noqa: BLE001
            self._warm_error = f"{type(e).__name__}: {e}"
            raise
        return self._pipeline

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state == "canceled":
                return
            job.state = "preparing"
            job.started_at = time.time()

        report = self._log_to(job)
        job_dir = self.job_dir(job_id)
        image = next(iter(sorted(job_dir.glob("input.*"))), None)
        if image is None:
            self._fail(job, "输入图片丢失")
            return

        try:
            pipeline = self._ensure_pipeline(report)
            with self._lock:
                job.state = "running"
            result = pipeline.run(
                image_path=image,
                job_dir=job_dir,
                on_progress=report,
                job_id=job_id,
            )
            with self._lock:
                job.state = "succeeded"
                job.progress = 100.0
                job.stage = "done"
                job.detail = "完成"
                job.result = result.as_dict()
                job.finished_at = time.time()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._fail(job, f"{type(e).__name__}: {e}")

    def _fail(self, job: Job, message: str) -> None:
        with self._lock:
            job.state = "failed"
            job.error = message
            job.detail = message
            job.finished_at = time.time()


__all__ = ["JobManager", "Job", "TERMINAL_STATES"]
