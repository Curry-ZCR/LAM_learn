"""lam3d —— 从 LAM 中抽取出的「单图生成 3D 人物」独立模块。

本包只依赖 /workspace/LAM 的**源码与权重**，不修改其中任何文件。
对外只暴露 :class:`lam3d.core.pipeline.SingleImage3DPipeline` 与
:class:`lam3d.core.jobs.JobManager` 两个入口。

典型用法::

    from lam3d.core.pipeline import SingleImage3DPipeline
    pipeline = SingleImage3DPipeline(device="cuda")
    result = pipeline.run("face.jpg", "job_dir", on_progress=print, job_id="demo")
    # 产物：result.face（重建人脸 PNG）/ result.glb / result.obj / result.gs_ply / result.thumb
    # 实时表情：POST /api/jobs/{job_id}/arkit_splat → LSPL v1 二进制（见 arkit_live.compute_splat）
"""

__version__ = "1.0.0"
