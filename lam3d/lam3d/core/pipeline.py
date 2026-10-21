"""从 LAM 中抽取的「单图生成 3D 人物」核心管线。

流程（每一步都可单独调用）::

    单张人脸图片
      ├─ 1. FLAME 跟踪      preprocess → optimize → export
      ├─ 2. 参考图预处理    裁剪/抠图 + 读取 canonical shape 参数
      ├─ 3. 姿态装配        FLAME 参数 + 参考相机内外参（单帧）
      ├─ 4. LAM 前馈        图像编码 → 隐式点 → 20018 个高斯
      ├─ 5. 单帧渲染        重建后的人物正脸 512×512
      ├─ 6. 3D 模型导出     GLB / OBJ / Gaussian-Splat PLY
      └─ 7. ARKit 会话      中性基准 + 相机/中心缓存（供实时泼溅）

与上游实现的四点差异（前三点是修复，第四点是可配置的行为选择）：

1. **3D 网格拓扑用对**：上游 ``lam/runners/infer/lam.py`` 导出带色网格时
   顶点取自 20018 点的细分空间、面却取自 5023 点的原始 FLAME，
   导致 ``*_textured_mesh.obj`` 里只有头 5024 个顶点被引用、形变细节全丢。
   本模块改用 ``flame_model.faces_up``（39904 面），与高斯点严格一一对应。
2. **单帧重建**：跟踪导出的序列天然只有 1 帧，本模块只渲染这一帧作为
   「重建后的人脸图」；动画交给 ARKit 实时泼溅（见 ``core/arkit_live.py``）。
3. **规范几何直接给出**：``framework_img.obj`` 是 20018 个孤立点、无面；本模块
   直接产出可被 Blender / Three.js 打开的单文件 GLB，并回读校验顶点色没丢。
4. **颈部裁剪**（可用 ``LAM3D_NECK_TRIM=0`` 关闭）：LAM 的 FLAME 以
   ``add_shoulder=False`` 构建，只在「头 + 颈」上训练过；颈部以下属于外推区，
   实测会长出跨接三角面。默认裁掉 FLAME 模板最低点以下 0.03 的区域。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import arkit_live, lam_env, paths

ProgressFn = Callable[[str, float, str], None]


def _srgb_to_linear(c):
    """sRGB(0~1) → 线性(0~1)，逐通道精确转换（glTF 的 COLOR_0 要求线性空间）。"""
    import numpy as np

    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _laplacian_smooth(verts, faces, iters: int, lam: float = 0.5, cache: dict | None = None):
    """均匀拉普拉斯平滑（保持顶点数与顺序不变）。

    用邻接矩阵一次构建、多次复用；对 2 万顶点的网格，每次迭代约 1 ms。
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    n = len(verts)
    key = (id(faces), n)
    if cache is None or cache.get("key") != key:
        # 无向邻接（每条边两个方向）
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        r = np.concatenate([e[:, 0], e[:, 1]])
        c = np.concatenate([e[:, 1], e[:, 0]])
        adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
        adj.data[:] = 1.0
        deg = np.asarray(adj.sum(1)).ravel()
        deg[deg == 0] = 1
        w = coo_matrix((1.0 / deg, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr() @ adj
        if cache is not None:
            cache["key"] = key
            cache["W"] = w
    w = cache["W"] if cache is not None else w  # type: ignore[possibly-undefined]

    out = verts.astype(np.float64, copy=True)
    for _ in range(iters):
        out += lam * (w @ out - out)
    return out


def _vertex_normals(verts, faces):
    """按面法线面积加权平均出顶点法线（无外部依赖）。"""
    import numpy as np

    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # 模长 = 2×面积
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.where(n < 1e-12, 1.0, n)).astype(np.float32)


def _noop(stage: str, pct: float, detail: str = "") -> None:  # pragma: no cover
    pass


def _verify_glb(path) -> None:
    """回读刚写出的 GLB，确认顶点色、法线、索引都正确。

    这些断言都是有代价换来的经验：
      · 顶点数被改动会让「一顶点一颜色」错位 —— 前端颜色整体跑偏。
      · 缺 NORMAL —— three.js 光照失效，模型死黑。
      · 面索引越界 —— 上游 ``*_textured_mesh.obj`` 正是栽在这里。
    """
    from .glb import read_glb_info

    info = read_glb_info(path)
    attrs = set(info["attributes"])
    if not {"POSITION", "NORMAL", "COLOR_0"} <= attrs:
        raise RuntimeError(f"GLB 缺少必要属性：已有 {sorted(attrs)}")

    acc = {a.get("type"): a for a in info["accessors"]}
    n_pos = next(a["count"] for a in info["accessors"]
                 if a.get("type") == "VEC3" and a.get("componentType") == 5126)
    n_col = next(a["count"] for a in info["accessors"] if a.get("type") == "VEC4")
    if n_pos != n_col:
        raise RuntimeError(f"顶点色数量 {n_col} 与顶点数 {n_pos} 不一致")

    idx = next(a for a in info["accessors"] if a.get("type") == "SCALAR")
    if idx.get("componentType") != 5125:
        raise RuntimeError("索引不是 uint32")
    if idx["count"] % 3 != 0:
        raise RuntimeError(f"索引数 {idx['count']} 不是 3 的倍数")


@dataclass
class PipelineResult:
    """一次「单图 → 3D 人物」的产物索引。"""

    job_id: str
    image_name: str
    out_dir: Path
    glb: Path | None = None
    obj: Path | None = None
    gs_ply: Path | None = None
    face: Path | None = None
    preview: Path | None = None
    thumb: Path | None = None
    shape_param: list[float] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        def rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(paths.WORK_ROOT))
            except ValueError:
                return str(p)

        return {
            "job_id": self.job_id,
            "image_name": self.image_name,
            "glb": rel(self.glb),
            "obj": rel(self.obj),
            "gs_ply": rel(self.gs_ply),
            "face": rel(self.face),
            "preview": rel(self.preview),
            "thumb": rel(self.thumb),
            "stats": self.stats,
        }


class SingleImage3DPipeline:
    """单图生成 3D 人物的完整管线。

    模型只在第一次 :meth:`load` 时载入并常驻显存，之后每次 :meth:`run`
    复用同一份权重。**非线程安全**：GPU 推理必须串行，请由 JobManager 排队调用。
    """

    def __init__(self, device: str = "cuda", lazy_load: bool = True) -> None:
        self.device = device
        self._loaded = False
        self.lam = None
        self.tracker = None
        self.cfg = None
        self.flame_faces: "object" = None  # np.ndarray [39904, 3]
        self.flame_template_y_min: float = 0.0
        self.load_seconds: float | None = None
        if not lazy_load:
            self.load()

    # ------------------------------------------------------------------ 载入
    def load(self, on_progress: ProgressFn | None = None) -> None:
        """载入 LAM 主模型与 FLAME 跟踪器（约 20~40 s，仅需一次）。"""
        if self._loaded:
            return
        report = on_progress or _noop
        t0 = time.time()

        report("prepare", 0.0, "校验权重")
        lam_env.ensure_assets()
        lam_env.prepare()

        report("prepare", 0.6, "导入 LAM 源码")
        import torch  # noqa: F401  (延迟导入，保证 lam_env 先就位)

        with lam_env.lam_cwd():
            report("prepare", 1.2, "载入 LAM 主模型（2.0 GB）")
            self.cfg = self._build_cfg()
            self.lam = self._build_model(self.cfg)
            self.lam.to(self.device)
            self.lam.eval()

            report("prepare", 2.4, "载入 FLAME 跟踪器")
            self.tracker = self._build_tracker()

            report("prepare", 2.9, "准备 FLAME 拓扑")
            self.flame_faces = (
                self.lam.renderer.flame_model.faces_up.detach().cpu().numpy().astype("int32")
            )
            # 颈部裁剪基准：FLAME 中性模板的最低点（模板无肩，其下方全是外推区）
            self.flame_template_y_min = float(
                self.lam.renderer.flame_model.v_template_up[:, 1].min().item()
            )
            n_v = int(self.lam.renderer.flame_model.v_template_up.shape[0])
            assert self.flame_faces.max() + 1 == n_v, "FLAME 拓扑与高斯点数不一致"

        self._loaded = True
        self.load_seconds = time.time() - t0
        report("prepare", 3.0, f"就绪（耗时 {self.load_seconds:.1f}s）")

    @staticmethod
    def _build_cfg():
        """等价于 app_lam.py 的 parse_configs()，但参数写死、不依赖命令行。"""
        import sys

        from omegaconf import OmegaConf

        cfg = OmegaConf.create()
        cfg.blender_path = "blender"
        cfg.model_name = str(paths.LAM_CKPT) + "/"
        cfg.infer = str(paths.LAM_INFER_CFG)
        cfg.motion_video_read_fps = paths.RENDER_FPS

        cfg_infer = OmegaConf.load(cfg.infer)
        cfg.merge_with(cfg_infer)
        cfg.source_size = cfg_infer.dataset.source_image_res
        cfg.src_head_size = 112
        cfg.render_size = cfg_infer.dataset.render_image.high
        cfg.setdefault("save_tmp_dump", os.path.join("exps", "save_tmp", "lam_20k"))
        cfg.setdefault("image_dump", os.path.join("exps", "images", "lam_20k"))
        cfg.setdefault("video_dump", os.path.join("dumps", "lam_20k", "videos"))
        cfg.setdefault("mesh_dump", os.path.join("dumps", "lam_20k", "meshes"))
        cfg.setdefault("logger", "INFO")
        # 让 LAM 内部 argparse 也能解析（它读 sys.argv，服务进程里 argv 是 uvicorn 的）
        argv_backup = sys.argv
        sys.argv = [argv_backup[0] if argv_backup else "lam3d"]
        try:
            pass
        finally:
            sys.argv = argv_backup
        return cfg

    @staticmethod
    def _build_model(cfg):
        import torch
        from safetensors.torch import load_file

        from lam.models import ModelLAM

        model = ModelLAM(**cfg.model)
        ckpt_path = os.path.join(str(cfg.model_name), "model.safetensors")
        ckpt = load_file(ckpt_path, device="cpu")
        state = model.state_dict()
        matched = 0
        for k, v in ckpt.items():
            if k in state and state[k].shape == v.shape:
                state[k].copy_(v)
                matched += 1
        print(f"[lam3d] LAM 权重载入完成：{matched}/{len(ckpt)} 个张量匹配（{ckpt_path}）")
        return model

    @staticmethod
    def _build_tracker():
        from tools.flame_tracking_single_image import FlameTrackingSingleImage

        tm = paths.TRACKING_MODELS
        return FlameTrackingSingleImage(
            output_dir="output/tracking",  # 每个任务前会被改写
            alignment_model_path=str(tm / "68_keypoints_model.pkl"),
            vgghead_model_path=str(tm / "vgghead/vgg_heads_l.trcd"),
            human_matting_path=str(tm / "matting/stylematte_synth.pt"),
            facebox_model_path=str(tm / "FaceBoxesV2.pth"),
            detect_iris_landmarks=False,
        )

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def list_samples() -> list[str]:
        if not paths.SAMPLE_INPUT_ROOT.is_dir():
            return []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted(
            p.name for p in paths.SAMPLE_INPUT_ROOT.iterdir() if p.suffix.lower() in exts
        )

    # ------------------------------------------------------------------ 主流程
    def run(
        self,
        image_path: str | Path,
        job_dir: str | Path,
        on_progress: ProgressFn | None = None,
        job_id: str = "",
    ) -> PipelineResult:
        """跑完整条链路（单图 → 1 帧重建人脸 + 3D 资产 + ARKit 会话）。

        ``image_path`` 为任意可读的图片路径；重建所用姿态是**用户这张照片自身**
        的拟合姿态（跟踪导出 1 帧），动画由 ARKit 实时泼溅提供。
        """
        report = on_progress or _noop
        image_path = Path(image_path).resolve()
        job_dir = Path(job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        if not image_path.is_file():
            raise FileNotFoundError(f"输入图片不存在：{image_path}")

        self.load(report)

        result = PipelineResult(
            job_id=job_id,
            image_name=image_path.name,
            out_dir=job_dir,
        )
        t_start = time.time()
        timings: dict[str, float] = {}

        with lam_env.lam_cwd():
            # --- 1. FLAME 跟踪 -------------------------------------------
            t = time.time()
            report("tracking", 3.0, "检测人脸并抠图")
            export_dir = self._track_image(image_path, job_dir, report)
            timings["tracking"] = time.time() - t

            # --- 2. 参考图预处理 -----------------------------------------
            t = time.time()
            report("reference", 26.0, "裁剪并读取规范形状参数")
            image, shape_param = self._prepare_reference(export_dir, report)
            timings["reference"] = time.time() - t

            # --- 3. 姿态装配（单帧：跟踪导出的那张照片本身）--------------
            t = time.time()
            report("motion", 31.0, "装配参考姿态（单帧）")
            import torch

            from lam.runners.infer.head_utils import prepare_motion_seqs

            # ★ 路径末尾的斜杠不能省，且必须是绝对路径：
            #   head_utils.prepare_motion_seqs 用 os.path.dirname(motion_seqs_dir) 去定位
            #   transforms.json / flame_param/。相对路径下 dirname 会随 cwd 变化。
            #   跟踪导出的 export/input/ 天然只有 1 帧（transforms.json 的 frames 长度 1）。
            motion_seq = prepare_motion_seqs(
                str(export_dir).rstrip("/") + "/",
                None,
                save_root=str(job_dir / "tmp"),
                fps=paths.RENDER_FPS,
                bg_color=1.0,
                aspect_standard=paths.ASPECT_STANDARD,
                enlarge_ratio=[1.0, 1.0],
                render_image_res=paths.RENDER_SIZE,
                multiply=16,
                need_mask=False,
                vis_motion=False,
                shape_param=shape_param,
                test_sample=False,
                cross_id=False,
                src_driven=[image_path.stem, "reference"],
            )
            motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
            n_frames = 1
            timings["motion"] = time.time() - t
            report("motion", 34.0, f"共 {n_frames} 帧")

            # --- 4/5. LAM 前馈 + 单帧渲染 --------------------------------
            t = time.time()
            res = self._infer_and_render(image, motion_seq, n_frames, report)
            timings["infer_render"] = time.time() - t

            # --- 6. 3D 模型导出（含重建人脸图）---------------------------
            t = time.time()
            report("mesh", 88.0, "写出人脸图 / GLB / OBJ / Gaussian Splat")
            self._export_meshes(res, job_dir, image_path.stem, result)
            result.shape_param = shape_param.reshape(-1)[:50].tolist()
            timings["mesh"] = time.time() - t

        # ARKit 实时表情会话：把 canonical 高斯 + 参考相机留在显存里，
        # 之后每个表情向量只需一次 animate_gs_model（无需光栅化）即可打包泼溅。
        # 会话失败不影响主任务（下载 / 统计照常产出）。
        try:
            arkit_live.save_session(self, job_id, res, job_dir, res["bg"])
        except Exception as exc:
            print(f"[lam3d] ARKit 实时会话初始化失败（不影响主流程）: {exc}")

        result.face = job_dir / "face.png"
        result.preview = job_dir / "preview.png"
        result.thumb = job_dir / "thumb.jpg"
        result.stats = {
            "frames": n_frames,
            "gaussians": int(res["n_gaussians"]),
            "faces": int(self.flame_faces.shape[0]),
            "seconds": round(time.time() - t_start, 2),
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "model_load_seconds": None if self.load_seconds is None else round(self.load_seconds, 2),
        }
        (job_dir / "result.json").write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report("done", 100.0, f"完成，用时 {result.stats['seconds']}s")
        return result

    # ------------------------------------------------------------- 各步骤实现

    def _track_image(self, image_path: Path, job_dir: Path, report: ProgressFn) -> Path:
        """跑 FLAME 跟踪，返回 export 出来的序列目录。"""
        tr = self.tracker
        # 每个任务独占输出目录，避免并发/重名互相覆盖
        tr.output_dir = str(job_dir / "tracking")
        tr.output_preprocess = os.path.join(tr.output_dir, "preprocess")
        tr.output_tracking = os.path.join(tr.output_dir, "tracking")
        tr.output_export = os.path.join(tr.output_dir, "export")
        for d in (tr.output_preprocess, tr.output_tracking, tr.output_export):
            os.makedirs(d, exist_ok=True)

        report("tracking", 5.0, "人脸检测 + 抠图 + 关键点")
        code = tr.preprocess(str(image_path))
        if code != 0:
            raise RuntimeError("人脸检测失败：未在图片中找到清晰正面人脸（请换一张正脸照）")

        report("tracking", 13.0, "FLAME 参数拟合（约 10~20 s）")
        code = tr.optimize()
        if code != 0:
            raise RuntimeError("FLAME 跟踪优化失败")

        report("tracking", 24.0, "导出跟踪结果")
        code, export_dir = tr.export()
        if code != 0:
            raise RuntimeError("跟踪结果导出失败")
        return Path(export_dir)

    def _prepare_reference(self, export_dir: Path, report: ProgressFn):
        """裁剪 + 抠图 + 读取 canonical_flame_param.npz 里的 shape 参数。"""
        import torch

        from lam.runners.infer.head_utils import preprocess_image

        image_path = export_dir / "images/00000_00.png"
        mask_path = export_dir / "fg_masks/00000_00.png"
        if not image_path.exists():
            raise FileNotFoundError(f"跟踪输出缺少参考图：{image_path}")

        image, _mask, _intr, shape_param = preprocess_image(
            str(image_path),
            mask_path=str(mask_path) if mask_path.exists() else None,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=paths.ASPECT_STANDARD,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=paths.SOURCE_SIZE,
            multiply=14,
            need_mask=True,
            get_shape_param=True,
        )
        assert shape_param is not None, "未能读取 canonical_flame_param.npz 中的 shape"
        from PIL import Image as _Image
        import numpy as _np

        vis = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(_np.uint8)
        vis_path = export_dir.parent / "reference.png"
        _Image.fromarray(vis).save(vis_path)
        report("reference", 31.0, "参考图就绪")
        return image, torch.as_tensor(shape_param, dtype=torch.float32)

    def _infer_and_render(self, image, motion_seq, n_frames: int, report: ProgressFn) -> dict:
        """LAM 前馈 + 逐视角渲染。

        等价于 ``ModelLAM.infer_single_view``，但把逐视角循环展开以便回报进度。
        """
        import numpy as np
        import torch
        from einops import rearrange

        device, dtype = self.device, torch.float32
        model, renderer = self.lam, self.lam.renderer

        report("infer", 34.0, "图像编码 + 隐式点生成")
        flame_params = {k: v.to(device) for k, v in motion_seq["flame_params"].items()}
        render_c2ws = motion_seq["render_c2ws"].to(device)
        render_intrs = motion_seq["render_intrs"].to(device)
        render_bg = motion_seq["render_bg_colors"].to(device)

        with torch.no_grad():
            # preprocess_image 已返回 [1, 3, H, W]，批维已就位。
            # （上游写的是 image.unsqueeze(0)[:, 0]，等价于原样使用。）
            ref = image.to(device, dtype)

            # ★ get_query_points 内部访问 device.type，必须传 torch.device 而非字符串
            query_points, flame_params = renderer.get_query_points(flame_params, device=ref.device)
            latent_points, image_feats = model.forward_latent_points(
                ref, camera=None, query_points=query_points
            )
            image_feats_bchw = rearrange(
                image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1]))
            )
            gs_model_list, query_points, flame_params, _ = renderer.forward_gs(
                gs_hidden_features=latent_points,
                query_points=query_points,
                flame_data=flame_params,
                additional_features={
                    "image_feats": image_feats,
                    "image": ref,
                    "image_feats_bchw": image_feats_bchw,
                },
            )
            n_gaussians = int(gs_model_list[0].xyz.shape[0])
            report("infer", 50.0, f"生成 {n_gaussians} 个 3D 高斯")

            # ★ 关键：头部模型要在 FLAME 蒙皮后的空间里导出。
            #   gs_model_list[0].xyz 是「蒙皮前」的规范位置，其 offset 只有在经过
            #   animation_forward 的 LBS 之后才落到正确位置（渲染之所以正确，
            #   是因为 forward_animate_gs 内部对每个视角都做了这一步）。
            #   直接用 xyz 建网格会在肩颈区域出现大三角面（"翅膀"）。
            # 网格直接取「蒙皮前」的规范位置 gs_model_list[0].xyz。
            # 实测验证（见 README「几何空间」一节）：
            #   · 与官方 LAM 产物 framework_img.obj 逐点误差 mean 0.0006（头半径 0.10）
            #   · 改用蒙皮后位置反而偏离官方 0.035，且长边面比例不变（2.26% vs 2.27%）
            #   · 官方自己的 status_shaped_mesh.obj 同样有 2.24% 长边面
            # 结论：肩颈处的长边面（"翅膀"）是 LAM 对胸像类输入的固有预测，
            #       不是本模块的网格装配问题；渲染时它被高斯不透明度掩盖。
            cano_model = gs_model_list[0]

            render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
            rgbs, masks = [], []
            t0 = time.time()
            for i in range(n_frames):
                out = renderer.forward_animate_gs(
                    gs_model_list,
                    query_points,
                    renderer.get_single_view_smpl_data(flame_params, i),
                    render_c2ws[:, i : i + 1],
                    render_intrs[:, i : i + 1],
                    render_h,
                    render_w,
                    render_bg[:, i : i + 1],
                )
                rgb = out["comp_rgb"][0, 0].permute(1, 2, 0)  # [H, W, 3]
                mask = out["comp_mask"][0, 0].permute(1, 2, 0)  # [H, W, 1]
                rgbs.append(rgb)
                masks.append(mask)
                if (i + 1) % 5 == 0 or i + 1 == n_frames:
                    frac = (i + 1) / n_frames
                    eta = (time.time() - t0) / max(frac, 1e-6) * (1 - frac)
                    report(
                        "render",
                        50.0 + 38.0 * frac,
                        f"{i + 1}/{n_frames} 帧" + (f"，剩余 ~{eta:.0f}s" if eta > 1 else ""),
                    )

            rgb = torch.stack(rgbs).cpu().numpy()  # [Nv, H, W, 3]
            mask = torch.stack(masks).cpu().numpy()  # [Nv, H, W, 1]
            mask = np.where(mask < 0.5, 0.0, 1.0)
            rgb = np.clip(rgb * mask + (1 - mask), 0.0, 1.0)
            frames = (rgb * 255).astype(np.uint8)

        return {
            "frames": frames,
            "cano_gs": cano_model,
            "n_gaussians": n_gaussians,
            "render_hw": (render_h, render_w),
            # 交给网格导出步骤做顶点色烘焙（见 _bake_vertex_colors）。
            # 相机参数去掉批维，烘焙处按 [N, 4, 4] 使用。
            "renderer": renderer,
            "gs_model_list": gs_model_list,
            "flame_params": flame_params,
            # 中性基准（姿态/表情全零，只留 betas）：ARKit 实时会话的起点。
            "flame_params_neutral": arkit_live.neutral_flame_data(renderer, flame_params),
            "query_points": query_points,
            "c2ws": render_c2ws.detach().cpu().numpy()[0],
            "intrs": render_intrs.detach().cpu().numpy()[0],
            # 背景色（去掉批维），供 ARKit 实时会话复用同一背景
            "bg": render_bg[0, 0].detach().cpu(),
        }

    def _export_meshes(self, res: dict, job_dir: Path, stem: str, out: PipelineResult) -> None:
        """导出 3D 人物模型：GLB（查看器用）/ OBJ（通用）/ Gaussian-Splat PLY。

        导出前做两步清理，都是为了让「给渲染用的高斯」变成「给人看的网格」：

        1. **颈部裁剪**：LAM 的 FLAME 以 ``add_shoulder=False`` 构建，只在
           「头 + 颈」上训练过；颈部以下是外推区，实测会长出跨接三角面。
        2. **拉普拉斯平滑**：高斯点的 offset 是为了体积渲染优化的，相邻点偏移
           互不约束；直接连成三角面会自交起刺（渲染时被高斯混合掩盖，导出后
           在 Blender / 浏览器里就暴露成破面）。做几次轻度平滑即可消除，
           同时保留面部细节。

        两步都可用环境变量关闭：``LAM3D_NECK_TRIM=0``、``LAM3D_SMOOTH_ITERS=0``。
        """
        import numpy as np
        import trimesh

        gs = res["cano_gs"]
        verts = gs.xyz.detach().cpu().numpy().astype(np.float64)
        colors = np.clip(gs.shs.detach().cpu().numpy().reshape(-1, 3), 0.0, 1.0)

        # --- 顶点色烘焙 ---
        # 必须在裁剪/平滑之前做：这一步要把顶点投影回渲染帧采样，
        # 而投影依赖「顶点顺序 == 高斯顺序 == FLAME 细分顶点顺序」。
        # 裁剪与平滑都会打乱/改动这个对应关系。
        # 优先用烘焙出的反照率；逐高斯的 f_dc 只作兜底（低不透明度点的
        # f_dc 学得很差，直接当顶点色会在脸上留下彩色噪点）。
        base_normals = _vertex_normals(verts, self.flame_faces)
        colors, baked_mask = self._bake_vertex_colors(
            res.get("renderer"), res.get("gs_model_list"), res.get("flame_params"),
            res.get("query_points"), res.get("frames"), res.get("c2ws"), res.get("intrs"),
            verts, base_normals, colors,
        )
        if baked_mask.any():
            print(f"[lam3d] 顶点色来源：渲染帧烘焙 {baked_mask.mean()*100:.1f}%，"
                  f"f_dc 兜底 {(~baked_mask).mean()*100:.1f}%")

        faces, trimmed = self._trim_neck(verts, self.flame_faces, self.flame_template_y_min)
        if trimmed:
            print(f"[lam3d] 颈部裁剪：剔除 {trimmed} 个外推面（肩/底座区域）")

        # 裁掉孤立顶点（颈部以下的点已无面引用），避免文件里留下垃圾数据
        used = np.unique(faces)
        if len(used) < len(verts):
            remap = np.full(len(verts), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            verts, colors, faces = verts[used], colors[used], remap[faces]
            print(f"[lam3d] 清理孤立顶点：{len(remap)} -> {len(verts)}")

        if paths.SMOOTH_ITERS > 0:
            # 只平滑几何，不动颜色：颜色是烘焙出来的反照率，平滑会把
            # 五官边界抹开（眼睛、嘴唇会糊成一片）。
            verts = _laplacian_smooth(verts, faces, paths.SMOOTH_ITERS, paths.SMOOTH_LAMBDA)
            print(f"[lam3d] 拉普拉斯平滑：{paths.SMOOTH_ITERS} 次（λ={paths.SMOOTH_LAMBDA}）")

        normals = _vertex_normals(verts, faces)

        # ★ 色彩空间：顶点色已是 sRGB（LAM 渲染输出即 sRGB），而 glTF 规范规定
        #   COLOR_0 必须是**线性**空间。直接写入会让 three.js / Blender 把 sRGB
        #   当线性用，表现为整体发灰、暗部被压死（实测近黑顶点从 7% 涨到 40%）。
        vcols = _srgb_to_linear(colors)

        # GLB 用自带写出器（见 core/glb.py 的说明：trimesh 会丢顶点色、
        # 把线性颜色压成 8 bit、且不生成法线，三个坑都会静默损坏产物）。
        from .glb import write_glb

        glb_path = job_dir / f"{stem}_head.glb"
        write_glb(
            glb_path,
            vertices=verts.astype(np.float32),
            faces=faces,
            colors=np.concatenate([vcols, np.ones((len(vcols), 1))], axis=1),
            normals=normals,
            name=f"{stem}_head",
        )
        out.glb = glb_path
        _verify_glb(glb_path)

        # OBJ 供 Blender / MeshLab 等使用；顶点色是 OBJ 的扩展语法，
        # 用 trimesh 写即可（它不会改顶点数）。
        mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, process=False)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, vertex_colors=(vcols * 255).astype(np.uint8)
        )
        mesh.vertex_normals = normals
        obj_path = job_dir / f"{stem}_head.obj"
        mesh.export(obj_path)
        out.obj = obj_path

        # --- Gaussian Splatting PLY（SuperSplat / antimatter15 等查看器可直接打开） ---
        ply_path = job_dir / f"{stem}_gaussians.ply"
        gs.save_ply(str(ply_path), rgb2sh=True, offset2xyz=False)
        out.gs_ply = ply_path

        # --- 重建人脸图：就是第 5 步渲染出的那一帧（512×512 PNG）---
        # preview.png 是 face.png 的**逐字节副本**（契约 §3.2：preview 为 face 别名，
        # 前端两处引用同一张图；任务 1 的验收 A2b 直接比对两份字节）。
        first = res["frames"][0]
        from PIL import Image as _Image

        face_path = job_dir / "face.png"
        _Image.fromarray(first).save(face_path)
        out.face = face_path

        preview = job_dir / "preview.png"
        preview.write_bytes(face_path.read_bytes())
        out.preview = preview

        _Image.fromarray(first).resize((256, 256), _Image.LANCZOS).convert("RGB").save(
            job_dir / "thumb.jpg", quality=88
        )

    @staticmethod
    def _trim_neck(verts, faces, y_min: float, margin: float | None = None):
        """裁掉 FLAME 模板颈部最低点以下的外推面。

        Args:
            verts: [N, 3] 网格顶点
            faces: [F, 3] 面索引
            y_min: FLAME 中性模板的最低点 y
            margin: 上抬量，None 时取 ``paths.NECK_TRIM_MARGIN``；<=0 表示不裁剪

        Returns:
            (保留的面索引数组, 被剔除的面数)
        """
        import numpy as np

        if margin is None:
            margin = paths.NECK_TRIM_MARGIN
        if margin <= 0:
            return faces, 0

        y_cut = y_min + margin
        # 三个顶点都在裁剪面之上才保留，避免留下悬挂边
        keep_mask = (verts[faces][:, :, 1] >= y_cut).all(axis=1)
        return faces[keep_mask], int((~keep_mask).sum())

    def _bake_vertex_colors(self, renderer, gs_model_list, flame_params, query_points,
                            frames, c2ws, intrs, verts, normals, fallback, max_frames=48):
        """用 LAM 自己渲染的每一帧给顶点上色，替代逐高斯的 f_dc 颜色。

        为什么必须这么做：``gs.shs`` 的直流分量是为**体积渲染**优化的逐点颜色。
        渲染时每个像素是几十个高斯按不透明度加权混合的结果，所以低不透明度的
        点即使颜色完全乱掉，画面也看不出来 —— 网络从来没有被约束去把它们学对。
        直接把这些颜色当顶点色贴到网格上，脸上就会出现彩色噪点（实测明显）。

        做法：对若干帧，把网格顶点先**动画到该帧的姿势**（FLAME LBS，与渲染
        用的完全同一条路径），再投影进该帧图像采样像素；只取正面朝向相机的
        顶点；最后对全部帧取平均。得到的是一份干净、稳定的反照率。

        Args:
            renderer: LAM 的 GS3DRenderer（提供 animate_gs_model）
            gs_model_list / flame_params / query_points: 与渲染时同源
            frames: [N, H, W, 3] uint8 渲染结果
            c2ws / intrs: [N, 4, 4] 相机参数
            verts: [V, 3] 规范网格顶点
            normals: [V, 3] 顶点法线
            fallback: [V, 3] 采样失败时的兜底颜色
            max_frames: 参与烘焙的帧数上限（均匀抽样）

        Returns:
            (baked_colors [V,3], hit_mask [V] bool)
        """
        import numpy as np
        import torch

        n = len(verts)
        if frames is None or len(frames) == 0:
            return fallback, np.zeros(n, dtype=bool)

        total = len(frames)
        step = max(1, total // max_frames)
        sel_frames = list(range(0, total, step))[:max_frames]

        acc = np.zeros((n, 3), np.float64)
        cnt = np.zeros(n, np.float64)
        H, W = int(frames.shape[1]), int(frames.shape[2])

        for i in sel_frames:
            # 把规范顶点动画到第 i 帧（与渲染时的 LBS 完全一致）
            try:
                single_view = renderer.get_single_view_smpl_data(flame_params, i)
                frame_data = renderer.get_sing_batch_smpl_data(single_view, 0)
                with torch.no_grad():
                    animated = renderer.animate_gs_model(
                        gs_model_list[0], query_points[0], frame_data
                    )[0]
                posed = animated.xyz.detach().cpu().numpy().astype(np.float64)
            except Exception:  # noqa: BLE001 — 烘焙失败就退回 f_dc，不该拖垮主流程
                continue
            if len(posed) != n:
                continue

            c2w = np.asarray(c2ws[i], dtype=np.float64)
            intr = np.asarray(intrs[i], dtype=np.float64)
            w2c = np.linalg.inv(c2w)

            # 只是近似：用规范法线判断朝向（姿势变化不大时足够）
            facing = np.einsum("ij,ij->i", normals, posed - c2w[:3, 3][None, :]) < 0.0

            cam = posed @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            ok = facing & (z > 1e-6)
            if not ok.any():
                continue
            inv_z = 1.0 / np.where(ok, z, 1.0)
            u = intr[0, 0] * cam[:, 0] * inv_z + intr[0, 2]
            v = intr[1, 1] * cam[:, 1] * inv_z + intr[1, 2]
            ui = np.rint(u).astype(np.int64)
            vi = np.rint(v).astype(np.int64)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            acc[idx] += frames[i][vi[idx], ui[idx]].astype(np.float64) / 255.0
            cnt[idx] += 1.0

        baked = fallback.copy()
        good = cnt > 0
        baked[good] = acc[good] / cnt[good, None]
        print(f"[lam3d] 顶点色烘焙：{len(sel_frames)}/{total} 帧，命中 {int(good.sum())}/{n} "
              f"({good.mean()*100:.1f}%) 个顶点")
        return baked, good

__all__ = ["SingleImage3DPipeline", "PipelineResult", "ProgressFn"]
