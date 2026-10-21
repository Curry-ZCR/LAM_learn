"""不占用 GPU 的快速自检：路径、权重、动作序列、拓扑一致性、HTTP 路由。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAIL = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global FAIL
    mark = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
    print(f"[{mark}] {name}" + (f"  — {extra}" if extra else ""))


def main() -> int:
    from lam3d.core import paths
    from lam3d.core.lam_env import missing_assets
    from lam3d.core.pipeline import SingleImage3DPipeline

    check("LAM_ROOT 存在", paths.LAM_ROOT.is_dir(), str(paths.LAM_ROOT))
    missing = missing_assets()
    check("LAM 权重齐备", not missing, f"缺 {len(missing)} 项" if missing else "全部就绪")

    # CONTRACT_v2 已删除动作序列：改为校验单图重建所需的静态资源。
    samples = SingleImage3DPipeline.list_samples()
    check("示例图片可用", len(samples) > 0, f"{len(samples)} 张")
    check("不再依赖动作序列(list_motions 已删)",
          not hasattr(SingleImage3DPipeline, "list_motions"))

    # 配置解析（不载模型）
    cfg = SingleImage3DPipeline._build_cfg()
    check("推理配置可解析", cfg.source_size == 512 and cfg.render_size == 512,
          f"source={cfg.source_size} render={cfg.render_size} model={Path(cfg.model_name).name}")

    # HTTP 路由
    from lam3d.server.app import app

    routes = {r.path for r in app.routes}
    need = {"/", "/api/config", "/api/status", "/api/generate", "/api/jobs/{job_id}",
            "/api/jobs/{job_id}/file/{name}"}
    check("HTTP 路由就绪", need <= routes, f"{len(routes)} 条")

    check("前端入口存在", (paths.STATIC_DIR / "index.html").is_file())
    from lam3d.core import paths as p2

    for lib in (
        "three/build/three.module.js",
        "three/examples/jsm/loaders/GLTFLoader.js",
        "three/examples/jsm/controls/OrbitControls.js",
        "three/examples/jsm/utils/BufferGeometryUtils.js",
    ):
        check(f"内置前端库 {lib}", (p2.VENDOR_DIR / lib).is_file())

    print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 项失败"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
