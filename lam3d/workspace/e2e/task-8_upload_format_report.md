# task-8 QA 报告：上传格式归一化（WEBP/BMP → PNG）独立端到端验证

- 验证方：qa-contract（独立验证，未复用 T7 作者脚本/结论）
- 被测目标：task-7（`lam3d/lam3d/server/app.py` 上传分支，14270 字节，mtime 2026-10-16 11:55:29）
- 服务：`python -m lam3d.server.app`，PID 53648 启动于 11:55:35（**晚于 app.py mtime** ⇒ 跑的是最终代码）
- 验证脚本：`/workspace/lam3d/scripts/upload_format_test.py`（qa-contract 独占）
- 结果：**51 passed / 0 failed（EXIT=0）**
- 原始日志：`/workspace/lam3d/workspace/e2e/upload_format_verify.log`

## 1. bug 曾经存在（逻辑复现 + 物证）

修复链条（`app.py` 修复前 = 按**后缀**落盘）：
1. `ALLOWED_SUFFIX = {".png",".jpg",".jpeg",".webp",".bmp"}`（app.py:19）→ `.webp`/`.bmp` 通过白名单；
2. 修复前直接 `dest.write_bytes(raw)`，落盘后缀沿用原后缀 → job 目录出现 `input.webp`；
3. `jobs.py` 的 `suffix = Path(image_name).suffix` → LAM 侧 `torchvision.io.read_image` 读到 webp/bmp
   → `RuntimeError: Unsupported image file. Only jpeg and png are currently supported.`

物证（现存 job 目录，正是后缀链的产物）：
- `workspace/jobs/09ef12e402a4/input.webp`（image_name=...fw658webp.webp）
- `workspace/jobs/98e7e5c4c438/input.webp`（同上）

独立复核 torchvision 支持面（本机实测）：
```
FAIL /tmp/uft_fix/webp.webp -> RuntimeError: Unsupported image file. Only jpeg and png are currently supported.
FAIL /tmp/uft_fix/bmp.bmp  -> RuntimeError: Unsupported image file. Only jpeg and png are currently supported.
OK   /tmp/uft_fix/true_png.png (3, 384, 384) torch.uint8
OK   /tmp/uft_fix/true_jpeg.jpg (3, 384, 384) torch.uint8
```
⇒ 修复前 webp/bmp 必在推理进程失败，逻辑链闭合。

（注：`logs/server.log` 已滚动，未保留当时的 failed 原文；job.json 只存 id/image_name，`GET /api/jobs/<id>` 现返回「任务不存在」——故改用 job 目录落盘文件 + torchvision 实测作证据。）

## 2. 端到端核心结果（真实 GPU 串行，无并发）

| 组 | 输入 | job id | 终态 | face.png | 非背景占比 |
|---|---|---|---|---|---|
| C | webp.webp (384×384) | dbda23951f2e | succeeded | 512×512 PNG | 52.09% |
| C | bmp.bmp (384×384) | 21e16f0b1590 | succeeded | 512×512 PNG | 51.91% |
| Z | true_png.png | ffb97c977204 | succeeded | 512×512 PNG | 51.91% |
| Z | true_jpeg.jpg | de2eda157ca5 | succeeded | 512×512 PNG | 52.21% |
| X | webp_renamed.png（内容 webp） | a62085f54182 | succeeded | 512×512 PNG | 52.09% |
| S | sample=status.png | feb2e0e969bb | succeeded | — | — |

## 3. 零回归逐字节（关键）— PASS

- true_png: uploaded sha256=`d00b74ea51336896`，stored `input.png`=`d00b74ea51336896`，identical=True
- true_jpeg: uploaded sha256=`667c4fa918380133`，stored `input.jpg`=`667c4fa918380133`，identical=True
⇒ JPEG/PNG 走「原样落盘」分支，**零重编码**。

## 4. 内容≠后缀 — PASS
把 WEBP 改名 `.png` 上传 → job a62085f54182 succeeded；job 目录 `input.png` 的 `PIL.Image.format=PNG` ⇒ 判定按内容而非后缀。

## 5. 坏文件 / 超限 / 空文件 — PASS
- README.md 改名 .png → **400** `{"detail":"无法解析为图片：文件内容不是有效图片，请改用 PNG/JPG/WEBP"}`
- 文案不含 `Unsupported image file` / `torchvision` / `Traceback`
- >20MB → **413** `{"detail":"图片过大（>20 MB）"}`
- 0 字节 → **400** `{"detail":"上传内容为空"}`
- `.tiff` / `.gif` → 400「不支持的图片格式」= **按设计拒绝，非 FAIL**（`ALLOWED_SUFFIX` 本就不含它们）

## 6. accept ↔ 真实能力一致性 — PASS
`accept="image/png,image/jpeg,image/webp,image/bmp"`，四种 MIME 逐个实测全部 succeeded；README 提到 WEBP / PNG/JPG，与 accept 一致。

## 7. 产品级不变量 — PASS
- `git status --porcelain`：无 `lam3d/static/` 改动
- LAM 条目三条（flame.py / gs_renderer.py / covariance_modulation.py）mtime 05:08–06:40，**均早于 T7 开工（11:00）** ⇒ 既有脏文件，非本次回归；`find /workspace/LAM -newermt "2026-10-16 11:00:00"` 为空
- `static/*.html` 数 = 1；`py_compile app.py` = OK

## 8. 结论

**T7 修复真正解决了用户报告的问题**：webp/bmp 上传不再落到推理进程抛 `Unsupported image file`，而是在 HTTP 层归一化为 PNG 后成功重建（512×512 PNG 产物，非背景占比 >51%）；真 PNG/JPEG 字节级零回归；判据按内容而非后缀；坏文件/超限给出干净的 400/413 中文文案。

未覆盖边界：TIFF/GIF 仍为设计拒绝（400），前端 accept 也未宣传它们；多帧动画 GIF 未进入白名单，故无对应测试面。

## 自查记录（测试缺陷，非产品缺陷）
首次运行报 `FAIL I.no_new_lam_changes`。判定为**测试缺陷**：脚本按 `porcelain` 行偏移 `ln[3:]` 取路径，而输出被 `.strip()` 吃掉首行前导空格导致 `M LAM/...` 被切成 `AM/...`，路径变成不存在 → 误报。已改为 `split(maxsplit=1)` 去状态码 + 按 mtime 与 T7 开工时刻比较（比硬编码 allowlist 更严格），复跑 51/0。**产品代码未改动。**
