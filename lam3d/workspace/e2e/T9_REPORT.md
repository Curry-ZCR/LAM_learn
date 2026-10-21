# T9 报告 — 浏览器真实「选本机文件 → 点生成」端到端验证

**任务**：task-9（独立 QA，只读生产代码）
**脚本**：`lam3d/scripts/e2e_upload_formats.py`（本任务新建，md5 `bf56634f9941676f4e10788af39aa72d`）
**命令**：`/usr/local/miniconda3/envs/lam/bin/python scripts/e2e_upload_formats.py --tag t9`
**结果**：`==== 64 passed / 0 failed ====`，退出码 `0`（flag `/tmp/t9/t9_upload_formats.done` = `exit=0`）
**结构化报告**：`/tmp/t9/t9_report.json`（24.5 KB，含每条断言的原始 extra）
**服务新鲜度**：`lam3d/server/app.py` mtime `2026-10-16 11:55:29` < 服务进程启动 `2026-10-16 11:55:35`（PID 53648）✔

## 环境
- 无头 Chromium（Playwright 1.x / chromium-1243），`--no-sandbox --disable-dev-shm-usage --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`
- viewport 1600×950、`device_scale_factor=2`；比对基准 `bg=#0f131c`、前景判定 = 任一通道差 > 6
- 素材由 `/workspace/LAM/assets/sample_input/status.png`（1024×1024 RGB PNG）用 PIL 转出，只落 `/tmp/t9/`

## 逐格式结果（原始数字）

| case | 上传文件名（内容格式） | job id | 终态 | 耗时 | `#faceImg` | 画布非背景覆盖率 | console err | 截图 |
|---|---|---|---|---|---|---|---|---|
| webp | `t9_true.webp`（真 WEBP，109,562 B） | `51323042d0e4` | succeeded | 14.3 s | 512×512 | **0.1208**（fg 58,112/481,052，52,071 色） | 0 | `workspace/e2e/t9_webp_ok.png` 1,172,157 B |
| bmp | `t9_true.bmp`（真 BMP，3,145,782 B） | `09a2e45b61d7` | succeeded | 14.4 s | 512×512 | **0.1211**（fg 58,255/481,052，51,586 色） | 0 | `workspace/e2e/t9_bmp_ok.png` 1,161,669 B |
| png | `t9_true.png`（真 PNG，1,030,321 B） | `47f4bcb3b896` | succeeded | 14.1 s | 512×512 | **0.1211**（fg 58,255/481,052） | 0 | `workspace/e2e/t9_png_ok.png` 1,162,333 B |
| mismatch | `t9_webp_mismatch.png`（内容 WEBP / 后缀 .png） | `fb43dd209771` | succeeded | 14.3 s | 512×512 | **0.1208**（fg 58,112/481,052，52,071 色） | 0 | (附加) `workspace/e2e/t9_mismatch_ok.png` 1,173,042 B |

三张截图尺寸均为 3200×1900（full_page @ dsf=2）。

## 关键断言的原始读数
- 页面加载：`title='LAM · 单图生成 3D 数字人'`，首屏 console error **0**、pageerror **0**
- `setInputFiles('#file', …)` 后 `File` 对象进入 input：
  - webp → `files=1 name=t9_true.webp size=109562 type='image/webp'`
  - bmp → `files=1 name=t9_true.bmp size=3145782 type='image/bmp'`
  - png → `files=1 name=t9_true.png size=1030321 type='image/png'`
  - mismatch → `files=1 name=t9_webp_mismatch.png size=109562 type='image/png'`（浏览器按后缀给 MIME，内容实为 WEBP）
- 选图后：`#preview` `hidden=False` 且 `naturalWidth=1024`；`#dropHint.hidden=True`；`#generate.disabled=False`（三格式一致）
- `POST /api/generate` 全部 HTTP **200**；例如 webp 响应 `{"id":"51323042d0e4","image_name":"t9_true.png","state":"queued",...}`
- `#viewerBadge` = `'已加载 · 20,018 高斯 · 可 360° 拖拽'`（四例一致）
- `#errorBox` 全程 `hidden=True`，`#errorText=''`；`dialogs=[] popups=[]`；`requestfailed` 0 条

## 落盘字节级交叉验证（独立于 HTTP 响应）
对每个 job 目录里的 `input.*` 重新用 PIL 打开并取 sha256：

| 上传 | 上传 sha256(前16) | job 目录内 input | 内容格式 | 字节数 | 与上传同字节 |
|---|---|---|---|---|---|
| `t9_true.webp` WEBP | `d2445049ba1134e7` | `input.png` | **PNG** | 876,212 | 否（已归一化）|
| `t9_true.bmp` BMP | `cc24b15f6d394f4c` | `input.png` | **PNG** | 1,030,321 | 否（已归一化）|
| `t9_true.png` PNG | `67beec5a3ac55ef6` | `input.png` | **PNG** | 1,030,321 | **是（逐字节相同）** |
| `t9_webp_mismatch.png` WEBP | `d2445049ba1134e7` | `input.png` | **PNG** | 876,212 | 否（已归一化）|

即：非 JPEG/PNG 真的被转成了真 PNG 落盘（`image_name` 后缀也同步改成 `.png`）；真 PNG 零重编码（同字节）。

## 结论
**修复真正解决了用户在浏览器里上传本机图片失败的问题。** 用户报错路径（点「选择文件」→ 点生成）对真 WEBP、真 BMP 均端到端 succeeded，产物脸图 512×512、点云 20,018 高斯、画布非背景覆盖率 ~12%（阈值 5%），全程 `#errorBox` 不显示、console/pageerror 各 0 条，且不再出现 `RuntimeError: Unsupported image file…`。真 PNG 主路径零回归（且保持字节级零重编码）。「后缀 .png 但内容是 WEBP」的错配也被按内容正确分流。

## 未覆盖边界（诚实登记，均由本任务范围决定，非缺陷）
1. **拖拽 / 粘贴上传**：`static/app.js:91` 的 `drop` 事件与 `:98` 的 `paste` 路径本次未走（它们绕开 `#file` input；drop 只做 `/^image\//` 弱校验，其余交给服务端）。**但**：拖入 `.gif`/`.tiff` 会因 `ALLOWED_SUFFIX` 不含该后缀而被服务端 400 拒绝（`#errorBox` 显示「不支持的图片格式 .gif，请用 PNG/JPG/WEBP」）——优雅失败，未实测。
2. **无后缀文件名**：`app.py:104` 按后缀白名单拦截，上传 `image`（无扩展名、内容是有效 PNG）会 400。浏览器正常选文件都会带后缀，风险低，未实测。
3. **>20 MB 图片**（413）、**空文件**（400）、**损坏图片**（400「无法解析为图片」）：修复前已有分支，本任务未覆盖。
4. **TIFF/GIF 真素材**：本次未生成；其行为同 (1)，属 HTTP 层用例（T8 已覆盖 51/0）。
5. 画布像素取自 **元素截图** 通道（`via=element_screenshot`，`toDataURL` 因未开 `preserveDrawingBuffer` 被回退），与 T4/T6 口径一致。

## 产物
- `/workspace/lam3d/scripts/e2e_upload_formats.py`
- `/workspace/lam3d/workspace/e2e/t9_webp_ok.png`、`t9_bmp_ok.png`、`t9_png_ok.png`、`t9_mismatch_ok.png`
- `/tmp/t9/t9_report.json`、`/tmp/t9/run.log`、`/tmp/t9/t9_upload_formats.done`、`/tmp/t9/t9_true.{webp,bmp,png}`、`/tmp/t9/t9_webp_mismatch.png`
- 未修改任何生产代码（`lam3d/lam3d/**` / `lam3d/static/**` / `/workspace/LAM/**` mtime 均未变）
