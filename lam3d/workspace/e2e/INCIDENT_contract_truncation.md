# 事故记录：CONTRACT_v2.md 被 rw_edit 超时截断为 0 字节（已完整恢复）

- 发生时间：2026-10-16 10:48:47（远端 mtime）
- 发现时间：2026-10-16 10:52 前后（Lead 例行核验 `wc -l CONTRACT_v2.md` 返回 0）
- 恢复时间：2026-10-16 10:52:35（远端 mtime）
- 影响范围：仅 `/workspace/lam3d/CONTRACT_v2.md` 一个文件。生产代码（lam3d/lam3d/**、static/**、scripts/**、LAM/**）零影响。

## 现象
Lead 用 `rw_edit` 给 CONTRACT_v2.md 的 §8 验收条目 14（清晰度）追加一条「判定口径澄清」。该次调用返回
`sftp operation timed out`，但工具已经在远端把文件 **truncate 到 0 字节**（read-modify-write 的写入阶段
先截断、后写内容，写内容时 SFTP 超时 ⇒ 文件停留在 0 字节）。md5 = d41d8cd98f00b204e9800998ecf8427e（空文件）。

## 为什么之前没发现
`rw_read_file` 对该文件返回 `(empty or out of range)` 是**正确**的空文件表现，但容易被误读成
「越界/参数问题」；真正暴露问题的是 `wc -l` 返回 0。

## 恢复过程（可复现）
1. 在远端全盘搜索副本：`find / -name "CONTRACT_v2*"`、`/proc/*/fd/*` 打开句柄、编辑器 swap/backup —— **均无副本**。
2. 转向压缩历史：`decompress(blockId="b44", full=true, toFile=...)` 把「写契约那一刻」的完整对话还原成
   474427 字符的文本（b44 覆盖 m02063–m02142，正是 Lead 创建契约的回合）。
3. 从该文本里用 `json.JSONDecoder().raw_decode()` 抓出 `write` 工具调用的 `{"content": ..., "path": ...}`
   JSON 对象，落盘得到 **20668 字节**的原始契约（与冻结时的字节数完全一致，U+FFFD=0，CJK 1960 字，334 行）。
4. 用脚本重新施加其后由 Lead/teammate 做过的 5 处编辑（A13 判定口径、A14 无、6.1 左栏、6.1 右栏、6.2 位置段），
   每个 `str.replace` 都断言命中次数（期望 1 次），任一不匹配即 `exit(1)` 拒绝产出。
5. 得到 22474 字节 / 338 行的最新版，`rw_upload` 覆盖回去；核验 10 个章节标题存在、3 处 T5 编辑均在。

## 校验
- `md5sum` = dc944eedcb189a43ec21572ba4e31e69，22474 字节，338 行
- 章节齐全：§1 硬约束 / §4 LSPL v1 / §5 splat.js API / §6.2 / §8 验收标准 / §10 上一轮既有事实
- T5 编辑均在：`交集面积必须为 0`、`绘制缓冲区尺寸`、`在前景 = 非背景像素上` 各命中 1 次
- 无 0 字节文本文件残留（除两个本就为空的 `__init__.py`）；`py_compile` 全部 core/server 模块 exit=0；
  `node --check` app.js / splat.js 均 OK

## 教训（Lead 侧）
1. **不要用 `rw_edit` 改长文件**：远端 read-modify-write 无版本保护，超时可能留下 0 字节或半截文件。
   改用「本地生成完整文件 + `rw_upload` 覆盖」。
2. **关键文件先留副本**：契约类文件在每次编辑前 `cp CONTRACT_v2.md /tmp/CONTRACT_v2.<ts>.bak`。
3. **`rw_read_file` 返回 `(empty or out of range)` 时必须再跑一次 `wc -c`/`md5sum` 确认**，
   不要直接当成参数问题忽略。
4. 长任务/长文件操作后必须立刻 `stat` + `md5sum` 复核，而不是等到下次读完才比对。
