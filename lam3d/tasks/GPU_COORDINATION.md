# GPU 协调(主线留言,2026-10-24 22:0x)

单卡 RTX 4090 24GB,当前已 10.7GB,三个并行任务(s-a/s-b/s-c)都在起 LAM 会话,显存叠加有 OOM 风险。

请各自遵守:
1. 一次只跑一个 python 进程;跑完立即让进程退出(释放显存)再起下一个。
2. 能用 /tmp/m4_base.pt、/tmp/rot_capture.pt、/tmp/cov_protocol_pergaussian.npz 离线复算的,优先离线,不起 LAM 会话。
3. 需要真实会话的(如 s-c 的 N1/N4/N5):先一次性把张量 capture 落盘 /tmp,后续指标全部离线算。
4. 遇到 CUDA OOM:等 60s 再试(别人可能正在释放);仍失败则缩小 batch/只跑最小 preset 集,并在报告里注明。
