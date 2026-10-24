# p99 尾部 fallback 研究整合(主线整合版 v2,并入 s-b 正式回报)

> 数据与实验:s-b 子任务(exp_p99_fallback.py 1324 行,五阶段 capture/offline/splitting/
> refstability/local_rotation,全 numpy 向量化)。子任务交付版:docs/exp_p99_fallback.md(16.7KB,含全表)。
> 本文件为主线整合视角。**所有数字为 slerp 半球修复后(HEAD 14:00)重算值**,未混用旧 npz 的 rot_sync 行。

## 判决:G2 FAIL(预声明判据 rigid p99<=60deg 全部 5 preset)

38 臂(含退化臂)无一通过,g2_arms_passing=[]。最好命名方案 F3_vtp_tau0.99:4/5 过,
extreme p99=99.55deg(n>60=487)。p50 辅助条件全臂满足(delta max 0.008deg,τ=0.99 时 F1 +0.505/F2 +1.312 越限)。

## extreme p99 全臂(其余 preset 见 s-b 文档表)

| 臂 | τ=0.3 | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 |
|---|---:|---:|---:|---:|---:|---:|
| F0 rot_sync 原样 | 158.27 | - | - | - | - | - |
| F1 回落 LAM | 154.66 | 152.98 | 150.61 | 143.45 | - | 135.90 |
| F2 回落 VTP(不门控) | 155.38 | 154.81 | 153.03 | 150.57 | - | 150.42 |
| F3 回落 VTP(门控) | 150.63 | 147.29 | 138.89 | 117.99 | 106.21 | **99.55** |

参考:VTP 原样 99.55 / LAM 原样 150.73 / ORACLE3(三选一)95.01 / **ORACLE4(+一环最好面旋转)60.91**。
排序恒定 F3 > F1 > F2;最优 = F3(回落 VTP)。

## 三条硬结论

1. **τ→1 不是「找到好 gate」而是「少用 rot_sync」**:coherence 中位 0.9988-1.0,frac<0.7 仅 0.2-2.5%;
   τ=0.3 时已有 66-78% 顶点被 t>=0.99 全关——F1/F2/F3 实际是全局平滑插值;F3_τ0.99 与 VTP 原样
   几乎无差(99.55 vs 99.55)。生产若采用必须标注「换方法」而非 fallback。
2. **尾部成因 = 参考系不稳,不是「一环 176° 矛盾」**:refstability 阶段——尾部 100% 落在 1-ring vs 2-ring
   差 >=5deg 的集合;稳定子集上 F0 p99 仅 4.17-6.29deg。任务书「一环面旋转两两夹角中位 176°」方向对
   但数值夸大:真实 tail 中位 53.8-64.4deg,>150deg 仅 0.4-1.6%,>170deg 全 preset 为 0。
   (教训:用非正交基 (e1,e2,e1×e2) 做 F1·inv(F0) 会得 p50 117-118deg 假象——极分解已非旋转。)
3. **信息下界**:逐高斯自由选择 + 允许从自己一环挑真实面旋转(ORACLE4)才到 60.91deg。
   60deg 不是调参可达——尾部需要的旋转不在一环可达集合内。

## 口径声明(评测方法学)

- **代码漂移**:HEAD rot_sync.py(mtime 14:00:11,slerp 半球修复)无法复现旧 npz 的 rot_sync 行
  (lam/vtp 逐位吻合,rot_sync 五 preset 全 mismatch)——旧 npz 是 bug 版计算。s-b 全部重算,口径干净。
- **参考系选择本身值 10-68deg p99**:换成面积加权 Markley 参考后 rot_sync extreme 158.27->121.19
  而 vtp 99.55->164.99,**方法排序直接翻转**。「158deg」含相当比例评测口径差。

## GSDeformer splitting:判死,不实现

arXiv 2405.15491v5,按论文原判据(175deg,跳过轴长<1e-2,k1k2k3=(1/2)^(1/3))实测:
- 单轮膨胀:位移场 2.098-3.977x(extreme 3.977x),旋转场(下界)1.046-1.476x;
- 递归 6 轮不收敛,全撞 2^18 上限(children_mean 到 67799);
- straddle(高斯两端 skinning 旋转差)中位仅 0.12-7.62deg,仅 0.3-16% 高斯任一轴 >45deg → 分裂不对症
  (尾部是参考系退化而非足迹太大);
- 任务书假设「论文报告 splitting 后高斯数增幅<1%」在原 PDF 0 命中,无出处——报告中如实指出未引用。

## 对主线的最终影响

- G2 记 FAIL;p99 尾部 = 逐面常值 J 的分辨率极限(参考系不稳顶点),接受现状;
- 生产维持 RotSync gate=0.3(或 RSB);不实现 VTP fallback(τ0.99 版 ≈ 换方法)与 splitting;
- 评测协议未来必须固定参考系口径(Markley vs Kabsch)并报告两者——差 10-68deg 足以翻转排序。

## 产物清单

- scripts/exp_p99_fallback.py(1324 行,五阶段)
- docs/exp_p99_fallback.md(s-b 交付版,16.7KB,全表)
- experiments/p99_fallback/{results,splitting,refstability,local_rotation}.json + 100 个 tail_ids_*.npy
  (20 臂 x 5 preset)+ cache/rot_capture.npz
- /tmp/p99_*.json x19(bend/f5/facerepro/footprint/full_res/nonrigid/offline_res/oracle/refstab/
  split_bound/split_faithful/split_mesh/straddle/two_metrics/inst_sweep/unst/allarms_unst)
- GPU:capture 阶段一次 89s,按 GPU_COORDINATION 串行,已全部结束。