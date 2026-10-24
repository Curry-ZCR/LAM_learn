# LAM 协方差改进工作总结(2026-10-24)

> 范围:LAM 单图数字人推理期协方差调制改进。基线=LAM 原版(面内 2x2 极分解+面积缩放)。
> 全部工作在推理期完成,不改 LAM 上游训练,单资产验证。验收口径:RTX 4090 / f32 / B=1。

## 一、做了什么(时间线)

1. **文献与机制调研**:2024-2026 协方差/旋转调制论文 19+ 篇(HyperGaussians/DLA/GALA/GTAvatar/
   Grassmannian Splatting/3D Gaussian Blendshapes 等),docs/lit_scan_covariance_2024_2026.md。
2. **方法原型与评测**:TGA(Σ'=JΣJᵀ)、VTP(先平均 J 再极分解)、RotSync(逐面极分解+Markley 平均)、
   RSB(RotSync+法向贴合)四方法实现+多口径评测(rig/jitter/false-move/无参考法向残差)。
3. **无真值评测协议**:gt_free_eval_protocol.md+eval_cov_protocol_v2.py,含 N1 穿透/N4 形变覆盖/N5 饱和。
4. **生产化落地**:rot_sync.py/RsbModulator/cov_switch.py(五方法热切换)+Web 前端参数面板+README。
5. **并行子任务验证**(3 个子智能体):Δε 法向差异/p99 尾部 fallback/N1N4N5。

## 二、改进了什么(核心机制)

### 改进 1:RotSync 旋转同步(lam3d/core/rot_sync.py)

问题:LAM 原版只做面内 2x2 旋转,离面薄片高斯(嘴唇/下巴/鼻翼)在皮肤剪切滚动时方向不跟。

方法:逐面 J_f=[d1 d2 n1][e1 e2 n0]⁻¹ -> Newton 极分解 R_f -> 逐面四元数 -> **Markley 平均**
(外积矩阵主特征向量,免半球对齐,coherence=1-λ2/λ1 免费副产物)-> 恒等吸附 -> 增益 -> 45° 限幅
+相干度门控单次 slerp -> q'=δq⊗q0, s'=s0(尺度透传不动)。

### 改进 2:RSB 法向贴合(lam3d/core/rsb.py,方法四终态)

在 RotSync 之上加一级最短弧修正:从形变后网格边直接算顶点法向 n_v(=nv_surf),
δq_fix=[cos(a·θ̃/2), sin(a·θ̃/2)·axis],a=g^1.5 共享门控,θ̃=25°·tanh(θ/25°) 软限幅,q'=δq_fix⊗δq⊗q0。

### 改进 3(生产 bug 修复):slerp_identity 半球对齐

原实现对 w<0 的四元数重建 w=cos(t·half) 恒≥0,丢失 w 符号——带限幅/门控的顶点插值方向整体反号
(随机四元数 t=1 时 |dot| 低至 0.11,即高达 ~40° 的错误)。修复:先半球对齐(sgn·v/||v|| 为轴)再插值。

### 改进 4:基础设施

- cov_switch.py:lam/tga/rot_sync/vertex_then_polar/rsb 五方法运行时热切换,/api/cov/methods REST+前端滑块。
- 评测脚本族 50+(scripts/),实验数据 experiments/p99_fallback/ 等。

## 三、效果(已测指标)

### 3.1 修复 slerp bug 后的刚性贴合重验(5 preset,一环 Kabsch 参考口径)

| 方法 | rig50 | rig90 |
|---|---:|---:|
| LAM 原版(不转) | 2.616° | 21.944° |
| RotSync gate=0 | **0.285°** | 5.849° |
| RotSync gate=0.3 | **0.283°** | 5.727° |
| RSB gate=0.3 | 0.308° | 6.464° |

要点:**slerp 修复使 RotSync rig50 从历史(带 bug)5.507°→0.285°,好 20 倍**;
RSB 法向贴合的刚性代价仅 +0.025°(p50)。

### 3.2 RSB 相对 RotSync 的旋转差异(E2E 真实 capture,5 帧)

转角差 mean 1.12° / p50 0.056° / p90 3.49° / p99 13.66° / max 24.96°(=25° 软限幅生效);
法向修正施加角 mean 0.90°;12.4% 顶点严格恒等。

### 3.3 性能(RTX 4090,B=1,调制器增量)

| | LAM | TGA | RotSync | RSB |
|---|---:|---:|---:|---:|
| 增量耗时 | 0 | ~2.4ms | 5.07ms | 5.88ms(+0.81) |

端到端单帧 ≈13.2ms(RSB),满足 ≤15ms 预算。

### 3.4 W8 断言套件 12/12 ALL PASS

恒等帧逐位透传/刚体输入残差 1.5e-05°/门控回退逐位/尺度透传逐位/法向修角≤cap(0.04°≤25°)/全 finite。

### 3.5 被否决的路线(负结果也是产出)

| 路线 | 判决 | 证据 |
|---|---|---|
| J⁻ᵀ 协变法向 | **负收益** | 相对 nv_surf 残差 15.47° vs ROT 14.30°,仅 16.5% 顶点更近 |
| SPD 测地插值 | 砍 | 上界收益为负(同上) |
| p99 尾部 fallback(38 臂) | **G2 FAIL** | ORACLE4 信息下界 60.91°>60° 判据;尾部=参考系不稳 |
| GSDeformer splitting | 不实现 | 单轮膨胀 2.1-4×,递归撞 2^18;straddle 中位 0.1-7.6° 不对症 |
| 拉伸支路(stretch) | 默认关 | 面积比与高斯厚度无因果绑定,实测零影响 |

## 四、未测/待办

- R11(逐高斯转角分布)带 slerp 修复的重验(可选,rig 重验已覆盖主口径)。
- 感知质量用户研究(协议已在 gt_free_eval_protocol.md 设计,未执行)。
- 多资产(非 messi)泛化(单卡单资产约束)。

## 五、文件索引

- 核心:lam3d/core/{rot_sync.py, rsb.py, cov_switch.py}
- 文档:lam3d/docs/{method4_rsb_design.md, rsb_reverify.md, exp_de_normal_cov.md,
  exp_n1n4n5_metrics.md, exp_p99_fallback.md, exp_p99_fallback_summary.md,
  gt_free_eval_protocol.md, lit_scan_covariance_2024_2026.md, cov_methods_eval.md}
- 验证脚本:lam3d/scripts/(check_rot_sync_mesh.py / proto_m4_*.py / exp_*.py / eval_*.py)
- 本轮子任务数据:lam3d/experiments/p99_fallback/

## 六、复现

```bash
# rig 重验(核心口径)
python lam3d/scripts/exp_p99_fallback.py --stage offline
# E2E: 启动 web 后在界面下拉切换 rot_sync/rsb
```

> 历史指标数字口径见 docs/exp_p99_fallback_summary.md「口径声明」节(旧 npz 为 slerp-bug 版,勿混用)。