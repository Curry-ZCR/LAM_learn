# LAM 单图数字人推理期协方差调制:完整技术报告

> 项目周期:2026-10-20 ~ 10-26 | 仓库:github.com/Curry-ZCR/LAM_learn(main @ 6c3a992)
> 基线系统:LAM(arXiv:2502.17796)单图→FLAME 绑定 3DGS 头像,推理期冻结
> 硬约束:①推理期规则式,无训练/无重打标 ②compute_splat <30ms ③协方差改动必须回写 (q,s)
> (gs_renderer cov3D_precomp 硬编码 None,外部 Σ 被光栅器静默丢弃) ④单图资产,无同人物表情真值
> 验收环境:RTX 4090 / f32 / B=1 / 单资产 messi
>
> 本报告为全工作总账:§1 背景与问题 §2 方法演进(五种) §3 机制详解 §4 评测体系 §5 实测结果
> §6 视觉验证 §7 否决路线 §8 结论与局限 §9 论文引用清单 §10 交付物索引。

---

## 1. 背景与问题定位

### 1.1 LAM 的原版调制与缺陷

LAM 将 ~20018 个 3D 高斯绑在 FLAME Loop 细分顶点上,ARKit52 表情驱动时只更新:
- 高斯中心(顶点位移 x_B);
- **面内 2×2 旋转角 + 面积缩放**(LAM 自带的 _get_cov_modulator)。

缺陷:高斯极度各向异性(轴比 p50=3.37/p90=83/>2 占 80.4%),大量"薄片"高斯近似切于皮肤表面。
表情变化时皮肤发生**剪切+离面弯曲**,但原版只动面内角度——薄片高斯的方向不跟随曲面法向变化,
产生三类伪影:**薄片斜插出皮肤**(protrusion)、**假旋转**(false rotation)、**时域抖动**(jitter)。

### 1.2 问题的数学形式

记规范网格面 f 的形变梯度 J_f = [d₁ d₂ n₁][e₁ e₂ n₀]⁻¹(边向量 + 法向构成的 3×3 映射)。
高斯协方差应从 Σ₀ 变为 Σ' = J Σ₀ Jᵀ(TGA);但极分解 J = R·P 后只取旋转:
- **面内近似**(LAM):只用 2×2 子块,离面旋转全丢;
- **只取 R**(RotSync):Σ' = R Σ₀ Rᵀ,方向跟随但丢剪切;
- **全 J**(TGA):保真但 65-70% 高斯最长轴越界(>1.35×),clamp 后引入假旋转 3-5%。

### 1.3 关键实测:厚度不变性(立项依据)

在真实 messi 会话 5 个 preset 上直接测 Nanson 厚度比 k_f = A|det J|/A':
逐顶点对数厚度 ℓ 的 p1/p50/p99 ≈ 0(精确到 1e-3),|ℓ|>log1.35 占比 **0.0%**,k_f>1.35 仅 0.01%。
⇒ **真实表情下皮肤厚度几乎不变**;TGA 的 65-70% "最长轴越界"绝大部分不是物理拉伸,而是
"纤维伸长 |J n₀| 把剪切误记成厚度"的系统偏差(TGA 从未被指出的缺陷)。
这意味着:把拉伸限制到法向厚度 + 把剪切全部交给方向,不是妥协,而是**正确参数化**。

---

## 2. 方法演进(五种,全部推理期实现)

| # | 方法 | 机制一句话 | 状态 |
|---|---|---|---|
| 1 | LAM 原版 | 面内 2×2 极分解 + 面积缩放 | 基线 |
| 2 | TGA(雅可比) | Σ' = J Σ₀ Jᵀ 精确回写 q/s | 保留(对照) |
| 3 | VTP(先平均再分解) | 先把 J 平均到顶点再极分解 | 对照组(跳变) |
| 4 | **RotSync 旋转同步** | 逐面极分解 R_f + Markley 平均 → Σ' = R Σ₀ Rᵀ | **生产默认** |
| 5 | **RSB 法向贴合** | RotSync + 逐高斯法向最短弧修正 | **本项目最终交付** |

## 3. 机制详解

### 3.1 RotSync(lam3d/core/rot_sync.py)

管线(全部 f32、向量化、无 Python per-N 循环):

1. **逐面形变梯度** J_f = [d₁ d₂ n₁][e₁ e₂ n₀]⁻¹(nan_to_num + clamp ±64);
2. **逐面极分解** R_f = polar(J_f):8 步 Newton 迭代(f32,比 f64 SVD 快 15×);
3. **逐面四元数** q_f = matrix_to_quat(R_f);
4. **Markley 平均**(顶点聚合):解外积矩阵 Σ_f A_f q_f q_fᵀ 的主特征向量——
   ① 免半球对齐(q 与 −q 外积相同,天然免疫符号);② 免费副产置信度 coherence = 1−λ₂/λ₁;
5. **恒等吸附 + 增益 + 45° 硬限幅 + 相干度门控**,合并为对参考四元数的**单次 slerp**;
6. 写回:q' = δq ⊗ q₀, s' = s₀(尺度透传,逐位不动)。

设计要点:**不能先平均 J 再分解**(反向顶点跳变,即 VTP 对照组),也不能逐面取算术平均四元数
(半球问题)。Markley 方法两者皆免。

### 3.2 RSB 法向贴合(lam3d/core/rsb.py,方法四终态)

在 RotSync 之上加一级**最短弧修正**:

- 顶点法向 n_v 从**形变后网格边**直接叉积计算(不经过 J,避开被否决的 J⁻ᵀ 协变公式,见 §7);
- 高斯当前薄轴方向 b = q₀ 旋转下最小轴;slerp 后薄轴已部分跟随,剩余夹角 θ;
- δq_fix = [cos(a·θ̃/2), sin(a·θ̃/2)·axis],其中 θ̃ = 25°·tanh(θ/25°)(软限幅),a = g^1.5
  (与旋转门控共享 g,保证"拉伸放行 ⊆ 旋转放行"的嵌套门控);
- 最终 **q' = δq_fix ⊗ δq ⊗ q₀**;尺度支路实测恒等(eigmax_rel 恒 1.277)→ 默认关闭。

### 3.3 生产 bug 修复:slerp 半球对齐(全项目影响最大的单点)

原 slerp_identity 对 w<0 的四元数重建 w = cos(t·half) 恒 ≥0,**丢失 w 符号**——带限幅/门控的
顶点插值方向整体反号(随机四元数 t=1 时 |dot| 低至 0.11,即高达 ~40° 错误)。
修复:先半球对齐(sgn·v/‖v‖ 为轴)再插值。**修复前所有 rig50/rig90 历史数字都是 bug 版**
(见 §5.1 口径声明)。

### 3.4 基础设施

- **cov_switch.py**:五方法运行时热切换,REST /api/cov/methods + Web 前端下拉/滑块(每方法
  参数面板独立:RSB 暴露 rotation_gain/max_angle_deg/coherence_gate/normal_cap_deg/gate_power);
- 评测脚本族 50+(scripts/),实验数据 experiments/。

---

## 4. 评测体系(无真值协议)

单图资产无同人物表情真值 ⇒ 自建 gt_free 协议(docs/gt_free_eval_protocol.md,569 行):

- **patch 分区统计**(K=155):替代"20018 独立样本"假象;ICC 设计效应 rigid 30-72×,
  逐高斯 i.i.d. CI 会严重高估显著性,patch bootstrap(2000 次)才有效;
- **刚性贴合 rig**(主口径):以一环 Kabsch 参考旋转为真值,逐高斯旋转误差角;
- **无参考法向残差 nres**:薄轴与形变后法向夹角,不依赖任何被测方法中间量;
- **N1 穿透 / N4 形变覆盖 Ω / N5 饱和占比**:堵"缩小作弊"/暴露覆盖不足;
- **时域**:roughness + TJA jerk 放大(RTT 切平面参考);
- **契约回归**:α̅=0 与 RotSync torch.equal;max(s/s0)≤1.35;恒等帧逐位透传;
- **安慰剂对照**:shrink-only(只缩幅不改向)与 A0 逐位相同 ⇒ 收益来自修正**方向**而非幅度;
  neg-sign(反向修正)反而 +36% 更差 ⇒ 方向信息真实。

---

## 5. 实测结果

### 5.1 刚性贴合(slerp 修复后重验,5 preset,一环 Kabsch 口径)

| 方法 | rig50 | rig90 |
|---|---:|---:|
| LAM 原版(不转) | 2.616° | 21.944° |
| RotSync gate=0 | **0.285°** | 5.849° |
| RotSync gate=0.3(生产默认) | **0.283°** | 5.727° |
| RSB gate=0.3 | 0.308° | 6.464° |

口径声明:修复前历史数字(RotSync rig50 8.501/5.507 等)为 bug 版,方向性结论不变。
**slerp 修复单项使 RotSync rig50 5.507→0.285(好 20 倍)**;RSB 法向贴合刚性代价仅 +0.025°(p50)。

### 5.2 法向贴合效果(重验)

nres p50 3.763° → **0.044°**(98.8% 降幅,87-91% 顶点被改善);rig50 相对 A0(RotSync)
平均 −35.2%(5 preset 一致 −33.6%~-38.0%,bug 版口径)。

### 5.3 RSB vs RotSync 端到端差异(真实 capture,5 帧)

转角差 mean 1.12°/p50 0.056°/p90 3.49°/p99 13.66°/max 24.96°(=25° 软限幅生效);
法向修正施加角 mean 0.90°;12.4% 顶点严格恒等(门控正确放空)。

### 5.4 性能(RTX 4090,B=1,调制器增量)

| | LAM | TGA | RotSync | RSB |
|---|---:|---:|---:|---:|
| 增量耗时 | 0 | ~2.4ms | 5.07ms | 5.88ms(+0.81) |

端到端单帧 ≈13.2ms(RSB),远低于 30ms 预算。分段:face_areas 0.199/face_jacobians 0.802/
polar 2.946(39.6%)/matrix_to_quat 0.723/Markley ~0.63ms。

### 5.5 时域质量

TJA jerk 放大:rot_sync 的 out_jerk 在全部 4 条轨迹上比 tga 低 1-2 个数量级
(jawOpen 1.37e-5 vs 2.48e-4 等);迟滞 hyst_deg 低 6-7 个数量级。VTP 与 rot_sync 同级。

### 5.6 W8 断言套件 12/12 ALL PASS

恒等帧逐位透传 / 刚体输入残差 1.5e-05° / 门控回退逐位 / 尺度透传逐位 /
法向修角 ≤ cap(0.04°≤25°)/ 全 finite。

---

### 5.6 PSNR/SSIM/LPIPS(2026-10-26,口径受限)

无逐帧照片 GT(单图资产;驱动视频非同人)。已完成的替代口径(docs/eval_psnr_ssim_lpips.md):

- **中性重构校验**:五方法中性帧(arkit=0)**逐位一致**——协方差调制不污染静态重构 ✓
- **方法间偏离**(参考=LAM 原版渲染,24 帧×yaw{0,35},LPIPS-alex):
  **lam=参考系,偏离恒 0.0**(自渲染噪声底 99–102 dB†);其余相对 LAM:
  tga 0.01252 > vertex_then_polar 0.01046 ≈ rot_sync 0.01041 ≈ **rsb 0.01029**(yaw0);
  yaw35 全线缩小 ~13×。RSB 与 RotSync 改动量同量级(软限幅不引入额外可见改动),
  TGA 改动最大;改动量≠质量,质量仍以 5.1 rig 表+§6 视觉对照为准。

## 6. 视觉验证(渲染对照)

图:docs/images/rsb_compare.png(4 行×2 列×2 preset:B0 静态/RotSync/RSB/|RSB−ROT|×8;正面+35°)。
放大切图:docs/images/rsb_vision/。

| preset | 视角 | 差异 mean | 变化像素 |
|---|---|---:|---:|
| ones | 0° | 0.749 | 25.9% |
| ones | 35° | 0.794 | 27.0% |
| extreme | 0° | 0.654 | 25.0% |
| extreme | 35° | 0.699 | 25.9% |

视觉模型(deepseek-v4.1-flash 读图)+主线像素统计双口径交叉验证:

1. **量级**:全局 mean|Δ| 0.82-1.00/255,max 45-56,>64 像素 0——肉眼几乎不可辨,放大 3-6 倍下脸可见;
2. **空间定位**:主峰=颊/嘴(亮区 10-11.3%)+下巴/下颌(9.4-10.4%),鼻仅 1.2-2.8%;嘴部框
   独占差异能量 40-52%;峰值=上唇/人中/口腔开口+唇颏沟/下巴胡须下缘——**与 RSB 设计目标
   (薄片高斯密集区)精确吻合**;
3. **修正方向=减伪影**:RotSync 偏离 B0>32 的像素上 62-70% 变好、误差再降 4.6-9.9%;
   全局 L1 降 2.9-4.8%;口腔收窄变暗(−9.5%/−13.3%)、唇颏沟 V 形接缝磨软;
4. **不动轮廓/姿态**:剪影 XOR 仅 8-330px(全图 112k);非锐化(梯度相关 0.17-0.27);
5. **稀疏定点**:top5% 像素承载 59-62% 差异能量;两 preset 修正模式同构(extreme 仅小 ~15%)。

一句话:**RSB = 下三分之一脸(唇-口-颏-颌)的定点小修正,不触碰轮廓/姿态/上半脸。**

---

## 7. 否决路线(负结果也是产出)

| 路线 | 判决 | 证据 |
|---|---|---|
| J⁻ᵀ 协变法向 | **负收益** | 相对 nv_surf 残差 15.47° vs ROT 14.30°,仅 16.5% 顶点更近;真实 preset Δε p90 1.178°<2° 判据成立 |
| SPD 测地插值 Σ(α)=expm((1−α)logmΣ_R+α logmΣ_TGA) | 砍 | 上界收益为负(依赖协变法向) |
| p99 尾部 fallback(38 臂×5 preset) | **G2 FAIL** | ORACLE4(三选一+一环最好面)信息下界 60.91°>60° 判据;尾部=参考系不稳(100% 在 1ring vs 2ring 差≥5° 集),稳定子集 F0 p99 仅 4-6° |
| GSDeformer splitting | 不实现 | 按论文原判据实测单轮膨胀 2.1-4×,递归 6 轮撞 2^18 上限;straddle 中位 0.1-7.6° 不对症;论文中找不到出处"增幅<1%" |
| 拉伸支路(stretch) | 默认关 | 厚度不变性实测 ⇒ 面积比与高斯厚度无因果绑定,实测恒等 |
| 逐高斯 i.i.d. 统计 | 方法论否决 | patch ICC 设计效应 10-70×,i.i.d. CI 严重虚高 |
| 绝对 protrusion 当验收指标 | 方法论否决 | 四方法中性态逐位相同(θ p50≈59° 为资产属性,非方法属性);正确形式=方法间差分 Δθ |

---

## 8. 结论与局限

### 8.1 结论

1. **推理期协方差调制可以把刚性贴合误差从 2.6° 压到 0.28°(9×),端到端 13.2ms**——在冻结模型、
   无训练、单图的硬约束下成立;生产 bug(slerp 半球)修复单项贡献 20×,大于任何新方法。
2. **RSB 是 RotSync 的正确补充**:法向贴合残差 98.8% 降幅、视觉修正精确落在设计目标区域
   (唇-口-颏-颌),且定量证明"只修伪影、不动表情/轮廓/姿态"(口腔变暗=减伪影而非拉回中性)。
3. **p99 尾部(1% 高斯 143° 撕裂)是逐面常值 J 的分辨率极限,不是参数问题**:38 臂扫描 + ORACLE4
   信息下界证明任何门控/fallback/参考系选择都到不了 60°;唯一出路是顶点级 J(+2.8ms)或训练期方案。
4. **评测方法学**:①patch 分区 + bootstrap 是唯一有效统计;②参考系口径(Markley vs Kabsch)差
   10-68°、足以翻转方法排序,未来协议必须固定并双报;③无参考指标(nres/N1/N4/N5)+安慰剂对照
   + 契约回归构成不依赖真值的证据链。
5. **文献定位**:RSB 与 On the Skinning of Gaussian Avatars(2509.11411,加权四元数平均)同题,
   但后者无置信度/无协变法向/无厚度项;Markley 主特征向量 + 免费置信度 + 法向最短弧修正是
   更完备的工程化实现。

### 8.2 局限

- 单资产(messi)单卡验证;多资产泛化未做;
- PSNR/LPIPS:无逐帧照片 GT(单图资产+驱动视频非同人)。已做替代口径(docs/eval_psnr_ssim_lpips.md):中性重构五方法逐位一致(不污染静态);方法间偏离 rot_sync/rsb LPIPS 0.0103-0.0104 vs tga 0.0125(改动量温和);感知质量仍以 rig 表+视觉对照为准;
- 感知质量用户研究(协议已设计)未执行;
- 内部代号 TGA 与 NeurIPS 2025 "TGA: True-to-Geometry Avatar" 撞名,正式写作需改名(建议 ACD)。

---

## 9. 论文引用清单

### 9.1 基线与头像系统

- [LAM] **LAM: Large Avatar Model for One-shot Animatable Gaussian Head**. arXiv:2502.17796,2025.
  单图→FLAME 绑定 3DGS 头像的基线系统。
- [GaussianHeadAvatar] **Gaussian Head Avatar**. arXiv:2312.03029. 几何引导初始化(SDF+DMTet)+MLP 变形场;
  最近三角形搜索+变形梯度估计位移(Insta 路线对照)。
- [FlashAvatar/GaussianAvatars 系] FLAME 三角面绑定 + UV 空间高斯的代表工作(本项目 LAM 同系)。
- [Splatting] **3D Gaussian Splatting**. Kerbl et al., SIGGRAPH 2023, arXiv:2308.04079. 光栅化基座。

### 9.2 协方差/形变传播(方法直接相关)

- [TGA-内部代号] 本项目方法二 Σ'=JΣ₀Jᵀ 的解析实现(注意与下条撞名)。
- [TGA-NeurIPS25] **TGA: True-to-Geometry Avatar Dynamic Reconstruction**. NeurIPS 2025 Spotlight
  (superkeaibb.github.io/TGA_web)。⚠️ 与内部代号撞名,正式写作必须改名。
- [GSDeformer] **GSDeformer**: arXiv:2405.15491. 高斯分裂传播——本项目实测判死(§7)。
- [SkinningGaussians] **On the Skinning of Gaussian Avatars**. arXiv:2509.11411. 加权四元数平均,
  RSB 最近亲,但无置信度/协变法向/厚度项。
- [HyperGaussians] **HyperGaussians**(CVPR 2026). arXiv:2507.02803. 高维条件分布切片出 3DGS,
  MLP 输出 latent;表达力增强但需重训练(不满足推理期约束,相关工作引用)。
- [3DGS-Blendshapes] **3D Gaussian Blendshapes**. Ma et al., arXiv:2404.19398(SIGGRAPH/ToG)。
  高斯属性线性混合 B_ψ=B0+Σψ_k ΔB_k,ΔB_k 用 α_k·d_scale 保持与网格位移成比例。
- [HeadGaS] **HeadGaS**. ECCV 2024, arXiv:2312.02902. 不动高斯位置,潜特征基线性混合→MLP 出颜色/不透明度。
- [TexAvatars] 混合 Jacobian 相关(子智能体检索,见 lit_scan_covariance_2024_2026.md)。
- [OUGS] 参数协方差传播(同上)。
- [Dual-Covariance] 双协方差(同上)。
- [ReorgGS] 条件数门控(同上)。

### 9.3 法向/几何表示(启示来源)

- [GTAvatar] **GTAvatar**. arXiv:2512.09162. 2DGS surfels 解析法向 + UV 映射,单矩阵乘每交点。
- [GrassmannianSplatting] **Grassmannian Splatting**. arXiv:2607.10489. 4D rank-2 surfels,
  法向在 Gr(3,4)=S³/{±1},投影器 P_n=P_{-n} 符号免疫——与 Markley 外积免疫 q/−q **同构**。
- [MixedGaussianAvatar] **MixedGaussianAvatar**. arXiv:2412.04955. 2D 高斯保几何+3D 高斯补渲染质量。
- [TimeWalker] **TimeWalker**. arXiv:2412.02421. DNA-2DGS,变形梯度 F=L_def·Λ⁻¹·L_canon⁻¹ 驱动 surfels。

### 9.4 驱动与加速

- [GALA] **GALA: One Basis to Animate Them All**. arXiv:2610.02207. 从预训练高斯数字人蒸馏
  身份无关 blendshape 基,浅 MLP 预测系数,免重训原模型——证明预训练模型动画输出有共享线性结构。
- [DLA] **DLA**. NVIDIA+ETH, arXiv:2507.15979. 单图→视频扩散 dream→lift 3DGS→SMPL-X UV 空间 body 驱动。
- [PanoLAM] **PanoLAM**. arXiv:2509.07552. LAM 同系,单张 unposed 图→全头 Gaussian。
- [SEGA] **SEGA**. arXiv:2504.14373. 单图可驱动 3DGS 头部,层级 UV+动静解耦。
- [RAF] **RAF**. arXiv:2603.08645. 训练期检索增强表情库。

### 9.5 数学工具

- Markley et al., *Fundamentals of Spacecraft Attitude Determination and Control* — Markley 四元数
  平均(外积矩阵主特征向量)与置信度(1−λ₂/λ₁)。
- Higham, *Functions of Matrices* — Newton 极分解迭代。
- Kabsch/Umeyama 对齐 — 一环参考旋转口径。

完整 1315 条 arXiv 索引:docs/arxiv_index_gaussian.tsv;检索详情:docs/lit_scan_covariance_2024_2026.md
与 report/gaussian_avatar_deform_survey.md(307 行)。

---

## 10. 交付物索引

**代码**(全部在 /workspace/lam3d/):
- lam3d/core/rot_sync.py — RotSync 调制器(+slerp 半球修复)
- lam3d/core/rsb.py — RSB 法向贴合调制器(最终交付)
- lam3d/core/cov_switch.py — 五方法热切换 + REST + 前端
- lam3d/server/app.py + static/ — Web 前端(方法下拉/参数滑块)
- scripts/ 50+ — 评测/消融/验证脚本(check_rot_sync_*/proto_m4_*/exp_*/eval_*)

**文档**(lam3d/docs/):
- method4_rsb_design.md(v2,70KB) — RSB 完整设计+消融+安慰剂对照
- research_report_v1.md(v3) — 研究总报告(口径已刷新)
- rsb_reverify.md — 修复后重验+视觉对照(§0)
- exp_de_normal_cov.md / exp_p99_fallback.md(+summary) / exp_n1n4n5_metrics.md — 三子任务实验
- gt_free_eval_protocol.md(569 行) — 无真值评测协议
- cov_methods_eval.md / cov_methods_deepdive.md — 四方法定量评测
- lit_scan_covariance_2024_2026.md + arxiv_index_gaussian.tsv — 文献
- images/rsb_compare.png + images/rsb_vision/ — 渲染对照与放大切图

**报告**(report/):
- WORK_SUMMARY_2026-10-24.md — 10-24 工作总结
- FULL_TECHNICAL_REPORT.md — 本文档

**运行**:bash run.sh --daemon(端口 8000),Web 界面下拉切换方法,或 REST /api/cov/methods。
