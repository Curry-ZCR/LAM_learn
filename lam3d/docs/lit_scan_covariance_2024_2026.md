# 文献侦察：LAM 表情驱动的推理期协方差/旋转调制（2024–2026）

检索方式：arXiv API 25 条组合查询，~1315 条去重条目，筛选 2024-01 之后、标题/摘要含 Gaussian splatting / surfel 的条目（1117 条），按机制关键词（Jacobian / normal / SPD-manifold / confidence-gate / surfel / rigidity / training-free）打分排序并逐篇读摘要。
**局限**：只读了 arXiv 摘要，未读正文/实验表；编号与日期为本次 arXiv API 返回值。

## A. 直接改 (q,s) 混合方式的（对本项目最相关）

1. **arXiv:2509.11411 — On the Skinning of Gaussian Avatars** (2025-09)
   机制：把 LBS 里的旋转部分换成 *加权四元数平均*（weighted rotation blending），只改 LBS、任意高斯光栅器可用。
   可借鉴：这是 RotSync（逐面极分解 + Markley 聚合）的"邻域同题"基线，可直接对照它把 quaternion averaging 放在顶点而不是面；它没有处理 shear，正好解释你 shape_err 0.857 > LAM 0.830 的来源。

2. **arXiv:2512.21099 — TexAvatars: Hybrid Texel-3D Representations for Stable Rigging** (2025-12)
   机制：UV 空间 CNN 预测局部几何属性，但驱动 3D 变形走 **mesh-aware Jacobians**，保证跨三角形边界平滑过渡。
   可借鉴：把"混合量"从旋转改成 **混合 Jacobian（3×3 仿射）再分解**，天然保留 shear，是修 RotSync 丢 shear 的最直接路线；跨面平滑权重也与"一环面反向须门控"对应。

3. **arXiv:2406.03394 — Gaussian Primitives for Deformable Image Registration** (2024-06)
   机制：每个图元带 center + covariance + **local rigid transformation**，逐 voxel 运动由 *blending 局部刚体变换* 得到。 
   可借鉴：混合完整刚体变换（旋转+平移耦合）而非只平均旋转，是 Markley 的第 5 种候选替代，避免旋转与位置解耦失真。

4. **arXiv:2606.06685 — RigPAPR: Rig-Based Animation of Static Neural Point Clouds** (2026-06)
   机制：把关节边界 gap/spike 归因于**表征本身**——每个 splat 的 shape 是在 canonical 姿态下"拼贴"好的，刚性 LBS 下 splat 不能弯折，拼贴必然破裂，"即使用逐图元修正也不行"。
   可借鉴：这是"46–76 顶点一环面反向、任何平均无解"的理论背书——门控不是工程妥协而是必然；同时给出上界直觉：任意逐高斯修正的幅度都应受限。

## B. 法向对齐 / J^{-T} 协变（可做门控信号）

5. **arXiv:2608.20687 — TopoSurfel** (2026-08)：mesh-guided surfel evolution，含 **normal alignment** 与 geometry-aware densification。
   可借鉴：把"法向一致性"做成显式判据。对你：用 J^{-T}n 与 n 的夹角（或 n_canon·n_posed）作逐面置信度，正好捕捉反向面。

6. **arXiv:2609.36969 — Prior-Driven Normals & Depths Regularization for 3DGS** (2026-09)
   机制：surface normal 正则"aligning Gaussian covariance with local surface structures"。
   可借鉴：给出 Σ 与法向的结构约束形式（最薄轴对齐 n），可当 (q,s) 回写时的合法性校验——回写后最薄轴偏离 n 太多即降权。

## C. 逐高斯置信度 / 门控

7. **arXiv:2608.06117 — Confidence matters: Leveraging Multi-view Geometric Priors** (2026-08)
   机制：用多视图模型自带的 **confidence map 加权** 每个先验，而非硬二值开关。
   可借鉴：把二值门控换成连续置信度权重 w∈[0,1] 乘在旋转修正上，避免 46–76 个反向顶点造成跳变。

8. **arXiv:2511.09397 — OUGS: Active View Selection via Object-aware Uncertainty in 3DGS** (2025-11)
   机制：把 (position, scale, rotation) 的参数协方差 **经渲染 Jacobian 传播**，得到可解释的逐图元不确定度。
   可借鉴：闭式、极便宜的逐高斯置信度（比学习型不确定度更适合 <30ms 约束）；传播矩阵在你的 TGA 里已算过 J，可直接复用。

9. **arXiv:2609.25746 — Dual Covariance Gaussian Splatting SLAM** (2026-09)
   机制：每个高斯保留 **两个协方差**：渲染协方差（优化器塑形）+ 跟踪/量测协方差（噪声模型），解耦两种互相冲突的需求。
   可借鉴：LAM 的冻结 Σ0 当"渲染协方差"原样保留，另存一份 **各向同性化/投影后** 的协方差只用于门控与 culling 判定——这样门控不会污染渲染保真度，也能缓解 TGA 65–68% 超 clamp 的问题。

10. **arXiv:2605.08739 — ReorgGS: Equivalent Distribution Reorganization** (2026-05)
    机制：指出"参数化退化"——冗余重叠簇造成 **近共线 Jacobian 响应**，梯度被高不透明度 floater 吃掉。
    可借鉴：把 J 的条件数/秩作为逐高斯退化指标：薄片（轴比 p90=83）上 J 近奇异时 TGA 解析解本就不可信，条件数就是天然门控量。

## D. 协方差流形 / clamp 的理论依据

11. **arXiv:2607.19043 — Local Second-Order Geometry Induced by Deformation Maps** (2026-07)
    机制：形变诱导的 **tangent covariance ≈ JΣJᵀ**，并**显式给出它与真协方差的偏差上界**（线性化误差界）。
    可借鉴：给 TGA 的 clamp 一个不是拍脑袋的判据——用局部线性化误差界决定 clamp 区间，替代固定 [0.25,1.35]。

12. **arXiv:2608.29538 — As-Rigid-As-Possible Deformation of Gaussian Radiance Fields** (2026-08)
    机制：几何编辑后再优化高斯，使 **rasterization 结果逼近"形变后的辐射场"**；并提出 **自适应各向异性空间低通滤波** 防走样。
    可借鉴：①"别只对齐几何、要对齐辐射场"——可作为四方法的统一评测目标；②极端拉伸时加各向异性低通（对最薄轴做下限保护）比硬 clamp 更平滑。

## E. 2D surfel vs 3D 薄片 / 残差化 / 几何一致性指标

13. **arXiv:2504.20607 — EfficientHuman: Articulated 2D Gaussian** (2025-04)
    机制：明确论证"Gaussian ellipsoid 无法准确表达动态物体表面"，改用 articulate 2D surfel。
    可借鉴：支撑你"薄片为主"的观察，也说明为何旋转回写对薄片比对各向同性高斯敏感得多（最薄轴方向误差被放大约 1/σ_min）。

14. **arXiv:2607.00959 — GaussianEmoTalker** (2026-07)
    机制：把表情动画写成 **neutral→emotional 的残差形变**问题（残差 + 强度编码），而非直接回归最终状态。
    可借鉴：把 (q,s) 改写成 **相对 LAM 原始的残差**：Δq=exp(β·δ) 形式，β=0 时逐位复现 LAM，保证不破坏原模型且便于扫参。

15. **arXiv:2606.05912 — SAGE: Self-Learning Expression Deformations** (2026-06)
    机制：2D surfel + SDF 联合优化，用 **geometric & appearance 自监督一致性** 替代长时序监督。
    可借鉴：设计无监督几何一致性指标（SDF/法向残差 + 渲染一致性）作为四方法的补充评测，不必等 3D 真值。

## 明确排除（已在你的"已知"清单内，检索时命中并跳过）
- arXiv:2607.10489 *Grassmannian Splatting I*（rank-2 时空 surfel，Σ=(P_n L)(P_n L)ᵀ）= 清单里的 GrassmannianSplatting
- arXiv:2610.02207 *One Basis to Animate Them All*（方法名即 GALA）= 清单里的 GALA
- arXiv:2603.08645 *Retrieval-Augmented Gaussian Avatars*（方法名 RAF）= 清单里的 RAF
- 2412.04955 / 2410.11682 / 2412.02421 = MixedGaussianAvatar / SurFhead / TimeWalker

## 三条最可执行的动作建议
1. **第 5 种方法候选**：混合 Jacobian（TexAvatars 式）或混合局部刚体变换（GaussianDIR 式）替代 Markley 旋转平均，直接针对 RotSync 的 shear 丢失（shape_err 0.857）。
2. **门控信号从"几何启发式"换成"法向翻转 + J 条件数 + 参数协方差传播"三重连续置信度**（#5/#8/#10/#7），对 46–76 反向顶点做软加权而不是硬门。
3. **双协方差思路（#9）**：冻结的 Σ0 只服务渲染，门控/culling 用另一份正则化协方差，可同时缓解 TGA 超 clamp 与保真度冲突。
