# 高斯泼溅头像形变：7 个技术方向的文献调研与可迁移性评估

> 面向对象：LAM (arXiv 2502.17796, SIGGRAPH 2025) 单图重建 3DGS 头像 + ARKit 52ch 实时驱动。
> 约束前提：①仅推理期改造，不重训 LAM；②gs_renderer 只接受 (quaternion, scale) 回写（cov3D_precomp 硬编码 None，外部预算协方差被丢弃）；③每表情 30 ms 预算（当前 12 ms）；④无同人真值表情，PSNR/SSIM/LPIPS 仅自指；⑤20018 个高斯与 FLAME Loop 细分顶点 1:1 绑定，无面局部坐标系；轴长比 p50=3.37 / p90=83，80.4% 为薄片。
> 已实现四法：LAM 原调制器（面内 2×2 极分解 + tanh 面积软截断）、**TGA**（内部代号；Σ′=JΣ₀Jᵀ，逐面 J=[d1 d2 n1][e1 e2 n0]⁻¹，面积加权顶点平均，eigh→(q,s)，逐轴 clamp[0.25,1.35]）、**RotSync**（Σ′=RΣ₀Rᵀ，逐面极分解 U→Markley 主特征向量聚合，conf=1−λ₂/λ₁，45° 上限 + 置信门控单次 slerp）、**VTP**（先平均 J 再极分解，对照）。
>
> ⚠️ **命名冲突警告**：本报告所有“TGA”均指项目内部的 Jacobian 形变实现，**不是** Bo Guo 等的 *TGA: True-to-Geometry Avatar Dynamic Reconstruction*（NeurIPS 2025 Spotlight）。正式写作时内部代号必须改名。

---

## 方向一 · 顶点绑定的仿射 / Jacobian 形变（"面局部标架"的等价物）

**问题定位**：我们的 `Σ' = J Σ₀ Jᵀ`（每面一个 J，顶点一环内**面积加权平均**后 eigh）本质上就是 JBS 家族的朴素实现。文献里这条路线已经很成熟，关键差异全在**"怎么混合"**。

### 1.1 SurFhead — Affine Rig Blending（2410.11682，ICLR 2025）
- **核心机制**：Jacobian Blend Skinning（JBS）。逐三角形仿射 `Σ = J Σ^ℓ Jᵀ`、`μ = J μ^ℓ + T`（T 为三角形质心）；**JBS 不逐元素混合矩阵，而是先在极分解（linearized）空间取 (R, S) 再混合**。附录 A.2.1 论证极分解唯一性，A.2.2 论证"带 Jacobian 梯度时 PSD 保持"。原文表述"precise geometric deformation … including those affecting normals"，并用 2D surfel 修掉了眼球空洞（hollow illusion）。
- **相对我们四法的优劣**：与 TGA 同族但**混合域不同**——我们在协方差/矩阵域平均，一旦相邻面 J 的奇异值交叉或某面近退化，平均矩阵会出现**假旋转**（我们实测 3–5%）与 12% 量级 >10× 的轴长爆点；在 (R,S) 域混合可**在数学上消除假旋转**。代价是丢掉面内剪切（与 RotSync 同病），但它把"旋转保真"和"保留剪切"做成两条可选路径，比我们二选一更工程化。
- **是否需要重训**：原论文需要（J 与 offset 由图像监督），但**机制本身推理期可直接用**——我们的 J 是解析算出来的。
- **能否直接用**：顶点绑定 ★★★★（1:1 完全兼容）；薄片 ✗（需改 2D surfel）；推理期 ✓（多一次 3×3 极分解 + 四元数平均）。
- **可迁移技巧**：把"面积加权平均 Σ"换成"面积加权平均 (R,S)"——修 TGA 假旋转最省事的一刀。

### 1.2 TexAvatars（2512.21099，3DV 2026，KAIST）
- **核心机制**：`Σ = J Σ^ℓ Jᵀ`、`μ = J μ^ℓ + T`；核心是 **quasi-Phong Jacobian 场**：把逐三角形 Jacobian 映射到 texel 空间、在采样前解析施加形变，`Σ = GridSample_lerp(J_uv Σ^ℓ_uv J_uvᵀ)`。原话："removes the rigid binding between Gaussians and mesh triangles, effectively binding them to a smoothly blended neighborhood"；强调"把 scale/rotation 先并进 covariance 再重采样"以保证 PSD。另有 η∈R^128 隐表情向量直接驱动 opacity+color（皱纹/褶皱），无 shading 分支。
- **可引用的关键论证**："Since face-level deformations are dominated by rotations and only mild shear or scaling, J is near-isometric with singular values close to unity, i.e. ‖J‖ ≤ C"。**这句直接判定我们 TGA 里 12% 轴长 >10× 的样本不是几何信号，而是折叠边界处的数值病态**，为 clamp 提供理论依据而非拍脑袋。文中还有 "Jacobian vs. Scaled Rotation" 消融节，正是我们 TGA vs RotSync 的对照。
- **相对我们四法的优劣**：我们做**一环离散平均**（顶点上的 box filter），它做**连续场重采样**（双线性），后者对"一环内三角形朝向差 176°"的退化区有天然平滑作用（权重随 uv 连续变化，无 0/1 突变）。优于 TGA；但同样依赖 J。
- **是否需要重训**：需要（texel 场要学）。**机制**可分拆使用。
- **能否直接用**：顶点绑定 ★★★★；薄片 ✗；推理期 ✓（把面积权重换成重心坐标双线性权重即可）。
- **可迁移技巧**：(a) 用重心插值权重替代面积权重；(b) 引 ‖J‖≤C 论证正当化 clamp；(c) 抄 "Jacobian vs Scaled Rotation" 的消融表述写进实验节。

### 1.3 GaussianAvatars（2312.02069）
- **核心机制**：`Σ = R S Sᵀ Rᵀ`，R/S 由逐三角形标架定义，即 **scaled rotation 形式的 J**；binding inheritance 把新增高斯挂到最近三角形。
- **优劣**：所有"顶点/三角形绑定 + 3DGS"的开山基线。TexAvatars 明确指出"J realized as a scaled rotation in GaussianAvatars and as a full Jacobian in SurFhead"。对我们 = RotSync 的祖先（只取旋转+缩放，无剪切）。
- **重训**：是。**直接用**：概念 ★★★。**迁移**：binding inheritance 可处理"一顶点挂多高斯"的归属问题。

### 1.4 SplattingAvatar（2403.05087）
- **核心机制**：高斯以**重心坐标 + 沿法线位移**定义在三角形上（Phong surface），lifted optimization 边走边优化；300 FPS 桌面 / 30 FPS 移动端。
- **优劣**：证明"给每个高斯一个面局部标架"是消除"无面局部坐标系"的正解；但需逐高斯优化。我们受限 1:1 绑定且不能重训，只能"借标架、不借优化"。
- **重训**：是。**直接用**：★★☆。**迁移**：为每个高斯记录"出生面"法线，从而支持方向二里的 SH 旋转与曲面贴合约束。

### 1.5 PIG（2506.07657）
- **核心机制**：物理仿真 + 多材质交互；关键句"embedded constraint scales into deformation gradients, specifically clamping the scaling and rotation properties of the Gaussian primitives to eliminate [artifacts]"。
- **价值**：为逐轴 clamp[0.25,1.35] 提供**已发表的理论外衣**——把 clamp 解释成"deformation gradient 里的约束缩放"，而非经验 hack。
- **重训**：否。**直接用**：★★★（仅作论证引用）。**迁移**：写成约束投影 `proj_[a,b](S)`，并说明其保持 PSD（对角正缩放）。

## 方向二 · 旋转/协方差的流形聚合（RotSync 的理论归宿）

**问题定位**：RotSync 的 Markley 主特征向量聚合 + 45° 上限 + 置信门控，是"在 SO(3) 上做加权平均"的一个特例。文献里已有**完全对应**的发表先例，以及更严格的一般化理论。

### 2.1 On the Skinning of Gaussian Avatars（2509.11411，2025-09；Moverse：Zioulis, Kotarelas, Albanis, Thermos, Chatzitofis）
- **核心机制**：**用加权四元数平均取代 LBS 对旋转的处理**。`A_i^t = Σ_j w_i^j (b_j^t)^T b_j^t`（Eq.7，累积 4×4 矩阵），`q̄_i^t` = A_i^t 最大特征值对应的特征向量（Eq.8），`q_i^t = q̄_i^t ⊗ q_i`。SH 用 **Wigner D-矩阵**旋转：`s_{i,1:4} = W^1 s`，`4:9 = W^2 s`，`9:16 = W^3 s`。作者原话："vertex-based Gaussians … animated … by **only modifying the linear blend skinning technique, and using any Gaussian rasterizer**"；Unity3D 600 FPS；无数神经网络。
- **相对我们四法的优劣**：这就是 RotSync 的**已发表先例**——我们不是首创，但我们的 Markov 聚合是 Perron–Frobenius 意义下的最优旋转平均（Eq.8 即 Markley 解），因此可放心作为正式方法陈述，并引用它背书。它比我们多的是 **SH 旋转**（我们完全没做——见下）。它比我们少的是**置信门控与 45° 上限**（我们的贡献点）。
- **是否需要重训**：否。**纯推理期，只改 skinning 函数**。
- **能否直接用**：顶点绑定 ★★★★★（它就是顶点高斯）；薄片 ✗；推理期 ★★★★★。
- **可迁移技巧**：**(1) 立刻可用的缺口——SH 旋转**。RotSync 只改了协方差，SH 系数仍按 view direction 直接查询，这在高斯被旋转后会带来颜色错位（细看鼻翼/唇边）。用 W^1/W^2/W^3 旋转 band-1/2/3 是闭式、每帧每高斯 15 次乘加，30 ms 预算内完全可负担。**(2) 引 Eq.7/Eq.8 作为 RotSync 聚合步骤的文献依据。**

### 2.2 Poisson-Sampled Fréchet Means on Gaussian Information Manifolds（2508.17382）
- **核心机制**：严格的 Fréchet 均值理论，"specialized to covariance-varying Gaussian models: the full univariate Fisher–Rao manifold and the affine-invariant manifold of multivariate covariance matrices, including arbitrary **noncommuting, spatially heterogeneous** covariance laws"。
- **价值与局限**：它正是"一环内 J 非交换（noncommuting）时如何定义平均"的数学答案——而我们目前的 Markley 只处理旋转部分，面积加权处理平移部分，两者**不在同一个度量下**，所以 shape_err 从 LAM 的 0.830 退化到 0.857。Fréchet 均值给出统一的 Riemannian 中心。**纯理论，无图形实验**，落地需自己实现迭代 log/exp 映射（每顶点 3–5 次迭代），30 ms 预算下对 2 万顶点是可行的（每次迭代是 3×3 矩阵 log/exp）。
- **重训**：否。**直接用**：★★★☆（实现成本中等）。**迁移**：作为"方向三里 SPD 测地插值"的理论基座。

### 2.3 Depth Anything V4（2608.18388）中的 Riemannian Flow Matching
- **核心机制**：在 4DGS 参数（scale, rotation, opacity）上做 Riemannian Flow Matching，路径定义在**非欧流形**上，"ensuring all intermediate states are valid"；孤立 RFM 增益 F-score 0.762→0.806。
- **价值**：这是"**别在欧氏空间插值旋转/尺度**"这一主张在 4DGS 上的最新实证支撑。可用于我们报告的方法论引言（为什么 RotSync 优于 TGA），但训练期方法、推理期不可用。
- **重训**：是。**直接用**：★（仅引用）。

### 2.4 Quadratic Gaussian Splatting（2411.16392）与 HyperGaussians（2507.02803）
- **核心机制**：前者用二阶（quadratic）高斯提高表达能力；后者用高维潜变量把各向同性高斯提升为可表达各向异性形状的族。
- **价值**：与"轴长 clamp"直接相关——它们说明**表达力不足时应换参数化，而非硬 clamp**。可作为我们"clamp 是权宜、非终局"论述的旁证。
- **重训**：是。**直接用**：★（引用）。

## 方向三 · 各向异性 / 薄片抑制与退化协方差的数值稳定

**问题定位**：80.4% 的高斯是薄片（轴长比 p90=83），任何一点朝向误差都会被放大成"穿透皮肤"的尖刺。这一方向直接对应我们最痛的症状。

### 3.1 Grassmannian Splatting I（2607.10489）
- **核心机制**：把 4D 协方差写成 `Σ_4D = (P_n L)(P_n L)^T`，其中 `P_n = I − n n^T`，`n ∈ S³/{±1} ≅ Gr(3,4)`，`L ∈ R^{4×3}`；对时间做条件化 → **rank-2 surfel**。运动闭式给出，**无学习形变场**；"the conditioned disk feeds a standard 3DGS rasterizer through its **precomputed-covariance interface**"；Schur 分母里的 soft clamp 同时正则化静态朝向并在 rank-3↔rank-2 之间平滑过渡；17 个 HyperNeRF MonoDyGauBench 场景训练快 4.9–5.6×。
- **相对我们四法的优劣**：**这是对我们"薄片"问题最对症的一篇**。我们被迫在 rank-3 上做 clamp，而它主张**显式降到 rank-2**——数学上就承认了薄片是 surfel 而非体积椭球，于是"沿法线方向的厚度"不再是自由度，也就不会出现"戳穿"的细长尖刺。缺点：(a) 它走 precomputed covariance 接口，而我们的 gs_renderer 把 cov3D_precomp 硬编码为 None——**接口不兼容**；(b) 是训练期方法。
- **重训**：是（但表示层的思想可推理期借用）。
- **能否直接用**：顶点绑定 ★★★；薄片 **★★★★★（思路最直接）**；推理期 ✗（接口）。
- **可迁移技巧**：**把 rank-2 作为"硬保险"**——RotSync 输出 (q,s) 后，若该高斯轴长比 > 阈值（如 50），把最短轴长度钳到 0（或一个极小 ε），让它退化成面片。无需预计算协方差，只改 scale 回写，**与我们的推理管线 100% 兼容**。这是本报告里性价比最高的单项改造之一。

### 3.2 As-Rigid-As-Possible Deformation of Gaussian Radiance Fields（2608.29538）
- **核心机制**：ARAP 编辑后**再优化高斯**，"it is the radiance field that rasterizes and renders the final image. The inconsistency between the deformed 3D Gaussians and the desired radiance field inevitably leads to artifacts"；用 radial features 描述径向差异 + 自适应各向异性低通。
- **价值**：指出**几何上正确的形变 ≠ 渲染上正确的图像**——这正是我们"PSNR/SSIM/LPIPS 自指"困境的镜像。它的补救（形变后再优化）我们做不到（推理期、无监督信号），但它的**诊断语言**可用：我们的假旋转/轴长爆点造成的正是"deformed Gaussians 与期望辐射场不一致"。
- **重训**：否（交互式优化，非训练）。**直接用**：★★（概念）。**迁移**：作为我们"为什么不能只看旋转误差、必须看渲染误差"的论证。

### 3.3 SAGE — Self-Learning Expression Deformations（2606.05912）
- **核心机制**：联合优化 2D 高斯 surfel 与一个 SDF，"to enforce compact, **surface-aligned** Gaussian distributions"；自监督表情学习取代长序列；单图可用、无预训练先验。
- **价值**：给出"**曲面贴合**"作为薄片问题的根本解——不是限制轴长，而是让盘面法线永远贴合表面法线。与方向一里 SplattingAvatar 的 Phong surface 是同一思想的两条实现路径。我们可推理期近似：用 RotSync 得到的 R 的第 3 轴与顶点法线 n 做一次**符号对齐 + 微调**（把 R 的第 3 列替换为 n，再正交化），成本极低。
- **重训**：是。**直接用**：★★★（可抄"法线对齐"这一条）。**迁移**：见上。

### 3.4 Texture / Surfel 混合（MixedGaussianAvatar 2412.04955，SplattingAvatar 2403.05087）
- **核心机制**：2D 高斯（surfel）负责表面，3D 高斯负责体积细节（头发、口腔）。
- **价值**：直接对应我们"皮肤区该是面片、毛发/口腔区该是体积"的分区想定。推理期可按区域**分别设置 clamp 策略**（皮肤区强压最短轴、口腔区内脏区放宽），不需要改表示。
- **重训**：是。**直接用**：★★★（分区策略可抄）。**迁移**：用 FLAME 的语义分区（皮肤/唇内/眼内）给不同 clamp 阈值——比全局 [0.25,1.35] 更合理，且无需任何训练。

## 方向四 · 推理期免训练形变（本项目的合法性依据）

**问题定位**：我们整个任务被限制在"推理期、不重训"，必须证明这不是权宜，而是已有范式。

### 4.1 GSDeformer（2405.15491，IEEE TVCG；NCCA Bournemouth + NCKU 成大）
- **核心机制**：cage-based deformation（CBD）作用于 3DGS，桥梁是**proxy point cloud**：由高斯生成点云 → 点云被 cage 形变（mean value coordinates，`x = Σ_j ω_j v_j`）→ 形变后的点驱高斯的变换；为处理弯曲引入 **splitting process**（把一个高斯沿变形方向切成多个，以逼近非仿射弯曲）。自动 cage 构建走 render-reconstruct-simplify，带**两阶段化简**（边坍缩 + 顶点位置梯度优化，最小化 `L_MVC`，抑制负 MVC 权重）；用 depth map + T-SDF 先重建干净代理，**绕过 floaters**。原文："does not extend or modify the core architecture of 3DGS; thus it can work with **any existing trained vanilla 3DGS** as well as its variants"；"does not require retraining for editing, runs in real-time (60FPS)"。
- **相对我们四法的优劣**：**这是我们方法的最近亲范式**，且它明确地把"免训练 + 不改渲染器 + 实时"作为卖点。差异：它是**外部编辑**（用户拖 cage），我们是**由表情驱动**；它用点云代理，我们直接解析 J。
- **重训**：**否（这正是它的卖点）**。
- **能否直接用**：顶点绑定 ★★★★（点云代理与顶点绑定不冲突）；薄片 ★★★（splitting 可把超长薄片拆成多个短段——但这会增加高斯数，30 ms 预算下需权衡）；推理期 ★★★★★。
- **可迁移技巧**：**(1) splitting 用于极端表情**——对轴长比 > 100 且处于嘴/眼褶皱区的少数高斯（我们统计到的 0.2–0.4% 不可解区），与其 clamp 成一条怪异长条，不如沿主轴拆成 2–3 个短高斯（只改 scale 与位置回写，不改渲染器）。**(2) T-SDF 去 floater** 的思路可用于预处理：我们的薄片若来自 LAM 重建噪声，可在离线时先筛掉。

### 4.2 "On the Skinning of Gaussian Avatars"（2509.11411）— 见方向二
- 再次强调：这是**免训练、只改 skinning、任意光栅器可用**的已发表先例，直接把"推理期改造"从"取巧"变成"有据"。

### 4.3 GALA — One Basis to Animate Them All（2610.02207）
- **核心机制**：把 blendshape 蒸馏成**身份无关的 block-local PCA 基**；"applies to various animation architectures **without retraining original models**"。
- **价值**：为"在已训练模型外面套一层动画算子而不重训"提供了第二个独立证据，且是 2026 年的新工作，可用来回应审稿人"为什么不重训"的质疑。
- **重训**：否（对原模型）。**直接用**：★★（论证）。**迁移**：block-local 分解的思想可用于我们把 20018 个高斯按 FLAME 区域分块、逐块设不同形变策略。

### 4.4 G-Skin（2608.01726）
- **核心机制**：生成式 skinning；核心痛点陈述："3D Gaussian primitives are **not restricted to the surface** and **lack explicit topological connectivity**"——这正是我们顶点绑定的假象：我们以为有拓扑，实际只有 1:1 顶点索引，没有面邻接的连续场。
- **价值**：为我们"一环平均在 176° 处失效"提供文献层面的病因学解释（缺显式拓扑连通性 → 局部平均无定义）。
- **重训**：是。**直接用**：★（引用）。

## 方向五 · 法线协变 / 曲面贴合与法向一致性

**问题定位**：我们未实现的 (a) `n_d = J^{-T} n_c`（法线协变纠正）——RotSync 只给 `R·n₀`，在存在非均匀缩放时法线方向会偏，导致薄片既不贴面也不朝外。

### 5.1 SurFhead（2410.11682）— 法线处理的正面表述
- 见方向一。它的关键主张"precise geometric deformation … **including those affecting normals**"意味着 JBS 天然给出正确法线：`n_d ∝ J^{-T} n_c`（面积/余子式意义下的协变）。**这套公式在推理期是纯解析的**，我们完全可以在 RotSync 之后补一步：`n' = normalize(R_face^{-T} n₀)`，或用整 J 的余子式。
- **可迁移技巧（本方向最实用的落地）**：对每个顶点，用其所属面 J 的**逆转置**求变形后法线，再与 R 的第 3 列做一次**最小旋转对齐**（把 R 绕轴 xi 旋转到法线一致）。成本：每顶点一次 3×3 求逆（可预算缓存）+ 一次旋转向量构造。

### 5.2 SplattingAvatar（2403.05087）/ SAGE（2606.05912）
- 两者都把"法向贴合"当作表示的核心约束（Phong surface / SDF-aligned）。见方向一、三。

### 5.3 GeoDiff4D（2602.24161）
- **核心机制**：几何感知扩散，同时合成肖像与**表面法线**，pose-free 表情编码器。
- **价值**：说明"法线作为显式监督目标"在 2026 年已成为头像生成的标配假设——反向印证我们"只驱动顶点不驱动法线"是表示层的缺失。
- **重训**：是。**直接用**：★（引用）。

### 5.4 PhysHead（2604.06467）
- **核心机制**：FLAME 网格 + 发丝 strand 的混合表示，高斯**同时挂在网格与发丝段上**；用 VLM 生成动态序列中被遮挡区域的 appearance。
- **价值**：对我们有两点：(a) 它的"双挂载"说明我们的 1:1 顶点绑定并非不可扩展；(b) VLM 补全遮挡区的思路，可用于我们"无同人真值"的评估困境——虽然不能替代 GT，但可生成**语义一致的参考**用于人工检查。
- **重训**：是。**直接用**：★★（评估侧启发）。

## 方向六 · 面向高斯原语的评价体系（破解"无同人真值"困境）

**问题定位**：PSNR/SSIM/LPIPS 只能自指（把 LAM 自身当 GT），因此**无法判断哪种形变更"对"**。我们需要的是对几何/协方差本身有定义的度量。

### 6.1 GaussianBench（2610.10554）
- **核心机制**：物理保真度评测套件，明确包含 "**Gaussian covariance transport and rendering**" 项；输出 PASS/FAIL/NA/INVALID 四态。
- **价值**：**这是最直接可用的一篇**——它把"协方差是否正确传输"作为被评测对象，恰好是我们 TGA/RotSync 的核心观测量。可照搬其四态判定，把"轴长比是否越界""是否出现假旋转""协方差是否保持 PSD"变成可自动判定的 PASS/FAIL。
- **重训**：否（评测）。**直接用**：★★★★（评测协议可直接借）。

### 6.2 2606.01031 — 时间对齐评测（Soft-DTW）
- **核心机制**：用 Soft-DTW 做时间对齐的评测，指出帧级指标"is a special case under rigid alignment"。
- **价值**：我们的"时间粗糙度 0.73–0.94"是帧间抖动指标，但直接逐帧比较会惩罚合理的相位差。Soft-DTW 给出**对齐后再比**的规范做法。
- **重训**：否。**直接用**：★★★★（把 temporal roughness 升级为 Soft-DTW 距离）。

### 6.3 THEval（2511.04520，CVPR 2026 Findings）
- **核心机制**：8 项指标覆盖 quality / naturalness / sync；85,000 视频、17 个模型的基准。
- **价值**：提供"自指指标之外"的**感知合理性**维度（naturalness）；虽为视频级而非高斯级，但可用于验收"表情是否自然"。
- **重训**：否。**直接用**：★★★。

### 6.4 Drop-In Perceptual Optimization（2603.23297，ECCV'26，Apple）
- **核心机制**：39,320 对主观评分；正则化 Wasserstein Distortion（WD-R）胜出，比 3DGS loss 被偏好 >2.3×，比 Perceptual-GS 高 1.5×。
- **价值**：若引入**分布级**（而非逐像素）指标，WD-R 是当前最强候选——它衡量"高斯分布本身"的差异，天然适合比较 TGA vs RotSync 的协方差场。
- **重训**：否。**直接用**：★★★★（评测指标）。

### 6.5 No-reference PCQA（2610.13219）
- **核心机制**：复杂度校准的重建差异——对失真点云再重建一次（更强平滑）作为复杂度代理，校准后融合几何差异与投影视图差异，配合冻结的感知特征。
- **价值**：**无需参考**即可评估点云质量，正好对应"无同人 GT"的处境。虽针对点云，其"再重建"思想可移植到高斯：把形变后的高斯再"重建成更平滑版本"，差异即失真度。
- **重训**：否。**直接用**：★★★。

### 6.6 其他相关
- SCANTalk（2410.11041）：热扩散拓扑鲁棒特征，批判现有 lip-sync 指标 —— 可作"现有指标不足"的引文。
- Talk Like You（2610.06658）：PLAD（articulation axes 上的指标）。
- Beyond Monoscopic Viewing（2609.38525）：VR 中的 3DGS 主观质量研究 —— 支撑"要在头显里看"的验收主张。
- FMReward（2608.15296，TVCG）：用人类偏好对齐/评测音频驱动 3D 面部动画。

## 方向七 · 头像专用的形变正则与区域自适应（可推理期借用的"先验"）

**问题定位**：全局统一的 clamp[0.25,1.35] 与全局统一的 45° 上限，忽略了面部的**区域异质性**（嘴内褶皱 vs 额头 vs 颈部）。

### 7.1 GeoAvatar（2507.18155）
- **核心机制**：Adaptive Pre-allocation Stage（APS）**无监督**地把高斯分成 rigid / flexible 两组，配自适应 offset 正则；基于口部解剖与动力学设计嘴部结构与 part-wise 形变策略；另有高斯-3DMM 面的精确 rigging 正则。附带 DynamicFace 数据集。
- **相对我们四法的优劣**：我们四法都是**全局同构**的形变策略；GeoAvatar 的核心洞察是"不同面部区域的几何偏差量级不同，应分配不同的形变自由度"。这与我们实测的"不可解区集中在嘴/眼内褶 + 颈部裁剪边界"高度吻合。**APS 是无监督的**，其"按需分配"思想可在推理期用几何量近似（如按顶点到唇线/眼裂的距离、按一环面法线散度）。
- **重训**：原论文是。**但 APS 的分组思想可推理期复现**。
- **能否直接用**：顶点绑定 ★★★★；薄片 ★★★★（分区 clamp）；推理期 ★★★★。
- **可迁移技巧**：**按 FLAME 语义分区设置三套参数**——皮肤区（强 clamp + 强法线对齐）、唇/眼内区（放宽上限，允许更大旋转，因为那里本来就剧烈折叠）、颈部裁剪边界区（用置信门控直接放弃，回退到恒等变换）。这不增加任何训练成本，只是把全局常量换成查表。

### 7.2 SpatialAvatar-0（2606.15659）
- **核心机制**：共享 FLAME-mesh-bound 高斯表示；前馈生成器 + 10K 迭代的 **layout-preserving 逐主体精修**，"freezes the FLAME-binding and Gaussian count and replaces densification with a **three-component anti-spike regularization**"。
- **价值**：**"anti-spike regularization" 与我们"薄片戳穿"是同一个问题的同一个词**。它用三项正则替代 densification，说明"抑制尖刺"在 2026 年已被当作独立的一等公民正则项。可作为我们"分区 clamp/法线对齐"的正名。
- **重训**：是。**直接用**：★★★（引用 + 正则设计借鉴）。

### 7.3 TexAvatars（2512.21099）的隐表情通道
- 见方向一。η∈R^128 直接驱动 opacity+color，产生皱纹/褶皱 —— 说明**几何形变之外还需要外观通道**才能让表情"活"。我们目前只动 (q,s)，颜色完全不动，这是可解释的质量上限。
- **迁移**：推理期若不重训，无法加 η，但可用**简单的光照-几何耦合近似**（如按面法线变化调制 SH band-0 的亮度）获得部分效果。这是"廉价的假皱纹"，需谨慎标注为近似。

### 7.4 FG-3DGS（2602.09736，原文标题 *Toward Fine-Grained Facial Control in 3D Talking Head Generation*）
- **核心机制**：频率解耦 —— 低频区（颊、鼻、额）一个 MLP，高频区（眼、嘴）**单独加掩码网络**；目标是降低 lip-sync 误差与面部抖动。
- **价值**：与我们"时间粗糙度 worst 0.73–0.94 集中在局部"的观察一致：抖动是**局部高频现象**，不该用全局平滑。可推理期实现为"按区域选择不同的时间平滑系数"（对高频区做更强的帧间 slerp 阻尼）。
- **重训**：是。**直接用**：★★★（分区思想）。

### 7.5 3DV 2026 "Real-Time Facial Animation of Gaussian Head Avatars via Mocap-to-Parametric Expression Mapping"（IEEE Xplore doc 11533249）
- **核心机制**："efficient regression model that maps mocap-derived blendshape parameters to the expression space of parametric face models, enabling **direct control over Gaussian avatars without iterative optimization**"；配"lightweight expression regularization mechanism … encouraging semantically disentangled, identity-specific deformations"；实现上**以 ARKit 为 mocap 源、FLAME 为目标**；优于 fitting-based 与 regression-based 基线，支持直播与离线重演。
- **价值**：**这是与我们设定（ARKit → FLAME → 高斯头像，无迭代优化）最接近的已发表系统**，是极有力的对标引用。
- **重训**：是（它训练了回归器）。**直接用**：★★★（论证 + 基线对比）。

### 7.6 FA-LAM（2607.20922）
- **核心机制**：LAM 的扩展——对称/语义注意力正则、双阶段训练、流式 4D。
- **价值**：**"别人怎么改 LAM"的参照系**。它的改动全在训练侧，反证我们的推理期路线是另一条独立赛道（不冲突、可叠加）。
- **重训**：是。**直接用**：★★（定位论述）。

### 7.7 PDB（2610.08672）
- **核心机制**：Point-Based Deformation Blending，mesh-agnostic 重定向：ReLU 保证非负权重且允许精确 0，行归一化；目标网格 = 权重 × 控制点；**权重每目标只算一次、跨帧复用**；无 cage、无可学习逐元素解码器、无全局重建求解。
- **价值**：**"权重只算一次、跨帧复用"正是我们 30 ms 预算下的关键工程手段**。我们目前的面积权重每帧重算（随表情变化），若改为"拓扑权重缓存 + 每帧只更新控制点"，可省下可观开销。
- **重训**：是（训练预测控制点），但**权重复用机制与我们的预计算缓存同构**。**直接用**：★★★★（工程优化思路）。

---

## 汇总表（按方向）

| # | 方向 | 代表工作 | 年份/出处 | 重训? | 顶点绑定 | 薄片 | 推理期 30ms |
|---|---|---|---|---|---|---|---|
| 1 | 仿射/Jacobian 绑定 | SurFhead 2410.11682 | ICLR 2025 | 机制可免训 | ★★★★ | ✗ | ✓ |
| 1 | 同上（连续场） | TexAvatars 2512.21099 | 3DV 2026 | 机制可免训 | ★★★★ | ✗ | ✓ |
| 1 | 同上（先驱） | GaussianAvatars 2312.02069 | 3DV 2024 | 是 | ★★★ | ✗ | ✓ |
| 1 | 面局部标架 | SplattingAvatar 2403.05087 | 2024 | 是 | ★★☆ | ★★★ | 300FPS/30FPS 移动 |
| 1 | clamp 的理论外衣 | PIG 2506.07657 | 2025 | 否 | — | ★★ | ✓ |
| 2 | 四元数平均 = RotSync 先例 | On the Skinning of Gaussian Avatars 2509.11411 | 2025 | **否** | ★★★★★ | ✗ | 600FPS |
| 2 | Fréchet 均值理论 | 2508.17382 | 2025 | 否 | ★★★☆ | ✗ | ✓（迭代） |
| 2 | 流形上的 RM | 2608.18388 | 2026 | 是 | ★ | ✗ | ✗ |
| 3 | rank-2 surfel / Grassmann | Grassmannian Splatting I 2607.10489 | 2026 | 是 | ★★★ | **★★★★★** | ✗（接口） |
| 3 | 形变后重优化 | ARAP of Gaussian Radiance Fields 2608.29538 | 2026 | 否 | ★★ | ★★ | ✗ |
| 3 | 曲面贴合 | SAGE 2606.05912 | 2026 | 是 | ★★★ | ★★★★ | — |
| 4 | 免训练 cage 形变 | GSDeformer 2405.15491 | TVCG | **否** | ★★★★ | ★★★ | 60FPS |
| 4 | 免训重渲 | GALA 2610.02207 | 2026 | 否（对原模型） | ★★ | — | — |
| 5 | 法线协变 | SurFhead / SplattingAvatar / SAGE | 见上 | — | — | — | — |
| 6 | 协方差传输评测 | GaussianBench 2610.10554 | 2026 | 否 | ★★★★ | ★★★★ | ✓ |
| 6 | 时间对齐评测 | 2606.01031 (Soft-DTW) | 2026 | 否 | ★★★★ | — | ✓ |
| 6 | 分布级感知指标 | WD-R 2603.23297 | ECCV'26 | 否 | ★★★★ | ★★★★ | ✓ |
| 6 | 无参考点云质量 | 2610.13219 | 2026 | 否 | ★★★ | ★★★ | ✓ |
| 7 | 区域自适应 | GeoAvatar 2507.18155 | 2025 | 思想可免训 | ★★★★ | ★★★★ | ✓ |
| 7 | 抗尖刺正则 | SpatialAvatar-0 2606.15659 | 2026 | 是 | ★★★★ | ★★★★ | — |
| 7 | 频率解耦 | FG-3DGS 2602.09736（*Toward Fine-Grained Facial Control…*） | 2026 | 是 | ★★★ | ★★★ | — |
| 7 | ARKit→FLAME→GS 对标 | 3DV 2026 (doc 11533249) | 3DV 2026 | 是 | ★★★ | — | 实时 |

---

## TOP 5 · 按"可迁移价值"排序

> 排序准则：**①推理期可落地（不改 LAM、不改渲染器、30 ms 内）②直接命中我们的两个痛点（薄片戳穿 / 假旋转）③改造成本低**。纯理论或必须重训的均被压低。

### 🥇 第 1 名 · Grassmannian Splatting I（2607.10489）的 **rank-2 退化保护**
- **为什么是第一名**：我们 80.4% 的高斯是薄片（p90 轴长比 83），"戳穿皮肤"的物理本质就是"把一个本该是面片的东西当成体积椭球去渲染"。这篇论文的主张——**显式降到 rank-2，让厚度不再是自由度**——是对症下药，而不是在 rank-3 上打补丁。
- **落地方式（改 3 行代码级别）**：RotSync 输出 (q, s) 后，对轴长比 > 阈值的少数高斯，把最短轴 s_min 钳到 ~0（如 1e-3），使其平面化。**只改 scale 回写，完全绕开 cov3D_precomp 的限制**。
- **预期收益**：直接消掉最刺眼的长条尖刺；薄片区视觉质量提升显著。
- **风险**：过度平面化会在斜视角出现"纸片翻面"。需用它的 soft clamp 思想（连续过渡而非硬阈值）。
- **成本**：O(n)，可忽略。

### 🥈 第 2 名 · On the Skinning of Gaussian Avatars（2509.11411）的 **SH 旋转（Wigner D-矩阵）**
- **为什么是第二名**：这是**我们目前完全没有做、且是闭式解、且每帧成本极低**的一项。RotSync 只旋转了协方差，SH 系数（颜色）仍按原方向查询，高斯被旋转后会出现颜色错位——这在薄片上是可见的。
- **落地方式**：`s_{1:4} = W^1(R)s`、`s_{4:9} = W^2(R)s`、`s_{9:16} = W^3(R)s`，R 即 RotSync 已算出的旋转。W^1/W^2/W^3 可预计算成常数张量或按 R 实时算。
- **预期收益**：在高光/镜面区域（鼻尖、唇）与斜视角下颜色一致性提升；顺带把 RotSync 的"方法完整性"补齐（协方差 + 颜色一起旋转，才是严格的刚体变换）。
- **风险**：若 LAM 的 SH 已在 canonical 空间被训练成 view-dependent，旋转后可能反而变差 —— **必须做 A/B 实测**。
- **成本**：每高斯 1+4+9 = 14 次基函数求值，2 万高斯 ≈ 可在 1–2 ms 内完成。

### 🥉 第 3 名 · GSDeformer（2405.15491）的 **splitting 策略处理不可解区**
- **为什么是第三名**：我们已确认有 0.2–0.4%（46–76 个）顶点在一环内面朝向差达 176°，**任何平均都无定义**。与其继续调门控阈值，不如承认"一个高斯表达不了这块折叠"，**把它拆成几个**。
- **落地方式**：对高置信度低的顶点，沿主轴把该高斯拆成 2–3 个较短高斯（共享 (q, s/√k) 与位置偏移），再分别做形变。GSDeformer 原文即为此设计，且明确"不改渲染器架构"。
- **预期收益**：嘴内/眼内褶皱区的伪影从"结构性错误"降级为"轻微模糊"。
- **风险**：高斯数增加 → 直接吃 30 ms 预算（当前 12 ms，余量约 18 ms）。若只对 76 个顶点拆分，增量 < 1%，完全可接受。
- **成本**：仅对极少数顶点，O(1)。

### 4️⃣ 第 4 名 · SurFhead（2410.11682）的 **在 (R,S) 域混合 + 法线协变**
- **为什么是第四名**：它同时解决我们的两个问题——(a) 把"面积加权平均 Σ"换成"面积加权平均 (R,S)"，从数学上消除 TGA 的 3–5% 假旋转；(b) `n_d ∝ J^{-T}n_c` 给出正确的变形后法线，而我们未实现的 (a) 项正是这个。
- **落地方式**：J 拆成 (R, S) → 面积加权分别聚合（R 用四元数，S 逐轴）→ 重组 → 再对 R 的第三列做法线对齐。
- **预期收益**：假旋转降到近 0；薄片朝向更贴面。
- **风险**：丢失剪切 → shape_err 可能进一步偏离（RotSync 已从 0.830 退化到 0.857）。**建议与我们未实现的 (c) SPD 测地混合联合使用**，用 α 在"保旋转/保形状"之间连续调节。
- **成本**：每顶点一次 3×3 极分解 + 四元数平均 ≈ RotSync 同量级。

### 5️⃣ 第 5 名 · GaussianBench（2610.10554）+ Soft-DTW（2606.01031）的 **评测协议改造**
- **为什么是第五名**：它不改善画面，但**决定我们能不能证明改善**。当前 PSNR/SSIM/LPIPS 自指，等于没有裁判；而"协方差是否正确传输"恰好是被 GaussianBench 正式列为评测项的东西。
- **落地方式**：把现有指标升级为三层——①**几何层**：PSD 保持率、轴长比越界率、假旋转率（PASS/FAIL 四态）；②**时间层**：Soft-DTW 对齐后的轨迹距离替代帧间粗糙度；③**感知层**：WD-R 分布距离（若可拿到实现）。
- **预期收益**：让 TGA / RotSync / VTP 的差异从"看起来差不多"变成可判定的数字。
- **成本**：纯离线。

---

## 诚实声明：未找到的与不可用的

1. **方向 (c) SPD 测地混合 `Σ(α) = exp(α log Σ_TGA + (1−α) log Σ_ROT)` 在 3DGS / 头像领域没有任何直接先例。** 我做了多轮 arXiv 检索（"SPD"、"log-Euclidean"、"Karcher mean"、"geodesic covariance interpolation" 与 "gaussian splatting" 的组合），只找到纯理论工作（2508.17382 Poisson-Sampled Fréchet Means）与统计学工具库（spd_learn）。**这意味着 (c) 是一个真正的空白点**——既是风险（无人验证），也是机会（可作为方法贡献）。
2. **方向 (b) 逐轴部分拉伸 `Σ' = R diag(s1, s2, s3·k_n) Rᵀ`（k_n = tanh(g·u)）未见直接对应**；最接近的是 PIG 2506.07657 的"clamp scaling/rotation"，但它是在形变梯度里 clamp，而非沿法线做单轴缩放。**这也是一个空白点**。
3. **若干论文只能确认题录、正文不可得**：LiveAvatar（ACM IMX 2026, DOI 10.1145/3788851.3815056）、Matrix-based representation of mesh surface deformation for mesh-driven 3DGS（Neurocomputing, DOI 10.1016/j.neucom.2026.135141）、DG-4DGS（Virtual Reality & Intelligent Hardware, S2096579626000227）。这些**未纳入 TOP5**，也不作为论据使用。
4. **命名冲突必须处理**：内部代号 "TGA" 与 NeurIPS 2025 Spotlight 的 *TGA: True-to-Geometry Avatar Dynamic Reconstruction* 撞名。正式写作前必须改名（建议 "JBS-Area" 或 "ACD: Analytic Covariance Deformation"）。

---

## 附：核心文献清单（全部经 arXiv / Crossref / 官方页面核实）

**绑定与形变**：GaussianAvatars 2312.02069 · SplattingAvatar 2403.05087 · Rig3DGS 2402.03723 · SurFhead 2410.11682 (ICLR'25) · TexAvatars 2512.21099 (3DV'26) · GSDeformer 2405.15491 (TVCG) · PIG 2506.07657 · ARAP of Gaussian Radiance Fields 2608.29538 · G-Skin 2608.01726 · PDB 2610.08672
**旋转与流形**：On the Skinning of Gaussian Avatars 2509.11411 · Fréchet Means on Gaussian Information Manifolds 2508.17382 · Depth Anything V4 (RFM) 2608.18388 · Grassmannian Splatting I 2607.10489 · Quadratic GS 2411.16392 · HyperGaussians 2507.02803
**头像系统**：LAM 2502.17796 (SIGGRAPH'25) · GeoAvatar 2507.18155 · FlashAvatar 2312.02214 · MixedGaussianAvatar 2412.04955 · SpatialAvatar-0 2606.15659 · SAGE 2606.05912 · HeadsUp 2605.04035 · EGGHead 2610.26898 · PhysHead 2604.06467 · GPiCA 2512.15711 · FA-LAM 2607.20922 · GALA 2610.02207 · OmniFaceRig 2606.08043 · FG-3DGS 2602.09736 · FaceGAN 2610.11070 · Mocap-to-Parametric Mapping (3DV'26, doc 11533249)
**评测**：GaussianBench 2610.10554 · Soft-DTW aligned eval 2606.01031 · THEval 2511.04520 · WD-R 2603.23297 · No-ref PCQA 2610.13219 · SCANTalk 2410.11041 · Talk Like You 2610.06658 · FMReward 2608.15296
**命名冲突参照**：TGA: True-to-Geometry Avatar Dynamic Reconstruction（Bo Guo, Sijia Wen, Ziwei Wang, Yifan Zhao，北航；NeurIPS 2025 Spotlight；项目页 http://superkeaibb.github.io/TGA_web ）——非 arXiv 编号，故此处不列 id。

> 说明：上表所有 arXiv id 与结论均来自本次会话内实际抓取的 abs/html 页面、Semantic Scholar 引用图、CrossRef 或作者项目页。凡未能取得正文的，均已在"诚实声明"中标注，未做任何推断性填充。
