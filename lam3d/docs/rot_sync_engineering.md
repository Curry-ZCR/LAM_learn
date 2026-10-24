# 高斯旋转同步（rot_sync）工程对照：门控 / 四元数平均 / 时域 / 病态 / 性能
> 对象 /workspace/lam3d/lam3d/core/rot_sync.py；基线 LAM = arXiv:2502.17796（https://arxiv.org/abs/2502.17796 ）。方法一：保持三轴长不变，只按局部形变梯度 J 的极分解旋转因子更新朝向（Sigma' = R Sigma0 R^T）。

> 标记：【原文】= 论文正文 / 官方代码可逐字核验；【实测】= 本文在 RTX 4090 上跑出，脚本见 /tmp/rs/；【推断】= 无公开来源的工程判断。

## 0. 结论速览

1. **没有找到任何系统在推理期按「高斯 / 面片尺寸比」衰减旋转。** GaussianAvatars 的 eps_scaling=0.6 是训练期正则（把高斯压小），SplattingAvatar 的 T_r=0.008 也是训练期正则且方向相反（只罚过大）。直接乘在旋转/四元数上的公开门控：**未找到**（§1）。
2. **半球对齐是四元数平均的正确性前提，不是可选项。** 逐元素面积加权平均（SplattingAvatar Eq.5 的原始写法）在有 q / -q 符号不一致时直接崩：仅 5° 散布就有 14% 顶点偏差超 30°；对齐后 0%（§2，【实测】）。
3. **同半球内，「(c) 半球对齐 + 加权平均」与「(b) Markley 主特征向量」方向完全一致**：半跨度 1°–89° 扫描实测方向差恒为 0.000°。Markley 唯一值得算的理由是它顺带给出不确定度（最大特征值缺口）。
4. **时域滤波应作用在最终四元数上，而不是四元数增量上。** 240 帧 / 60fps 合成实验（alpha=0.2）：final-q EMA 误差 1.98° vs 增量 EMA 6.11°；增量域无低频约束，会把每帧独立噪声重新注入并沿链式乘法累积漂移（§3）。
5. **真实链路里 J 并不病态**：真实 blendshape 下 cond(J) 的 p99 < 10，**det(J)<0 实测占比 0.00%**。抖动主因是局部帧的选择与 blendshape 高频成分，不是条件数（§4）。
6. **性能完全不是瓶颈**：N=20018、f32、RTX 4090 上 Newton 10 步 = **1.383 ms**，SVD f32 = 4.451 ms，SVD f64 = 24.046 ms，帧间矩阵相乘 0.175 ms；相对 30 ms 契约有 20 倍以上余量（§5）。
7. **仓库当前实现有一处反向**：极分解前的 Frobenius 预归一化使收敛更慢，应去掉（§5.3）。

## 1. 尺度门控：别人有没有「按高斯相对面片大小衰减形变」

### 1.1 GaussianAvatars（arXiv:2312.02069 §3.4）【原文】—— https://arxiv.org/abs/2312.02069
动机原文：「if a 3D Gaussian is large in comparison to its parent triangle, small rotations of the triangle – barely noticeable at the scale of the triangle – will be magnified by the scale of the 3D Gaussian, resulting in unpleasant jittering artifacts.」正则 Eq.(9)：L_scaling = || max(s, eps_scaling) ||_2，eps_scaling = 0.6，原文说它是「a threshold that disables this loss term when the scale of a Gaussian less than 0.6x the scale of its parent triangle」。
**这是训练期正则，不是推理期门控。** 它没有「把旋转乘一个衰减系数」的机制，而是从结构上保证高斯不会长得比父三角面大。消融（§4）原文：去掉小误差容限后「all metrics drastically deteriorate ... all the splats are regularized to be infinitely small」——阈值本身是必需的。

### 1.2 SplattingAvatar（arXiv:2403.05087 §3 末）【原文】—— https://arxiv.org/abs/2403.05087
L_scaling(i) = |s_hat_i| 当 s_hat_i > max(T_s, T_r * s_check_i)，否则 0；lambda_s=1.0、T_s=10.0、T_r=0.008。原文意图「prevent Gaussians from growing long and thin」，判据是「既长（>T_s）又薄（>T_r 倍父面尺寸）」。与 GaussianAvatars **方向相反**：只罚过分大的高斯，T_r=0.008 极宽松。仍作用在**缩放**上，不是旋转。

### 1.3 GaussianSpeech（arXiv:2411.18675 Eq.31）【原文】—— https://arxiv.org/abs/2411.18675
L_scaling = || max(s, eps_scaling) ||_2，与 GaussianAvatars 同形（照抄），无独立门控。

### 1.4 SurFhead（arXiv:2410.11682 §2.1）【原文】—— 反面论据，对我们最有用 —— https://arxiv.org/abs/2410.11682
原文批评上述做法：把高斯按父三角面的「相对旋转矩阵 / 面积比（各向同性标量）/ 重心位置」变换，「falls short in capturing the stretch and shear deformations」；并指出 GaussianAvatars 用「一边长与其垂线均值」、SplattingAvatar 用「形变前后面积比」，当「the deformed and canonical triangle areas ... are the same but the shape is different, the Gaussians become misaligned」。
→ 从**反面**支持我们「不改轴长、只转朝向」的设计：任何用单个标量吸收形变的方案，在纯剪切下都会失配。

### 1.5 FlashAvatar / Rig3DGS / 3DGS-Avatar / 3DGS 编辑器

- FlashAvatar（arXiv:2312.02214，https://arxiv.org/abs/2312.02214 ）：检索 scal(e|ing) 与 triangle / patch / parent / larger than / eps_scal 的全部组合，只命中 perceptual loss 与 VGG/LPIPS，**无 scale-vs-triangle 正则 → 未找到**。
- Rig3DGS（arXiv:2402.03723，https://arxiv.org/abs/2402.03723 ）§3.2：正则是 Eq.5（对齐 FLAME 顶点形变）/ Eq.6（远离网格的点形变置零）/ Eq.7（抑制修正量）；§3.3「Rotating and Scaling Gaussians」内无 scale-ratio 正则 → **未找到**。
- 3DGS-Avatar（arXiv:2312.09228，https://arxiv.org/abs/2312.09228 ）：用 skin MLP 直接回归四元数（Eq.6–8），**没有几何法旋转，也就没有门控** → 不适用。
- 3DGS 编辑器中的「gaussian to patch size ratio」类做法：**未找到**任何可引用的实现或文档。

### 1.6 对我们的结论【推断】
我们的轴长是**固定**的（apply() 里 s' = s0 逐位相等），所以 GaussianAvatars 担心的「大高斯放大三角面旋转」这个**增益恒为 1**。真正的抖动来源变成**旋转角本身的误差**（≈ J 的噪声 / 三角面尺寸），因此正确对策是 §4 的逐面置信度 + 角度限幅（仓库已有 max_angle_deg=45），**不是** scale gating。若将来放开轴长，才需要补「按 s / sqrt(2A) 衰减 delta_q」的门控。

## 2. 四元数平均：(a) 弦距线性 (b) Markley (c) 先半球对齐
设 F 个邻面旋转 q_k、非负权重 w_k（取面积）。

- **(a) 逐元素线性加权再归一化**（chordal L2 mean）：q = normalize(Sum w_k q_k)。SplattingAvatar Eq.5 原文即此式：q_V = (Sum_{k in Omega_V} A_k q_k) / (Sum_{k in Omega_V} A_k)，**正文没有半球对齐步骤**【原文】。几何意义：在 R^4 里最小化 Sum w_k ||q - q_k||^2 后投影回单位球。
- **(b) Markley 4x4 主特征向量**：M = Sum w_k q_k q_k^T，取 M 最大特征值对应的特征向量（Markley / Cheng / Crassidis / Oshman, Averaging Quaternions, JGCD 30(4):1193–1197, 2007, DOI [10.2514/1.28949](https://doi.org/10.2514/1.28949)；PDF https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf ）。几何意义：最大化 Sum w_k (q·q_k)^2，等价于最小化 Sum w_k ||R(q) - R(q_k)||_F^2。
- **(c) 先半球对齐再做 (a)**：取参考 q_ref，q_k' = sign(q_k·q_ref) * q_k，再归一化加权和。仓库 rotation_field() 已实现（sign = where((qf3*pref).sum < 0, -1, 1)）。

### 2.2 失效条件【实测 /tmp/rs/num3.py，F=6 邻面，轴随机、角 ~ N(0,sigma)，权重 uniform(0.5,1.5)】
符号一致（各库函数都做了四元数规范化）时，平均四元数与真值 q0 的测地角（度）：

| sigma | (a) 无对齐 | (c) 半球对齐 | (b) Markley |
|---|---|---|---|
| 5° | 1.71 | 1.71 | 1.71 |
| 20° | 7.32 | 7.32 | 7.24 |
| 45° | 15.50 | 15.50 | 14.84 |
| 90° | 34.57 | 33.28 | 31.26 |
| 150° | 64.34 | 45.09 | 47.45 |
**sigma <= 45° 时三者相差不到 1°。** 但若每个面的四元数**独立随机翻符号（p=0.5，模拟不同库函数分别给出 q 或 -q）**：

| sigma | (a) 无对齐（括号内是偏差 >30° 的顶点占比） | (c) 半球对齐 | (b) Markley |
|---|---|---|---|
| 5° | 5.98（**14%**） | 1.81（0%） | 1.81（0%） |
| 20° | 22.48（**39%**） | 7.42（0%） | 7.37（0%） |
| 45° | 49.54（**74%**） | 16.24（6%） | 15.40（5%） |
| 90° | 85.11（**92%**） | 34.18（60%） | 31.77（53%） |
→ **(a) 的失效条件就是符号不一致，与角度散布无关**：哪怕散布只有 5°，只要有一个面翻符号，(a) 就塌。这不是数值精度问题，而是 q + (-q) = 0 的代数事实。**(b) 天生免疫**（q q^T 对 q <-> -q 不变）；**(c) 必须显式对齐才能达到 (b) 的鲁棒性**。

### 2.3 (c) 与 (b) 什么时候等价【实测 /tmp/rs/num3.py B2/B3】

- 对称扇形邻域、uniform 权重、半球半跨度从 1° 扫到 89°：**(c) 与 (b) 的方向差恒为 0.000°**。F=3 等权时 chordal mean 与 Markley 方向**恒相同**，只差模长（归一化后一致）。
- 分叉只出现在「权重不对称 + 大散布」：sigma=120° 时 (c) 39.56° vs (b) 40.38°。
数学解释【推断 + 文献】：只要所有 q_k 落在同一个开半球内（等价于存在 q 使全部 q·q_k > 0），M 的主特征向量就等于 normalize(Sum w_k q_k)，且主特征值严格占优；散布超过 90° 后 M 趋近 (Sum w_k / 4) I，主特征值不再占优，两解分叉且都失去意义。这与 Moakher 关于 chordal 与 geodesic 均值在直径 < 90° 时接近的结论一致（Moakher, SIAM J. Matrix Anal. Appl. 24(1):1–16, DOI [10.1137/S0895479801383877](https://doi.org/10.1137/S0895479801383877)）；系统理论见 Hartley et al., Rotation Averaging, IJCV 103(3):267–305, DOI [10.1007/s11263-012-0601-0](https://doi.org/10.1007/s11263-012-0601-0)。
**建议【推断】**：生产代码只用 **(c)**——area-weighted scatter_add + 归一化，O(F)、无特征分解。Markley 只在需要诊断量时才算：lambda_max - lambda_2nd 就是该顶点的「旋转不确定度」，可直接作为 §4 置信度的一项。仓库 average='face' 走的就是 (c)，方向正确。

## 3. 时域稳定性：滤波放增量上还是最终四元数上

### 3.1 直接对比【实测 /tmp/rs/diag12.py，T=240 帧 @60fps，观测加每轴 3° 噪声 + 1% 帧随机翻符号；jitter = 相邻帧角速度的均值绝对差，raw = 2.887°】

| alpha | final-q EMA：误差均值 / p95 / jitter | 增量 EMA：误差均值 / p95 / jitter |
|---|---|---|
| 0.2 | **1.981** / 3.256 / **0.464** | 6.111 / 9.112 / 0.577 |
| 0.4 | 2.550 / 4.337 / 0.917 | 5.115 / 7.716 / 1.154 |
| 0.7 | 3.581 / 6.134 / 1.780 | 4.275 / 7.362 / 2.021 |
→ **在最终四元数上做 EMA，误差小 2–3 倍、jitter 也更低。** 原因：增量 d = q_t * conj(q_{t-1}) 里**不含低频姿态信息**，对它 EMA 只是把每帧独立噪声重新注入，而误差沿链式乘法**随机游走累积**；final-q EMA 则是对绝对姿态的有界低通，不会漂移。

### 3.2 q / -q 翻转
修号（q·q_prev < 0 时取 -q）后 err 2.23° / max 7.3° / jitter 0.684；不修 err 2.33° / max 8.7° / jitter 0.745。1% 翻转率下均值差异不大（受影响帧被平滑吸收），但**不修号时单帧会出现接近 180° 的瞬时跳变**——q 与 -q 线性混合时模长趋 0，归一化方向由噪声决定，60fps 流里就是一次闪帧。**必须修号。** 增量域滤波的附带好处是可强制 d.w >= 0 从而天然规避，代价是 3.1 的误差与漂移。

### 3.3 实际系统的做法

- **MediaPipe**：通用滤波层提供 one-euro 实现，源码已核验 https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/util/filtering/one_euro_filter.h （接口 OneEuroFilter::Create(frequency, min_cutoff, beta, derivate_cutoff)）。原始文献 Casiez / Roussel / Vogel, 1€ Filter, CHI 2012, DOI [10.1145/2207676.2208639](https://doi.org/10.1145/2207676.2208639)，参考实现 https://github.com/casiez/OneEuroFilter 。**但** FaceLandmarker 任务层是否默认启用、参数名如何暴露，**未核实**。
- **ARKit 官方示例**：逐顶点 / 逐高斯旋转滤波 → **未找到**（ARKit 只输出 52 通道 blendshape 与 head pose，没有高斯层）。
- **SplattingAvatar 实时 demo**：论文声称 300 FPS（RTX 3090）/ 30 FPS（iPhone 13）【原文】，但**未描述任何时域滤波** → **未找到**。
- **Rig3DGS / FlashAvatar / 3DGS-Avatar / SurFhead**：全文检索 temporal / smooth / filter / jitter 均无旋转滤波机制 → **未找到**。

### 3.4 建议【推断】

- **固定 alpha**：作用在**最终四元数**上，先修号，alpha 取 0.2–0.4（60fps 下等效时间常数约 40–80 ms），在 slerp 域而非线性域混合，避免模长塌陷。
- **速度自适应（one-euro / 自适应 EMA）**：放**增量**域更自然——one-euro 的 beta 项需要速度，而增量本身就是速度。实现为对 d 做 one-euro 后再 q_t = d * q_{t-1}，并强制 d.w > 0。
- **仓库现状**：temporal_smoothing 默认 0.0，且作用在 **delta_q** 上（带符号翻转）。按 3.1，默认关闭是对的；一旦打开，建议改到 final-q 域。

## 4. 病态 J 的逐高斯可信度

### 4.1 文献侧

- **SurFhead 附录 A.2.1 Lemma 1**【原文】：可逆矩阵的极分解唯一（反证法）。**反过来读**：J 不可逆（sigma_min = 0）时极因子 U 的对应列**任意**——极分解存在但不唯一，这就是「病态不可信」的数学根源。
- **SurFhead 对退化 J 没有专门兜底**：检索 determinant / singular / degenerate 只命中「eyeball 退化透明解」「LBS 在矩阵空间退化解」，与 J 病态无关 → **未找到**。
- **Mueller / Kim / Chentanez, A robust method to extract the rotational part of deformations, MIG 2016**，DOI [10.1145/2994258.2994269](https://doi.org/10.1145/2994258.2994269)——正是「退化形变下鲁棒提取旋转」的专门文献；**标题与 DOI 已核验，其具体算法未逐条核验**（见 §6）。
- ARAP / 形变图文献的通用兜底（det 修正、SVD 奇异值截断、条件数加权）本文**未逐个核验原文**。

### 4.2 我们自己的实测（比文献更关键）

| 场景 | cond(J) | det(J) < 0 占比 | 极分解转角 |
|---|---|---|---|
| 真实 blendshape nb=1, amp=1.0 | p99 1.6 | 0.00% | p99 25.2°, max 51.4° |
| nb=5, amp=1.0 | p99 1.55, max 4.1 | 0.00% | p99 28.0°, max 65.3° |
| nb=8, amp=0.5 | p99 2.6 | 0.00% | p99 55.2°, max 178.4° |
| nb=16, amp=1.0 | p99 5.0, max 35 | 0.00% | p99 86.3°, max 179.9° |
| 500 个最小面积面人为塌成 sliver | inf | 0.00% | 5 个 **180.0°** 假旋转 |
| 整头镜像（det A = -1） | — | **0.00%** | — |
三个关键事实：(1) **真实链路的 J 并不病态**（p99 < 10），病态来自**局部帧的选择**与单帧 blendshape 的高频成分；(2) **det(J) < 0 从未发生**——混合单位帧第三列取单位法向 n，把翻面吸收进了正交化，整头镜像也一样，所以「det<0 翻最后一列」几乎不会触发，但它仍是唯一能兜住翻面的地方，**保留**；(3) **顶点噪声 -> 法向误差很陡**：以 h = sqrt(2A) 为尺度加顶点噪声，1e-3 h 时法向 p50 0.08° / max 3.3°，**1e-2 h 时 p50 0.80° / max 48.1°**，1e-1 h 时 max 179.4°——face_jacobians() 直接吃 n1，这是噪声进入旋转的最短路径。

### 4.3 建议的逐面 / 逐高斯可信度标量【推断，量纲已在 /tmp/rs/diag12.py C 段验证】
```text
c_f = clamp( (1 - min(1, (cond(J_f) - 1) / (c_max - 1)))      # 条件数项：主项
             * exp(-|| R_f^T R_f - I ||_F / r_0)              # 正交残差项：数值健康度
             * (A_f / A_med)^0.5 , 0, 1 )                     # 面积项：小面降权
c_max = 100,  r_0 = 1e-3
```
用法：把 c_f 作为面积权重的**额外乘子**（w_f = A_f * c_f）进 scatter_add，而不是丢弃低可信面——这样凸组合仍归一化，低可信面自动让位给同顶点邻面。
**必须诚实标注**：exp(-||R^T R - I||_F / r_0) 这一项在 f32 / f64 极分解下**几乎恒为 0**（实测正交残差 p50 = 0、max 2.46e-7），只在「手写迭代且没收敛」时有判别力。实测判别力：合成 cond=1 时 c_p50 = 1.000（c<0.5 占 0%）；cond=5（实际中位 11.2）时 0.897（0%）；cond=50（实际中位 353.6）时 0.000（100%）。→ **cond 项才是主项**；c_max = 100 偏保守，真实链路 p99 < 10，不会误杀。
比公式更管用的两条**结构性**兜底：(1) **角度限幅**（仓库已有 max_angle_deg = 45）直接切断 sliver 面的 180° 假旋转，实测是唯一 100% 有效的手段；(2) **邻面一致性** c_f *= exp(-angle(R_f, R_vertex) / theta_0)，R_vertex 取上一帧该顶点已平滑的旋转——比 cond 更能抓「单个面被噪声带飞」，且能复用 §2 的 Markley 不确定度。

## 5. 性能：40k 个 3x3 极分解怎么快
环境：RTX 4090 / torch 2.14.1+cu130 / cuda.Event 计时，warmup 30、rep 300、f32、N = 20018。【实测 /tmp/rs/diag11.py、/tmp/rs/diag9.py、/tmp/rs/bench_polar.py】

| 路线 | 耗时（ms） |
|---|---|
| Newton–Schulz / Higham 迭代 4 步 | 0.588 |
| 6 步 | 0.865 |
| 8 步 | **1.136** |
| 10 步 | 1.383 |
| SVD 3x3 f32（含 U @ V^T） | 4.451 |
| SVD 3x3 f64（含 U @ V^T） | 24.046 |
| 3x3 批量求逆（单次） | 0.094 |
| 3x3 批量矩阵乘 | 0.032 |
| 局部帧相乘 Q1 @ Q0^T | 0.175 |
N = 38890 的加严基准（/tmp/rs/bench_polar.py）：f64 SVD 68.124 / f32 SVD 15.023 / Newton10 1.910 / Newton5 1.011（**5 步不收敛**，det 出现 1.348）。量级与上表一致，随 N 近线性。**注意：与仓库注释 rot_sync.py:212-221 的 f64 SVD 45.8 / f32 SVD 6.66 / Newton 3.20 ms 量级一致但绝对值不同，仓库注释无对应脚本，建议以本表替换。**

### 5.2 收敛性：多少步才够
判据 max ||X^T X - I||_F 降到 1e-4 / 1e-6 所需步数随 cond(J)：1.5 -> 4/5；2 -> 5/5；4 -> 6/7；10 -> 8/9；30 -> 10/11；100 -> 13/13；1000 -> 18/18；1e5 -> 36/36；**1e6 -> 不收敛（跑到 inf）**。真实链路 cond p99 < 10，**6–8 步足够**；POLAR_ITERS = 10 是稳妥上界。注意 f32 迭代在 cond > 1e4 时是**溢出到 inf**，不是缓慢退化。

### 5.3 两个反直觉的实测结论

1. **不要先做 Frobenius 归一化。** cond=30 时：x0 = J（原始）8 步到 4.0e-7；x0 = J / ||J||_F 需 10 步才 1.9e-5；x0 = J / sqrt(||J|| * ||J^-1||) 10 步仍有 2.1e-2。Higham 迭代的收敛域与奇异值的**绝对位置**有关，把 sigma 压到 1 附近反而落进慢区。**仓库当前先做 Frobenius 归一化，方向反了。**
2. **迭代原样保留 det 符号，不做投影。** 对 det 正负各半的随机矩阵，迭代 10 步后 det(R) < 0 仍占 **50.2%**，正交残差同样优秀（2.8e-7）。这不是 bug：U V^T 本就是「最近**正交**矩阵」，det = -1 时是反射。「取最近旋转」是**应用层决策**，必须显式 `if det(X) < 0: X[:, :, 2] *= -1`，且要在**转四元数之前**做。仓库已实现，语义正确。
文献依据：Higham, Computing the Polar Decomposition—with Applications, SIAM J. Sci. Stat. Comput. 7(4):1160–1174, DOI [10.1137/0907079](https://doi.org/10.1137/0907079)；3x3 专用算法 Higham & Noferini, Numer. Algorithms 73(2):349–369, 2016, DOI [10.1007/s11075-016-0098-7](https://doi.org/10.1007/s11075-016-0098-7)。SurFhead 附录 A.2.2【原文】给出同样判断「PyTorch's SVD routine scales linearly in time as the batch size increases」，建议改用 RoMa（Brégier, 2021）的 Procrustes 或 Higham & Noferini (2016) 的 3x3 专用 routine。

### 5.4 推荐路径

- **默认：Newton–Schulz / Higham 8 步 + f32 + 不预归一化 + 显式 det 修正。** 实测 1.136 ms，占 30 ms 契约的 3.8%；与 f64 SVD 相比在真实 J 上角误差 p99 = 0.017°、max 0.023°。
- **精度 / 诊断：只在出问题时切 f64 SVD**（24 ms 一次性可接受，但不进逐帧热路径）。
- **更激进：2x2 面内闭合解** theta = atan2(F10 - F01, F00 + F11)，实测 0.034 ms（比 Newton10 快 40 倍），对 sliver 面完全稳定（3x3 路线出现 180° 假旋转时它仍有界）。代价是只给绕面法向的转角，丢掉面外旋转分量——对「面片近似刚体、法向变化主导」的场景够用，但需先确认 apply() 要的是完整 R 还是只要转角。
- **batching 不是瓶颈**：N = 20018 时所有路线都在 0.03–1.4 ms 量级，GPU 利用率很低。真正值得做的是把 J 构造、极分解、四元数转换**融合成一个 kernel**，省掉 4–5 次 launch（单次 3x3 inv 只要 0.094 ms，说明 launch 开销在几十微秒量级）。

## 6. 未核实（不要当作事实引用）

1. **3DGS 编辑器 / Nerfstudio 类工具里的「gaussian to patch size ratio」**：未找到任何可引用的实现、文档或论文。
2. **FlashAvatar / Rig3DGS / 3DGS-Avatar 是否存在推理期旋转门控**：全文检索无命中，但「无命中」不等于「不存在」。
3. **ARKit 官方示例的逐顶点 / 逐高斯旋转时域滤波**：未找到（ARKit 输出只到 blendshape 层）。
4. **MediaPipe FaceLandmarker 任务层的 smoothing 参数名与默认值**：只核验到通用 OneEuroFilter 类，任务层封装未核验。
5. **Mueller et al. MIG 2016 的具体鲁棒化算法**：只核验标题与 DOI。
6. **SplattingAvatar 实时 demo 的滤波实现**：论文未描述，官方 Unity 工程未审。
7. **GaussianAvatars eps_scaling = 0.6 的敏感性**：论文只有定性消融，无逐参数扫描。
8. **顶点数 19494 vs 20018**：任务描述称 19494；本仓库 README.md:302 与 FLAME 模板为 5023 顶点 / 9976 面，LAM 的 20018 个高斯是 5023 顶点经 Loop 细分后的位置，本文所有数字按 **20018** 计。19494 可能来自另一版本，**未核实**。
9. **Markley 特征向量解在 GPU 热路径上的数值稳定性**：本文只做 numpy 离线验证，未做 GPU eigh 基准。

## 7. 一句话给实现者
把「旋转可信度」做成**面积权重的乘子**（§4.3），把「时域平滑」放在**最终四元数**上并先修号（§3.4），把「极分解」换成 **8 步 Higham + 不预归一化 + det 修正**（§5.4），并且**先检查局部帧是不是正交归一**——这才是抖动最大的单一来源。
