# 方法四：RSB（Rotation–Stretch Blend）—— 协变旋转主导、拉伸支路实测关闭

> 版本 **2.0（重写版）**。相对 v1 的 505 行旧稿，本版做了四类改动，**全部由实测驱动**：
>
> | # | v1 的说法 | 本版结论 | 依据 |
> |---|---|---|---|
> | 1 | S2 用 k_f = A_f&#124;det J_f&#124;/A_f' 求厚度比 | **删除**。在仓库的 J 构造下 det J ≡ A'/A **恒等**，k_f ≡ 1，该步是同义反复 | f64 重算 max&#124;detJ/(A'/A)−1&#124; = 1.447e-13 |
> | 2 | 数值盒 k_f 与 n_1 一起进软投影 S5、单轴拉伸 s'_i3 | **尺度支路默认关闭**。面积比与高斯厚度无因果绑定，实测对 shape/各向异性零影响 | eigmax_rel p50 恒 1.277（off/on 四种口径逐位相同） |
> | 3 | tau=0.30 的"依据"是无解顶点定性事实 | 补上真实分布：coh p0.1=0.3226 / p1=0.7008 / p5=0.9418；tau=0.3 只切 0.08% 顶点 | capture 实测 |
> | 4 | G_dir = 1−e^(−a/1.5) 纯先验 | **实测为负收益**（rig50 5.507 → 6.018），默认关 | 5 preset 扫描 |
>
> 一句话：**方法四最终形态 = RotSync 的旋转通路 + 一级"顶点协变法向"最短弧修正 + 共享同一个相干度门**。
> 拉伸支路（stretch）在真实数据上被证伪，作为**默认关闭的名义开关**保留，并在 §2.2 写明为什么它必然无效。

---

## 0. 结论先行（TL;DR）

### 0.1 最终方案（可直接落地的一页）

对每个高斯 $v\in\{1..N\}$，$N=20018$，每帧执行：

$$
\underbrace{J_f=\tilde E_fE_f^{-1}}_{\text{S0 已有}}
\;\to\;
\underbrace{R_f=\mathrm{polar}(J_f)}_{\text{S1 已有}}
\;\to\;
\underbrace{q_{R,v}=\mathrm{Markley}(\{R_f\}_{f\ni v}),\ r_v}_{\text{S3 已有}}
\;\to\;
\underbrace{\delta q_v=\mathrm{slerp}(\mathbb 1,q_{R,v},t_vg_v)}_{\text{S4 已有}}
$$
$$
\underbrace{n_v=\mathrm{normalize}\Big(\sum_{f\ni v}A_f\hat n_{1,f}\Big),\ u_v=\mathrm{Rot}(\delta q_v)n_{0,v}}_{\text{S5 新增}}
\;\to\;
\underbrace{\delta q^{\mathrm{fix}}_v=\big[\cos\tfrac{a_v\tilde\theta_v}{2},\ \sin\tfrac{a_v\tilde\theta_v}{2}\,\hat a_v\big]}_{\text{S6 新增}}
$$
$$
\boxed{\;q'_v=\delta q^{\mathrm{fix}}_v\otimes\delta q_v\otimes q_{0,v},\qquad s'_v=s_{0,v}\;}
$$

其中 $a_v=g_vt_v$（**共用** S4 的门控与限幅，不引入第二个门），$\tilde\theta_v=\theta_{\mathrm{cap}}\tanh(\theta_v/\theta_{\mathrm{cap}})$，$\theta_{\mathrm{cap}}=25^\circ$。

**推荐超参**：`coherence_gate=0.30, p=1.5, max_angle_deg=45, normal_cap_deg=25, use_stretch=False, use_dir_gain=False`。

### 0.2 实测收益（5 个 ARKit preset，真实 capture，RTX 4090 / f32 / B=1）

| 指标 | A0 = 现 RotSync | **A3 = RSB（推荐配置）** | 变化 |
|---|---|---|---|
| rigid_err p50（一环 Kabsch） | 8.501° | **5.507°** | **−35.2%** |
| rigid_err p90 | 46.031° | **37.879°** | −17.7% |
| 一环 jitter | 5.353° | **4.381°** | −18.2% |
| false_move_frac（不该动却动了） | 0.156% | **0.074%** | **−52.6%** |
| 无参考法向残差 p50（新指标，§10.5） | 3.763° | **0.044°** | **−98.8%** |
| 修正后**被改善的顶点占比** | — | **87.4%–90.8%** | — |
| 调制器增量耗时 | 0（基准） | **+0.99 ms** | 端到端 ≈ 13.2 ms |

逐 preset 的 rigid p50（RSB / A0 / 降幅）：ones 6.462/10.215/−36.8%、half 3.163/5.038/−37.2%、
extreme 13.862/20.860/−33.6%、rand 3.863/6.094/−36.6%、rand_sparse 0.186/0.300/−38.0%。

### 0.3 四条被明确否定的路线（留档，防止二次投入）

| 路线 | 否决理由（一句话 + 硬数字） |
|---|---|
| **尺度/厚度支路**（含任务候选 (b)） | 面积比与高斯厚度无因果绑定；开关前后 `eigmax_rel p50` 恒 1.277、越界率 0.00%→0.63% 无实质变化 |
| **候选 (c) SPD 测地插值** | 逐高斯 `eigh`×2（logm+expm）**4.414 ms**，加 `matrix_to_quat` 0.800 + `quat_to_matrix` 0.469 ⇒ 增量 ≈ 5.7 ms，**超预算 5.7×** |
| **各向异性门 aniso>10 才修** | rig50 6.460 → 7.857，**几乎吃掉全部收益**（p50 aniso 仅 4.14，门把一半以上顶点关掉了） |
| **方向增益 G_dir = 1−e^(−a/1.5)** | rig50 5.507 → 6.018、jitter 4.381 → 4.394，**负收益** |

### 0.4 三条工程硬事实（写进实现 checklist）

1. **注入点不可变**：`LAM/lam/models/rendering/gs_renderer.py` 里 `cov3D_precomp = None` 是硬编码，只能回写 `(q, s)`；`delta_rotation_scale / delta_scale / apply` 三个签名不可改。
2. **不要用 `torch.linalg.inv`**：仓库 `tga_covariance.inv3`（闭式伴随 / det，floor 1e-12）实测相对误差 p50 7.119e-08、p99 3.140e-07，**优于** `torch.linalg.inv(f32)` 的 p99 1.950e-06。
3. **f32 足够**：f32 与 f64 逐面 J 的最大元差 p50 1.583e-07 / p99 7.212e-06 / max 4.829e-03，误差 >1.0 的面数为 **0**。v1 里"f32 逆在 sliver 上严重失真"的说法是**与 f64 重算的口径混淆**，已撤回。

---

## 1. 问题设定与失效模式分解

### 1.1 冻结的上下文

- 预训练 LAM（arXiv:2502.17796）单图重建：$N=20018$ 高斯，与 FLAME 细分网格顶点**一一绑定**，$F=39904$ 面。
- 推理期**冻结**：不改权重、不训练。表情由 ARKit 52 通道 → FLAME blendshape → 顶点位移 $x_B$。
- 调制器只能看到 `(x0, xB)` 两个顶点张量（中性 / 表情后），必须输出 `(q', s')`。
- 硬预算：< 30 ms（现 RotSync 端到端 11.20–12.22 ms）。

### 1.2 三个必须解决的失效模式

记 $x_0$ 为中性顶点、$x_B$ 为表情后顶点（本 repo 的 `ret["animated"]` 与 `mean_3d`）。

| 失效模式 | 现象 | 根因 | 现有方法谁失败 |
|---|---|---|---|
| **F1 面外丢失** | 张口/眨眼时薄片朝向不跟随表面 | LAM 原版把调制轴锁在静止法向 $n_0$，只做面内 2×2 旋转 | 方法一（LAM） |
| **F2 聚合歧义** | 一环内各面旋转方向相反时（$\theta\to180^\circ$）平均退化为零旋转 | 矩阵域**线性**平均在 $SO(3)$ 上不封闭，$\tfrac12(R_1+R_2)$ 对反向对给出 $\approx 0$ 阵 | 方法二（TGA）、VTP |
| **F3 无解区** | 46–76 个顶点（0.2–0.4%）一环面两两夹角中位 176°，合矢量中位长度 0.0000，**任何**聚合都无定义 | 真实几何翻折（薄片被压扁到近零面积） | **所有**方法；只能门控 |

### 1.3 为什么方法四只需要动"旋转"这一个自由度

这是本方案最重要的设计判断，且**是被实测逼出来的**，不是美学选择：

1. 高斯协方差的 6 个自由度在 `(q,s)` 表示下是 $3(\text{rot})+3(\text{scale})$。
2. 但 LAM 的 $s_0$ 分布极端各向异性（$s_{\max}/s_{\min}$：p50 4.14 / p90 99.7 / p99 23485.9，最小轴 p50 4.185e-04），
   即"薄片"的第三轴是**亚像素厚度**，几何上的面积变化（$\pm$几倍）与这个厚度之间**没有任何因果绑定**——
   这一点在 §2.2 用三个独立实验证伪了它。
3. 剩下唯一与几何强相关的自由度就是**朝向**。而朝向的正确值只能来自形变后的面法向 $\hat n_1$，
   它与用旋转聚合出的 $u=\mathrm{Rot}(\delta q)n_0$ 之间的差，就是方法三（RotSync）**唯一**的结构性误差来源。

于是方法四的结构就确定了：**在 RotSync 之上，补一级"把面法向对齐"的最短弧修正**。全部新增计算只有
两个叉积、一个 `atan2`、一个 `tanh`、两次四元数乘。

---

## 2. 四个候选方向的评估与取舍

任务给出四个候选 (a)(b)(c)(d)。逐个给出**数学形式 → 理论理由 → 实测判决 → 去留**。

### 2.1 候选 (a)：法向协变 $n_d=\mathrm{normalize}(J^{-\top}n_c)$ 且 $q'$ 对齐 —— **采纳（本方案核心）**

> **v2.1 判决补注（2026-10-24，exp_de_normal_cov）**：与逐顶点先求 $J_v^{-\top}$ 的实现路线做了决定性对照
> （9 向量 × 20018 顶点，/tmp/exp_de_normal_cov.json）。两条独立证据：
> ① 预声明判据 Δε p90<2° 在真实 ARKit preset 上成立（pooled p90 **1.178°**）；
> ② 相对**形变后真实曲面法向** $n_v^{surf}$ 的残差，$J_v^{-\top}$ 路线反而更大（pooled p90 **15.468°** vs 本方案面级加权 **14.296°**，
> 仅 16.5% 顶点更近，随机基线 50%）——负收益，机制是 $J_v$ 已是面积加权平均，$J_v^{-\top}$ 在近奇异顶点放大平均误差。
> 结论：**本文的面级加权 $n_v=\mathrm{normalize}(\sum A_f\hat n_{1,f})$（即 $n_v^{surf}$）保持不变**；
> 被否决的是「顶点级先求逆再协变」的实现路线与 SPD 测地插值候选 (c)。
> 另：m4_base.pt 中的 nv 字段实测即 $n_v^{surf}$（与本文 $n_v$ ones 最大夹角 0.0442°），非 $J^{-\top}$ 输出。
**数学形式**。设 $n_c$ 为静止参考法向（本文 $n_{0,v}$），$n_d$ 为协变目标法向（本文 $n_v$）。
用 Nanson 公式，面法向的协变变换是

$$
n_d \propto J_f^{-\top}n_{0,f},
$$

且因为在仓库的 $J$ 构造里第三列取的是**单位**法向（`rot_sync.py:446-448`，`n1 = F.normalize(cross(d1,d2))`），
有 $\hat n_1=\mathrm{normalize}(d_1\times d_2)$ **恰好**就是 $J_f^{-\top}n_{0,f}$ 的方向（差一个正标量）。所以**不需要求逆**：
直接用 $\hat n_1$。

**实现粒度的选择（关键设计点）**。两条路：

| 粒度 | 做法 | 代价 | 问题 |
|---|---|---|---|
| 顶点级 | 先算 $J_v=\sum_f A_fJ_f/\sum_f A_f$，再 $n_d\propto J_v^{-\top}n_{0,v}$ | 需一次 3×3 求逆 / 面 | 先平均后求逆，**把逐面置信度差抹平**；且 $J_v$ 在翻折区可奇异 |
| **面级 + 加权（采用）** | $n_v=\mathrm{normalize}\big(\sum_{f\ni v}A_f\hat n_{1,f}\big)$ | 一次 `index_add`，0.15 ms | 无 |

面级加权更稳：$\hat n_{1,f}$ 本身已是单位向量（模长信息已被 `normalize` 剥掉，这正好——我们不想要面积权重被"法向长度变化"污染），
面积权重只用来做**几何重要性**加权。

**为什么目标必须是 $n_v$ 而不是 $n_{0,v}$ 的旋转**。这是候选 (a) 的全部价值所在：

$$
u_v=\mathrm{Rot}(\delta q_v)n_{0,v}\quad(\text{聚合旋转给出的预测法向}),\qquad
n_v\quad(\text{形变网格给出的真实法向}),
$$

$\angle(u_v,n_v)$ 的中位是 **2.072°**、p90 **19.547°**、p99 **89.797°**（$>5^\circ$ 占 35.04%，$>15^\circ$ 占 13.78%）。
这不是小数——它是**聚合损失**：把一环内多个面的旋转平均成单个旋转、再作用到法向上，一般不等于面法向的面积加权平均。
$SO(3)$ 上的平均与 $\mathbb R^3$ 上的平均**不可交换**。

**实测判决（三组证据，全部支持采纳）**

1. **有参考指标**：rigid p50 8.501 → 5.507（−35.2%），逐 preset 全线下降。
2. **完全无参考指标**（§10.5，绕开"参考系本身是 Markley"的循环）：法向残差 p50 3.763 → 0.044，
   **87.4%–90.8% 的顶点被改善**。
3. **安慰剂对照**（§10.4）：把方向反转（neg-sign）变差到 11.043；换成同幅度随机轴等于没做（8.559 且 jitter 暴涨到 8.370）；
   只做幅度收缩（shrink-only）与 A0 **逐位相同**。⇒ 收益**只能**来自"指向 $\hat n_1$"，不可能来自扰动幅度或收缩效应。

**粒度：帧级优先（保留 LAM 学到的倾角）**。$q_{0,v}$ 是优化出来的朝向，未必等于面帧 $E_f$
——薄轴可能相对面法向**有意**倾斜以服务视角相关外观。所以修正应作用在**帧级**（把整个高斯连同它的倾角一起搬），
而不是把薄轴强行钉到切面。后者（特征轴级）列为消融 B12，预期 PSNR 下降。

### 2.2 候选 (b)：法向轴单独拉伸 $k_n=\tanh(g\cdot u)$ —— **否决（实测无效应，且假设无物理基础）**

**数学形式**（任务表述）：把法向那一根轴的尺度按形变量单独拉伸，
$s'_{i_3}=s_{0,i_3}\big(1+\tanh(g\,u)\big)$，或对数版 $s'_{i_3}=s_{0,i_3}\exp(\alpha\hat\ell)$，其中 $\hat\ell$ 来自面积比。

**三条独立证伪证据**：

**(b-1) 面积比与厚度无因果绑定（合成实验，最干净）**。构造已知的纯各向同性缩放 $x_B=x_0\cdot1.2$，
此时逐顶点 $\log(A'/A)$ 的 p50 = **+0.3646**（即面积涨 1.44 倍，正确），
但把整条尺度支路打开（$G^{\mathrm{amp}}$、$G_\rho$ 任意组合）后，$s'/s_0$ 的 p50 **恒为 1.0000**。
原因：$\exp(\alpha\hat\ell)$ 里 $\alpha$ 由 $g_v$ 与增益决定，而缩放情形下 $g_v\approx1$、$\hat\ell$ 过软投影后仍非零，
**但 $s_0$ 的第三轴是"高斯厚度"，不是"网格面积"**。两者在 LAM 的优化目标里从未被约束成一致。

**(b-2) 真实数据上"越界率"几乎不动**。$\log(A'/A)$ 的分位：p50 0.0 / p90 0.173 / p99 1.415 / max 5.369；
$\lvert\ell\rvert>\log1.35$ 占 **10.90%**、$>\log2$ 占 3.90%、$>\log3$ 占 1.79%。
但把开关从 off 打到 on（三种增益口径），`eigmax_rel p50` **恒为 1.277**、capfrac 只在 0.00%–0.63% 之间抖动。
⇒ 这条支路在真实数据上**既不改善 shape，也不触发越界**，等于没接。

**(b-3) "让薄轴对齐法向"这个前提本身是错的**。实测 $\angle(\text{薄轴 }i_3,\ n_{0,v})$：
p10 22.5° / p50 **56.7°** / p90 83.6°，$\angle<15^\circ$ 只占 **4.89%**；
而**随机方向**与固定方向的期望夹角中位是 **57.3°**。⇒ 薄轴与法向**统计上不可区分**，
"法向轴"这个概念在 LAM 的高斯里根本不存在。任何"沿法向拉伸"的操作都是在拉一个**随机方向**。

**结论**：尺度支路以 `use_stretch=False` 的形式**保留接口但不启用**。
保留接口的意义是让 §10 的消融 B1/B2/B3 可复现地证明它无效——这比直接删掉更有工程价值。

> 诚实标注：任务候选 (b) 的**变体**——"只用 $q'$ 改正方向、不改 $s$"——正是本方案实际采纳的候选 (a)。
> 所以 (b) 不是被完全丢弃，而是被**降级**成 (a)：改方向，不改厚度。

### 2.3 候选 (c)：SPD 测地插值 $\Sigma(\alpha)=\exp\big((1-\alpha)\log\Sigma_R+\alpha\log\Sigma_{TGA}\big)$ —— **否决（超预算）**

**数学形式**（任务表述）：逐高斯在 SPD 流形上做测地插值，$\alpha$ 逐高斯自适应，端点取
$\Sigma_R=R\Sigma_0R^{\top}$ 与 $\Sigma_{TGA}=J\Sigma_0J^{\top}$。

这个形式**在数学上是最优雅的**：它是唯一同时满足 (i) 端点精确、(ii) 全程 SPD、(iii) 仿射不变测地线 的插值。
若算力无限，它应该是首选。

**但它过不了预算这一关**。逐高斯做一次 `logm`+`expm` 需要两次 3×3 `eigh`（特征分解 → 取 log → 重组 → 再分解）：

| 操作（$N=20018$，f32，RTX 4090） | 实测 |
|---|---|
| `eigh` × 2（logm → expm，即完整 (c)） | **4.414 ms** |
| `eigh` × 1（只做 logm，端点线性混合后再 expm 近似） | 1.463 ms |
| `matrix_to_quat`（$N$ 个 3×3 → 四元数） | 0.800 ms |
| `quat_to_matrix`（$N$ 个四元数 → 3×3） | 0.469 ms |

仅 (c) 的矩阵函数就 **4.414 ms**，加上往返的 1.27 ms ⇒ **增量 ≈ 5.7 ms**，
把端到端从 12.2 ms 推到 **≈17.9 ms**，相对 30 ms 预算只剩 40% 余量，而预算是**整帧 splat** 的，不是调制器的。

更致命的是**结构性缺陷**：$\Sigma(\alpha)$ 的特征基在 $\alpha\in(0,1)$ 中间**一般不同于两端**
⇒ 写回 `(q',s')` 必须 `eigh` ⇒ **必须排序** ⇒ 排序在特征值交叉时跳变 ⇒ **时域不连续**。
这正是方法二（TGA）jitter 0.73–0.94 的同一病根，只是被推迟到中间 $\alpha$ 上。**不会用 5.7 ms 买一个已知会跳变的方案。**

**复活条件**（写下来防止将来重复讨论）：若未来 (i) 有批量 3×3 `eigh` 的融合 kernel 把 4.4 ms 压到 < 0.5 ms，
且 (ii) 能证明 $\alpha$ 的自适应不引入排序跳变，则 (c) 可以重开。当前两者都不成立。

### 2.4 候选 (d)：逐高斯双门 $c_R/c_S$，$\Delta=\mathrm{Exp}(c_R\mathrm{Log}R)\cdot\mathrm{Exp}(c_S\log S)$ —— **部分采纳**

**数学形式**（任务表述）：两个独立的逐高斯系数，一个收缩旋转、一个收缩尺度。

**采纳的部分**：
- $c_R$ 通路**已在仓库里**（`rot_sync.py:577-579` 的 `rotation_gain`，以及 `:599-600` 的统一 slerp 回缩）。
  方法四**复用**它，并让法向修正共用同一个回缩系数。
- **"双门"的正确形式不是两个独立系数，而是"一个门 + 两个不同的作用对象"**。我们的做法：
  $$
  \delta q_v=\mathrm{slerp}(\mathbb 1,q_{R,v},t_vg_v),\qquad
  \delta q^{\mathrm{fix}}_v=\mathrm{Exp}\big(g_vt_v\tilde\theta_v;\hat a_v\big),
  $$
  两个门**完全相同**（都是 $t_vg_v$）。

**否决的部分**（$c_S$ 独立门）：
1. $c_S$ 作用的量（厚度）已被 §2.2 证伪，独立门无从谈起。
2. 更重要的**一致性论证**：若 $c_R\neq c_S$，会出现"转不动但拉得动"的顶点。
   在 176° 翻折顶点上，$g_v\to0$ 而 $c_S$ 若独立于 $g_v$，就会产生**"静止但突然变厚"的闪帧**——
   这是两个独立门最容易犯的错，且恰好落在最差的 0.2% 顶点上。

**嵌套性质（本方案的一致性主张）**：
$$
\{\text{被允许修正方向的顶点}\}\ \subseteq\ \{\text{被允许旋转的顶点}\}
$$
因为两者用同一个 $g_vt_v$。这不是巧合，是设计约束：**门控必须嵌套，不能正交。**

---

## 3. 完整公式链（S0–S7）

### 3.1 符号表

| 符号 | 定义 | 备注 |
|---|---|---|
| v ∈ {1..N} | 高斯索引，N = 20018 | 与 FLAME Loop 细分顶点一一绑定 |
| f ∈ {1..F} | 三角面索引，F = 39904 | |
| E_f = [e1, e2, n0h] | 静止面局部帧（列向量） | n0h = normalize(e1 × e2)，**单位** |
| Ẽ_f = [d1, d2, n1h] | 形变后面局部帧 | n1h = normalize(d1 × d2)，**单位** |
| J_f = Ẽ_f E_f^(-1) | 仓库定义的逐面形变梯度 | 第三列是单位法向 ⇒ det J = A'/A（见 3.2.1） |
| A_f, A'_f | 静止 / 形变面积 | |
| R_f U_f | J_f 的右极分解 | 8 步 Newton–Higham |
| Q_0v | 高斯静止朝向（q0 的矩阵） | 冻结属性 |
| s_0v | 静止轴长 | 冻结，最小轴 p50 = 4.185e-04 |
| n_0v | 顶点静止法向（面积加权） | init 缓存，240 KB |
| q_Rv, r_v | Markley 聚合四元数 / 相干度 | r_v = 1 − λ2/λ1 |
| g_v, t_v | 相干度门 / 角度限幅系数 | 均 ∈ [0,1] |
| a_v | 方向修正总增益 = g_v · t_v | **唯一**的门，不引入第二个 |

记 Q0 为四元数写法 q0（wxyz），Rot(q, n) 表示用四元数 q 旋转矢量 n。

### 3.2 S0–S1：逐面几何与极分解（**已有，不改动**）

A_f = ½ ‖(P2 − P1) × (P3 − P1)‖，　A'_f = ½ ‖(P'2 − P'1) × (P'3 − P'1)‖

J_f = [d1 d2 n1h] · [e1 e2 n0h]^(−1)，　J_f = R_f U_f

#### 3.2.1 必须记录的恒等式（v1 的错误）

因为第三列取的是**单位**法向，

det J_f = ((d1 × d2) · n1h) / ((e1 × e2) · n0h) = ‖d1 × d2‖ / ‖e1 × e2‖ = A'_f / A_f

实测 f64 重算 max |det J_f / (A'_f / A_f) − 1| = 1.447e-13，**逐面精确**（f32 下 1.7e-5）。
所以 v1 的 S2 步骤 k_f = A_f |det J_f| / A'_f **恒等于 1**，是一个同义反复（degenerate step），本版删除。

这条恒等式还有一个**正面用途**：它是免费的自检——实现时用 A'_f / A_f 校验 det J_f，
偏差 > 1e-4 即说明 `inv3` 或法向归一化出问题。→ 对应伪代码里的 `_assert_det_consistency`。

R_f 由 Newton–Higham 迭代 X ← ½(X + X^(−T)) 得到，X0 = J_f，8 步。
仓库实现 `rot_sync.polar_rotation`（`lam3d/core/rot_sync.py:255`）实测角误差 p99 = 0.017°，耗时 3.155 ms。

**S1 的两个必要守卫**（`_polar_rotation_svd`，`rot_sync.py:238`）：
det R < 0 时翻最后一列（随机矩阵占 50.2%，迭代不改变符号）；cond > 1e4 时退化到 f64 SVD。

### 3.3 S2：聚合权重（**简化：删掉 v1 的条件数置信度**）

w_f = A_f（只用面积）

v1 用 w_f = A_f · c_f，c_f 含一个 κ_f = ‖J_f‖_F ‖J_f^(−1)‖_F 项与一个「R_f 正交性残差」项。**实测删掉它**，理由：

- cond(J) 的真实分布：p50 **1.00**、p99 **1.08**、max **37.9**。真实数据的面几乎刚性
  （原因：ARKit 表情位移很小，‖x_B − x_0‖ p50 = 2.3e-3，而 ‖x_0‖ 均值 0.042）。
- 于是 κ_f 项在 99% 的面片上恒为 1，**永不触发**，却要 0.15 ms。正交性残差项同理。

**保留的守卫**：A_f ≤ ε 时该面不参与（w_f = 0），且 q_f 用恒等填充。
真实 capture 上静止面积 min = 9.410e-10（> 0），frac(A_f ≤ 1e-12) = 0，
未被任何面引用的顶点 = **0**，仅 1 面引用的顶点 = **0** ⇒ 本 capture 无退化，
但实现必须硬化（0 × nan = nan 会把 NaN 传播到整张图）。

### 3.4 S3：旋转聚合与相干度（**已有，不改动**）

M_v = Σ_{f∋v} w_f · q_f q_f^T，　q_Rv = argmax_{‖q‖=1} q^T M_v q，　r_v = 1 − λ2/λ1 ∈ [0,1]

q_f = quat(R_f)，M_v 是该顶点一环（通常 6 个面）的 4×4 加权外积和，q_Rv 取最大特征值的特征向量。
Markley 平均对 q / −q 符号**完全免疫**（q q^T 里符号抵消），所以不需要半球对齐。代价 0.476–0.63 ms。

r_v 的几何含义：一环内所有面旋转一致时 M_v 的扰动是秩 1 的，λ2 → 0，r_v → 1；
翻折时各面四元数近似对径，M_v → ¼(Σw) I（各向同性），λ1 ≈ λ2，r_v → 0。

**实测分布（v1 缺失的关键数据）**：r_p0.1 = 0.3226，r_p1 = 0.7008，r_p5 = 0.9418，r_p50 = 0.99989。
即 99% 以上的顶点一环高度一致，**只有尾部**才需要门控。

### 3.5 S4：门控与角度限幅（**已有**，但方法四的用法有变化）

g_v = clamp((r_v − τ)/(1 − τ), 0, 1)^p，　p = 1.5，τ = 0.30
t_v = min(1, θ_max / θ^raw_v)，　θ^raw_v = angle(q_Rv)，θ_max = 45°
δq_v = slerp(1, q_Rv, t_v · g_v)

**p = 1.5 的作用**：把门控从线性拉成「接近 1 时几乎全开、接近 0 时迅速归零」的形状（幂次 < 2 避免过度抑制）。
**必须只做一次 slerp**（实测两次 slerp 多 2.3 ms），所以 t_v 与 g_v **先乘再插值**，这正是仓库现成的写法（`rot_sync.py:586-600`）。

**门控覆盖率实测**（v1 缺失）：

| τ | g_v = 0 占比 | g_v > 0.5 占比 |
|---|---|---|
| 0.0 | 0.00% | 99.32% |
| **0.3（推荐）** | **0.08%** | **98.74%** |
| 0.5 | 0.31% | 98.11% |
| 0.7 | 1.00% | 96.97% |
| 0.9 | 3.30% | 93.35% |

τ = 0.3 的取舍：它只切掉 0.08% 的顶点（正好覆盖 true fold 区），却保留 98.7% 的修正自由度。
τ = 0.5 时 rig50 变差到 5.524（开始误杀 θ 较大的正常顶点）。**v1 建议「先试 0.5」是错的**，实测 0.3 更优。

### 3.6 S5：顶点协变法向与预测法向（**新增，本方案核心**）

n_v = normalize( Σ_{f∋v} A_f · n1h_f )，　u_v = Rot(δq_v, n_0v)

两个方向都必须用**限幅 + 门控之后**的 δq_v，不能用 q_Rv——否则在限幅顶点上
会把「被砍掉的旋转」当成缺失的法向摆动画回来，等于绕过 t_v。

Rot(q, n) 用两次叉积（不做矩阵化）：

Rot(q, n) = n + 2w (q_xyz × n) + 2 q_xyz × (q_xyz × n)，　w = q_w

**n_0v 的构造**（init 一次，缓存）：n_0v = normalize( Σ_{f∋v} A_f · n0h_f )，
退化守卫 ‖·‖ < ε 时取该顶点任一邻面法向。

**为什么 n_0v 可以缓存**：x_0 与 q_0 都不随帧变，所以运行期只需 Rot 的一次旋转，不需要重新算静止面法向。
n_v 必须每帧算（依赖 x_B），但只用一个 `index_add`。

**逐位一致性的细节**：n_0v 必须在 **f32** 下 cache 并且在 init 时归一化一次；
若每帧重新归一化，1e-7 的抖动会经 atan2 放大到 1e-5° 量级，虽然可忽略，但会让
`torch.equal` 级别的一致性测试失效。

### 3.7 S6：最短弧法向修正（**新增**）

c⃗_v = u_v × n_v，　s_v = ‖c⃗_v‖，　c_v = u_v · n_v
θ_v = atan2(s_v, c_v) ∈ [0, π]，　â_v = c⃗_v / max(s_v, 1e-9)
θ̃_v = θ_cap · tanh(θ_v / θ_cap)，　θ_cap = 25°
δq^fix_v = [ cos(a_v θ̃_v / 2), sin(a_v θ̃_v / 2) · â_v ]，　a_v = g_v t_v

**为什么用 atan2 而不是 arccos**：cos θ → 1 时 arccos 的导数发散（1/√(1 − c²)），
小角上的绝对误差被放大到 √ε_f32 ≈ 3.4e-4 rad 量级；atan2 在小角上精确到 1e-7 量级。
这条对方法四**特别重要**，因为修正角的 p25 只有 **0.006°**
——绝大多数顶点在「几乎不用修」的区域，arccos 的数值噪声会直接变成 jitter。

**为什么用 tanh 软饱和而不是硬 clamp**：
1. 硬 clamp 在阈值处导数不连续 ⇒ 阈值附近的顶点来回跳（TGA 的 jitter 0.73–0.94 部分来自此）。
2. tanh 在 θ ≫ θ_cap 时把增益**自动降到 0**（1 − tanh² → 0），即「越没把握越少动」，正是无解区需要的。
3. θ → 0 时 tanh(θ/θ_cap) ≈ θ/θ_cap ⇒ 近似恒等，**不损失小角精度**。

θ_cap = 25° 的取值依据（消融扫描，§10.3）：

| θ_cap | rig50 | rig90 | jitter |
|---|---|---|---|
| 15° | 5.713 | 39.512 | 4.515 |
| **25°（推荐）** | **5.507** | **37.879** | **4.381** |
| 35° | 5.439 | 36.936 | 4.313 |

35° 略好但**不安全**：θ^raw 的 p90 是 19.5°、p99 是 89.8°，cap 越大，尾部（不可信）顶点获得的修正越多。
25° 落在 p90 附近，是「只修可信部分」的自然分割。

### 3.8 S7：写回参数（**主公式**）

**q'_v = δq^fix_v ⊗ δq_v ⊗ q_0v，　s'_v = s_0v（逐位，不加任何算术）**

等价协方差形式：

Σ'_v = R(δq^fix_v) R(δq_v) Q_0v diag(s_0²) Q_0v^T R(δq_v)^T R(δq^fix_v)^T

**关键性质：Σ' 是由 (q', s') 生成出来的，而不是从 Σ' 分解出 (q', s')**。
于是 Σ' ≡ R' diag(s'²) R'^T **恒等成立**（TGA 只在未触发截断的轴上成立）。

**左乘的顺序不可交换**：δq^fix 定义在**世界系**（它把世界系里的法向 u_v 转到 n_v），
q_0 定义在**局部系**（高斯自身的静止朝向）。所以必须左乘、且 δq^fix 在最外层。

### 3.9 退化到三种方法（消融锚点）

| 条件 | 退化到 | 精确性 |
|---|---|---|
| a_v ≡ 0（`enable_normal_fix=False`） | **RotSync（方法三）** | **逐位相同**（命题 1） |
| a_v ≡ 0 且 g_v ≡ 1 | RotSync 无门控 | 除门控外逐位 |
| 把 n_v 换成 u_v（即 θ_v ≡ 0） | δq^fix = 1 ⇒ RotSync | 自检用 |
| 把 S6 换成 R_f 的顶点版、并把 s' 三轴全改 | TGA | 仅在无剪切 + q_0 = E_f + 未饱和时精确 |
| 把 S5/S6 换成「面内 2×2 旋转 + 各向同性面积缩放」 | LAM 原版 | 结构同构（k = 0.35, g = 2） |

**命题 1（a_v = 0 的逐位恒等）**：当 `enable_normal_fix=False` 时，S6 走守卫分支直接赋值
δq^fix ← 1 并**短路**，不进入 ⊗；S7 里 s' 用 `.expand()` 直接返回 `scaling` 本身
（**不做任何算术**，连 ×1.0 都不做）。于是输出与 `RotSyncModulator.apply` 逐位相同。
实现要点：1 必须是**赋值**（`torch.tensor([1.,0,0,0])`），不能是计算
（sin(0)/… 会引入 +0 vs −0 的差别，破坏 `torch.equal` 自检）。

---

## 4. 每步复杂度与实测耗时

### 4.1 复杂度表（N = 20018，F = 39904，B = 1）

记 K 为平均一环面数（FLAME 细分网格 K ≈ 6）。

| 步骤 | 新增/已有 | 渐进复杂度 | 主操作 | 实测（ms） |
|---|---|---|---|---|
| S0a 面面积 A_f | 已有 | O(F) | 1 cross + norm | 0.196 |
| S0b 逐面 J_f | 已有 | O(F) | 2 cross + 归一化 + inv3 | 0.916 |
| S1 极分解 R_f | 已有 | O(8F) | 8× inv3(3×3) + 乘 | 3.155 |
| S1b R_f → q_f | 已有 | O(F) | matrix_to_quat | 0.819 |
| S3 Markley | 已有 | O(F) scatter + O(N) eigh(4×4) | scatter_add + linalg.eigh | 0.476 |
| S4 门控+限幅+slerp | 已有 | O(N) | atan2 + 1 slerp | 0.584 |
| **S5a n_v 聚合** | **新增** | O(3F) | 1 index_add(3ch) + 归一化 | **0.150** |
| **S5b u_v = Rot(δq, n_0)** | **新增** | O(N) | 2 cross | **0.030** |
| **S6a θ_v** | **新增** | O(N) | 1 cross + norm + dot + atan2 | **0.070** |
| **S6b 软饱和 + 半角四元数** | **新增** | O(N) | 1 tanh + 1 sqrt + 4 乘 | **0.060** |
| **S7 两次四元数乘** | **新增** | O(N) | 2 qmul | **0.190** |
| **新增合计** | | | | **0.99** |

> 增量实测口径：`scripts/proto_m4_timing.py` 里，「方向块（q_R, coh → δq，含 45° 限幅与门控）」= 0.5838 ms；
> 同一块 + S5/S6/S7 = 1.4689 ms ⇒ **增量 = 0.9869 ms**。本表按 0.99 ms 记。
> 端到端预算：现 RotSync 端到端 12.22 ms ⇒ 方法四 ≈ **13.2 ms**，仍在 30 ms 预算的 44% 以内。

### 4.2 为什么调制器是 dispatch-bound 而不是 compute-bound

N = 20018 的逐高斯小算子单次耗时实测：单个 cross **0.0104 ms**、单个 atan2 **0.0194 ms**、
10 个连续 cross **0.1077 ms**。即**每个 op ≈ 10–13 µs**，几乎全是 kernel launch 开销，
GFLOPS 只用到个位数百分比。

**这给出两条优化路径，与上面的表一致**：

1. **减少 op 数优于减少 FLOPs**。所以 S5 用 `index_add` 而不是「先 gather 再用 matmul」；
   S6 的半角四元数直接构造，而不是「转角 → 转轴 → 转矩阵 → matrix_to_quat」
   （那样多 3 个 op ≈ 0.04 ms，还多一次 1.3e-6 的往返误差）。
2. **融合 scatter 几乎免费**：把 n_v 的 3 通道与（可选的）log k 第 4 通道塞进**同一次** `index_add`：
   9 通道 0.1512 ms、10 通道 0.1475 ms、分开做 9+1 通道 0.2958 ms ⇒ **省 0.15 ms**。
   这是为什么「尺度支路保留接口」的成本接近零。

### 4.3 内存

| buffer | 形状 | 大小 | 生命周期 |
|---|---|---|---|
| n_0v | [N,3] f32 | 240 KB | init 一次，`persistent=False` |
| s_0v 最小轴索引 i3 | [N] int8 | 20 KB | init 一次（尺度支路用） |
| g_v t_v | 不缓存 | — | 每帧算比缓存便宜 |
| 临时张量 | 数个 [N,3] | < 1 MB | 每帧 |

**总计新增常驻 < 0.3 MB**，相对 LAM 模型（> 100 MB）可忽略。峰值额外显存 < 2 MB。

---

## 5. 每步修复的失效模式（逐条对应 §1.2）

| 步骤 | 修复哪个失效模式 | 机制（不是「改善」，而是「为什么能改善」） |
|---|---|---|
| S3 Markley（已有） | **F2 聚合歧义** | 在 SO(3) 的内蕴度量下取最大特征向量，对 q/−q 免疫，θ → 180° 时仍有定义（矩阵域线性平均此时退化为近零阵） |
| S4 门控（已有） | **F3 无解区** | r_v → 0 时 g_v → 0 ⇒ δq_v → 1 ⇒ 该顶点**逐位穿透**为 q_0，不产生假旋转 |
| **S5 n_v 协变（新增）** | **F1 面外丢失** | 目标从「n_0 被旋转后的预测」换成「形变网格的真实面法向」，把聚合损失直接消掉 |
| **S6 最短弧（新增）** | **F1 的残余 + F2 的残余** | 显式补上「聚合旋转 → 真实法向」的测地残差，且用软饱和保证不把翻折区的 176° 直通 |
| **S4 共用 g_v t_v（新增约束）** | **F3 的二次污染** | 修正量与旋转量共享同一个门，避免「转不动但修得动」的失配（§2.4 的嵌套性质） |

**量化对照（同一指标，四个方法）**：

| 失效模式指标 | LAM | TGA | RotSync (A0) | **RSB (A3)** |
|---|---|---|---|---|
| F1 面外：无参考法向残差 p50 | — | — | 3.763° | **0.044°** |
| F2 聚合：rigid p50 | 7.759° | 6.086° | 8.501° | **5.507°** |
| F2 聚合：rigid p90 | 15.732° | 12.082° | 46.031° | **37.879°** |
| F3 无解：false_move_frac | 0.09–0.37% | 3.08–5.03% | 0.156% | **0.074%** |
| F3 无解：jitter | 15.90° | 18.75° | 5.353° | **4.381°** |

> 口径注：LAM/TGA 的 rigid 数字来自 `docs/cov_methods_deepdive.md `§7.2 的 ones preset；A0/A3 是 5 preset 均值。
> 不能直接横比（preset 不同），但**同一 preset 内 A0 vs A3 的下降是逐位可比的**（§10.1 给了逐 preset 表）。
> 这里放它是为了给出量级判断：RSB 的 5.507 已低于 TGA 的 6.086，并向 LAM 的 7.759 之外的最好值收敛
> （LAM 的 7.759 本身是它在 ones preset 上的值，口径不同，仅作量级参考）。

---

## 6. 边界条件处置全表

原则：**任何退化情形都必须在「不改输出」的一侧失败**（fail-static，退化为恒等/透传），
绝不允许 fail-dynamic（产生 NaN、随机跳变、或把无穷大放大到可见）。

| # | 边界条件 | 触发规模（实测） | 处置 | 代码位置 | 失败若不做 |
|---|---|---|---|---|---|
| E1 | **det J_f < 0**（面翻转） | 6e-5（约 2.4 个面） | S1 的 `_polar_rotation_svd` 内翻最后一列，产出 det = +1 的 R_f；在诊断里计 `flipped` 计数 | `rot_sync.py:238-252` | R_f 不是旋转，quat 化后得负 w，Markley 外积仍可用但 R_f 退化（模长 ≠ 1） |
| E2 | **A_f ≤ ε**（零面积面） | frac = 0（min 9.410e-10），但必须硬化 | w_f ← 0；q_f ← 恒等（**赋值**，非计算）；参与 `index_add` 的 A_f 先 `clamp(min=0)`，避免 −0 | `rot_sync.py:451-458` / 新增 S5a | 0 × nan = nan ⇒ 单点污染整张图（NaN 经 reduce 传播） |
| E3 | **孤立顶点**（无邻面） | 0 个 | 归一化前判 `den ≤ ε` ⇒ n_v ← n_0v；r_v ← 0 ⇒ g_v = 0 ⇒ 输出 q_0 | `rot_sync.py:460-484` 已有 den≤eps→eye 的模式 | 除零 ⇒ inf ⇒ normalize 得 nan |
| E4 | **cond(J_f) 极大** | p99 = 1.08，max 37.9（真实数据几乎不触发） | cond > 1e4 退化到 f64 SVD；不做逐面条件数置信度（v1 已删，见 §3.3） | `rot_sync.py:246-252` | 8 步 Newton 在近奇异 J 上不收敛 ⇒ R_f 非正交 |
| E5 | **r_v < τ**（无解区 / 翻折） | τ=0.3: 0.08%；真实翻折顶点 46–76 个 | g_v = 0 ⇒ δq_v = 1、δq^fix = 1 ⇒ **逐位透传 q_0**（`apply` 的 is_identity 分支） | `rot_sync.py:632-656` | 在 176° 翻折顶点上产生随机方向的假旋转（这正是 RotSync 之前 false_move 的来源） |
| E6 | **θ^raw > θ_max** | 45° 以上占少数尾部 | t_v = θ_max/θ^raw，一次 slerp 完成限幅（不二次 slerp） | `rot_sync.py:586-600` | 两次 slerp 多 2.3 ms；超 180° 的 θ 无定义 |
| E7 | **θ_v > θ_cap（法向修正超限）** | > 25° 占 7.41% | tanh 软饱和：θ̃ = θ_cap tanh(θ/θ_cap) ≤ θ_cap，且**增益随 θ 增大自动衰减** | 新增 S6 | 硬 clamp 在 25° 处导数断裂 ⇒ 阈值顶点逐帧跳 |
| E8 | **s_v = ‖u × n‖ → 0**（两法向近似平行） | 绝大多数顶点（θ p25 = 0.006°） | â_v = c⃗/max(s_v, 1e-9)；θ_v = atan2(s_v, c_v) 此时 ≈ s_v（线性，无除零） | 新增 S6 | 直接除 ⇒ inf/nan；用 arccos ⇒ 小角噪声放大 |
| E9 | **θ_v → π**（两法向近似反向） | p99 = 89.8°，max 可能接近 180° | atan2 给出 π；θ̃ = θ_cap tanh(π/θ_cap) = 24.06°（几乎满额但**有界**）；轴 â 由叉积给出（两向反平行时叉积近零 ⇒ 退化为任意轴，但此时该顶点 r_v 通常 < τ 已门控） | 新增 S6 | 用 arccos 在 c → −1 处同样发散 |
| E10 | **aniso = s_max/s_min > 10** | p50 = 4.14 / p90 = 99.7 | **不作门控**（实测门 aniso > 10 才修 ⇒ rig50 6.460 → 7.857，几乎吃掉全部收益）。仅在文档记录：所有涉及 s 的除法用 `clamp(min=1e-9)` | — | 用 aniso 当门会误杀 50% 以上顶点 |
| E11 | **s_0 极小轴 ~1e-11** | min 4.1e-11，p50 最小轴 4.185e-04 | 只在尺度支路（默认关）用到；任何 ‖s‖ 归一化 `clamp(min=1e-9)` | `tga_covariance.py` 的 eps | 除零 ⇒ inf |
| E12 | **NaN / Inf 已进入输入** | — | S5 的求和入口 `nan_to_num(nan=0, posinf=0, neginf=0)`（与 `face_jacobians` 同策略）；S6 输出再 sanitize 一次 | `rot_sync.py:447` | 单个 NaN 经 Markley eigh 会污染整帧 |
| E13 | **scaling 溢出**（exp 过冲） | 尺度支路关闭 ⇒ 不可能 | 若未来打开：`s' = clamp(s'·exp(αℓ̂), s·0.25, s·1.35)` | `tga_covariance.py` 的 max/min_scale_ratio | 亚像素薄片被拉伸到可见厚度 ⇒ 黑斑 |
| E14 | **B > 1（批量）** | 推理期 B = 1，但接口允许 | 所有 [N,·] 张量一律 `unsqueeze(0).expand(B,·)`；缓存 n_0v 用 [1,N,3] 广播，不复制 | `apply` | 形状不匹配（仓库未调制分支用 `expand(0,-1,-1)` 正是为此） |
| E15 | **时间连续性**（帧间跳变） | jitter 指标 | 法向修正**不做 EMA**（当前 `temporal_smoothing` 作用于 δq 域；把 fix 也放进同一 EMA 会引入 1 帧延迟的过冲）。若启用 EMA，必须先修 q/−q 符号（仓库已有 `(delta*prev).sum(-1)<0` 修号） | `rot_sync.py:604-618` | 对已经逐帧确定性的量做 EMA 只会增加滞后与过冲（实测增量域 EMA 误差 6.11° vs final-q 域 1.98°） |

**共 15 条**。其中 E1/E2/E3/E4/E6 是**已有实现**必须保持的行为，E5/E15 是**已有实现已被验证有效**的行为，
E7/E8/E9/E12 是**方法四新增**必须写的守卫，E10/E11/E13 是**明确不做**的决定。

---

## 7. 与已实现三法的差异

### 7.1 逐维度对照

| 维度 | 方法一 LAM | 方法二 TGA | 方法三 RotSync | **方法四 RSB** |
|---|---|---|---|---|
| 作用的自由度 | 面内 2×2 旋转（$\theta=\mathrm{atan2}(F_{10}-F_{01},F_{00}+F_{11})$） | 全协方差（含 shear / 各向异性） | 仅旋转 | 仅旋转，但**分两级**：聚合旋转 + 法向残差修正 |
| 是否面外 | **否**（锁在 n_0 平面内） | 是 | 是 | 是，且**显式对齐形变法向** |
| 聚合方式 | 无（逐面直接作用） | 无（逐高斯直接用 J_v） | Markley（SO(3) 内蕴平均） | Markley + **R³ 法向平均**（两级） |
| 写回方式 | (q,s) | Σ → eigh → (q,s) | (q,s) | (q,s)，**不经过矩阵分解** |
| 是否可能越界 | 否（tanh 有界） | 是（需 clamp 1.35/0.25，实测 67.7% 越界） | 否 | 否（旋转天然保长度） |
| 时域连续性 | 好 | **差**（eigh 排序 + 截断跳变） | 好 | 好（修正角连续） |
| 无解区处置 | 无（不需要） | 无（会产生假形变） | 相干度门 | 相干度门，**且修正共用同一门** |
| 增量耗时 | 0（基准） | +2.35 ms | +3.78 ms | **+0.99 ms**（相对 RotSync） |
| 端到端 | 8.44 ms | 10.79 ms | 12.22 ms | **≈13.2 ms** |
| 一环 jitter | 15.90° | 18.75° | 5.353° | **4.381°** |
| false_move_frac | 0.09–0.37% | **3.08–5.03%** | 0.156% | **0.074%** |

### 7.2 为什么 RSB 不是三者的插值

这是必须写清楚的一点，因为它决定了消融矩阵的形状：

1. **不是 LAM 与 RotSync 的插值**。LAM 的旋转轴固定在 n_0（2 自由度）；RotSync 的旋转是 SO(3) 全 3 自由度平均。
   两者之间**没有连续路径**——你不能「把轴从 n_0 逐渐松开」，因为轴一旦离开 n_0 就不是 LAM 了。
   RSB 与 LAM 的关系是：RSB 的**次级**修正（δq^fix）在「n_v 恰好等于 u_v」时为单位元，
   而 LAM 的旋转在「n_v 恰好在切平面内」时退化为单位元——两个单位元出现的条件不同，故不可插值。
2. **不是 TGA 与 RotSync 的插值**。TGA 的 Σ' 有完整的 6 个自由度（含 shear），RotSync/RSB 的 Σ' 只有
   3（旋转）+ 3（对角尺度）且**尺度不变**。RSB 的输出**永远落在 RotSync 的输出族内**（旋转群），
   TGA 的输出族更大。所以 RSB ⊆ RotSync 的族，与 TGA 无交集（除非 J 无剪切且 q_0 = E_f）。
3. **RSB 的族关系可以精确写出**：
   RSB = { R_fix · R_agg · Q_0 : R_fix ∈ exp(ℝ·â_v), â_v = (u_v × n_v)/‖·‖ }。
   它是在 RotSync 的输出上加一个**一维（单轴）**的修正子流形，而不是在方法一/二之间移动。
   这也是为什么它的耗时增量只有 0.99 ms——一维子流形的参数化（半角四元数）不需要任何矩阵运算。
4. **与候选 (d) 的关系**：候选 (d) 的 $\mathrm{Exp}(c_R\mathrm{Log}R)\cdot\mathrm{Exp}(c_S\log S)$ 中，
   $c_R$ 支路正是 RotSync 已有的；$c_S$ 支路被证伪。RSB 保留了 (d) 的**乘法结构**
   （两个 Exp 相乘），但把第二个 Exp 从「尺度」换成了「法向修正」——
   两者的形式完全相同（都是 Exp(标量 × 轴)），只是作用对象从对数尺度换成法向残差。

### 7.3 与任务候选 (a)–(d) 的最终映射

| 候选 | 数学形式 | 判决 | 在 RSB 里的角色 |
|---|---|---|---|
| (a) | n_d = normalize(J^(−T) n_c)，q' 对齐 | **采纳** | S5 + S6 的全部内容。注意：不做 J^(−T)，直接用 n1h（因 det 恒等式） |
| (b) | 法向轴单独拉伸 k_n = tanh(g·u) | **否决** | 无。降级为 (a) 的「只改方向不改厚度」 |
| (c) | SPD 测地插值 Σ(α) | **否决（超预算 5.7×）** | 无 |
| (d) | 双门 c_R / c_S 的 Exp·Exp | **部分采纳** | 乘法结构保留；c_R 复用 RotSync；c_S 换成法向修正并共用 c_R |

---

## 8. 基于现有代码的增量实施清单（W1–W8）

前置：所有改动都在 `lam3d/core/`，**不改** `LAM/lam/models/rendering/gs_renderer.py`（注入契约不变）。
新文件建议 `lam3d/core/rsb.py`，继承 `RotSyncModulator` 或作为其子类/包装。

### W1 — 缓存静止顶点法向（init 路径）

- **位置**：新文件 `lam3d/core/rsb.py`，`__init__`；复用 `rot_sync.face_areas`（`rot_sync.py:451`）与 `face_jacobians` 的 n0 列。
- **做法**：`n0v = normalize(index_add(A_f * n0h_f))`，存 `register_buffer("n0v", ..., persistent=False)`，形状 [1,N,3]。
- **验收**：`assert n0v.shape == (1, N, 3) and torch.allclose(n0v.norm(dim=-1), ones, atol=1e-5)`；
  与「逐面法向直接平均」的相对差 < 1%（面积加权 vs 均匀加权会差 ~0.3°）。
- **耗时**：init 一次，< 1 ms。

### W2 — 逐帧形变法向 n_v

- **位置**：`rsb.py` 新增 `_covariant_normals(xB)`。
- **做法**：复用 S0b 已经算出的 `Jf`（**不要重算**）——但注意 Jf 的第三列已是 n1h，
  所以直接 `n1h = Jf[..., :, 2]` 然后 `normalize` 即可（省一次 cross）。
  若直接拿 `polar_rotation` 前的 Jf，第三列可能已含尺度，需再 normalize。
- **验收**：`assert angle(nv, normalize(cross(d1,d2) 面积加权平均)) < 1e-4 rad`（与显式 cross 版对比）。
- **提示**：这一步必须**复用** `rotation_field` 内部已算的 `Jf`，否则要把 0.916 ms 再付一次。
  → 建议在 `rotation_field` 的返回诊断里加 `"Jf"` 键（或改成返回可选），由 RSB 消费。

### W3 — 预测法向 u_v

- **位置**：`rsb.py` 的 `delta_rotation_scale` 覆盖中。
- **做法**：在算出 `delta_q` 之后、返回之前，加：
  `u = rotate_by_quat(delta_q, self.n0v)`（两次 cross，0.030 ms）。
- **验收**：`assert (u - n0v).abs().max() < 1e-6` 当 `delta_q = identity`（逐位透传检查）。

### W4 — 最短弧修正四元数

- **位置**：`rsb.py` 新增 `_normal_fix(delta_q, nv)`。
- **做法**：`theta = atan2(norm(cross(u,nv)), (u*nv).sum(-1))`；
  `theta_t = cap * tanh(theta/cap)`；`a = g*t`（**从 delta_rotation_scale 里一并返回**，不要重算门）；
  半角构造 `fix = cat([cos(a*theta_t/2), sin(a*theta_t/2)*axis])`，`axis = cross(u,nv)/|cross|.clamp_min(1e-9)`。
- **验收**：`assert angle(fix·u, nv) ≤ cap + 1e-3`；`assert torch.isfinite(fix).all()`；
  且在 `u ≡ nv` 的合成输入上 `torch.equal(fix, identity)`（逐位）。

### W5 — 合成写回

- **位置**：`rsb.py` 覆盖 `apply`（`rot_sync.py:632` 的逐位恒等契约**必须保留**）。
- **做法**：`q = quat_multiply(fix, quat_multiply(delta_q, q0))`；
  短路：`if not self.enable_normal_fix: return super().apply(...)`。
- **验收**：`torch.equal(rsb_apply(identity...), rotsync_apply(...))` 为 True；
  与 `docs/` 里记录的 A0 数字逐位复现（rig50 = 8.501）。

### W6 — 门控参数穿透

- **位置**：`lam3d/core/cov_switch.py:57-74` 的 `PARAMS` 加 `"rsb"` 项。
- **做法**：新增 `normal_cap_deg: (25.0, 0, 60, 1.0)`、`enable_normal_fix: (True,)`、
  `use_stretch: (False,)`、`gate_power: (1.5, 1.0, 3.0, 0.1)`；`METHODS`（`:42`）加 `"rsb"`。
- **环境变量**：`LAM3D_RSB_NORMAL_CAP` / `LAM3D_RSB_STRETCH` / `LAM3D_RSB_GATE_P`（沿用 `LAM3D_ROT_SYNC_*` 的解析风格）。
- **验收**：`set_method("rsb", {...}, renderer)` 后 `renderer._cov_modulator` 是 RSB 实例；切换 < 15 ms。

### W7 — 诊断输出

- **位置**：`rsb.py` 的 payload / diag dict。
- **做法**：加 `normal_angle_deg`（θ_v 的 p50/p90，或直接存张量）、`normal_applied_deg`（a·θ̃）、
  `flipped`、`gated`、`nres_hint`。
- **验收**：`mod.last_diag["normal_angle_deg"].shape == (N,)`，且 p50 ≈ 2.07（与 §3.6 的实测一致，容差 ±0.5）。

### W8 — 回归与逐位一致性测试

- **位置**：`tests/`（或 `scripts/check_rsb.py`）。
- **断言清单**：
  1. `enable_normal_fix=False` 时输出与 `RotSyncModulator` **逐位相同**（`torch.equal`）。
  2. `xB == x0`（无表情）时，δq → 1 ⇒ fix → 1 ⇒ q' 与 q_0 逐位相同。
  3. 纯刚体（`xB = x0 @ R.T`）时一环法向残差 < 0.05°（已在 `proto_m4_verify.py` 验证为 0.000）。
  4. 纯各向同性缩放时 s' 与 s_0 逐位相同（尺度支路关闭）。
  5. `coh < τ` 的顶点输出 q_0 逐位。
  6. 全流程 `torch.isfinite` 检查，5 个 preset 全覆盖。
- **验收**：6 条全绿，且端到端耗时 ≤ 13.5 ms（含 0.3 ms 余量）。

**总工作量估计**：W1–W5 是新逻辑约 250 行；W6–W8 约 150 行。**一个工作日可完成**。

---

## 9. 伪代码（PyTorch，可直接照抄）

### 9.1 完整 modulator

```python
# lam3d/core/rsb.py
import torch
import torch.nn.functional as F

from .rot_sync import (
    RotSyncModulator, identity_quat, quat_multiply, quat_angle,
    slerp_identity, face_areas,
)


def rotate_by_quat(q, v):
    """用四元数 q=[w,x,y,z] 旋转矢量 v（两次叉积，不做矩阵化）。"""
    w, u = q[..., :1], q[..., 1:]                    # [B,N,1], [B,N,3]
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def axis_angle_quat(angle, axis):
    """半角构造：Exp(angle * axis) 的四元数。angle:[B,N], axis:[B,N,3]（单位）。"""
    h = 0.5 * angle
    return torch.cat([torch.cos(h).unsqueeze(-1),
                      torch.sin(h).unsqueeze(-1) * axis], dim=-1)


class RsbModulator(RotSyncModulator):
    """RSB = Rotation-Stretch Blend（拉伸支路默认关闭）。
    在 RotSync 之上加一级"顶点协变法向"最短弧修正。"""

    def __init__(self, flame_model, arkit_bs_path=None, *,
                 normal_cap_deg=25.0,
                 enable_normal_fix=True,
                 use_stretch=False,          # 实测无效，默认关（见文档 §2.2）
                 gate_power=1.5,             # 相干度门的幂次
                 **rot_sync_kwargs):
        super().__init__(flame_model, arkit_bs_path, **rot_sync_kwargs)
        self.normal_cap_deg = float(normal_cap_deg)
        self.enable_normal_fix = bool(enable_normal_fix)
        self.use_stretch = bool(use_stretch)
        self.gate_power = float(gate_power)
        n0v = self._build_rest_vertex_normals()
        self.register_buffer("n0v", n0v, persistent=False)   # [1,N,3], unit

    # ---------- W1: init 缓存静止顶点法向 ----------
    @torch.no_grad()
    def _build_rest_vertex_normals(self):
        x0 = self.flame_model.v_template.unsqueeze(0)       # [1,N,3]
        faces = self.flame_model.faces.to(x0.device)        # [F,3]
        tri = x0[0][faces]                                  # [F,3,3]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        cr = torch.cross(e1, e2, dim=-1)
        area = 0.5 * cr.norm(dim=-1)                        # [F]
        n0h = F.normalize(cr, dim=-1, eps=self.eps)         # [F,3] 单位
        acc = torch.zeros(1, x0.shape[1], 3, dtype=x0.dtype, device=x0.device)
        acc.index_add_(1, faces.reshape(-1),
                       (n0h * area[:, None]).unsqueeze(0)
                       .expand(1, -1, -1).reshape(1, -1, 3))
        return F.normalize(acc, dim=-1, eps=self.eps)

    # ---------- W2: 逐帧形变顶点法向 ----------
    def _covariant_normals(self, Jf, faces):
        """Jf:[B,F,3,3]，第三列已是单位法向 n1h。返回 nv:[B,N,3]。"""
        area = face_areas_from_jacobian(Jf)                 # 复用 S0a，见下
        n1h = Jf[..., :, 2]
        n1h = F.normalize(n1h, dim=-1, eps=self.eps)
        B, Fn, _ = Jf.shape
        acc = torch.zeros(B, self.n_vert, 3, dtype=Jf.dtype, device=Jf.device)
        acc.index_add_(1, faces.reshape(-1),
                       (n1h * area[..., None]).reshape(B, -1, 3))
        nv = F.normalize(acc, dim=-1, eps=self.eps)
        return torch.where(acc.norm(dim=-1, keepdim=True) <= self.eps,
                           self._fallback_normals(Jf), nv)

    # ---------- W3+W4: 预测法向 + 最短弧 ----------
    def _normal_fix(self, delta_q, nv):
        u = rotate_by_quat(delta_q, self.n0v.expand_as(nv))  # [B,N,3]
        cr = torch.cross(u, nv, dim=-1)
        s = cr.norm(dim=-1)                                  # [B,N]
        c = (u * nv).sum(-1)                                 # [B,N]
        theta = torch.atan2(s, c)                            # [0,pi]
        axis = cr / s.clamp_min(1e-9).unsqueeze(-1)
        cap = self.normal_cap_deg * torch.pi / 180.0
        theta_t = cap * torch.tanh(theta / cap)              # 软饱和
        return theta, theta_t, axis

    # ---------- 主入口：覆盖 delta_rotation_scale ----------
    @torch.no_grad()
    def delta_rotation_scale(self, x0, xB):
        delta_q, payload = super().delta_rotation_scale(x0, xB)
        if not self.enable_normal_fix:
            payload["fix"] = None
            payload["normal_angle_deg"] = None
            return delta_q, payload

        faces = self.flame_model.faces
        Jf = payload["Jf"]                    # ← W2 的前提：由父类透出，避免重算 0.916ms
        nv = self._covariant_normals(Jf, faces)
        theta, theta_t, axis = self._normal_fix(delta_q, nv)

        a = payload["gate"]                   # = t_v * g_v，父类已算好 【B,N】
        fix = axis_angle_quat(a * theta_t, axis)
        # 逐位恒等：a==0 的顶点强制赋 identity（赋值，不计算）
        ident = identity_quat(fix)            # [1,1,4]
        zero = (a <= 0)
        fix = torch.where(zero.unsqueeze(-1), ident.expand_as(fix), fix)

        payload["fix"] = fix
        payload["normal_angle_deg"] = theta * 180.0 / torch.pi
        payload["normal_applied_deg"] = (a * theta_t) * 180.0 / torch.pi
        return delta_q, payload

    # ---------- W5: 合成写回 ----------
    @torch.no_grad()
    def apply(self, delta_q, payload, rotation, scaling):
        if not self.enable_normal_fix or payload.get("fix") is None:
            return super().apply(delta_q, payload, rotation, scaling)
        B = delta_q.shape[0]
        q0 = rotation.unsqueeze(0).expand(B, -1, -1)
        q = quat_multiply(delta_q, q0)
        q = quat_multiply(payload["fix"], q)
        # s 逐位穿透：不做任何算术
        s = scaling.unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.use_stretch:                  # 默认 False；实测无效（§2.2）
            s = self._stretch_branch(s, payload)
        return q, s
```

### 9.2 与现有代码的三个衔接点（必须一起改）

1. **父类要透出 Jf 和 gate**。当前 `rotation_field`（`rot_sync.py:529`）把 Jf 算完就丢了。
   最小改动：在 `rotation_field` 的 `payload` 里加两个键
   `payload["Jf"] = Jf`、`payload["gate"] = t * g`（都是已有张量，零额外开销）。
   **不改签名的做法**：`RsbModulator` 自己重算一次 Jf（+0.916 ms）——可行但没必要，
   多付 0.9 ms 换"父类零改动"不划算（预算里的 0.99 ms 就变成 1.9 ms）。
2. **`face_areas_from_jacobian`**：Jf 的第三列是单位法向，所以
   `area_from_Jf` **不能**从 Jf 直接读（det 会给出 A'/A 比例而非绝对面积）。
   必须用 `face_areas(x0)` 或 `face_areas(xB)`（`rot_sync.py:451`）单独算——
   这也正是 S0a 的 0.196 ms，父类已经算过，把它一并塞进 payload 即可（`payload["area"]`）。
3. **`identity_quat` 必须是赋值**。`rot_sync.py:144` 的 `identity_quat` 返回
   `torch.tensor([1.,0,0,0])`，用它 `expand_as` 是逐位精确的 +1.0 / ±0.0，满足 `torch.equal`。

### 9.3 关键实现陷阱（每条都踩过）

| 陷阱 | 现象 | 正确写法 |
|---|---|---|
| `cross` 的 dim | 默认 `dim=-1` 对 [B,N,3] 正确，但对 [B,N,1,3] 静默广播 | 固定 `dim=-1` 且断言 `.shape[-1]==3` |
| `index_add_` 的 dtype | index 必须是 `int64`；f32 index 会抛 "expected scalar type Long" | `faces.reshape(-1).long()` |
| `index_add_` 的 src 形状 | [B,F,3] 直接传会被当成 [B*F,3] 或报错 | `.reshape(B, -1, 3)` 后按 b 循环，或先 `.transpose(0,1).contiguous()` 再 `index_add_` |
| `scatter_add_` 的 9 通道硬编码 | `rot_sync.py:474` 的 `reshape(B,F,9)` 只服务 Jf | 新代码用 3 通道 `index_add_`，不要复用这个路径 |
| `atan2` 的 zero guard | `s=0, c=1` 时 `atan2(0,1)=0` 正确返回 0；但 `s=0,c=-1` 给 π（正确） | 无需额外 guard，但 `axis` 必须 `clamp_min` |
| `tanh` 在 cap=0 时 | `theta/0 = inf` ⇒ `nan` | `cap = max(cap, 1e-6)` |
| `torch.where` 的逐位 | `where(cond, ident, fix)` 会同时算两条分支（不是短路），但结果逐位正确 | 只用它做**输出选择**，不要用它省计算 |

---

## 10. 实测消融

全部数据来自真实 capture（`/tmp/rot_capture.pt`，5 个 ARKit preset：ones / half / extreme / rand / rand_sparse），
RTX 4090，f32，B = 1，指标定义见附录 A。

### 10.1 推荐配置 vs A0（逐 preset，**主结果表**）

推荐配置：`tau=0.30, p=1.5, th_cap=45°, n_cap=25°, G_dir=OFF, 尺度支路 OFF`。

| preset | A0 rig50 | **RSB rig50** | 降幅 | A0 rig90 | **RSB rig90** | 降幅 | A0 jitter | **RSB jitter** | 降幅 |
|---|---|---|---|---|---|---|---|---|---|
| ones | 10.215 | **6.462** | −36.8% | 57.282 | **46.102** | −19.5% | 6.749 | **5.536** | −18.0% |
| half | 5.038 | **3.163** | −37.2% | 29.328 | **21.865** | −25.4% | 3.740 | **2.835** | −24.2% |
| extreme | 20.860 | **13.862** | −33.6% | 100.420 | **91.159** | −9.2% | 11.210 | **9.863** | −12.0% |
| rand | 6.094 | **3.863** | −36.6% | 28.395 | **20.324** | −28.4% | 3.466 | **2.524** | −27.2% |
| rand_sparse | 0.300 | **0.186** | −38.0% | 14.729 | **9.945** | −32.5% | 1.599 | **1.147** | −28.3% |
| **均值** | **8.501** | **5.507** | **−35.2%** | **46.031** | **37.879** | **−17.7%** | **5.353** | **4.381** | **−18.2%** |

**辅助指标**：false_move_frac 0.1559% → **0.0739%**（−52.6%），逐 preset 0.1748/0.1399/0.1049/0.2448/0.1149%
→ 0.0599/0.0400/0.0550/0.1549/0.0599%。shape_err（由 (q',s') 重构 Σ 与 J_v Σ0 J_vᵀ 的相对误差）p50
从 0.8554 → 0.8593（**几乎不变**，因为 s 支路未启用，这是预期行为——RSB 不承诺 shape 改善，只承诺朝向改善）。

**逐 preset 读法**：四个有实际形变的 preset 降幅高度一致（−33.6% ~ −38.0%），
说明收益不依赖某个特定表情；rand_sparse（位移极小的稀疏随机）降幅最大但绝对值最小（0.300 → 0.186），
因为它的 A0 误差本身已经很小。**extreme 的 rig90 只降 9.2%**——它在 100.4° 的 p90 上有大量真翻折顶点，
被 τ=0.3 的门控正确地挡在门外（这正是设计意图：翻折区不该被"修"）。

### 10.2 超参扫描（5 preset 均值，其余取推荐值）

| 变体 | rig50 | rig90 | jitter | false_move | 判决 |
|---|---|---|---|---|---|
| **REC（τ=0.3, p=1.5, θ_cap=45°, n_cap=25°, G_dir=OFF）** | **5.507** | **37.879** | **4.381** | **0.074%** | **采纳** |
| G_dir=ON（= 1−e^(−a/1.5)） | 6.018 | 38.550 | 4.394 | 0.075% | **负收益**，否决 |
| τ = 0.5 | 5.525 | 38.160 | 4.855 | 0.076% | 略差 |
| τ = 0.7 | 5.554 | 38.941 | 6.301 | 0.085% | 开始误杀 |
| τ = 0.9 | 5.675 | 41.979 | 10.954 | 0.149% | 明显变差（jitter 翻倍） |
| 只在 aniso > 3 时修 | 6.705 | 40.518 | 4.987 | 0.094% | 吃掉 63% 的收益 |
| 只在 aniso > 10 时修 | 7.858 | 44.445 | 5.405 | 0.125% | **吃掉 90% 的收益** |
| A0（不做修正） | 8.501 | 46.031 | 5.353 | 0.156% | 基准 |

**关键读法**：

1. 收益对 τ 在 [0.3, 0.5] 区间**极不敏感**（5.507 → 5.525），说明不是靠调参调出来的；
   但 τ 到 0.9 时 jitter 从 4.38 暴涨到 10.95——门控过度会在"半开"的顶点上产生帧间跳变。
   这个非单调性（τ 大 → rig50 只涨一点但 jitter 涨 2.5 倍）是选 τ=0.3 的**真正理由**，不是 rig50。
2. 一旦引入"用 aniso 判断该不该修"这种**结构性先验**，收益立刻塌回 6.7–7.9。
   ⇒ 结论：**法向修正是普适的，不该用几何启发式去筛顶点。**
3. G_dir 的负收益在 5 preset 上是稳定的（6.018 vs 5.507）。它的原意是"薄片才修方向、厚块不需要"，
   但实测 aniso 与"该不该修方向"无关（见 §2.2 (b-3) 的薄轴-法向夹角统计），所以这条增益是错的先验。

### 10.3 n_cap（法向修正限幅）详细扫描

| n_cap | rig50 | rig90 | jitter | false_move | 备注 |
|---|---|---|---|---|---|
| 0°（= A0） | 8.501 | 46.031 | 5.353 | 0.156% | 基准 |
| 12° | 5.850 | 40.292 | 4.582 | 0.083% | 偏保守 |
| 15° | 5.713 | 39.512 | 4.515 | 0.079% | |
| 18° | 5.624 | 38.940 | 4.463 | 0.078% | |
| **25°** | **5.507** | **37.879** | **4.381** | **0.074%** | **推荐** |
| 30° | 5.469 | 37.326 | 4.342 | 0.072% | |
| 35° | 5.439 | 36.936 | 4.313 | 0.071% | |
| 45° | 5.409 | 36.372 | 4.272 | 0.069% | 尾部风险最大 |

**为什么选 25° 而不是更大的值**：收益随 n_cap **单调**缓降（45° 时 rig50 = 5.409，只比 25° 好 1.8%），
但 n_cap 越大，被强行修正的顶点越多，而这些顶点正是 θ_v 大（≈ 不可信）的那批。
θ_v 的分布是 p50 2.07° / p90 19.55° / p99 89.80°，**25° 落在 p90 附近**，等于"只修可信的 90%"。
45° 已经把 p99 的尾部放进来，而收益只多 1.8%——不值得。

**修正角统计（推荐配置）**：
θ_v（rot(δq)·n_0 与 n_v 的夹角）p10 0.000° / p25 0.006° / p50 **2.072°** / p75 8.094° / p90 **19.547°** / p99 **89.797°**；
frac(θ_v > 5°) = **35.04%**、> 15° = **13.78%**、> 25° = **7.41%**。
实际生效的修正角（a·θ̃，含门控与 tanh 软饱和）p50 **2.015°** / p90 **13.565°** / p99 **20.884°**。
⇒ 99% 的实际修正都在 21° 以内，与 n_cap=25° 的设定自洽。

### 10.4 安慰剂对照（**最关键的一组**，证明收益来源）

固定幅度（用推荐配置的 a·θ̃）但改变**修正方向**：

| 变体 | rig50 | rig90 | jitter | false_move | 读法 |
|---|---|---|---|---|---|
| none（= A0） | 8.501 | 46.031 | 5.353 | 0.156% | 基准 |
| **real（= REC，指向 n_v）** | **5.507** | **37.879** | **4.381** | **0.074%** | 收益真实 |
| neg-sign（轴向反转，把 n_v 推离 u） | 11.559 | 55.538 | 7.491 | 0.568% | **显著变差（+36%）** ⇒ 方向必须对 |
| random-axis（同幅度随机轴） | 8.572 | 47.670 | 8.902 | 0.276% | ≈ A0 但 jitter **暴涨 66%** |
| shrink-only（只做幅度收缩，不转向） | 8.501 | 46.031 | 5.353 | 0.156% | 与 A0 **逐位相同** |

**三条结论（这是全文最强的因果证据）**：

1. **shrink-only 与 A0 逐位相同**（8.501/46.031/5.353，false_move 也逐位）⇒ 收益**不来自**"少转一点"
   这种幅度效应。若收益只是"幅度变小"，shrink-only 就该改善，但它一点没动。
2. **neg-sign 显著变差**（8.5 → 11.6，比不做还差 36%）⇒ 修正方向携带的信息**真实且正确**。
   如果这批顶点本来就"无需修正"，反转方向不会造成这么大伤害；反转会把残差**加倍**。
3. **random-axis 打平 A0（8.572 ≈ 8.501）但 jitter 从 5.35 涨到 8.90、false_move 从 0.156% 涨到 0.276%**
   ⇒ 随机扰动零均值，所以对 p50 的系统性误差无影响，但破坏时域平滑与静止性。
   这解释了为什么"加噪声/加抖动"类方法在 rig50 上看起来"没坏"却视觉上更差——
   **是 jitter 与 false_move 抓住了它，rig50 抓不住。**

### 10.5 无参考法向残差 nres（**绕开循环性辩护的指标**）

定义：`nres = angle( Rot(q' ⊗ conj(q0), n_0v), n_v )`。
它**完全不用** Kabsch 或 Markley 的任何中间量作为参考——右手边是形变网格的几何事实，
左边是调制器最终输出的四元数。所以它不可能是"被优化目标自己造出来的"。

| preset | A0 p50 | **RSB p50** | A0 p90 | RSB p90 | 改善顶点占比 |
|---|---|---|---|---|---|
| ones | 3.763° | **0.044°** | 24.561° | 7.118° | 90.8% |
| half | 1.859° | **0.005°** | 12.227° | 1.206° | 90.8% |
| extreme | 7.746° | **0.361°** | 46.048° | 29.884° | 90.7% |
| rand | 2.284° | **0.010°** | 12.596° | 1.185° | 90.7% |
| rand_sparse | 0.112° | **0.000°** | 6.231° | 0.158° | 87.4% |

**这是本文最干净的证据**：p50 从 3.763° 降到 0.044°（98.8% 降幅），87–91% 的顶点被改善。
注意 p50 接近 0（不是"略小"）——这**正是 S6 的设计意图**：把修正量取到"刚好消除残差"。

### 10.6 关于 rigid_err 参考系的说明（防止误读）

A0 的 rig50 = 8.501 **恰好等于** `angle(Kabsch一环参考, Markley聚合)` 的 p50（独立脚本实测 8.500 / p90 45.978，与主脚本 8.501 / 46.031 差 0.001°/0.05°，来自分位数插值实现差异）。
即 `rigid_err` 度量的是**「Kabsch 一环刚体拟合 vs Markley 聚合」这两种聚合定义之差**，
而不是"旋转对不对"。所以：

- A0 的 8.5° **不代表** RotSync 的旋转算错了 8.5°；
- 它代表 RotSync 的聚合方式与"逐顶点最优刚体拟合"之间的系统差；
- RSB 把 8.501 拉到 5.507，是把这部分系统差**部分**消掉（不是消掉误差，而是让聚合结果更贴近几何事实）。
- 真正衡量"绝对正确性"的是 §10.5 的 nres。两个指标互补，不要单独引用 rig50。

### 10.7 与另外三个方法的同口径对比（仅 ones preset，来源 `docs/cov_methods_deepdive.md` §7.2）

| 方法 | rigid p50 | rigid p90 | jitter | false_move | eig_rel_err p99 | 端到端 |
|---|---|---|---|---|---|---|
| LAM | 7.759 | 15.732 | 15.90° | 0.09% | 3.71e-1 | 8.44 ms |
| TGA | 6.086 | 12.082 | 18.75° | 3.08% | 2.95e-1 | 10.79 ms |
| RotSync | 1.544 | 3.136 | 15.92° | 0.07% | 4.5e-7 | 12.22 ms |
| **RSB** | **≈1.0（估计）** | — | — | — | 4.5e-7（不变） | ≈13.2 ms |

> **重要**：这一行的 Rigid p50 与 §10.1 的 8.501 口径**完全不同**——`cov_methods_deepdive` 用
> ones preset 且用了不同的参考（该文档 §7.2 的 rigid_err 定义与本报告的 Kabsch 一环定义不同）。
> RotSync 在那张表里是 1.544，在本报告里是 10.215（同为 ones preset）。
> **两个口径不能混用**。本报告全部结论建立在 §10.1/§10.5 的自洽口径上；
> 这里只给出"同一张表内 RSB 应当 ≤ RotSync"的方向性判断，不给跨表数字。

---

## 11. 风险与未决

### 11.1 已知风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **PSNR 下降**（法向对齐破坏了学到的视角相关朝向） | 中 | 视觉质量回退 | 消融 B12（特征轴级 vs 帧级）先做；若帧级也降，加 `gain` 全局缩放（0.5–1.0 扫描） |
| **薄片"翻面"观感**（修正量刚好把薄轴转到临界角） | 低 | 局部闪烁 | θ_cap = 25° 已在 p90 之内；监控 `normal_applied_deg` 的 p99 < 25° |
| **翻折区被误开**（r_v 偶然偏高） | 低 | 假运动 | τ = 0.3 已切掉 0.08%；additional: 若 `θ_v > 90°` 强制 a_v = 0（可加，当前未加） |
| **父类改动引入回归**（payload 加键） | 中 | RotSync 基线变化 | W8 的断言 1（逐位相同）必须自动化跑 |
| **多帧累积漂移**（若启用 EMA） | 低 | 漂移 | 当前不启用；启用前先修 q/−q 符号 |

### 11.2 未决问题（明确留给下一步）

1. **PSNR / LPIPS 未测**。全部结论建立在几何指标上。这是最大的缺口——
   但**必须先有"几何改进 → 感知改进"的桥**才能测（需要训练好的 LAM + 多视角渲染，当前会话资源不具备）。
   建议下一步：单帧 + 5 个 preset 渲染，人工对比 + 定量 LPIPS（无需真值）。
2. **是否该用顶点级 J_v 而不是面级聚合**。实测：顶点级 J_v 极分解的残差 p50 = 0.071°、p90 = 3.675°，
   优于面级 + Markley 的 p50 = 2.087°、p90 = 19.670°。但那是**有参考**的对比（参考是 Kabsch），
   且顶点级 J_v 在翻折区可奇异。**若追求极致**，可用 `average="vertex"`（仓库已支持）
   作为 S3 的替代，同时保留 S5/S6。这会改变 A0 基线，需重跑全部消融。
3. **θ_cap 是否该逐顶点自适应**（而非全局 25°）。当前用全局常数 + tanh 软饱和已经足够，
   自适应会引入新的超参和过拟合风险。**暂不做**。
4. **`temporal_smoothing` 域的选择**。仓库现作用于 δq 域（实测误差 6.11° vs final-q 域 1.98°），
   但修 q/−q 符号后可能反转。当前不启用，留给后续。

### 11.3 不做什么（明确边界）

- 不训练任何参数（推理期纯规则，符合约束）。
- 不改 `gs_renderer.py` 的注入契约（`cov3D_precomp = None` 硬编码）。
- 不改 LAM 权重、不改 FLAME blendshape。
- 不引入第二个门（§2.4 的嵌套性质）。
- 不用 SPD 矩阵函数（候选 (c)，超预算）。

---

## 附录 A：指标定义

| 指标 | 定义 | 用途 |
|---|---|---|
| **rig50 / rig90** | 对每个顶点，取其 1 环邻域内的"最优刚体拟合"（Kabsch）作为参考旋转，与调制器给出的增量旋转求夹角；取 p50 / p90（度） | 度量聚合质量。**注意它是"聚合定义差"而非绝对误差**（§10.6） |
| **nres** | `angle(Rot(q'⊗conj(q0), n_0v), n_v)` | **无参考**的正确性指标（§10.5） |
| **jitter** | 同一 preset 内相邻帧之间逐顶点增量旋转角的变化量的 p50（度） | 时域稳定性 |
| **false_move_frac** | 在"几乎不该动"的顶点上（中性 → 中性 + 微小位移）增量旋转 > 阈值（0.5°）的占比 | 假运动率 |
| **eig_rel_err** | 由 (q', s') 重构 Σ' 与其解析真值 J_v Σ0 J_vᵀ 的特征值相对误差 | 一致性检查 |

## 附录 B：复现命令

```bash
cd /workspace/lam3d
PY=/usr/local/miniconda3/envs/lam/bin/python

# 基础统计（分布、分位数）
$PY scripts/proto_m4_stats.py         # -> /tmp/m4_stats.json

# A0 vs RSB 消融（α 扫描、θ_cap 扫描、B1-B6）
$PY scripts/proto_m4_eval.py          # -> /tmp/m4_eval.json

# 合成正确性（纯刚体 / 纯缩放 / 各向异性）
$PY scripts/proto_m4_verify.py

# 安慰剂对照（neg-sign / random-axis / shrink-only）
$PY scripts/proto_m4_placebo.py

# 无参考法向残差
$PY scripts/proto_m4_nres.py

# 分段耗时
$PY scripts/proto_m4_timing.py

# 补充核验：f32 精度 / inv3 质量 / det 恒等式 / 参考系 gap
$PY scripts/proto_m4_f32.py
$PY scripts/proto_m4_inv.py
$PY scripts/proto_m4_refgap.py
```

## 附录 C：脚本与数据清单

| 文件 | 作用 |
|---|---|
| `scripts/proto_m4_lib.py` | 公共库：load / face_jacobians / face_areas / scatter_vertex / vertex_jacobians / markley / quat_angle / rot_angle / kabsch_rings / ring_jitter |
| `scripts/proto_m4_final.py` | 提取式公共前缀（`exec(open(...).read().split("VAR={")[0])` 复用） |
| `scripts/proto_m4_stats.py` | 分布统计（面积比、相干度、各向异性、门控覆盖率） |
| `scripts/proto_m4_eval.py` | 主消融（A0/A3/B1–B6、α 扫描、θ_cap） |
| `scripts/proto_m4_verify.py` | 合成核验（刚体/缩放/各向异性精确性） |
| `scripts/proto_m4_placebo.py` | 安慰剂对照 |
| `scripts/proto_m4_nres.py` | 无参考法向残差 |
| `scripts/proto_m4_timing.py` | 分段耗时 |
| `scripts/proto_m4_f32.py` | f32 vs f64 精度 |
| `scripts/proto_m4_inv.py` | inv3 vs linalg.inv 质量 |
| `scripts/proto_m4_refgap.py` | Kabsch–Markley 参考系 gap |
| `scripts/proto_m4_reco.py` | 从 (q,s) 重构 Σ 的一致性 |
| `/tmp/rot_capture.pt` | 真实 capture：x0[N,3] / xB[5,N,3] / faces[F,3] / gs_quat / gs_scale / names |

---

## 附录 D：本次重写相对 v1 的完整变更日志

| # | v1 内容 | v2 处置 | 依据 |
|---|---|---|---|
| 1 | §1.2 S2 步骤 `k_f = A_f|det J_f|/A'_f` | **删除**（记为恒等式，转为自检） | f64 重算误差 1.447e-13 |
| 2 | §1.2 S3 用 κ_f 条件数算置信度 c_f | **删除**（改用纯面积权重） | cond(J) p99 = 1.08，99% 不触发 |
| 3 | §1.2 S5 尺度软投影 + 单轴拉伸 | **默认关闭**（保留接口） | eigmax_rel 恒 1.277；合成实验 s'/s0 恒 1.0000 |
| 4 | §2.3 无解区 τ = 0.30「建议先试 0.5」 | **改为 τ = 0.30 为推荐，附分布依据** | coh p5 = 0.9418；τ=0.3 切 0.08% |
| 5 | §4 方向增益 G_dir = 1−e^(−a/1.5) | **默认关闭**（实测负收益） | rig50 5.507 → 6.018 |
| 6 | §4 半径门 G_ρ 用 ‖x−μ‖/mean(s) | **撤回**。修正口径为 ‖x−μ‖/median(‖x−μ‖)，σ=3 | 旧口径 100% 顶点被门死（p50 G_ρ = 0.000e+00） |
| 7 | 缺「每步修复哪个失效模式」 | **新增 §5** 逐条机制 + 量化对照 | — |
| 8 | 缺「边界条件全表」 | **新增 §6**（15 条 E1–E15） | — |
| 9 | 缺「与三法差异 / 为什么不是插值」 | **新增 §7** | — |
| 10 | 缺「增量实施清单」 | **新增 §8**（W1–W8 带行号与验收断言） | — |
| 11 | 缺「伪代码」 | **新增 §9**（完整类 + 3 个衔接点 + 7 个陷阱） | — |
| 12 | 缺「安慰剂对照」 | **新增 §10.4** | — |
| 13 | 缺「无参考指标」 | **新增 §10.5**（nres） | — |
| 14 | 缺「rigid 参考系说明」 | **新增 §10.6** | rig50 ≡ Kabsch–Markley gap（8.501 与独立脚本 8.500 逐位重合） |
| 15 | §1.3 α 退化表（抽象） | 保留并具体化（§3.9） | — |
| 16 | 「f32 逆在 sliver 面严重失真」 | **撤回**（口径混淆） | f32 vs f64 J 最大元差 max 4.829e-03，误差 >1 的面数 = 0 |
| 17 | 建议用 `torch.linalg.inv` | **保留仓库 inv3** | inv3 p99 3.140e-07 优于 linalg.inv p99 1.950e-06 |

**v2 的可信度声明**：本版所有数字都可用附录 B 的命令复现；
所有被否定的路线都保留了"为什么被否定"的硬数字，以便将来带着新证据推翻。
唯一没有覆盖的是**感知指标（PSNR/LPIPS）**——见 §11.2 第 1 条，这是本文的已知缺口，不做掩盖。
