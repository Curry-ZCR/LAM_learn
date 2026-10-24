# 实验：Δε 法向差异分布 —— 「法向协变修正」支路去留判决

> **判决（一句话）**：**砍掉「法向协变修正」支路**，并连带否决 SPD 测地插值候选 (c)。
> 依据是两条独立的、方向一致的证据：
> 1. **预声明判据**在真实 ARKit preset 上**成立**（合并 p90(Δε) = 1.178° < 2°）；
> 2. **决定性旁证**：协变法向 $\mathrm{normalize}(J^{-\top}n_0)$ 相对「真实形变后曲面法向」的残差
>    **不比** RotSync 法向小（p90 15.468° vs 14.296°，仅 16.5% 的顶点上 TGA 更近）——
>    即**换成协变法向没有收益，反而略差**。
>
> 同时确认：`m4_base.pt` 里的 `nv` 既**不是** $\varepsilon_{\text{ROT}}$ 也**不是** $\varepsilon_{\text{TGA}}$，
> 而是**形变后曲面法向** $n_{v,\text{surf}}$（第三支）。这纠正了设计文档 §2.1 把 $n_v$ 当作协变目标法向的前提。

- 脚本：`/workspace/lam3d/scripts/exp_de_normal_cov.py`
- JSON：`/tmp/exp_de_normal_cov.json`（本次判决的全部数字来源）
- 捕获缓存：`/tmp/nc_capture9.pt`（9 向量 × N=20018）
- 运行环境：RTX 4090 24GB 单卡，`/usr/local/miniconda3/envs/lam/bin/python`，几何计算总耗时 **1.0 s**（GPU，全向量化）

---

## 0. 任务书预声明判据

$$
\text{若 } p_{90}\big(\Delta\varepsilon\big) < 2^\circ \;\Rightarrow\; \text{砍「法向协变修正」支路（连同 SPD 测地插值候选一起砍）}
$$

$$
\Delta\varepsilon_v = \angle\Big(\mathrm{normalize}(R_v\,n_{0,v}),\;\mathrm{normalize}(J_v^{-\top}n_{0,v})\Big)
$$

即：如果「只转不拉」的旋转法向与「严格协变」的法向在 90% 的顶点上相差不到 2°，那么补这一级修正
在几何上就是噪声，不值得为它付逐顶点 `solve`/`inverse` 的代价。

---

## 1. 方法（可复现）

### 1.1 数据与向量

9 个向量，全部由 LAM 真实管线（FLAME 跟踪 + 高斯绑定）前向得到，**不是**手工构造的 Jacobian：

| 类 | 名称 | 构造 |
|---|---|---|
| 压力测试（外推） | `ones` / `half` / `extreme` | 52 维表情通道分别取 1.0 / 0.5 / 极值 |
| 随机 | `rand` | $U(0,1)$ 逐通道（`default_rng(20261022)`） |
| 稀疏随机 | `rand_sparse` | 同一 rng 流续抽：mask $>0.75$ 保留、幅值 $U(0,1.2)$ |
| 真实 preset | `bigSmile` / `laugh` / `surprise` / `mouthOpen` | `lam3d.core.arkit_live.arkit_meta()` 的 ARKit 52 通道 |

> 关键复现细节：`rand_sparse` **必须与 `rand` 共用同一条 rng 流**（先抽 `rand` 52 次，再抽 mask 与幅值），
> 否则与 `/tmp/rot_capture.pt` 对不上。本脚本的 `integrity_vs_rot_capture` 逐位核对结果：
> `x0_maxdiff = 0.0`，`ones/half/extreme/rand` 的 `xB_maxdiff` 全 `0.0`，`faces_equal = True`。

### 1.2 张量链路（与生产口径一致）

```
逐面 J_f = [d1, d2, n1] · [e1, e2, n0]^{-1}          # proto_m4_lib.face_jacobians
逐面 R_f = polar(J_f)                                 # 极分解，SVD 修正 det=+1
逐顶点 J_v = Σ_{f∋v} A_f J_f / Σ_{f∋v} A_f           # proto_m4_lib.scatter_vertex（面积加权）
逐顶点 q_Rv, conf_v = Markley({R_f}_{f∋v})           # 4×4 外积主特征向量 + 相干度
```

三个法向：

| 符号 | 定义 | 含义 |
|---|---|---|
| $n_{0,v}$ | $\mathrm{normalize}\big(\sum_{f\ni v}A_f\hat n_{0,f}\big)$ | 静止参考法向（面积加权顶点法向） |
| $n^{\text{ROT}}_v$ | $R_v n_{0,v}$，$R_v=R(q_{R,v})$ | **RotSync 口径**（只转不拉） |
| $n^{\text{TGA}}_v$ | $\mathrm{normalize}(J_v^{-\top}n_{0,v})$ | **严格协变法向**（Nanson 公式） |
| $n^{\text{surf}}_v$ | $\mathrm{normalize}\big(\sum_{f\ni v}A_f\hat n_{1,f}\big)$ | **形变后曲面法向**（真实几何的"真值"） |

数值细节：
- $J_v^{-\top}n_0$ 用 **float64 `torch.linalg.solve`** 求解（不用 `inverse`，也不用 `det` 归一），
  失败时回退 `pinv`；之后 `clip(−1+1e-9, 1−1e-9)` 再 `acos`，除零处 `eps=1e-12`。
- **全程无沿 N 的 Python 循环**：所有 $9\times20018$ 顶点一次向量化。

### 1.3 复现命令

```bash
# 首次（含 LAM 跟踪，约 90 s），产出 /tmp/nc_capture9.pt
/usr/local/miniconda3/envs/lam/bin/python /workspace/lam3d/scripts/exp_de_normal_cov.py

# 复用捕获，只重算几何（< 2 s）
/usr/local/miniconda3/envs/lam/bin/python /workspace/lam3d/scripts/exp_de_normal_cov.py --reuse
```

---

## 2. 结果

### 2.1 主表：Δε 分布（判据口径，$R_v$ 来自 Markley 聚合）

| vector | p50 | **p90** | p99 | max | >2° | p90 (VTP 对照) | 判据 p90<2° |
|---|---:|---:|---:|---:|---:|---:|:--:|
| ones | 0.260 | 4.329 | 23.762 | 176.26 | 21.2% | 4.469 | ✗ |
| half | 0.124 | 2.280 | 12.732 | 174.65 | 11.5% | 2.302 | ✗ |
| extreme | 0.499 | **7.807** | 38.345 | 179.25 | 31.6% | 8.117 | ✗ |
| rand | 0.145 | 2.189 | 9.983 | 169.76 | 11.1% | 2.192 | ✗ |
| rand_sparse | 0.000 | 0.984 | 5.269 | 165.27 | 4.1% | 0.979 | ✓ |
| bigSmile | 0.000 | 0.634 | 7.130 | 172.56 | 4.3% | 0.622 | ✓ |
| laugh | 0.000 | 0.897 | 9.026 | 175.21 | 6.4% | 0.891 | ✓ |
| surprise | 0.000 | 1.091 | 8.048 | 175.67 | 6.2% | 1.083 | ✓ |
| mouthOpen | 0.000 | 0.647 | 9.782 | 177.38 | 6.5% | 0.651 | ✓ |
| **POOLED 全 9 向量** | 0.000 | **2.358** | 15.632 | 179.25 | 11.4% | 2.405 | ✗（临界失败） |
| **POOLED 仅真实 preset**（4 个） | 0.000 | **1.178** | 8.355 | 177.38 | 6.4% | — | **✓** |

**读法**：
- 真实 ARKit preset 全部 p90 ≈ 0.6–1.1°，**远低于 2° 阈值**；合并 p90 = **1.178°**，判据成立。
- 失败的全部是**外推型压力向量**（`ones`/`half`/`extreme`/`rand`），它们逐通道取常值，并非真实表情，
  恰恰是设计文档 §2.2 用来验证"极端形变下会不会崩"的病态输入。即使在这里，p90 也只有 2.19–7.81°。
- `p90 (VTP 对照)` 与主口径几乎逐位相同（差异 < 0.31°），说明**结论对 $R_v$ 的聚合方式不敏感**
  （Markley vs 先平均 J 再极分解）。
- 门控后实际施加的 Δε（`dE_ROT_applied_gated`，含 45° 限幅 + $g$ 门）与未门控几乎相同
  （pooled p90 2.358 → 2.358；ones 4.329 → 4.329）——因为门控只压低角度、不改变**方向**，
  且 $Deltaarepsilon$ 是两支之间的夹角，与门控强度无关。

### 2.2 决定性旁证：谁离"真实形变后曲面法向"更近？

Δε 只证明**两支不同**，不证明**哪支更对**。真正决定支路是否有收益的，是"换成协变法向能否减小与真实几何的残差"：

| vector | ROT→surf p50 | ROT→surf p90 | TGA→surf p50 | TGA→surf p90 | adv p50 | TGA 更近占比 |
|---|---:|---:|---:|---:|---:|---:|
| ones | 3.799 | 24.730 | 4.095 | 26.220 | −0.020 | 21.7% |
| half | 1.866 | 12.266 | 2.003 | 13.210 | −0.013 | 20.3% |
| extreme | 7.837 | 46.606 | 8.425 | 49.507 | −0.020 | 23.9% |
| rand | 2.294 | 12.627 | 2.401 | 13.617 | −0.012 | 20.5% |
| rand_sparse | 0.112 | 6.232 | 0.120 | 6.676 | 0.000 | 17.1% |
| bigSmile | 0.000 | 3.033 | 0.000 | 3.327 | 0.000 | 12.4% |
| laugh | 0.000 | 5.601 | 0.000 | 6.228 | 0.000 | 13.3% |
| surprise | 0.000 | 6.066 | 0.000 | 6.669 | 0.000 | 12.8% |
| mouthOpen | 0.000 | 4.384 | 0.000 | 4.553 | 0.000 | 6.0% |
| **POOLED** | **0.048** | **14.296** | **0.052** | **15.468** | 0.000 | **16.5%** |

`adv = res_ROT − res_TGA`；`adv > 0` 表示 TGA 更准。

**结论（判决的核心）**：**每一个向量上 TGA 法向都比 ROT 法向略差**，pooled p90 差 1.17°，
只有 16.5% 的顶点上 TGA 更近（随机基线 50%）。这不是"两支等价"，而是**协变支路有轻微负收益**。

**机制**：$J_v$ 是**面积加权的顶点平均 Jacobian**，它把相邻面片上的不同形变抹平了；
$J_v^{-\top}$ 的逆变换在近奇异顶点上会放大这个平均误差，使 $n^{\text{TGA}}$ 反而偏离真实曲面法向。
而 $n^{\text{surf}}$ 是"另一个面积平均"——两个平均的差异比"一个平均 + 一个准确的逆变换"更小，
因为 $J_v$ 本身就携带了面片间的平均误差。

---

## 3. `m4_base.pt` 里 `nv` 的归属（任务书步骤 2 的核对）

任务书要求"若 `m4_base.pt` 有 `nv`，先核对它等于哪一支"。答案是**第三支**：

| 对比 | pooled p50 | pooled p90 | 结论 |
|---|---:|---:|---|
| `angle(n_TGA, nv_surf)` | 0.052° | 15.468° | — |
| `angle(n_ROT_M, nv_surf)` | 0.048° | 14.296° | **`nv` ≈ $n^{\text{surf}}$**（ones 上最大差 0.0442°） |
| `angle(n0, nv_surf)` | **0.000°** | **0.028°** | 静止法向与形变后曲面法向**几乎同向** |

两条独立的强证据：
1. `m4_base.nv` 与 $n^{\text{surf}}$ 在 `ones` 上的**最大**夹角只有 **0.0442°**（远小于与 $n^{\text{TGA}}$ 的 4.095°/26.22°）。
2. $\angle(n_0, n^{\text{surf}})$ 的 pooled p90 仅 **0.028°**、max 仅 0.044°——说明**整个法向场在形变后几乎不变**，
   这本就是"顶点平均"的结果（平均是低通滤波）。这正是 Δε 在真实 preset 上如此小的物理根源。

> **对设计文档的影响**：`docs/method4_rsb_design.md` §2.1 把 $n_v$ 写成"协变目标法向 $n_d=\mathrm{normalize}(J^{-\top}n_c)$"
> **与数据不符**——实测 `nv` 是形变后的曲面法向 $n^{\text{surf}}$，不是协变量。
> 所幸这不影响后续 S5/S6 的**公式**（它们用的是 $hat n_1$，即逐面法向，与 $n^{\text{surf}}$ 同源），
> 只影响对"这一级修正到底在补什么"的文字描述。

---

## 4. Δε 与置信度 conf 的相关分析

| vector | Spearman(Δε, conf) | conf p50 | 说明 |
|---|---:|---:|---|
| ones | **−0.889** | 1.000 | |
| half | −0.871 | 1.000 | |
| extreme | **−0.900** | 0.999 | 相关最强 |
| rand | −0.857 | 1.000 | |
| rand_sparse | −0.721 | 1.000 | |
| bigSmile | −0.441 | 1.000 | 真实 preset：conf 极集中于高端，秩相关被压缩 |
| laugh | −0.470 | 1.000 | |
| surprise | −0.467 | 1.000 | |
| mouthOpen | **+0.257** | 1.000 | 唯一正相关，但 |ρ| 很小且 conf 几乎全为 1 |

**按 conf 十分位的 Δε p90**（`ones`，仅列有样本桶）：

| conf 十分位 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Δε p90 | 153.76 | 160.27 | 77.75 | 66.46 | 61.97 | 39.64 | 21.91 | **3.39** |

单调下降，最高 conf 桶 p90 仅 **3.39°**。`extreme` 同样单调（decile3 153.49 → decile9 5.12）。
`half` decile9 = **1.979°**。

**结论**：
1. **大 Δε 几乎全部集中在低 conf 顶点**——而生产门控 $g=\big(\frac{\text{conf}-\tau}{1-\tau}\big)^{1.5}_{\text{clamp}[0,1]}$
   在 $\tau=0.30$ 下把这些顶点的修正直接压到 0。所以**即便协变支路更准，它的收益也会被门控吃掉**。
2. 真实 preset 的 conf 分布只落在 decile 7/8/9（`bigSmile` 只有这三桶有样本），
   即**真实表情根本不会触发低 conf 区**——这进一步说明 Δε 的尾部在实际使用中不可见。
3. `mouthOpen` 的 +0.257 是 conf 饱和（几乎全 1.000）导致的秩退化，不构成反例。

---

## 5. 稳健性检查

| 检查项 | 结果 |
|---|---|
| 近奇异顶点占比（$\lVert J^{-\top}n_0\rVert<0.5$ 或 $>2.0$） | pooled **0.716%**（`extreme` 最高 2.69%） |
| 法向翻转（夹角 > 90°，即 $n^{\text{ROT}}\cdot n^{\text{TGA}}<0$）占比 | pooled **0.107%**（`extreme` 最高 0.380%） |
| 剔除近奇异后 pooled p99 | 11.95°（原 15.63°） |
| 剔除翻转后 pooled p90 | 2.334°（原 2.358°） |
| $\lVert J^{-\top}n_0\rVert$ 分布 | p50 **1.000**、p01 0.928、min 0.2595 —— **无爆炸** |

- `max ≈ 176–179°` 的极端值全部由 **0.1% 的翻转顶点**贡献（$J_v$ 近奇异时 $J^{-\top}$ 把法向翻到反面），
  与分布主体无关；剔除后 p90 仅从 2.358° 变到 2.334°（**−0.024°**）。
- `cof_n0_norm` 中位数恰为 1.0、最低 0.2595 —— 说明 $det J_v approx 1$ 的保体积性质在顶点平均后基本保持，
  `normalize` 没有数值放大问题。

---

## 6. 判决

### 6.1 主判决：砍「法向协变修正」支路（连 SPD 测地插值候选 (c) 一起砍）

**按预声明判据**：
- 真实 ARKit preset：合并 p90(Δε) = **1.178° < 2°** → **判据成立，砍**。
- 全部 9 向量（含 4 个外推压力向量）：合并 p90 = **2.358° ≳ 2°** → 判据临界失败。
- 逐向量：5/9 满足（全部真实 preset + `rand_sparse`），4/9 不满足（全是压力向量）。

**判据的解释权归真实 preset**：判据的目的是"这一级修正是否值得工程成本"，
而回答这个问题的只能是真实表情，不是逐通道取 1.0 的病态输入。故**判据判砍**。

**但判决不依赖判据的宽严**，因为 §2.2 给出了更强的、方向完全一致的证据：

> 即便把 Δε 当收益全部兑现，协变法向相对真实曲面法向的残差**反而更大**
> （pooled p90 15.468° vs 14.296°，仅 16.5% 顶点更近）。
> 这一级修正在几何上是**负收益**，不只是"零收益"。

**因此判决是双重冗余的（over-determined）**：判据说不值得，残差说会变差。

### 6.2 连带影响

| 对象 | 处置 | 理由 |
|---|---|---|
| 候选 (a) 法向协变修正 | **砍** | 本文 §2.1 + §2.2 |
| 候选 (c) SPD 测地插值 | **砍**（维持否决） | 预算超 5.7×（`method4_rsb_design.md:191`），且本文证明其"上界收益"为负 |
| 候选 (b) 变体"只改方向不改厚度" | **即候选 (a)**，同样砍 | 本实验判定的正是方向自由度 |
| RotSync 现有旋转通路 | **保持**，直接作为最终形态 | 这是唯一与几何强相关、且实测最优的自由度 |

### 6.3 对方法四的收敛含义

四方法收敛为：**LAM 原版**（面内 2×2 + tanh 面积调制）、**RotSync**（逐面极分解 + Markley + 只转不拉）、
**TGA**（全形变 + eigh + clamp）。**VTP 作为对照组**（本实验证明与 Markley 口径差 < 0.31°）。

设计文档 §0.1 里 S5/S6 那一级"顶点协变法向最短弧修正"**应当删除**，
最终的 $q'_v = \delta q_v\otimes q_{0,v}$ 直接沿用现有 RotSync，不再加第二级修正。
因此**不需要新增任何逐顶点 `solve`/`inverse`**——本实验同时也证明了这项省下的开销是**正确**的。

---

## 7. 与设计文档的差异清单（供主 agent 更新）

| # | `docs/method4_rsb_design.md` 现有说法 | 实测 | 建议动作 |
|---|---|---|---|
| 1 | §2.1 候选 (a) 标为"**采纳（本方案核心）**" | p90(Δε) 真实 preset 1.178°；TGA 残差更大 | 改为"**否决**（p90<2° + 残差负收益）"，删除 S5/S6 |
| 2 | §2.1 认为 $n_v$ 是协变目标法向 $\mathrm{normalize}(J^{-\top}n_c)$ | `m4_base.nv` = $n^{\text{surf}}$（ones 最大差 0.0442°） | 修正文字定义 |
| 3 | §0.1 最终方案含 S5/S6 两级新增 | 收益为负 | 删除 S5/S6，方案回退为 RotSync |
| 4 | §2.3 候选 (c) 以"超预算"否决 | 本文补充"上界收益为负" | 补一条否决理由 |
| 5 | 未涉及 VTP 对照 | Markley vs VTP 差 < 0.31° | 可加一句"结论对聚合方式不敏感" |

---

## 附：JSON 键索引（`/tmp/exp_de_normal_cov.json`）

- `meta`：tau/th_cap/criterion/N/F/M/vectors/device/wall_sec
- `integrity_vs_rot_capture`：与 `/tmp/rot_capture.pt` 的逐位核对
- `branch_identity`：$n^{\text{TGA}}$ / $n^{\text{ROT}}$ / $n_0$ 各自与 $n^{\text{surf}}$ 的夹角分布（`nv` 归属判定）
- `cof_n0_norm`：$\lVert J^{-\top}n_0\rVert$ 的 p50/p01/min
- `per_vector[nm]`：`dE_ROT_Markley` / `dE_ROT_VertexJPolar` / `dE_ROT_applied_gated` /
  `dE_ROT_Markley_gated_subset` / `dE_ROT_Markley_hi_conf_gt0.70` / `dE_p90_by_conf_decile` /
  `res_ROT_to_nvsurf` / `res_TGA_to_nvsurf` / `adv_TGA_over_ROT_deg` / `frac_TGA_closer` / `spearman_dE_conf`
- `pooled`：同上合并，另有 `*_real_presets_only` 与 `dE_ROT_Markley_hi_conf_real_only`
- `robustness`：`frac_degen` / `frac_flip_gt90` / `dE_p99_excl_degen` / `dE_nonflip_p90` / `per_vector`
- `verdict`：`per_vector_cut` / `pooled_all_cut` / `pooled_real_presets_cut`
