# LAM 单图数字人：四种 ARKit 协方差调制方法深度技术对比

> 对象：`/workspace/lam3d/lam3d/core/` 下四种调制器，由 `cov_switch.py` 的 `METHODS` 定义（`lam` / `tga` / `rot_sync` / `vertex_then_polar`）。
> 基线 LAM（arXiv:2502.17796），N = 20018 个规范高斯，F = 39904 个 Loop 细分面；不改 `/workspace/LAM/` 下任何文件，全部从 `renderer._cov_modulator` 注入。
> 行号来自本次逐字阅读的源码快照；实测数字注明出处文件，未实测处标「未核实 / 推定」。

## 1. 总览

### 1.1 四方法对比

| 维度 | `lam`（LAM 原版） | `tga`（TGA 雅可比） | `rot_sync`（旋转同步） | `vertex_then_polar`（对照） |
|---|---|---|---|---|
| 定义 | `covariance_modulation.py:224` | `tga_covariance.py:204` | `rot_sync.py:337`，`average='face'` | 同一类，`average='vertex'` |
| 输入依赖 | 逐面**面内** 2x2 标架（rest 边对 + rest 法向），完全不用形变后法向 | 逐面 3x3 `J = Ẽ E⁻¹`，第三列取**单位**法向 | 逐面 `polar(J_f)` 的旋转因子，再在 SO(3) 上按面积加权聚合到顶点 | 面积加权平均后的 `J_v`，再做一次极分解 |
| 改什么量 | 绕 rest 法向的面内转角 θ + **各向同性**面积缩放 | 完整 `Σ' = J Σ0 Jᵀ`：朝向 + 逐轴拉伸/剪切 | `q' = normalize(δq ⊗ q0)`，`s' = s0` | 同上，只换 δq 的来源 |
| 保轴长 | 否，各向同性，`s'/s0 ∈ [0.65, 1.35]` | 否，逐轴 ratio 夹在 `[0.25, 1.35]` | **是**，`s'` 与 `s0` 逐位相等 | **是**（同一 `apply` 通道） |
| 保 PSD | 是（旋转 + 正标量缩放） | 是（`J Σ Jᵀ` 天然 PSD，另显式对称化） | 是（`R Σ0 Rᵀ` 特征值逐位不变） | 是 |
| 端到端耗时 | **5.07 ms**（增量 3.63） | **7.56 ms**（6.13） | **11.20 ms**（9.77） | 未单独实测（推定 6~7 ms，§7.3） |
| 关键风险 | 丢面外倾角、丢方向性剪切、「橡皮感」 | 法向噪声直通、eigh 近简并、夹比破坏精确性 | P 被丢弃；0.2%~0.4% 顶点数学上无解 | 平均 J 近奇异 → R 跳变 |
| 前端参数 | 无 | 3 项 | 4 项 | 2 项（无 deadzone / coherence） |

耗时全部来自 `docs/rot_sync_plan.md:231-238`（`scripts/bench_rot_sync.py`，WARM=6 / REP=30，RTX 4090）。

### 1.2 四者在同一条调用链上的位置

调用点是 LAM 里**冻结**的三行（`gs_renderer.py:668-674`）：

    delta_q, stretch = modulator.delta_rotation_scale(ret["animated"], mean_3d)
    gs_rotation, gs_scaling = modulator.apply(
        delta_q, modulator.delta_scale(stretch, gs_attr.scaling),
        gs_attr.rotation, gs_attr.scaling)

| # | 步骤 | 行号 | 说明 |
|---|---|---|---|
| 0 | `modulate = cov_modulation and arkit_params is not None and not DynamicCovarianceModulator.is_neutral(arkit_params)` | 624-625 | 唯一短路判据；B=0 时四方法都不被调用 |
| 1 | `prepare_params(arkit_params, num_view, device, dtype)` | 646-647 | → `[num_view, 52]` |
| 2 | `arkit_offsets(arkit)` | 650 | → `[Nv, N, 3]`，一次 GEMM（把 `arkit_bs_up` 的 `[52, N, 3]` reshape 成 `[52, N*3]`） |
| 3 | 第一次 `animation_forward(static_offset=None)` → `x0` | 629-643 | 中性/静止顶点 |
| 4 | 第二次 `animation_forward(dynamic_offset=dynamic_offset)` → `xB` | 651-665 | 同一组 pose，只多 `dynamic_offset` |
| 5 | `delta_rotation_scale(x0, xB)` → `(delta_q, stretch)` | 668 | **四方法的第一个分岔点** |
| 6 | `delta_scale(stretch, gs_attr.scaling)` | 671 | 第二个分岔点（rot_sync 原样透传） |
| 7 | `apply(delta_q, payload, rotation, scaling)` → `(q', s')` | 669-674 | 第三个分岔点 |
| 8 | 逐视角建 `GaussianModel(xyz=..., rotation=gs_rotation[i], scaling=gs_scaling[i], ...)` | 683-690 | 光栅器只吃 `(q, s)` |

**为什么必须沿 (q, s) 回写**：`gs_renderer.py:534` 的 `cov3D_precomp = None` 是硬编码，535-536 行无条件用 `gs.scaling` / `gs.rotation` 覆盖，556-558 行只把 `scales/rotations` 传给光栅器。`scripts/diag_cov_path.py` 实测：(B) 同一 Σ 走 raw rasterizer 与基准差 1.7e-6，(C) 挂成属性走官方 `forward` 与基准**逐位相同**（被丢弃），(C') 走 raw rasterizer `mean|diff| = 0.067`。所以任何协方差层改动最终都要分解回 `(q', s')`。

**为什么两次 animation_forward 只差一个 dynamic_offset**：x0 与 xB 用同一组 `betas/expr/rotation/neck/jaw/eyes/translation`，逐面差分把 LBS 刚体部分约掉；`arkit_live.neutral_flame_data()` 又把 pose 全 `zero_()` 只留 `betas`，未调制分支把 `gs_attr.rotation` 原样 `expand`（`gs_renderer.py:679-680`），基线 B = I，于是 `C = J B⁻¹ = J` —— 「全局姿态不能重复计算」在这条链上自动成立。

### 1.3 切换层（`cov_switch.py`，208 行）

`METHODS`（`42-47`）只放下拉框短名，长说明放 `DESC`（`49-54`）当 `title`；`_STATE`（`76-77`）默认 `rot_sync`。`_make()`（`80-107`）是唯一构造点，`rot_sync` 与 `vertex_then_polar` **共用同一分支**，只差一行（`92-93`）：

    elif mod_name in ("rot_sync", "vertex_then_polar"):
        avg = "face" if mod_name == "rot_sync" else "vertex"

`set_method`（`142-158`）先按 `PARAMS` 填默认再 `merged.update(params)`，**重建实例整体替换**（`154`：`r._cov_modulator = mod   # 契约允许的唯一注入点`），不原地改参数——rot_sync 有 EMA 状态 `_prev_delta`（`399`）。`rebind_session`（`175-198`）复用 live 会话，`183-184` 注释写明「B=0 静态基准无需重算：ARKit 全 0 时 is_neutral 短路，四种方法的 B=0 载荷逐位相同」，切方法只要 3~12 ms 而非重新生成 60~90 s。`register`（`161-172`）构造失败时回退 LAM 原调制器。

## 2. 方法一 `lam`：LAM 自带的各向同性调制器

### 2.1 数学定义

在**静止面**上建面内正交标架（`covariance_modulation.py:377-381`）：`normal = cross(e1,e2)/|·|`、`t1 = e1/|e1|`、`t2 = cross(normal, t1)`。把 e1/e2/d1/d2 投影进来得 `p00..p11`、`q00..q11`，则

    A = Q P⁻¹                                                              (面内 2x2 形变梯度)
    R = A (AᵀA)^(-1/2)                                                     (2x2 闭式极分解)
    θ = atan2(R₁₀, R₀₀)                                                    (绕 **rest** 法向的转角)
    u = log(det Q / det P)                                                 (面积雅可比的 log)
    δq_f = [cos(θ/2), sin(θ/2)·normal]                                     (w ≥ 0 的逐面四元数)
    δq_v = normalize(Σ_{f∋v} δq_f)                                         (一环**等权**平均)
    u_v  = Σ A_f u_f / Σ A_f                                               (一环**面积加权**标量)
    s'   = s0 · (1 + k·tanh(g·u_v)),  k = 0.35,  g = 2.0

三个要点：① 旋转轴是 **rest 法向**，形变后法向从不参与 → 面外倾角整个丢失；② 尺度是**一个标量**乘三个轴（各向同性）→ 无法表达方向性拉伸；③ 四元数平均**等权**且不面积加权，靠 `cos(θ/2) ≥ 0` 约定规避符号问题（不是对齐，是约定）。

### 2.2 源码定位与逐字引用

| 语义 | 函数 | 行号 |
|---|---|---|
| 面内标架 | `DynamicCovarianceModulator.delta_rotation_scale` | 377-381 |
| 2x2 极分解 A→R | 同上 | 388-403 |
| 转角 → 四元数 | 同上 | 407-413 |
| log 面积雅可比（f64 重算） | 同上 | 419-421 |
| 一环等权四元数平均 | 同上 | 430-438 |
| 一环面积加权 stretch | 同上 | 440-448 |
| 2x2 极分解闭式 | `_polar_rotation_2d` | 452-475 |
| **缩放** | `delta_scale` | 477-480 |
| 合成 | `apply` | 482-501 |

    a00 = (q00 * p11 - q01 * p10) / det_p                                        # :397
    r00, r01, r10, r11 = self._polar_rotation_2d(a00, a01, a10, a11)             # :403
    theta = torch.atan2(r10, r00) * self.rot_strength                             # :407
    face_quat = torch.stack((torch.cos(half), sin_half*normal[...,0], ...))       # :410-413
    log_jac = torch.log((det_q64 / det_p64).clamp(min=1e-4, max=1e4)).to(x0.dtype)  # :421
    gain = self.scale_strength * torch.tanh(self.stretch_gain * stretch)          # :479
    return scaling.unsqueeze(0) * gain                                            # :480

`det_p` 在 `:393` 已 `.clamp(min=self.eps)`；`_polar_rotation_2d` 内部三重 clamp（`:464-467`）：`det_g`、`det_m`、`scale` 全部 `.clamp(min=self.eps)`。docstring 明写「the cross terms are NOT optional: dropping them (a Hadamard product) silently destroys the rotation signal」（`:456-459`）—— R 必须是矩阵乘 `A·G^(-1/2)`，不能逐元素乘。

### 2.3 参数与推荐

`PARAMS["lam"] = {}`（`cov_switch.py:58`），前端无旋钮。固化的三超参在 `:245-247`：`scale_strength=0.35`（k）、`stretch_gain=2.0`（g）、`rot_strength=1.0`。k 是 `ds/s0` 的**上界**，`:256-260` 强制 `0.0 < scale_strength < 1.0`，理由逐字：「at k >= 1 the additive delta can drive s' <= 0 and hand a negative variance to the rasterizer」。k=0.35 ⇒ `s'/s0 ∈ [0.65, 1.35]`。g 是 tanh 斜率，调小（1.0）会让小形变更线性、幅度更小。rot_strength 是 θ 乘子，等价 `Exp(cR Log R)`。推荐全部保持默认。

### 2.4 数值保证

- **B=0**：`is_neutral`（`:326-331`）在 `gs_renderer.py:624-625` 短路，本方法不执行。强行执行时 `u = log 1 = 0` ⇒ `tanh(0)=0` ⇒ `delta_scale` 恒 0 ⇒ `s' = s0 + 0` **逐位**等于 s0；θ=0 ⇒ δq = [1,0,0,0]，但 `apply` **无条件** `F.normalize`（`:493`），所以 q' 与 q0 差一次重归一化残差（~1e-7），**不是逐位**——这是它与 rot_sync 的真实差别。
- **保 PSD**：`|k·tanh| < 1` 严格成立 ⇒ `s' > 0` ⇒ `diag(s'²) ≻ 0`，R 正交 ⇒ 特征值全正。
- **保轴长**：否，三轴**同比例**缩放，比例在 `(1-k, 1+k)` 内。
- **限幅**：只有 tanh 软饱和；无角度限幅、无置信度门控、无 det 修正。

### 2.5 已知失效模式

1. **面外分量丢失**：旋转轴固定为 `x0` 的叉积法向（`:378-379`），`n1` 从不进入计算；纯剪切/倾斜区域薄片朝向与皮肤解耦。
2. **各向同性缩放 → 「橡皮感」**：`tga_covariance.py:12-13` 转述 SurFhead 的批评为「在拉伸/剪切方向上 over-blend、在压缩方向上 under-blend」。
3. **一环等权四元数平均在反向区相消**：`:430-438` 是 `scatter_add_` 后除以 `quat_norm.clamp(min=1e-12)`，只有 `quat_norm < 1e-8` 才回落单位四元数；相邻面 θ 反号时归一化方向由噪声决定（`rot_sync.py:52-60` 把同一现象量化为「5° 散布 + 50% 翻号 ⇒ 14% 顶点偏差 > 30°」）。
4. **翻面（det P < 0）无处理**：`det_p` 被 clamp 成正的 `eps`，A 分量量级可达 1e12，2x2 极分解仍给正交阵但角度无意义，且没有置信度区分。
5. **sliver 面 f32 抵消**：`:416-418` 说明 det 可到 1e-8，所以 `log_jac` 在 f64 里重算——这是本方法唯一的数值防护；**无时域状态**，抖动直接进画面。

## 3. 方法二 `tga`：3x3 雅可比的完整协方差形变

### 3.1 数学定义

    E  = [e1, e2, n0],  Ẽ = [d1, d2, n1],  n₀ = normalize(e1×e2),  n₁ = normalize(d1×d2)
    J_f = Ẽ E⁻¹                                (逐面形变梯度 / PGT)
    J_v = Σ_f A_f J_f / Σ_f A_f                (面积加权摊到顶点)
    Σ'  = J_v Σ0 J_vᵀ,  Σ0 = R0 diag(s0²) R0ᵀ   (朝向 + 拉伸/剪切)
    Σ'  = V Λ Vᵀ（eigh，升序）,  s' = √Λ,  q' = quat(V)
    ratio = s'_i / sort(s0)_i,  clamp 到 [0.25, 1.35],  s'_i = sort(s0)_i · ratio

第三列取**单位**法向是关键（`tga_covariance.py:283-287` 原文：「第三列取 **单位** 法向, 与论文一致 (只有前两列承载尺度), 保证 J 在法向上不注入虚假膨胀」）。推导记两点：① `Σ0` 的特征值恰是 `s0²` 这个**集合**，故 `Σ'` 与**升序** s0 逐轴对应，夹比才有意义；② `Σ'` 实对称正定 ⇒ `R' diag(s'²) R'ᵀ ≡ Σ'` 逐元素（`:41-43`），光栅器从 (q', s') 现算的就是 TGA 协方差。

### 3.2 源码定位与逐字引用

| 语义 | 函数 | 行号 |
|---|---|---|
| 逐面 J | `face_jacobians` | 282-307 |
| 逐顶点 J（面积加权） | `vertex_jacobians` | 309-339 |
| payload 出口 | `delta_rotation_scale` | 342-354 |
| 透传 | `delta_scale` | 356-358 |
| Σ' → (q', s') | `apply` | 360-436 |
| 闭式 3x3 求逆 / 矩阵→四元数 | `inv3` / `matrix_to_quat` | 181-200 / 146-178 |
| 默认值 | `DEFAULTS` | 94-101 |

    J = torch.matmul(Eb, inv3(E))                                                   # :306
    return torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).clamp(-64.0, 64.0)  # :307
    Jv = num / den.clamp(min=self.eps).unsqueeze(-1)                                # :336
    Jv = torch.where((den <= self.eps).unsqueeze(-1), eye_flat.expand_as(Jv), Jv)   # :338
    S0 = R0 @ torch.diag_embed(s0 * s0) @ R0.transpose(-1, -2)                      # :381
    Sig = J64 @ S0.double() @ J64.transpose(-1, -2)                                 # :386
    Sig = 0.5 * (Sig + Sig.transpose(-1, -2))                                       # :387
    lam, vec = torch.linalg.eigh(Sig)                                               # :388（升序）
    ratio = s_eig / s0_sorted.unsqueeze(0).clamp(min=self.eps)                      # :399
    ratio = ratio.clamp(min=rmin, max=rmax)                                         # :400
    dots = (vec_out * R0.double()).sum(dim=-2)                                      # :419
    vec_out = vec_out * torch.where(dots < 0, -1.0, 1.0).unsqueeze(-2)               # :420
    q = matrix_to_quat(vec_out).to(torch.float32)                                   # :432
    s_final = s_out.to(torch.float32).clamp(min=self.eps)                           # :433
    Jv = eye + strength * (Jv - eye)                                                # :353（strength != 1.0）

三处必须一起读，否则会误判：① `:403-411` 用 `scatter_(-1, s_idx, s_capped)` 把夹过的尺度**写回 s0 原始槽位**，注释理由是「LSPL 消费者按 **下标** 比较 s'/s0（`scripts/arkit_splat_test.py:524` 逐点断言 `max(s/s0) <= 1.35`）」；② `:413-420` 的符号对齐——`eigh` 列符号任意，而 Σ 对「偶数次翻列」是盲的，不对齐会得到与原朝向差 180° 的等价 R'；③ `:422-428` 的兜底，`det(vec_out) < 0` 时翻最小特征值列（不改 Σ）。

### 3.3 参数与推荐

| 参数 | 中文名 | 默认 | 前端范围 | 后端校验 | 语义 |
|---|---|---|---|---|---|
| `jacobian_strength` | 形变强度 | 1.0 | 0.0 / 2.0 / 0.05 | `(0,1]` ⚠ | `J_eff = I + s(J-I)`，1.0 = 论文原式 |
| `max_scale_ratio` | 逐轴缩放上限 | 1.35 | 1.0 / 2.0 / 0.05 | `> 1.0` ⚠ | `s'/s0` 硬上限（契约 A6） |
| `min_scale_ratio` | 逐轴缩放下限 | 0.25 | 0.1 / 1.0 / 0.05 | `(0,1)` ⚠ | 允许 4x 压缩 |

推荐 strength=1.0、max=1.35、min=0.25。`:97-99` 解释下限必要性：「不设下限时一个被强剪切的轴可以缩到 ~0, 高斯在该方向塌成一条线、深度排序里直接消失」。**⚠ 契约缺口（读码发现）**：前端区间比后端校验宽（允许 strength=0.0、max=1.0、min=1.0），拖到边界会抛 `ValueError` 并透传到 API；当前二者不一致，未修。

### 3.4 数值保证

- **B=0**：短路同上。强行执行时 J=I ⇒ `Sig = S0` ⇒ `eigh(S0)` 给 `lam = sort(s0²)`、`vec ≈ R0`（顺序经 `scatter_` 复原、符号经 `:419-420` 对齐）⇒ `q' ≈ q0`、`s' ≈ s0`。**不是逐位**：eigh + matrix_to_quat 有 1e-6 量级残差，全靠渲染器短路。
- **保 PSD**：`J Σ Jᵀ` 对任意实 J 半正定；`:387` 显式对称化；`nan_to_num` + `clamp(min=eps)` 兜住 `lam.sqrt()` 的负零。
- **保轴长**：否，三轴独立变化。
- **限幅**：逐轴 ratio `[0.25, 1.35]`（`:400`）、J 逐元素 `±64`（`:307`）、`inv3` 的 `det_floor=1e-12`（`:191`）；**无**角度限幅与置信度门控（不改朝向的「角度」，直接由 eigh 给终点朝向）。

### 3.5 已知失效模式

1. **法向噪声直通**：`n1 = normalize(cross(d1,d2))` 进 `Ẽ`。`docs/rot_sync_engineering.md:122` 实测（`h = sqrt(2A)`）：`1e-3h` ⇒ p50 0.08° / max 3.3°；`1e-2h` ⇒ p50 0.80° / **max 48.1°**；`1e-1h` ⇒ max 179.4°。这是噪声进入旋转的最短路径。
2. **矩阵空间平均的固有风险**：`vertex_jacobians` 是逐元素线性平均，相邻面反向时 J_v 近单位阵/近奇异 ⇒ `Σ'` 特征值挤在一起 ⇒ eigh 的特征子空间内部任意旋转 ⇒ **R' 跳变**（TGA 不取 polar(J_v)，所以跳变表现 R' 而非角度）。
3. **夹比破坏精确性**：一轴触发 `ratio.clamp` 后 `R' diag(s'²) R'ᵀ ≠ Σ'`（`:41-43` 自认：「只有触发 1.35*s0 上限截断的极端轴会有偏差」）。契约优先于精确性，有意取舍。
4. **f64 eigh 是成本主体**：`:383-384` 注释「尺度动态范围大 (s 跨几个数量级), float32 的近退化特征分解会带来 1e-3 量级的相对误差」，代价是对 20018 个 3x3 做 f64 特征分解。
5. **strength 不能为 0**：`0.0 < jacobian_strength` 使「关掉 TGA」只能切方法或 `LAM3D_TGA_COV=0`（`:440-442`），不能把滑块拖到 0 做消融。

## 4. 方法三 `rot_sync`：只取旋转因子，严格保轴长

### 4.1 数学定义

    J = R S（极分解：R 正交 det=+1，S 对称正定）
    Σ' = R Σ0 Rᵀ                     ← 只保留 R，把 S 换成 I
    eig(Σ') = eig(Σ0)                ← 逐位不变 ⇒ 三轴长严格不变
    q' = normalize(δq ⊗ q0),  s' = s0
    δq_v = argmax_q  qᵀ (Σ_f A_f q_f q_fᵀ) q      （Markley，4x4 主特征向量）
    门控链：恒等吸附 → 死区 → Exp(cR Log R) → 限幅×置信度（单次 slerp）→ (可选) EMA → normalize

与 TGA 是**同一族两端**（`rot_sync.py:9-16`）：`mode="rot"` 保轴长只改朝向，`mode="full"` 连 S 一起用；共用同一套逐面/逐顶 J 与 (q,s) 回写通道，因此互为消融对照。极分解用 **Newton 迭代** `X ← (X + X⁻ᵀ)/2`（Higham 1986），对非奇异实矩阵必收敛到正交极因子且与 X 同号 det。

### 4.2 源码定位与逐字引用

| 语义 | 函数 | 行号 |
|---|---|---|
| 极分解（Newton + SVD 精修） | `polar_rotation` | 255-333 |
| 批量求逆（不抛异常） | `_safe_inv` | 208-235 |
| f64 SVD 参照路径 | `_polar_rotation_svd` | 238-252 |
| 逐面 J / 逐面面积 | `face_jacobians` / `face_areas` | 434-449 / 451-458 |
| 逐顶点 J（对照通路） | `vertex_jacobians` | 460-484 |
| Markley 平均 | `_markley_mean` | 487-527 |
| 旋转场 + 全部门控 | `rotation_field` | 529-619 |
| payload 出口 / 合成 | `delta_rotation_scale` / `apply` | 622-626 / 632-656 |

    n0 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=self.eps)             # :444
    n1 = F.normalize(torch.cross(d1, d2, dim=-1), dim=-1, eps=self.eps)             # :445
    J = torch.stack((d1,d2,n1), dim=-1) @ inv3(torch.stack((e1,e2,n0), dim=-1))     # :448
    return torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).clamp(-64.0, 64.0)  # :449

    nrm = x.reshape(x.shape[:-2] + (9,)).norm(dim=-1).clamp(min=1e-12)              # :297
    x = x / nrm[..., None, None]                                                    # :298
    x = 0.5 * (x + _safe_inv(x).transpose(-1, -2))                                  # :301（循环 8 次）
    bad = (~finite) | (scale <= 1e-6) | (orth > _POLAR_TOL) | (det <= 0) | neg_in    # :310

坏样本只走**原始 J** 的 f64 SVD 精修（`:315-323`：`index_select` → SVD → `index_copy`）；`:317-320` 强调必须对原始 J 做而不是对已收敛的 x 做——镜像样本的 Newton 极限本身就是 det=-1 的正交阵，对它做 SVD 只会原样返回。

    pw = (q_face.unsqueeze(-1) * q_face.unsqueeze(-2)) * area[:, :, None, None]     # :512
    wire = (pw / 3.0).reshape(B, n_face, 16).repeat_interleave(3, dim=1)            # :513
    ev, evec = torch.linalg.eigh(acc.view(B, n_vert, 4, 4))                         # :521
    q = evec[..., -1]                                                               # :522
    conf = (1.0 - ev[..., -2] / lam1).clamp(0.0, 1.0)                               # :526

    snap = (angle < IDENT_EPS).unsqueeze(-1)                                        # :568
    dead = (angle < self.deadzone_deg * DEG).unsqueeze(-1)                          # :574
    if self.rotation_gain != 1.0: delta = slerp_identity(delta, self.rotation_gain)  # :579
    clipped = angle > cap                                                           # :588
    t_shrink = torch.where(clipped, cap / angle.clamp(min=1e-9), ones)               # :589-591
    g = ((coh.squeeze(-1) - tau) / max(1.0 - tau, 1e-9)).clamp(0.0, 1.0)            # :596
    t_shrink = t_shrink * g                                                         # :598
    delta = slerp_identity(delta, t_shrink.clamp(max=1.0))                          # :600

    q = quat_multiply(delta_q, q0)                                                  # :645
    is_identity = (delta_q[..., 0] == 1.0) & (delta_q[..., 1:] == 0.0).all(-1)      # :649
    q = torch.where(is_identity.unsqueeze(-1), q0, F.normalize(q, dim=-1, eps=self.eps))  # :650-652
    s = scaling.unsqueeze(0).to(dtype=delta_q.dtype).expand(delta_q.shape[0], -1, -1).contiguous()  # :653-655

`:646-648` 解释 `is_identity` 的必要性：「quat_multiply(I, q0) 在浮点上逐位等于 q0，再 normalize 反而会引入 1e-8 级重归一化残差，破坏 "gain=0 与不启用逐位一致" 的可验证性」。`:581-582` 解释合并：`t_shrink = 限幅系数 × 门控系数`，只做一次 slerp，「两次 slerp 会被测出约 2.3 ms 的额外开销」。

### 4.3 参数与推荐

| 参数 | 默认 | 前端范围 | 语义 | 推荐 |
|---|---|---|---|---|
| `rotation_gain` | 1.0 | 0.0 / 1.0 / 0.05 | `Exp(cR Log R)`，沿测地线回缩 | 1.0；调试 0.5 |
| `max_angle_deg` | 45.0 | 0.0 / 180.0 / 5.0 | 逐高斯转角硬上限，`<=0` 关闭 | 30~45；抖动大降到 20~30 |
| `deadzone_deg` | 0.0 | 0.0 / 90.0 / 1.0 | 低于该角度清零 | 0；高频抖动试 1~2 |
| `coherence_gate` | 0.0 | 0.0 / 0.95 / 0.05 | 一环合矢量长度低于 tau 不转 | 生产先 0；嘴/眼穿模试 0.5 |

后端校验（`:376-389`）：gain `[0,1]`、deadzone `[0,180)`、average ∈ `(face, vertex)`、temporal_smoothing `[0,1)`、coherence_gate `[0,1)`。环境变量（`:683-690`）：`LAM3D_ROT_GAIN / _MAX_ANGLE / _DEADZONE / _AVG / _SMOOTH / _COH`，总开关 `LAM3D_ROT_SYNC`（`:667-668`）。**注意：生产路径走 cov_switch.set_method，不读这些 env**，只有 `install()` 读。

### 4.4 数值保证

- **B=0 恒等（最强）**：`is_neutral` 短路（`:423-425` 委托基类）；即便被调用，`IDENT_EPS = 1e-6`（`:131`）的吸附 + `apply` 的 `is_identity` 透传保证**逐位**。实测 `verify_rot_sync_live.py` R6 断言 `scaling` 与 B=0 基准 `torch.equal`，PASS（`rot_sync_plan.md:216`）。
- **保轴长**：`s = scaling.expand(...)`，与 s0 逐位相等；R5 实测 `s'/s0` 恰好 `1.000000`（对比 LAM 1.349918 / TGA 1.350000）。
- **保 PSD**：`Σ' = R diag(s²) Rᵀ`，特征值 = s² > 0；R7 实测 `Σ'` 与 `Σ0` 特征值 `max rel err = 1.31e-06`。
- **限幅**：沿测地线回缩（`:586-591`）；实测 max_angle = 10/45/0 ⇒ 上限 `0.1745 / 0.7854 / 3.116` 度（`rot_sync_plan.md:227`），`0` 表示**关闭**而非全零。
- **det 修正**：Newton 保留 det 符号，必须在转四元数**之前**显式翻列（`:327-328`、`:317-322`）。`docs/rot_sync_engineering.md:157` 实测：det 正负各半的随机矩阵迭代 10 步后 `det(R) < 0` 仍占 **50.2%**（正交残差同为 2.8e-7），「取最近旋转」是应用层决策，必须显式翻列且要在转四元数之前，这一步不能省。

### 4.5 已知失效模式

1. **P 被丢弃**：纯剪切区（嘴角外拉、下颌外扩）薄片倾角略小于真实皮肤；`rot_sync_plan.md:307-311` 明确这是「方法一的定义性质，不是缺陷」，要补只能上 TGA 或 §7.4 的轴级部分拉伸。
2. **面内扭转 5.38% 下界**：逐面 J 在面内是常值，量化见 §5.3。
3. **0.2%~0.4% 顶点数学上无解**：46~76 个，一环面夹角 p50 = 176°，合矢量长度 p50 = 0.0000，零面积面 0 个、坏面 0 个——是嘴内/眼内翻折与颈部裁剪边界的真实反向，唯一对策是门控 + 限幅。
4. **`average='vertex'` 时 coherence_gate 失效**：`:548` 把 `coh` 置 `torch.ones` ⇒ `g ≡ 1`；前端 `vertex_then_polar` 也不暴露该参数（`cov_switch.py:70-73`），两条通路一致，但读码易误以为门控生效。
5. **EMA 作用在增量域**：`:603-609`；`docs/rot_sync_engineering.md:11` 结论相反——「应作用在最终四元数上」，240 帧/60fps、alpha=0.2 时 final-q EMA 误差 1.98° vs 增量 EMA 6.11°，jitter 0.464 vs 0.577。默认关闭是对的，打开也不对。
6. **Frobenius 预归一化与自家文档冲突**（`docs/rot_sync_engineering.md:156`）：`:296-298` 仍做 `x = x / ||x||_F`，而 `docs/rot_sync_engineering.md` §5.3 实测「不要先归一（cond=30 时 x0=J 8 步到 4.0e-7，x0=J/||J||_F 需 10 步才 1.9e-5）→ 仓库当前实现方向反了」。**这处张力如实记录**：文档以代码为准，建议未落地。
7. **sliver 假旋转**：J 夹到 ±64 后极分解角可达 170°+；R11 实测限幅前逐高斯转角 `mean 11.885 / p50 5.679 / p95 46.592 / max 178.548`，`clipped 1078`。没有 `max_angle` 时直接进画面。

### 4.6 `average='face'` 与 `'vertex'` 的行为差异

同一条 `rotation_field` 里的 if/else 两条分支，必须分开读：

| 维度 | `average='face'`（rot_sync） | `average='vertex'`（vertex_then_polar） |
|---|---|---|
| 代码 | `549-560` | `545-548` |
| 中间量 | `q_face = F.normalize(matrix_to_quat(polar_rotation(Jf)))` | `Jv = self.vertex_jacobians(x0, xB, Jf=Jf)` |
| 聚合空间 | **SO(3)**（4x4 外积 + eigh） | **矩阵空间**（9 通道 `scatter_add_`，面积加权） |
| 极分解对象 | F = 39904 个**面**矩阵 | N = 20018 个**顶点**矩阵 |
| `coh` | Markley 特征值间隔，真实分布 | `torch.ones(B,N,1)`，**恒 1** |
| 孤立顶点 | `wsum <= eps` ⇒ 单位四元数 + `coh = 0` | `den <= eps` ⇒ `Jv = I` ⇒ `polar(I) = I`（自然回落） |
| 前端参数 | gain / max_angle / deadzone / coherence_gate | 只有 gain / max_angle |
| 相邻面反向 | 每面给出**合法** R，聚合是加权平均，合矢量短但不跳变 | 平均 J 近奇异/近单位阵 ⇒ R 在 I 附近由噪声决定 ⇒ **跳变** |
| 面内扭转 | 逐面常值 J ⇒ 5.38% 下界 | 顶点级 J 可表达面内线性变化 ⇒ 分辨率更高 |

一句话：**vertex 分支用「聚合稳健性」换「面内空间分辨率」**。`check_rot_sync_mesh.py:110-117` 的 `face vs vertex averaging agree` 实测 `median|d| < 1.0` 度，说明平滑场下两者接近；差别只在局部反向与强扭转两种极端场景。

## 5. 为什么必须「逐面极分解再聚合」，不能「先平均 J 再分解」

### 5.1 反例：下颌/嘴角相邻面方向相反

顶点 v 的一环只有 a、b 两面，面积相等，形变是绕**同一轴**方向相反的两个小旋转：

    J_a ≈ R(+θ),  J_b ≈ R(-θ)
    J_v = ½(J_a + J_b) ≈ ½(R(θ) + R(-θ))

两个旋转在矩阵加法里**互相抵消**，只剩 `1 - cosθ` 量级的残余。θ = 25° 时 `cosθ ≈ 0.906`，J_v 三个奇异值挤在 0.9 附近、旋转角接近 0，但它**不是**恒等——残余方向完全由两面面积权重的微小差异决定（真实网格 `A_a ≠ A_b`，权重一偏 R 就跳到一侧）。更糟是翻折：θ→180° 时 `R(θ)+R(-θ) ≈ 0`（零阵），极分解无定义，只能回单位阵，等于「这个顶点完全不转」。

这解释三件事：① `rot_sync.py:49-51` 原文「下颌边界一带有相邻面朝相反方向转，平均后的 J 可能接近奇异，极分解出来的 R 会跳变。逐面取 R 再在 SO(3) 上聚合（ASAP 的旋转场）才稳」；② `rot_sync_plan.md:317-320` 实测那 46~76 个顶点一环面四元数最大两两夹角 **p50 = 175.9~176.7 度**、合矢量长度 **p50 = 0.0000**，就是 `R(θ)+R(-θ)→0` 的极端实例，而非数据缺陷；③ `rot_sync_plan.md:16` 的结论「真正决定成败的是怎么把逐面旋转聚到顶点」。

### 5.2 数学上的不可交换性

`polar(·)` 含 `(JᵀJ)^{-1/2}` 属非线性映射，一般地 `polar(Σ w_f J_f) ≠ Σ w_f polar(J_f)`：左边丢掉「每个面各自转了多少」，只保留平均后的**净**变换；右边保留逐面朝向再在 SO(3) 上取加权重心。SurFhead 的 JBS 是同一思想（`J = U·P`，`U_b = exp(Σ w_i log U_i)`，`P_b = Σ w_i P_i`），并在 §2.3 明确批评「逐元素线性混合相邻面变换会产生几何畸变（Fig.2b）」。

### 5.3 实测数字（`scripts/check_rot_sync_mesh.py`）

25 度绕 y 轴**刚体旋转**（`L79-92`）：

    check("rigid rot recovered (%s)" % avg,
          abs(ang.mean().item() - 25.0) < 0.5 and ang.std().item() < 0.5,
          "mean=%.3f std=%.3f max=%.3f deg" % (ang.mean(), ang.std(), ang.max()))

结果 **mean = 25.000，std = 0.000，max = 25.000**，face 与 vertex 两种模式**都是**。这恰恰说明「不能拿刚体旋转证明 vertex 模式坏」：刚体旋转下所有面转得一样，平均 J 就是那个 R，两种聚合等价；vertex 的失败只出现在**局部反向**（§5.1）与**强扭转**，而这两个场景在合成网格上需要显式构造。

空间变化扭转场（`L94-109`，`ang_v = linspace(0, 40°, V)`）：

    lin = (diag["angle_deg"][0] - ang_v / DEG).abs()
    ratio = lin.median().item() / 40.0
    check("varying field tracked (rel. to span)", ratio < 0.08,
          "median|err|=%.3f deg = %.2f%% of span" % (lin.median(), ratio * 100))

实测 **median|err| = 2.154 度 = span 的 5.38%**。span 扫 2/5/10/20/40 度时误差严格线性 **0.110 / 0.275 / 0.549 / 1.095 / 2.154**，比例恒为 5.38%。`L100-102` 注释写清这是**理论下界**：「逐面 J 是"面内常值线性映射"，天生无法表示面内扭转（同一个面三个角转的角度不同）。所以误差必然随"每个面跨越的角度"线性增长，不能按绝对值卡。」物理含义：每个面跨越的扭转角里只有约 5.4% 漏成角度误差，网格越密越小。**这不是 bug，是逐面常值 J 的固有分辨率**；突破只能上顶点级 J（`average='vertex'` 或对 J 做重心插值）。

## 6. 为什么 Markley 主特征向量优于「线性平均 + 半球参考」

### 6.1 外积对符号不敏感 ⇒ 不需要参考四元数

q 与 -q 是同一旋转。逐元素求和 `Σ w_k q_k` 在两面符号相反时直接相消（`q + (-q) = 0`），而 4x4 外积在符号翻转下**逐位不变**：`(-q)(-q)ᵀ = q qᵀ`。所以 M = Σ w_f q_f q_fᵀ 天然免疫双覆盖，**不需要任何参考四元数**做半球对齐。

省掉多少？`_markley_mean` docstring（`:495-497`）逐字：「代价：一次 [3F,16] 的 scatter_add + 一次 [N,4,4] 的 eigh。实测 0.63 ms，其中 eigh 只有 0.12 ms；这比它替换掉的那条 "J_f 摊到顶点 → 极分解 → matrix_to_quat"（约 3.5 ms）便宜得多。」`rot_sync.py:52-60` 补充来源与占比：「原实现为了定半球额外算了 polar(J_v) + matrix_to_quat，共 3.5 ms，占当时旋转场的 37%；改用 Markley 后这两步全部省掉。」`rot_sync_plan.md:261` 拆成 `polar_rotation(J_v)` 2.8 ms + `matrix_to_quat` 0.7 ms，并记录 v4 优化正是这一条（rotation_field 9.33 → 7.43 ms，端到端 12.36 → 11.20 ms）。

### 6.2 反例：把 Markley 只当「半球参考」再做线性平均

真实踩过的 bug，`rot_sync_plan.md:205` 逐字：

> 这一层曾经出现过 2 项 FAIL（rigid rot recovered (face) mean=24.947 std=1.658、axis == y frac=0.9983）。根因是当时用 Markley 结果**只当半球参考**再做线性平均——eigh 特征向量在近简并时方向不稳，反而把个别面翻错。改成「Markley 直接作为最终平均」后 20/20 全绿。**这个 bug 说明：拿一个可能不稳的量当参考系，比不要参考系更危险。**

1.658 这个 std 与 §5.3 的 std = 0.000 是**同一项检查的前后对照**：它证明的不是「vertex 模式坏」，而是「半吊子方案最坏」——只做线性平均，符号不一致时整体塌（5° 散布 + 50% 翻号 ⇒ 14% 顶点 > 30°）；只做 Markley，符号免疫，近简并时方向仍由噪声决定但**每面各算各的**；Markley 当参考再线性平均，参考系本身在近简并处不稳 ⇒ 被它翻错的面更多 ⇒ std 反而更大。

### 6.3 何时两者等价

`docs/rot_sync_engineering.md:10` 实测：半跨度 1°~89° 扫描，(c) 半球对齐与 (b) Markley 的**方向差恒为 0.000°**；F=3 等权时 chordal mean 与 Markley 方向恒相同，只差模长。数学解释：所有 q_k 落在同一开半球（存在 q 使全部 q·q_k > 0）时，M 的主特征向量 = `normalize(Σ w_k q_k)` 且主特征值严格占优；散布超 90° 后 M 趋近 `(Σw_k/4) I`，主特征值不再占优，两解分叉且都失去意义。失效对照（`engineering.md:61-67`，F=6，每面独立随机翻符号 p=0.5，单位度）：

| sigma | (a) 线性平均无对齐 | (c) 半球对齐 | (b) Markley |
|---|---|---|---|
| 5° | 5.98（14% 顶点 > 30°） | 1.81 | **1.81** |
| 20° | 22.48（39%） | 7.42 | **7.37** |
| 45° | 49.54（74%） | 16.24（6%） | **15.40**（5%） |
| 90° | 85.11（92%） | 34.18（60%） | **31.77**（53%） |

结论：**Markley 唯一不可替代的价值是「不需要正确性前提」**。对齐后线性平均在等价区间一样好、还更便宜（O(F)、无 eigh），但要先有「对的」参考系——而参考系恰好是这里最不可靠的东西。所以选 Markley，并把副产品（特征值间隔）当置信度。

## 7. 复杂度与耗时账

### 7.1 逐方法复杂度（N = 20018 顶点，F = 39904 面）

| 方法 | 主要算子 | 复杂度 | 是否迭代 |
|---|---|---|---|
| `lam` | 面内 2x2 闭式极分解 + 2 次 `scatter_add_` | O(F) | 否 |
| `tga` | `inv3` + `scatter_add_` + N 个 3x3 **f64** `eigh` + `matrix_to_quat` | O(F + N) | 否 |
| `rot_sync` | `inv3` + F 个 3x3 极分解（8 步 × 每步一次批量求逆）+ F 个 `matrix_to_quat` + N 个 4x4 `eigh` | O(8F + N) | **是** |
| `vertex_then_polar` | `inv3` + `scatter_add_` + N 个 3x3 极分解 + N 个 `matrix_to_quat` | O(8N) | **是** |

### 7.2 端到端与分段实测（`docs/rot_sync_plan.md:231-250`）

| 配置 | median ms | 相对 B=0 增量 |
|---|---|---|
| B=0 静态短路 | 1.44 | — |
| `lam` | 5.07 | 3.63 |
| `tga` | 7.56 | 6.13 |
| `rot_sync` | 11.20 | 9.77 |

`rot_sync` 旋转场分段（GPU Event median ms，合计 7.433）：`face_areas` 0.199（2.7%）、`face_jacobians` 0.802（10.8%）、**`polar_rotation(J_f)` 8 步 2.946（39.6%）**、`matrix_to_quat(39904)` 0.723（9.7%）、Markley ~0.63（8.5%）、其余 ~2.13（28.7%）。极分解是最大单项，也是唯一值得继续优化的段。

**为什么牛顿而不是 SVD**（`rot_sync.py:269-274` 逐字，RTX 4090）：float64 SVD **45.8 ms**（早期实现，单这一项吃掉 138 ms 的 1/3）；float32 SVD **6.66 ms**；float32 Newton **3.20 ms**（10 步，正交残差 2.4e-7）。**为什么 inv_ex 而不是 inv 或 inv3**（`_safe_inv` docstring `:208-235`）：`torch.linalg.inv` 在 batch 里只要有一个奇异元素就整批抛 `LinAlgError`（原文报错：`linalg.inv: (Batch element 1): The diagonal element 1 is zero, the inversion could not be completed because the input matrix is singular`），而真实数据里有 1~6 / 39904 个 det ≤ 0 的 sliver 面；手写闭式 `inv3` 反而**慢 3.7 倍**（39904 个矩阵 `inv3 = 0.380 ms` vs `inv_ex = 0.102 ms`，9 个 stack 元素展开成三条 elementwise kernel 链），整个 polar 从 4.14 降到 2.06 ms；**CUDA 上 inv_ex 的 info 恒为 0**，不能靠它判坏样本，只能 `nan_to_num` + `orth/det/finite` 三重判据。**8 步的依据**（`:194-196`）：真实 J 条件数 `p50 = 2.22 / p99 = 7.73 / max = 22.0`，8 步残差 `1.8e-7`、与 10 步差 `4.8e-7`（`POLAR_ITERS = 8`，`:197`；`_POLAR_TOL = 1e-6`，`:205`，超限才送 SVD，真实数据命中率 0）。

| 版本 | polar 内核 | rotation_field | 端到端 ROT |
|---|---|---|---|
| 初版 | float64 SVD | 45.80（仅 polar） | 138 ✗ |
| v2 | `inv3` + Newton 10 步 | 15.77 | 18.41 |
| v3 | `inv_ex` + Newton 8 步 | 9.33 | 12.36 |
| **v4（当前）** | `inv_ex` + 8 步 + **Markley 取代第二次极分解** | **7.43** | **11.20** |

### 7.3 `vertex_then_polar` 的耗时（未单独实测）

极分解对象从 F = 39904 个面矩阵换成 N = 20018 个顶点矩阵，理论更便宜；但多一次 9 通道 `scatter_add_`（`rot_sync.py:474-483`）。净额无法从已有数据严格推出，**标「推定 6~7 ms，与 TGA 同阶」**；要给准数需给 `scripts/bench_rot_sync.py` 加一个配置。`rot_sync_plan.md:305` 的 `2.8 ms` 是「同时保留面/顶点两条极分解」的增量，不是纯 vertex 模式成本。

### 7.4 30 ms 契约与后续优化

最贵的 `rot_sync` 11.20 ms，契约 30 ms，余量 2.7 倍。极分解占 39.6%，两条可行路子（`docs/rot_sync_engineering.md` §5.4）：① 2x2 面内闭合解 `theta = atan2(F10-F01, F00+F11)` 实测 `0.034 ms`（快 40 倍，对 sliver 稳定，代价是只给绕面法向转角、丢面外分量）；② 轴级部分拉伸 `Sigma' = R diag(s1, s2, s3·k_n) Rᵀ`、`k_n = tanh(g·u)`（`rot_sync_plan.md:355-360` 称之为「方法一 ↔ 方法二之间最有性价比的折中」）。Newton 步数已经没有空间（6 步不够、10 步只多 4.8e-7）。

## 8. 与三篇论文的对应关系

### 8.1 GaussianAvatars（arXiv:2312.02069，SIGGRAPH 2024）

- **它的一步**：每个三角面绑一个高斯，`μ/r/s` 定义在面局部帧；驱动时 `r' = R·r`、`μ' = k·R·μ + T`、`s' = k·s`，R 由 LBS 后的三角面直接算出（已含全局姿态）。
- **对应我们的哪一步**：`rot_sync` / `tga` 的 `q' = δq ⊗ q0` 就是 `r' = R·r` 的四元数版；`lam` 只取了它的**面内 2x2 子块**（绕 rest 法向的 θ），把 `s' = k·s` 退化成各向同性面积缩放。
- **我们的差异**：① 绑定粒度——它「1 高斯/面」，LAM「1 高斯/细分顶点」，所以我们多了一整层「一环面旋转 → 顶点旋转」的聚合（§5/§6 全部内容，它不需要）；② k 是单标量，我们用 3x3 J（TGA）或干脆不用（ROT）；③ `eps_scaling = 0.6` 是**训练期正则**（原文："if a 3D Gaussian is large in comparison to its parent triangle, small rotations of the triangle ... will be magnified by the scale of the 3D Gaussian, resulting in unpleasant jittering artifacts"），不是推理期门控；我们轴长固定，这个放大增益恒为 1，真正要防的是旋转角本身的噪声（`max_angle` + 置信度）。

### 8.2 SurFhead（arXiv:2410.11682，ICLR 2025）—— 与本方案最接近

- **它的一步**：2D surfel + `J = ẼE⁻¹`（第三列叉积构造），`Σ^{1/2} = J·R_c·S_c`；法向 `n_d = J^{-T} n_c`；混合走 JBS：`J = U·P`，`U_b = exp(Σ w_i log U_i)`（so(3) 对数空间），`P_b = Σ w_i P_i`（线性空间，保 PSD），`J_b = U_b·P_b`；附录 A.2.1 Lemma 1 证明极分解唯一。
- **对应我们的哪一步**：`J = ẼE⁻¹` ⇒ `rot_sync.face_jacobians`（`:434-449`）与 `tga_covariance.face_jacobians`（`:282-307`）**逐式相同**；`Σ^{1/2} = J R_c S_c` ⇒ `Σ' = J Σ0 Jᵀ` ⇒ **`tga`**（tga 的 `apply` 是「先算 Σ' 再 eigh」，与「直接对三个轴向量施加 J」等价，但走 (q,s) 回写通道更稳）；JBS 的旋转分支 ⇒ **`rot_sync` = 只取 U、丢掉 P**；JBS 的 `P_b = Σ w P_i` ⇒ 我们**没有**对应物，`vertex_then_polar` 是 `polar(Σ w J)`，把 U 和 P 混在一起平均再分解，**这是对照组比 JBS 更粗糙的地方**。
- **我们的差异**：① 做了它没做的三件事——`det(J) < 0` 投影到最近旋转、近退化/孤立兜底、逐高斯角度限幅与置信度门控（`docs/rot_sync_papers.md:59`：它把 J 非奇异当**前提**，对近退化**未讨论**）；② **没做** `n_d = J^{-T} n_c` —— LAM 高斯没有独立法向自由度，法向就是 Σ0 最小特征值的特征轴、已被 q0 携带；`rot_sync_plan.md:352-353` 把这条列为下一步（约 0.5 ms）。

### 8.3 GeoAvatar（arXiv:2507.18155，ICCV 2025）

- **它的一步**：沿用面局部帧（`T: μ ↦ S_face·R_face·μ + C_face, s ↦ S_face·s, r ↦ R_face·r`）；**旋转不是它的贡献**，贡献是 APS 分区（rigid / flexible / mouth，`τ_r = 0.1`、`τ_f = 2.0`）与极坐标正则 `L_p(r) = ReLU(r - τ_p)`、`L_angle(φ) = 1_{r > τ_r}·ReLU(φ - τ_φ)`（`τ_φ = 45°`）。
- **对应我们的哪一步**：`r ↦ R_face·r` 与 GaussianAvatars 相同（左乘合成），即 `q' = δq ⊗ q0`；分区思想对应我们两层替代机制——`max_angle_deg = 45` 是「离面越远越不敢转」的粗粒度等价，`coherence_gate` 是「局部帧本身不可信时不动」。`rot_sync_plan.md:90-95`：「**没有**做的是原论文那种显式区域分类，因为 LAM 的 ARKit 管线里没有『贴面/离面』标注，而 ‖mu‖ 阈值（0.1/2.0）是针对 FLAME 单位量纲定的，直接搬到 LAM 需要重新标定。」
- **我们的差异**：`τ_φ = 45°` 与我们 `max_angle_deg = 45°` 是**数值巧合、语义不同**——它限柔性高斯的方位角偏离，我们限逐高斯旋转增量角，不要在汇报里混为一谈。

### 8.4 补充参照

- **SplattingAvatar（arXiv:2403.05087）**：`R_k = R_cano·R_pose⁻¹`、面积加权 `q_V = Σ A_k q_k / Σ A_k`、`δq = u q₁ + v q₂ + (1-u-v) q₃` 重心插值。**它就是我们 §6 批评的 (a) 方案**（逐元素加权、原文未提半球对齐），我们的 Markley 是对该式的直接修正；我们顶点数已是 Loop 细分后的 4 倍，省掉了重心插值。`docs/rot_sync_papers.md:169` 标注「官方代码是否另有对齐**未核实**」，所以只能说「原文未提」，不能说「它一定错了」。
- **Rig3DGS（arXiv:2402.03723）**：`R_def = R_can·R_mesh·R*`，`R_mesh` 对 KNN 顶点做 Kabsch/Procrustes，`R*` 为 MLP 残差，离网格 > D 强制 `R → I`（`λ = 1e-1`）。对应我们的 `coherence_gate`（「绑不住就不转」），但输入是**一环旋转一致性**而非**到网格距离**——LAM 的高斯全在顶点上，没有「离面距离」自由度。
- **FlashAvatar（arXiv:2312.02214）**：`Δr / Δs / Δμ` 全由 offset MLP 学（Eq.5-6），自述全局姿态误差导致细节丢失。**反证**：纯学出来的旋转残差抗不住姿态误差，所以我们的方向是几何驱动而非学习。

## 9. 参数速查表与取舍建议

### 9.1 四方法 × 全部参数

| 方法 | key | 中文名 | 默认 | 前端 min/max/step | 后端校验 | 环境变量 |
|---|---|---|---|---|---|---|
| `lam` | — | — | — | — | — | 无旋钮（k=0.35 / g=2.0 固化） |
| `tga` | `jacobian_strength` | 形变强度 | 1.0 | 0.0 / 2.0 / 0.05 | `(0,1]` ⚠ | `LAM3D_TGA_STRENGTH` |
| `tga` | `max_scale_ratio` | 逐轴缩放上限 | 1.35 | 1.0 / 2.0 / 0.05 | `> 1.0` ⚠ | `LAM3D_TGA_MAX_RATIO` |
| `tga` | `min_scale_ratio` | 逐轴缩放下限 | 0.25 | 0.1 / 1.0 / 0.05 | `(0,1)` ⚠ | `LAM3D_TGA_MIN_RATIO` |
| `rot_sync` | `rotation_gain` | 旋转增益 cR | 1.0 | 0.0 / 1.0 / 0.05 | `[0,1]` | `LAM3D_ROT_GAIN` |
| `rot_sync` | `max_angle_deg` | 逐高斯转角上限 | 45.0 | 0.0 / 180.0 / 5.0 | 无（`<=0` 关闭） | `LAM3D_ROT_MAX_ANGLE` |
| `rot_sync` | `deadzone_deg` | 死区角度 | 0.0 | 0.0 / 90.0 / 1.0 | `[0,180)` | `LAM3D_ROT_DEADZONE` |
| `rot_sync` | `coherence_gate` | 置信度门控 tau | 0.0 | 0.0 / 0.95 / 0.05 | `[0,1)` | `LAM3D_ROT_COH` |
| `vertex_then_polar` | `rotation_gain` | 旋转增益 cR | 1.0 | 0.0 / 1.0 / 0.05 | `[0,1]` | 同上（走 install 才读） |
| `vertex_then_polar` | `max_angle_deg` | 逐高斯转角上限 | 45.0 | 0.0 / 180.0 / 5.0 | 无 | 同上 |
| （隐藏） | `temporal_smoothing` | 增量 EMA | 0.0 | 前端不暴露 | `[0,1)` | `LAM3D_ROT_SMOOTH` |
| （隐藏） | `average` | 聚合方式 | `face` | 由方法名决定 | `(face,vertex)` | `LAM3D_ROT_AVG` |

⚠ = 前端区间宽于后端校验（§3.3）。隐藏项只能经环境变量或直接构造类设置，前端面板改不到。

### 9.2 前端下拉框四选项的取舍

前端链路：`static/app.js:813-825` 走 `GET /api/cov/methods`、`:921` 走 `POST /api/cov/method`、`:945` 的 `COV_COMPARE_ORDER = ['lam','tga','rot_sync','vertex_then_polar']`、`:1021` 是中文标签表；后端是 `lam3d/server/app.py:247` 与 `:258`。

| 场景 | 选哪个 | 参数 | 理由 |
|---|---|---|---|
| **生产默认 / 交付演示** | `rot_sync` | gain=1.0, max_angle=45, deadzone=0, coh=0 | 数学定义最干净（保轴长、`Σ'` 特征值逐位不变，R7 误差 1.31e-6），11.2 ms 在契约内，三层自检 15+20+24 项全绿 |
| **要「最大表达力」/ 上界对照** | `tga` | strength=1.0, max=1.35, min=0.25 | 唯一能表达方向性拉伸/剪切；但 1.35 截断破坏 `Σ'` 精确性、法向噪声直通、有「橡皮感」风险 |
| **基线 / 回滚 / 最省** | `lam` | 无 | 5.07 ms 最快；也是「不启用任何改进」的严格基线 |
| **消融「逐面聚合是否必要」** | `vertex_then_polar` | gain=1.0, max_angle=45 | 与 rot_sync 只差 `average` 一个字段（`cov_switch.py:92-93`），最干净对照；刚体/平滑场与 rot_sync 一致（25° / std 0.000），嘴内翻折区显著变差 |
| **嘴/眼边界穿模、个别高斯被甩飞** | `rot_sync` | max_angle 降 20~30；coh 试 0.5 | 限幅是实测唯一 100% 有效手段（`engineering.md:133`）；coh=0.5 关掉那 0.2%~0.4% 无解顶点 |
| **静态近景、帧率高、抖动明显** | `rot_sync` | deadzone 1~2 度；**不要**开 temporal_smoothing | 死区直接清零小角度噪声；EMA 作用在增量域方向不对（`rot_sync_plan.md:329-333`） |
| **纯剪切区需要厚度变化** | `tga`（或等 §7.4 轴级部分拉伸） | strength=1.0 | rot_sync 丢弃 P 是定义性质；TGA 是当前唯一带拉伸的选项 |

一句话：**`rot_sync` 是默认，`tga` 是上界，`lam` 是基线，`vertex_then_polar` 是实验对照（不进生产）**。四者 B=0 载荷完全一致，切换只换 `renderer._cov_modulator`，A/B 成本 3~12 ms，不必重新生成 60~90 s。

## 10. 未核实与已知张力

1. `vertex_then_polar` 的端到端耗时**未单独实测**（§7.3），只有「同阶」推定。
2. `rot_sync.py:296-298` 的 Frobenius 预归一化与 `docs/rot_sync_engineering.md` §5.3 的实测建议**方向相反**，代码未改（§4.5-6）；文档以代码为准，记录这处张力。
3. `temporal_smoothing` 的「作用在最终四元数」改法**未实现**（`:603-609` 仍是增量域）；`n_d = J^{-T} n_c` 的协变法向**未实现**（`rot_sync_plan.md:352-353`）。
4. `coherence_gate` 默认关闭，**没有渲染对照证据**说明那 0.2%~0.4% 顶点该不该转（`rot_sync_plan.md:327`：「建议先用 0.5 试，再看渲染」）。
5. `tga` 的参数前后端区间不一致（§3.3）**未修**；`tga` / `rot_sync` 的 `install()` 环境变量在 `cov_switch` 生产路径上**不生效**，易误判「设了 env 就是设了参数」。
6. SurFhead JBS 在 `log U` 接近 θ=π 时的分支处理、SplattingAvatar 官方代码是否另有半球对齐、FlashAvatar 的 `⊕` 定义——三处均**未核实**（`docs/rot_sync_papers.md:165-176` 已列）。
