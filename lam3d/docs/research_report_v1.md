# LAM 单图表情数字人:高斯旋转与拉伸研究报告与方法四(RSB)设计

> 课题:在预训练 LAM(arXiv:2502.17796)推理期冻结、无训练、<30ms 的硬约束下,改善单图数字人在
> ARKit52 表情变化时高斯椭球方向/尺度与中心运动的失配(薄片斜插、假旋转、时域抖动)。
> 本报告综合:lam3d/docs 六份前序工程研究文档、主线文献检索(两轮)、三个子智能体
> (buddy/deepseek-v4.1-flash)专题产出(文献侦察 15 篇 / 方法四 RSB 设计 / 无真值统计评测协议),
> 以及 2026-10-24 在真实 messi.png 会话上的补充实测。

---

## 0. 结论速览

> **v3 更新(10-24 23:30,数字口径刷新)**:①发现并修复 rot_sync.py `slerp_identity` 生产 bug(半球未对齐,w<0 顶点插值方向反号,随机四元数 t=1 时 |dot| 低至 0.11)——**本报告 v2 段与 method4_rsb_design.md 的 rig50/rig90/jitter 绝对值均为 bug 版历史数字**,方向性结论(RSB 优于 A0)不变,绝对值以下表为准;②修复后重验(5 preset,一环 Kabsch 口径,/tmp/rerig2.py):LAM 2.616/21.944°,RotSync gate=0 0.285/5.849°,gate=0.3 **0.283/5.727°**,RSB gate=0.3 0.308/6.464°(法向贴合刚性代价 +0.025°);③W8 断言 12/12、RSB 进 cov_switch 五方法下拉、live E2E 通过、GPU 计时 RotSync 7.16→RSB 7.97ms(+0.81);④三个子任务判决落地:Δε(砍 J⁻ᵀ 协变+SPD 插值,负收益)/p99(G2 FAIL,ORACLE4 下界 60.91°,splitting 判死)/N1N4N5(全过);⑤全部已推送 GitHub(dee9027),总结见 report/WORK_SUMMARY_2026-10-24.md。
>
> **v2 更新(10-24 12:44,历史——数字为 slerp-bug 版,仅方向性结论有效)**:方法四 RSB 已从纯设计推进到**带真实数据消融的可执行版**(docs/method4_rsb_design.md 扩至 70KB,含 W1-W8 实施清单/完整伪代码/实测消融/安慰剂对照/无参考指标 nres)。核心实测:rig50 平均 **−35.2%**(5 preset 一致 −33.6%~−38.0%),无参考法向残差 nres p50 3.763°→**0.044°**(98.8% 降幅,87-91% 顶点被改善),安慰剂对照证明收益来自修正**方向**而非幅度(shrink-only 与 A0 逐位相同;neg-sign 反而 +36% 更差)。尺度支路实测恒等(eigmax_rel 恒 1.277,合成实验 s'/s0 恒 1.0000)→**默认关闭**,v1 的 G_dir 增益实测负收益→否决。rig50 的口径澄清:它度量 Kabsch 一环 vs Markley 聚合的定义差(A0 的 8.5° 不代表 RotSync 错了 8.5°),绝对正确性看 nres。已知缺口:PSNR/LPIPS 未测(需训练好的 LAM 多视角渲染)。

1. **问题定位已经闭环**。LAM 的 20018 个高斯绑在 FLAME Loop 细分顶点上、极度各向异性
   (轴比 p50=3.37 / p90=83 / >2 占 80.4%),原版调制只动中心+面内 2×2 角度,面外旋转缺失
   ⇒ 薄片高斯斜插出皮肤。四方法(LAM/TGA/RotSync/VTP)的定量评测已完成
   (docs/cov_methods_eval.md):RotSync 旋转保真最好(rigid_err 1.54° vs LAM 7.76°),
   但按定义丢剪切(shape_err 0.857 vs LAM 0.830);TGA 形变最全但 65–70% 越界必 clamp,
   clamp 后假旋转 3–5%、时域最糙(0.73–0.94)。
2. **尾部安全警告(s3 补充实测;v3 已系统验证:G2 FAIL 且不可救)**:RotSync 改善集中在主体
   (extreme p50 −80%/p90 −70%),**p99 仅 −5%(143.12°,VTP 99.55° 反而最优)**——1% 高斯仍在撕裂。
   v3 判决(docs/exp_p99_fallback.md,38 臂扫描):尾部=参考系不稳顶点(1-ring vs 2-ring 差≥5° 集合 100% 覆盖尾部),
   ORACLE4(逐高斯三选一+一环最好面旋转)信息下界 60.91°>60°——**60° 判据不可达,属逐面常值 J 的分辨率极限**,
   接受现状,不实现 fallback;splitting(GSDeformer)单轮膨胀 2.1-4× 判死。评测口径注意:换 Markley 参考后方法排序翻转
   (rot_sync 158→121,vtp 99.5→165),「158°」含 10-68° 口径差——未来协议必须固定并双报参考系。
3. **关键新实测证据(本次报告新增,决定方法四形态)**:在真实会话 5 个 preset 上直接测
   "法向厚度比" k_f = A|det J|/A'(Nanson 公式,见 §3):逐顶点对数厚度 ℓ 的 p1/p50/p99 全部
   ≈0(精确到 1e-3),|ℓ|>log1.35 的占比 **0.0%**,k_f>1.35 仅 **0.01%**,k_f<0.741 为 0%。
   ⇒ **真实表情下皮肤厚度几乎不变;TGA 解析解 65–70% 的"最长轴越界"绝大部分不是物理拉伸,
   而是"纤维伸长 \|J n₀\| 把剪切误记成厚度"的系统偏差**(TGA 从未被指出过的缺陷)。
   这意味着:把拉伸自由度限制到"法向厚度"+把剪切全部交给"方向",不是妥协,而是**正确参数化**。
4. **方法四 = RSB(Rotation–Stretch Blend,旋转–拉伸测地混合调制器)**,完整设计见
   docs/method4_rsb_design.md(子智能体产出,490 行,含七步公式链/退化证明/预算表/消融矩阵)。
   一句话:**在 (q,s) 参数上直接构造结构有界的 Σ′**,而不是 TGA 的"先造 Σ′再分解再夹比":
   q′ = δq_fix ⊗ δq ⊗ q₀(旋转同步 + 法向协变最短弧修正),s′ 仅薄轴 s_{i₃}·e^{α·softproj(ℓ)}
   (对数域软投影,tanh 饱和,越界率构造性为 0),α 由逐高斯连续增益(各向异性比 a、离面距离 ρ̂、
   Markley 置信度 r_v)门控。α̅=0 时与 RotSync **逐位相同**(消融可判);端到端预算 ≈12.2ms < 30ms。
5. **文献定位**:RSB 与 2509.11411(On the Skinning of Gaussian Avatars,加权四元数平均)同题,
   但后者无置信度/无协变法向/无厚度项;我们的 Markley 主特征向量 + 免费置信度 r_v 更完备。
   子智能体检索的 15 篇(含 TexAvatars 混合 Jacobian、OUGS 参数协方差传播、Dual-Covariance、
   ReorgGS 条件数门控)提供了 RSB 三个组件的独立佐证与后续升级路径(docs/lit_scan_covariance_2024_2026.md)。
6. **评测协议 v2**(子智能体产出,docs/gt_free_eval_protocol.md + scripts/eval_protocol_stats.py 已在真实会话跑通):
   patch 分区(K=155)+ICC 设计效应(10–70×)+配对 bootstrap+统计功效(n_patch=155 ≫ 80% 功效所需
   6–15 个);新增无真值指标 normal_track(薄轴-法向夹角变化,rot_sync −0.12° vs LAM +1.61°,
   TGA +0.26°)与 protrusion/edge(薄片出皮量,rot_sync 中性级 ±0.0002 vs TGA +0.0192 变差)。
   ⇒ rot_sync 在统计上显著优于 LAM/TGA(rigid: −17.7°[−19.6,−13.7] 与 −14.4°[−15.6,−11.6],
   patch-bootstrap 95% CI)。

---

## 1. 问题定义与约束

- 数据通路:ARKit52 → FLAME blendshape → 顶点位移 x_B(中心)→ 高斯 (q,s) 冻结。
- 硬约束:①推理期规则式、无训练、无重打标;②compute_splat <30ms 契约;③一切协方差改动
  必须回写 (q,s)——gs_renderer.py:526-559 cov3D_precomp 硬编码 None,外部 Σ 被光栅器静默丢弃;
  ④ARKit 会话 FLAME pose 全零(B=I),J 已是残差形变;⑤单图资产,无同人物表情真值。
- 失效模式:LAM 面外旋转缺失→薄片斜插;TGA clamp 失真+假旋转+时域糙;46–76 个顶点一环面
  两两夹角中位 176°(真实翻折区),任何平均法无定义,只能门控。

## 2. 已有方法与证据(docs/cov_methods_eval.md,RTX 4090 实测)

| 方法 | 机制 | rigid_err p50 | 假旋转 | eig_rel_err_p99 | shape_err p50 | 越界 | 时域 roughness | 耗时 |
|---|---|---|---|---|---|---|---|---|
| LAM 原版 | 面内2×2极分解+各向同性tanh | 7.76° | 0.09–0.37% | 3.7e-1 | 0.830 | 无 | 0.13–0.47 | 8.44ms |
| TGA(方法二) | Σ′=JΣ₀Jᵀ,eigh回写+clamp | 6.09° | **3.1–5.0%** | 3.0e-1 | **0.821** | **65–70%** | 0.73–0.94 | 10.79ms |
| RotSync(方法一) | 逐面极分解+Markley,Σ′=RΣ₀Rᵀ | **1.54°** | **0.07–0.14%** | **4.5e-7** | 0.857 | 无 | 0.04–0.13 | 12.22ms |
| VTP(对照) | 先平均J再极分解 | 1.54° | 0.06–0.16% | 4.5e-7 | 0.857 | 无 | **0.03–0.09** | 12.27ms |

关键机制结论:①TGA 的 clamp 不是可选而是必需(12% 高斯解析解超 10 倍,最大 2.2e4),而 clamp
一旦命中,Σ′ 与真实局部旋转失去对应——这是它假旋转/抖动/各向异性膨胀(4.14→4.53)的统一根源;
②RotSync 的 shape_err 0.857 略差是"丢剪切"的定义级代价;③像素级指标被高斯重叠吸收
(pixel_flicker 四方法几乎相同),几何/动力学指标才有区分度。

## 3. 新实测:真实厚度比 k_f 的分布(方法四的立项依据)

在真实 messi.png 会话(5 preset: ones/half/extreme/rand/rand_sparse)上,对 39904 面直接计算
Nanson 厚度比 k_f = A_f·|det J_f|/A′_f(=|J^{-T}n₀|⁻¹,即垂直于形变面的真实厚度变化,含剪切的
正确不变量),并按面积加权聚合到顶点 ℓ_v = Σ w_f log k_f / Σ w_f(scripts/proto_m4_eval.py,
张量存 /tmp/m4_base.pt):

| 量 | 值 |
|---|---|
| ℓ_v 分位 p1/p50/p99 | −0.000 / −0.000 / 0.000 |
| \|ℓ_v\| > log 1.35 占比 | **0.0%** |
| k_f > 1.35 逐面占比 | 0.01% |
| k_f < 0.741 逐面占比 | 0.0% |
| det J ≤ 0 逐面占比 | 5.5e-5 |
| Markley 置信度 conf<0.3 / <0.1 | 0.085% / 0.013%(p0.1=0.32, p50≈1.0) |

**结论**:真实表情下皮肤厚度基本不变(±1% 量级);TGA 解析解的 65–70% 长轴越界并非"皮肤真的
拉长 1.35 倍以上",而是它取的"纤维伸长 \|J n₀\|"在剪切下偏离真实厚度。

**理论佐证(s2 命题,k_f ≤ ν 恒不等式)**:TGA 薄轴长度取纤维伸长 ν=\|U n₀\|,真实厚度是
k_f=1/\|U⁻¹n₀\|(U 为极分解的对称部分);由 ‖Un₀‖·‖U⁻¹n₀‖≥1 恒有 **k_f ≤ ν,等号当且仅当
n₀ 是 U 的主拉伸方向**。⇒ 剪切下 TGA **系统性高估厚度**,把"纤维被拉斜"误记成"板变厚"——
这是 TGA 各向异性从 4.14 膨胀到 4.53 的部分来源,也是它 65–70% 越界的机制解释。⇒ 方法四只需**一个**
拉伸自由度(法向厚度),且该自由度在真实数据上天然有界——RSB 的"仅薄轴 + 软投影"参数化
与数据分布精确匹配,预计拉伸通道越界率恒 0,同时消灭 TGA 的假旋转与时域糙。

Markley 置信度分布同时回答了 method4_rsb_design.md §8.2 的遗留问题:τ=0.30 门控阈值下
被门控顶点 <0.1%,与 46–76 个无解顶点(0.2–0.4%)同量级,**阈值不拍脑袋,有分布支撑**。

## 4. 文献全景与定位

两轮主线检索 + 子智能体 arXiv API 25 组查询(1315 条去重索引 docs/arxiv_index_gaussian.tsv),
按"推理期是否可用"分三类:

**A. 同题最近邻(顶点绑定+旋转混合)** — 2509.11411 On the Skinning of Gaussian Avatars:
LBS 旋转部分换加权四元数平均,任意光栅器可用,Unity3D 600FPS。与 RotSync 同族,佐证
"顶点绑定+四元数聚合"路线;但它无置信度、无协变法向、不处理厚度(正好对应我们 shape_err 0.857
的缺口)——**我们与其是"同题更完备"关系,引用为最近邻基线**。
**B. 机制组件佐证**(RSB 三组件各有独立文献支撑):
- 协变法向 J^{-T}n:SurFhead(2410.11682)"Orthogonality must be maintained in any space";
  TopoSurfel(2608.20687)normal alignment;Prior-Driven Normals(2609.36969)协方差对齐表面。
- 连续置信度门控:Confidence matters(2608.06117)连续 w∈[0,1];OUGS(2511.09397)参数协方差
  经渲染 Jacobian 传播(可复用已算的 J);ReorgGS(2605.08739)条件数门控;Rig3DGS(2402.03723)
  离网格>D 强制 R→I。
- 软饱和/流形:Local Second-Order Geometry(2607.19043)线性化偏差上界;ARAP-GRF(2608.29538)
  各向异性低通替代硬 clamp;Dual Covariance SLAM(2609.25746)双协方差解耦。
**C. 表达力上界(需训练,不满足约束,引用即可)**:HyperGaussians(2507.02803)条件分布切片;
GALA(2610.02207)动画基蒸馏;TexAvatars(2512.21099)混合 Jacobian;GeoAvatar(2507.18155)APS 分区;
FlashAvatar/HeadGaS/GaussianHeadAvatar/3D Gaussian Blendshapes 等。
**D. 失效理论**:RigPAPR(2606.06685)gap/spike 归因于 canonical 拼贴不可弯折——46–76 反向
顶点"任何平均无解"的理论背书;EfficientHuman(2504.20607)椭球无法表达动态表面,最薄轴方向
误差被 ~1/σ_min 放大——解释为何旋转回写对薄片收益最大。

(注:26xx 编号为 arXiv API 返回的 2026 新稿,正式引用前需核验;全部条目只读摘要。)

**s1 深度调研补充(docs/../report/gaussian_avatar_deform_survey.md,307 行,TOP5 可迁移技术)**:
- ① **SH 旋转缺口(重要,本项目完全没做)**:2509.11411 指出旋转高斯时球谐系数(SH,视角相关颜色)
  必须用 Wigner D 矩阵(W¹/W²/W³,14 项/高斯,2 万高斯 1–2ms)同步旋转——RotSync/RSB 只转了
  协方差没转 SH。LAM 的 SH 可能已训练成 view-dependent,须 A/B 验证(P3,风险项)。
- ② **GSDeformer(2405.15491)splitting**:把 46–76 无解顶点的长高斯拆 2–3 个短高斯,
  高斯数增量 <1%——G2 尾部问题的 fallback 替代方案(与门控二选一或混合),实时 60FPS 免重训。
- ③ **Grassmannian rank-2 退化保护**:轴长比>阈值的高斯把最短轴钳到 1e-3,只改 scale 回写,
  O(n) 零成本——薄片渲染稳定性的廉价保险。
- ④ TexAvatars 原话"face-level deformations are dominated by rotations and only mild shear,
  ‖J‖≤C"——直接判定 TGA 12% 轴长>10× 的样本是折叠边界数值病态而非几何信号,为 clamp/软投影
  提供理论依据;其 "Jacobian vs. Scaled Rotation" 消融节正是 TGA vs RotSync 的对照。
- ⚠️ **命名冲突警告**:内部代号 "TGA" 撞名 NeurIPS 2025 Spotlight "TGA: True-to-Geometry Avatar
  Dynamic Reconstruction"(北航,superkeaibb.github.io/TGA_web)。正式写作前必须改名
  (建议 ACD: Analytic Covariance Deformation 或 JBS-Area)。
- **两个真实空白点(可发表贡献)**:SPD 测地混合与逐轴部分拉伸在 3DGS/头像领域无先例
  (多轮检索仅纯理论 2508.17382 与 spd_learn 库)——RSB 的 ℓ 软投影正是逐轴部分拉伸的实例。

## 5. 方法四 RSB:核心设计(完整版 docs/method4_rsb_design.md)——已实测(v2)

> 状态:设计→原型→真实 capture 消融完成(10-24 12:44);**v3(10-24 23:30):仓库级落地完成**——lam3d/core/rsb.py(RsbModulator)+cov_switch.py 注册(`rsb` 方法,前端下拉)+W8 断言 12/12+live E2E 通过。注意:本节及 method4_rsb_design.md 的 rig50 5.507/8.501 等绝对值为 slerp-bug 版历史数字,修复后重验见 §0 v3 段(RotSync rig50 0.283/RSB 0.308)。数据 /tmp/rot_capture.pt(5 preset),脚本 scripts/proto_m4_*.py(12 个),复现命令见 method4_rsb_design.md 附录 B。v1→v2 变更 17 条见附录 D,其中 3 条 v1 路线被实测否决:k_f 公式降级为自检恒等式(f64 误差 1.4e-13)/条件数置信度删除(cond(J) p99=1.08)/G_dir 增益否决(负收益 rig50 5.507→6.018)。尺度支路实测恒等(eigmax_rel 恒 1.277)→默认关闭。新增最强证据链:安慰剂对照(§10.4,shrink-only 与 A0 逐位相同证明收益来自方向而非幅度;neg-sign +36% 更差证明方向信息真实)+无参考指标 nres(§10.5,p50 3.763°→0.044°,98.8% 降幅,87-91% 顶点改善,完全不依赖被测方法中间量)。rig50 口径澄清(§10.6):度量 Kabsch 一环 vs Markley 聚合之差,A0 的 8.5° 不代表 RotSync 错 8.5°。

**七步公式链**(全部 f32、可向量化、无 eigh):
S0 逐面 J_f=[d₁ d₂ n₁][e₁ e₂ n₀]⁻¹(已有)→ S1 极分解 R_f=Higham8(已有 2.95ms)→
S2 两个协变标量:n₁=normalize(J^{-T}n₀)=normalize(d₁×d₂)(免求逆),k_f=A|det J|/A′(免 eigh)→
S3 逐面置信度 c_f(cond 上界 κ_f=‖J‖_F‖J⁻¹‖_F)→ S4 顶点聚合:Markley(已有,免费 r_v)+
法向几何平均 n_v+对数厚度 ℓ_v(共用 scatter 权重)→ S5 log-Euclidean 软投影
ℓ̂=ℓ·ρ(\|ℓ\|), ρ(r)=(L/r)tanh(r/L)(C∞、Lipschitz≤1、自动降增益)→ S6 法向协变最短弧
修正 δq_fix(atan2 角,θ_cap=25° tanh 饱和,半角四元数闭式)→ S7 写回:
**q′ = δq_fix ⊗ δq ⊗ q₀;s′ = (s₁,s₂,s_{i₃}·e^{α̂ℓ̂})**,i₃=argmin s₀(init 缓存)。

**三个性质**:
- **嵌套门控**:旋转/方向/幅度共用 g_v=clamp((r_v−τ)/(1−τ),0,1)^1.5 ⇒ "拉伸放行 ⊆ 旋转放行",
  不会出现转不动但拉得动的失配;
- **逐位退化**:α̅=0 ⇒ δq_fix=1(短路)、e⁰=1.0 逐位 ⇒ 与 RotSync torch.equal 级相同,
  消融可判、上线可回退;
- **构造性无越界**:s′/s₀ ∈ (e^{−L⁻},e^{L⁺}) ⊂ (0.741,1.35),arkit_splat_test.py 的
  max(s/s0)≤1.35 断言无需修改即通过;另两轴比值恒 1.0。

**连续分区增益**(替代 GeoAvatar 离散分区,init 常量零运行成本):
G_dir=1−e^{−a/a₁}(薄片必须修方向,a₁=1.5);G_amp=1/(1+(a/a₂)²)(薄片不要拉厚,a₂=4.0);
G_ρ=e^{−(ρ̂/σ_ρ)²/2}(离面远的自由高斯不被牵引,σ_ρ=1)。

**时域**:delta 域 EMA 与 final-q 域 EMA 数学等价(右乘 q₀ 保内积),签名不变,λ=1 时逐位等于
目标;修号 + <0.9995 退化为 nlerp;C1 恒等吸附/C2 逐帧确定性(scatter 用排序+segment sum)/C3 逐位可重放。

**预算**(RTX 4090,f32,N=20018,F=39904):新增 11 步共 ≈1.0ms,总计 ≈8.4ms(调制器内),
端到端 ≈12.2ms < 30ms 契约,余量 17.8ms。无需 eigh/logm/expm——这是"流形上"与"<15ms"
同时成立的唯一原因。

**与三方法的关系**:α̅=0 → RotSync(逐位);无剪切+q₀=面帧+不饱和三条件同时成立 → TGA
(精确端点但不可达);族外性可分解为面内剪切残差 η_f/聚合域/饱和三差项。

## 6. 评测协议 v2(docs/gt_free_eval_protocol.md 569 行 + scripts/eval_protocol_stats.py 已跑通)

**0. 尾部安全新发现(s3 补充实测,改变验收格局)**:RotSync 的改善集中在主体,尾部几乎没动——
extreme 向量 rigid_err: p50 LAM 15.73°→RotSync 3.14°(−80%),p90 78.28°→23.80°(−70%),
但 **p99 150.73°→143.12°(仅 −5%,VTP 99.55° 反而最优)**。⇒ ①只报 p50 的验收会放过
"1% 高斯在 143° 撕裂"的方案;②尾部安全成为唯一硬否决项 **G2: rigid p99 ≤60°(当前不通过)**;
③46–76 无解顶点极可能是尾部主源,必须做 fallback 实验(coherence_gate/回落 LAM/回落 VTP)。

1. **分析单元**:patch 分区(K=155,中位 91 顶点)替代"20018 独立样本"假象;ICC 设计效应
   rigid 30–72×、shape 10–11× ⇒ 逐高斯 i.i.d. CI 会严重高估显著性,patch bootstrap 才有效。
2. **统计检验**:配对 patch bootstrap(2000 次)95% CI:rot_sync−LAM rigid = −17.7°[−19.6,−13.7];
   tga−rot_sync rigid = +14.4°[+11.6,+15.6];80% 功效所需 n_patch 仅 6–15 个(现有 155)⇒
   结论统计稳固。
3. **新无真值指标**(不需照片,跨方法可比):
   - normal_track(薄轴与顶点法向夹角变化,负=更贴皮肤):rot_sync −0.12° vs LAM +1.61°、
     TGA +0.26°(ones);extreme 下 LAM +2.39°/TGA −0.67°/rot_sync −0.14°;
   - protrusion/edge(s_max·sinθ 归一化出皮量):中性 0.3019;rot_sync 保持 ±0.0002,
     TGA +0.0192(变差),LAM −0.0063;
4. **契约回归**:α̅=0 与 RotSync torch.equal;max(s/s0)≤1.35 断言;cov3D_precomp 路径不变;
   B0 静态与不传 arkit 逐位一致。
5. **消融矩阵**(method4_rsb_design.md §7.1,A0–A3+B1–B14 预注册方向):B1/B2 分离方向与厚度
   贡献;B3/B4 反向验证增益先验;B8 扫 EMA λ(建议 0.3);B14(log 域三轴全拉伸)给表达力天花板,
   若 A3 与 B14 的 shape 差 <5% 则永久放弃剪切项。

## 7. 实施路线图

> **v3 执行记录(10-24):本节全部完成**——①Δε 已测:真实 preset p90 1.178°<2°,判据成立,砍 J⁻ᵀ 协变修正+SPD 插值(docs/exp_de_normal_cov.md);②p99 fallback 已测:38 臂 G2 FAIL,ORACLE4 信息下界 60.91°,尾部=参考系不稳,splitting 判死(docs/exp_p99_fallback.md);③N1/N4/N5 已测全过:Ω∈[0.97,1.03],饱和 1.2%<2%(docs/exp_n1n4n5_metrics.md);④P0-P3 全部完成(rsb.py 落地/消融/前端/e2e)。下表为原始计划留档。

**先做 s3 的四步高性价比实验(≈11h,决定 RSB 参数取舍)**:
1. 2h 测 Δε=ε_ROT−ε_TGA(normal_cov_err)→ 若 p90<2° 则砍掉法向协变修正与 SPD 插值两个候选项;
2. 4h 测 p99 尾部三种 fallback(coherence_gate/回落LAM/回落VTP)→ 决定 G2 能否过;
3. 4h 实现 N1 penetration(BVH SDF 7 点采样)→ 唯一与薄片穿出强相关的指标;
4. 1h 实现 N4 coverage Ω 与 N5 saturate_frac → 堵"缩小作弊"+暴露 45° 楔形。

fallback 候选(第 2 步的三个之上再加):GSDeformer splitting(拆高斯,免门控的物理解);
P3 加验 SH 旋转(Wigner D)是否对 LAM 的 view-dependent 外观有可见影响。

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0(0.5d) | 实现 RSB(新 modulator,复用 rot_sync 全部基建;stretch 通道升级 3 通道对数因子) | A1 逐位=RotSync;C1–C3 契约全绿 |
| P1(0.5d) | 消融 A3/B1/B2/B3/B4/B6/B8 + patch bootstrap 统计 | rigid ≤1.6°;越界 0;jitter ≤RotSync;shape ≤0.845 |
| P2(0.5d) | 感知扫描 α̅∈{0,0.3,0.6,1}×θ_cap∈{12°,25°},渲染对照图(正面+yaw35°) | 出双盲可看的对照图组;选默认参数 |
| P3(0.5d) | 前端 cov_switch 接入 LAM3D_RSB_* 环境变量+文档 | e2e 前端全绿;README 更新 |

**风险与缓解**:①外观耦合(LAM 高斯在无面外旋转下优化,δq_fix 会改外观)→ α̅ 从 0.3 起步、
θ_cap 从 12° 起;②τ=0.30 现有 conf 分布支撑(§3);③f32 Higham 在 cond>10⁴ 溢出 → κ 判定
+面内 2×2 闭式解兜底(命中率 0 但必在);④删 Frobenius 预归一化会改 RotSync 数值 → 回归基线
同版本重录。

## 8. 交付物索引

- 本报告:docs/research_report_v1.md(v3 口径刷新)
- **仓库级落地(10-24 新增)**:lam3d/core/rsb.py(RsbModulator,法向贴合+软限幅 25°)/
  lam3d/core/cov_switch.py(lam/tga/rot_sync/vertex_then_polar/rsb 五方法热切换+REST+前端)/
  rot_sync.py 的 slerp_identity 半球对齐修复(生产 bug)
- 三子任务实验文档:docs/exp_de_normal_cov.md(Δε 判决)/docs/exp_p99_fallback.md(+summary,38 臂 G2 FAIL)/
  docs/exp_n1n4n5_metrics.md(N1/N4/N5)/docs/rsb_reverify.md(slerp 修复后重验)/
  report/WORK_SUMMARY_2026-10-24.md(全工作总结,已推送 GitHub dee9027)
- 方法四完整设计:docs/method4_rsb_design.md(v2,约 1080 行/70KB:七步链+W1-W8 实施清单+完整伪代码+实测消融+安慰剂对照+附录 D 十七条变更日志)
- 文献侦察:docs/lit_scan_covariance_2024_2026.md(15 篇)+ docs/arxiv_index_gaussian.tsv(1315 条)
  + report/gaussian_avatar_deform_survey.md(307 行深度调研,TOP5 可迁移技术/命名警告/空白点)
- 评测协议完整版:docs/gt_free_eval_protocol.md(569 行:指标分层/2AFC 主观/统计检验/验收判据 G1-G11/合成故障注入);
- 统计协议脚本:scripts/eval_protocol_stats.py(369 行,已跑通,/tmp/cov_protocol_stats.json;逐高斯原始值 /tmp/cov_protocol_pergaussian.npz)
- 协议可执行实现:scripts/eval_cov_protocol_v2.py(SCE 轮廓覆盖误差/MAD 多视角伪影离散/TJA jerk 放大/RTT 切平面旋转参考/STAT 效应量审计;全方法无关锚;三资产试点中 FLAME 跟踪 alpha_maps 缺失为已知待修项)

**协议试点首批数字(messi 资产,/tmp/cov_v2/)**:
- **TJA(时域 jerk 放大,ramp 60帧)**:rot_sync 的 out_jerk 在全部 4 条轨迹上比 tga 低 1-2 个数量级(jawOpen 1.37e-5 vs 2.48e-4;mouthSmile 2.76e-7 vs 6.21e-4;browInner 1.12e-8 vs 2.77e-4;eyeBlink 2.04e-6 vs 1.03e-4),**迟滞 hyst_deg 比 tga 低 6-7 个数量级**(1e-9 vs 1e-2 量级);VTP 与 rot_sync 同级且常略优。LAM 原版居中。这是"时域平滑"从 rot_roughness 到独立 jerk 指标的二次确认。
- **protrusion 可辨识性发现(重要方法论)**:四方法的中性/表情 protrusion **逐位相同**(theta p50≈59°/poe p50≈823 全同)——因为四方法都只改 (q,s) 不改中心 x,薄轴-法向夹角的大基数(59°)是**资产属性**(LAM 学到的朝向本身),不是方法属性。⇒ N1 穿透指标的正确形式是**方法间差分 Δθ**(表情态 vs 各自中性态),绝对量无信息。这与评测协议 §normal_angle 删除结论一致,并进一步否定"绝对 protrusion 当验收指标"。
- **repocmp 一致性验证**:仓库版 RotSync 与原型库逐位一致(faces/area/Jf maxdiff 0,qR 4.8e-7,dq gap p50/p90=0.000/0.000 全 5 preset;唯一 extreme p90 12.3° 出现在翻折区,rigid 差异同量级)——原型消融结论可直接外推到仓库实现。
- 方法四原型库:scripts/proto_m4_*.py(约 30 个:lib/eval/verify/placebo/nres/timing/stats/f32/inv/refgap/reco 等;实测张量 /tmp/m4_base.pt 与 /tmp/rot_capture.pt)
- 前序六文档:cov_methods_deepdive / cov_methods_eval / gt_protocol_feasibility /
  rot_sync_engineering / rot_sync_papers / rot_sync_plan
