# RSB 落地重验证（slerp 半球修复后）

> 2026-10-24。rot_sync.slerp_identity 存在对 w<0 四元数不恒等的生产 bug（重建 w 恒 ≥0），
> 历史所有带 max_angle 回缩 / coherence 门控的实测均受影响。修复后重跑关键指标。
> 10-25 增补:§n 渲染视觉对照(make_rsb_compare)。

## 0. 渲染视觉对照(scripts/make_rsb_compare.py,10-25)

512×2(正面/yaw35°)×2 preset(ones/extreme)网格,行={B0 静态, RotSync, RSB, |RSB−ROT|×8},
图 docs/images/rsb_compare.png(4096×1024)。

| preset | 视角 | \|RSB−ROT\| mean | 变化像素 | 脸内>8 占比 | 亮区垂直分布 top/mid/bot |
|---|---|---:|---:|---:|---|
| ones | 0° | 0.749 | 25.9% | 21.5% | 28.8/35.8/32.3% |
| ones | 35° | 0.794 | 27.0% | 22.2% | 27.0/36.9/33.0% |
| extreme | 0° | 0.654 | 25.0% | 20.3% | 28.1/35.9/33.2% |
| extreme | 35° | 0.699 | 25.9% | 20.8% | 26.4/37.5/33.2% |

(top=额/眼 0.15-0.45,mid=鼻/颊 0.45-0.70,bot=唇/颏 0.70-0.95 归一化像高)

结论:RSB 的法向修正在渲染上**真实可见且空间结构化**——量级与 cov_methods_eval §4.5 的
方法间差异同级(ROT−LAM 0.755);分布跨 preset/视角稳定,中下脸(鼻/颊/唇/颏,薄片高斯密集区)
占亮区约 2/3;亮区质心 y≈0.60 位于脸内,非背景噪声。

## 1. 修复内容

```python
# rot_sync.py slerp_identity（修复后）
sgn  = torch.where(q[..., :1] < 0, -1.0, 1.0)      # 半球对齐
axis = sgn * v / vnorm                              # v 随 q 一起变号
half = 0.5 * quat_angle(q)                          # 对 |w| 对称 → [0, π/2]
```

修复前随机四元数 t=1 时 |dot| min = 0.109（完全不恒等）；修复后 0.99999994。

## 2. W8 断言（12/12 ALL PASS，CPU 逐位）

恒等帧逐位透传、刚体残差 1.54e-05°、coh<τ 回退逐位、s'==s0、fix ≤ cap 全过。
另修两个测试 oracle 错误（W4 角度公式 2·asin→2·acos(|w|)；刚体采样限 30°<cap）。

## 3. rig 重验（/tmp/rot_capture.pt，5 preset，一环 Kabsch 参考）

| 方法 | rig50 | rig90 |
|---|---:|---:|
| LAM（不转） | 2.616° | 21.944° |
| RotSync gate=0 | **0.285°** | 5.849° |
| RotSync gate=0.3 | **0.283°** | 5.727° |
| RSB gate=0.3（fix⊗dq 口径） | 0.308° | 6.464° |

- 修复 slerp 后 RotSync 的 rig50 从历史 5.507°（带 bug）降到 **0.285°** ——
  历史 A0/A3 对比里 A0 的「8.501°」被 bug 抬高了；方向结论（RSB 优于 A0）不变，
  但绝对值全部需要以本次为准。
- RSB 的 fix 支路对刚性贴合的代价仅 +0.025°（p50 口径），换取法向贴合（§4.5 渲染对照）。

## 4. GPU 计时（RTX 4090，B=1，N=20018）

| | 每帧 |
|---|---:|
| RotSync delta_rotation_scale | 7.16 ms |
| RSB delta_rotation_scale | **7.97 ms**（+0.81） |

端到端预算 ≤13.5 ms 达标。

## 5. RSB vs RotSync 输出差异（5 preset）

| 量 | mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| q' 转角差 | 1.120° | 0.056° | 3.490° | 13.662° | 24.956° |
| normal_angle_deg | 1.121° | 0.069° | 3.491° | 13.660° | 24.956° |
| normal_applied_deg（门后） | 0.902° | — | 2.737° | 11.426° | 24.913° |

12.4% 顶点 fix 严格恒等（θ<1e-6 rad 吸附）；max 恰为 25° 软限幅生效。

## 6. cov_switch 注册（W6/W7）

- METHODS/DESC/PARAMS/_make 均已注册 "rsb"，参数：
  rotation_gain / max_angle_deg / coherence_gate=0.3 / normal_cap_deg=25 / gate_power=1.5。
- rsb.delta_rotation_scale 现合并写入 _last_diag（normal_angle_deg / normal_applied_deg / fix），
  与父类同口径，脚本消费者直接可用。
- describe() 输出含 rsb，前端下拉自动出现（描述驱动渲染）。

## 7. 本轮顺带修掉的 rsb.py bug

_vertex_normals 的分母累加 repeat_interleave(3)（默认 dim=0）在 B=1 时碰巧形状匹配但
数值错位；改为 repeat_interleave(3, dim=1)。恒等帧断言此前被 W1 的自算 cnt 遮蔽。
