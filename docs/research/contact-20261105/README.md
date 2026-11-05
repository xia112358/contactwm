# 接触衔接与 loss 对齐进度（2026-11-05）

本快照记录 2026-11-05 已完成的接触监督、损失归因和 `climb_58` 衔接实验。数据仍为
[`partition-v8`](../contact-20261104/README.md)（12 条来源、3774 个保留控制帧、161 段）。
统计汇总见 [summary.json](summary.json)，逐轮/逐段见
[handoff-training-summary.json](handoff-training-summary.json)。本次同步只发布已完成实验的
代码和证据，没有启动新的训练。

## 本轮实现

- **固定目标模式的子步 witness。** `fixed_goal_mode` 下由
  `GPUContactLossMonitor` 在原生求解子步上维护 `contact_stable`、
  `contact_loss_point`；`SubstepLossWitness` 保留每个控制区间内最后一个合格实测点。
  不再用控制帧末接触掩码代替子步证据。
- **loss 与指标共用同一 witness。** `contact_loss_sampling.stable_loss_samples`
  选取每条轨迹最后一个稳定采样；`loss_aligned_attempt_metrics` 用完全相同的
  witness 计算成功、接触误差和缺失项。**缺失稳定 witness 记为失败，不再合成接触点。**
- **成功来源池与稳定段目标。** `contact_source_pool.successful_source_pool`
  只接受完整成功的原生执行，重复执行不能重复投票；`build_stable_segment_targets`
  在成功的稳定支撑放置内聚合固定目标。population 边界数据继续使用 v4。
- **DAgger 初始化路径。** `contact_train.py` 增加 `--initialize_from_dagger`：
  在新 slices 上初始化权重，同时重建帧几何和训练状态，避免沿用旧 slices 的缓存。
- **初始支撑与短时掉触。** `_filter_initial_support` 允许“已经在目标上”不需要先释放，
  也允许初始悬空后落地；未知状态不能当作释放，重置会清空历史。

新增/更新的回归覆盖 `contact_gpu_stability`、`contact_initial_support_loss`、
`contact_loss_aligned_metrics`、`contact_loss_sampling`、`contact_source_pool` 及
稳定起点/时序相关测试。

## `climb_58` 衔接实验

两个 100 轮续训都从同一个 `checkpoint_000500.pt` 出发：

| run | 入口来源 | native 成功率 | macro | 覆盖 goal |
|---|---|---:|---:|---:|
| pure handoff | 只从递归成功池采样后继入口 | 0.378 | 0.353 | 3 |
| reference producer | 前驱从各自参考起点 rollout，后继用池 | 0.521 | 0.507 | 10 |

单段 native 成功率（600 轮，按段）：

| 段 | pure | reference producer |
|---:|---:|---:|
| 1 | 0.955 | 0.962 |
| 2 | 0.105 | 0.181 |
| 3 | 0.000 | 0.021 |
| 4 | — | 0.815 |
| 5 | — | 0.617 |
| 6 | — | 0.173 |
| 7 | — | 0.938 |
| 8 | — | 0.396 |
| 9 | — | 0.411 |
| 10 | — | 0.553 |

两个 run 的**完整链式成功都是 `0/4085`**。reference-producer 最好出现在 520 轮：
4015 个尝试进入第 3 段，但没有一个通过。pure 在 520 轮有 2295 个进入第 3 段，同样
没有通过。pure 的池只覆盖到第 3 段；第 3 段池在 590 轮从 `256` 降到 `0`，此后
该段不再进入 actor 更新。

## 82 → 83 衔接诊断

`climb58-82-83-diagnose` 把同一个 #82 成功态分别作为参考起点和真实衔接入口重放第 2 段：

- 参考起点：`501/501` 成功
- 真实衔接入口：`0/492` 成功

入口偏差（中位 / p90）：根位置 `6.6 / 7.9 cm`，姿态 `0.095 / 0.174 rad`，关节 RMS
`0.10 rad`，关节速度 RMS `2.85 / 5.07 rad/s`，history RMS `0.20 / 0.47`，
sensor RMS `0.50 / 2.32`，上一动作 RMS `0.63 / 0.78`。实际进入第 2 段比参考早约 4 帧。
终端根误差 `25–33 cm`（25cm 容差），姿态 `0.39–0.58 rad`（0.4 容差），最近接触
`18–49 cm`（5cm 容差）。

`climb58-handoff-paired-audit` 对 246 个 #82 成功态做“同一实际状态不中断继续”与
“恢复到独立同构环境”对照：两边都是 `0` 成功，`outcome_disagreements=0`，初始
state/sensor/history/action 最大差都接近 0。**这不是恢复或环境 bug。**

详细诊断见 [handoff-diagnosis-82-83.json](handoff-diagnosis-82-83.json) 和
[handoff-paired-audit.json](handoff-paired-audit.json)。

### 如果只放宽接触距离

对同一批 492 个真实衔接入口，**只去掉接触距离约束，仍然 `0/492` 通过**，因为根位置
和姿态没有同时满足：

| 判据 | 通过 |
|---|---:|
| root<25cm & angle<0.4（原判据，含接触） | 0/492 |
| root<25cm & angle<0.4（忽略接触距离） | **0/492** |
| 只 root<25cm | 40/492 |
| 只 angle<0.4 | 101/492 |
| root<30cm & angle<0.5 | 264/492 |
| root<35cm & angle<0.6 | 492/492 |

根位置和姿态是**互相排斥**的：单独看各有约 8%/20% 能过，交集为 0。完整阈值表见
[handoff-contact-distance-ablation.json](handoff-contact-distance-ablation.json)。

作为对照，**独立参考起点**的 `stable-support-v2-345` 数据里，接触距离才是主要失败项
（[reference-start-contact-ablation.json](reference-start-contact-ablation.json)）：

| motion | attempts | 原成功 | root&angle 通过 | 纯接触失败占比 |
|---|---:|---:|---:|---:|
| `climb_58_original` | 264 | 0.792 | **1.000** | 0.208 |
| `climb_58_mirror` | 336 | 0.634 | **0.994** | 0.360 |

也就是说：参考起点上放宽接触距离几乎能把单段通过率抬到 100%；真实衔接入口上，
放宽接触距离一段也过不了。

## 梯度与监督归因

`climb58-handoff-regression-audit` 在 500/600 轮 checkpoint 上做了同样的 4085 次
入口重放和梯度归因。600 轮：

- 全梯度范数 `866`
- 第 10 段（case 91）梯度范数 `611`，稳定接触误差中位 `44.7 cm`
- 第 5 段（case 86）`437`，terminal guidance 贡献 `10.18`
- 第 2 段（case 83）`374`，稳定误差 `24.3 cm`
- 第 9 段（case 90）梯度余弦 `-0.35`

ref 600 的 `stable_contact_loss=5.07`、`terminal_guidance_loss=8.25`，guidance 项
在 hard case 上远大于稳定接触项（case 86：guidance 10.09 / stable 0.09）。pure 在
575 轮则出现 `stable=3.96 / guidance=26.06`。训练中的 `clip_grad_norm_(..., 1.)`
把更新方向交给这些 hard case；已学会的段随后被冲掉：

| 轮次 | 第 7 段 | 第 8 段 | 第 9 段 | 第 10 段 |
|---:|---:|---:|---:|---:|
| 541 | 0.94 | 0.73 | 0.11 | 0.75 |
| 600 | 0.68 | 0.005 | 0.016 | 0.00 |

原始归因见 [handoff-gradient-attribution-600.json](handoff-gradient-attribution-600.json)、
[handoff-attribution-500.json](handoff-attribution-500.json)、
[handoff-attribution-600.json](handoff-attribution-600.json)。

## 当前判断（待验证）

现有证据支持以下判断，但它们是待验证的工作假设，不是已证明的因果结论：

1. **衔接处的瓶颈不是接触距离，而是 root/angle。** 只放宽接触距离仍为 `0/492`；
   需要同时放宽到约 `30cm / 0.5rad` 才能让一半的入口通过第 2 段。
2. **入口偏离本身是可学的；失败集中在联合优化和信用分配。** 两个 run 都在同一批
   hard 段上失败，而 reference-start 单段可以很高；shared actor 被少数 hard case 的
   terminal 梯度支配，已学会的段被遗忘。
3. **接触和 root/angle 不在同一时间/尺度上。** 稳定接触是局部 witness 监督；
   root/orientation 只有终点 loss，并穿过冻结 WM 的 26–49 步反传。最后一轮 native
   batch 的第 2 段 root 均值：pure `46.8 cm`，ref `41.9 cm`，都大于入口的 `6.6 cm`；
   这不是接触阈值能解释的。
4. **success-only + `max_age=10` 的池是 loop 级放大器。** 第 3 段池在 590 轮过期后，
   frontier 不再进入更新；失败会直接导致断供。
5. **链式指标是乘法的。** ref 600 的单段 native 乘积约 `2.6e-5`，4085 个环境的期望
   成功约 `0.1`，所以 `0/4085` 是数学上必然的，不等于“完全没有学习”。

需要单独做的定位实验：只取第 2 段、用同一批真实衔接入口单独训一个 actor；以及
per-segment loss 归一化/PCGrad、在稳定 witness 帧增加 root/orientation 监督。
[`summary.json`](summary.json) 里保留了 500/600 的逐段和逐轮数据供这些实验对照。

## 其他同步证据

- [loss-aligned-status.json](loss-aligned-status.json)：loss 与指标共用 witness 的回放，
  151 个 goal、1305 次尝试、`0.944` native 成功率、`0.874` macro。
- [stable-loss-attribution.json](stable-loss-attribution.json)：150 个 case 的 stable
  witness 归因；`N0=4037`、`N275=4050`，连续接触贡献从 `2.30` 变为 `9.11`。这是
  witness 迁移的审计，不直接等同于任务成功率提升。
- [contact-old-support-fix-status.json](contact-old-support-fix-status.json)：旧支撑修复后
  425 轮回放，151 个 goal、`0.845` 成功率、`0.754` macro。
- [population159-new-support-interrupted.json](population159-new-support-interrupted.json)
  和 [population159-redagger-training.json](population159-redagger-training.json)：population
  159 的新支撑实验（前者在第 44 轮中断，后者 500 轮完成，`0.924` 成功率、`0.888` macro）；
  [population159-redagger-dagger.json](population159-redagger-dagger.json) 为 80 轮 DAgger。
- [pipeline-review.json](pipeline-review.json)：12 条 trace 中 11 条成功；166 个目标项里
  158 个控制目标没有稳定点。这个审计是“目标存在但缺少稳定实物证据”的直接来源。
- [previous-learning-audit.json](previous-learning-audit.json)：451–500 轮训练批次的
  161 个 case 低成功率归因；17 个低成功 case 中 11 个受数据修订影响，6 个尚未被现有
  检查解释。该审计明确说明训练批次不是独立评估，边界变化不能单独证明因果。

## 验证与复现

本轮同步的代码改动通过全量测试：**434 passed，28 pytest subtests passed**。涉及
接触、子步 witness、loss 对齐、source pool、稳定段目标和时序/切分的回归都在其中。

本次没有发布 checkpoint、replay buffer、`contact-evidence-*.npz`、完整 source pool 或
`contact_binding.json`；这些仍保留在外置 runtime 目录。衔接训练的命令和参数见对应 run 的
`training-plan.json`，回放/诊断入口见 `climb58-82-83-diagnose-20261105/entry.py`。
