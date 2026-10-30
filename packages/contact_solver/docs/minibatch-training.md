# Predictor 普通 batch 训练接入

本轮把第 14 帧实验中的更新机制接入普通数据 batch。每个 batch 只提出一次
AdamW 参数更新，最多 6 次缩步验证，不在 batch 内执行 480 步姿态修正。
训练不使用 QP teacher，推理保持普通网络 forward。

## 代码边界

- `constraint_training.py`：与机器人和场景无关的事务式训练更新器。
  `ConstraintBatch` 提供标量 prior、带稳定身份的 residual、保护 residual、
  实际接触证据和逐样本诊断；closure 每次重新 forward 和查询物理后端。
- `predictor_constraints.py`：当前 canonical G1 与 box/ground 数据集的 adapter。
  不包含第 14 帧、特定足部、手或膝的任务配置。其他机器人/场景需提供相应
  adapter，不能把 box 的解析几何冒充任意 mesh 的距离场。
- `generator.train_full1000_position`：维护后的训练入口。
  `tmp/train_full1000_position.py` 为兼容入口。数据准备和 Newton worker 仍沿用
  当前仓库已有的 tmp 辅助脚本；这不是脱离仓库即可运行的独立安装包。

## 训练目标与保护

接触吸引使用当前 Newton 点对的 `includemargin`，留 1 微米数值跨界余量。
实际激活、分配且经主表面筛选的接触不再受到间距吸引。穿透独立使用已有
`DEFAULT_ACCEPTANCE.shallow_penetration_m`，不改变接触真值或仿真 margin。
缺失点对时，形状/平面只提供接近方向，不产生接触标签。

全身包含 47 个权威碰撞形状：mesh 完整顶点、sphere 解析支撑、cylinder
解析平面支撑及采样点。箱体平面分离量是保守几何指导，不是精确 mesh 碰撞距离。
完整形状、地面及 Newton 最坏穿透联合检查；各形状、自碰撞点对也单独保护。
消失的受保护自碰撞点对仍视为未知并拒绝，不静默声明安全。

保持接触的材料点来自输入姿态的**新 Newton 查询**；明确的 persistent role
才请求保持。切向漂移有独立 1 cm 任务容差，不用于定义接触激活。
释放/意外接触残差使用原有额外激活接受比例。关节限位独立检查。
原有姿态/布局/计划监督保留，root 偏移另加可配置高代价；它不构成接触标签。
本接入分支不再优化旧 execution gap loss，但旧指标仍保留用于对照。

## 更新契约

1. 所有样本的活动约束联合进入参数空间投影，不能用 batch 均值掩盖坏样本。
2. 投影 **AdamW 实际提出的增量**，包含动量、自适应缩放及 weight decay。
3. 投影需收敛；只有证实内缩余量不可行才尝试无余量的非恶化方向。
4. 每个候选重新预测、查询 Newton、检查逐样本保护和目标下降。
5. 接受后提交 AdamW moments 一次，应用的增量可被投影和缩步；更新 AL。
   拒绝/异常则回滚参数、optimizer、AL、模块 buffers 和 Torch RNG。
6. AL 状态按 sample/context/constraint/component 保存。每个样本的身份不能是
   batch 下标。step/state 的加载与 optimizer/state 的加载分别显式进行。

当前 CLI 的 `--initial-position-checkpoint` 仍然是 **warm start**，不会自动恢复
AdamW/AL。约束模式的当前权重 checkpoint 保存 optimizer 和 constraint_training_state，
供显式恢复使用；历史 best 权重快照不伪装携带同一步 optimizer 状态。
不支持 AMP、多个参数 dtype/device 或分布式训练；显式限制活动行数，避免无界显存。

## 小 batch 运行

```bash
source scripts/source_isaaclab3_newton_setup.sh
python -m generator.train_full1000_position \
  --constraint-training --gpu-pipeline --rollout-batch-fraction 0 \
  --pilot-samples 8 --batch-size 4 --query-worlds 8 \
  --steps 1 --evaluation-every 1 --training-prediction-budget 8 \
  --no-event-roles \
  --initial-position-checkpoint tmp/full1000_position_no_surface_gpu1000_20260922_run2/step_1000.pt \
  --output tmp/my_constraint_training_pilot
```

`--pilot-samples` 从原 train/validation split 各自均匀抽取索引，不混用两者。
这不是完整验证集；需新输出目录。固定 batch 中的示范接触意图作为条件，训练
planner 的原有监督仍保留。暂不支持 parallel rollout、递推 unroll、unified-contact
或不可见 teacher 计划；不兼容组合直接报错。默认不开启此训练分支。

`constraint_steps.jsonl` 记录接受/拒绝、活动行数、实际步长和逐样本原因。
pilot 的 `constraint_queries/` 保留原始 Newton 点对、形状/link/face 映射、
姿态、资产身份与静态查询时序。验证另记录 `constraint_verified_penetration_cm`
和 `constraint_contact_satisfied`；后者只表示所需接触与穿透接受，不代表承重、
无滑动、关节限位或闭环执行全部通过。旧 validation loss 是诊断指标，不能和
新 AL 训练标量直接比较。

报告：`tmp/constraint_training_report_20260926/RESULTS.md`。
这些小步测试验证工程接入，不证明泛化或替换当前正式训练配方的收益。

## 等预算继续训练验证

64 条不同 motion 各取一个训练样本，batch=4，学习率 3e-5，从同一 baseline1000
继续训练 10 轮；普通/约束两组均处理 640 次样本，约束组接受 158/160 次更新。
motion 8 和原第 14 帧递推输入未参与本轮训练。

结果仍有深穿透：固定第 14 帧输入，左脚完整网格顶点嵌入为 baseline 9.00 cm、
普通训练 7.17 cm、约束训练 7.18 cm。64 个留出单步输入的平均最坏穿透虽从
6.49 mm 降至 5.69 mm，但递推仍失败，不能据此提升为新的训练基线。
详见 `tmp/contact_training_comparison_20260926/RESULTS.md`。29 步序列是失败后
继续预测的诊断，正常 endpoint 验收仍在首次失败处停止；它不是成功 rollout。
