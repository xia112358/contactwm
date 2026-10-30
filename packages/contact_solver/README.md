# Contact solver

与 `generator`、`motion_edit`、`somaforge_core` 并列的接触解算包。

登记的固定比较配方见
[v1基线](../../baselines/predictor_v1_latest_20261010/README.md)。
该历史配方由`unified_region_objective`、`ShapeTargetIntervalProvider`、
完整实体距离、位置区间及keep材料区域保持组成；共享查询的上下界保留同一材料身份，
其他实体和自碰撞独立覆盖。罚函数为`log1p(r²)`，经真实几何/FK导数回传。
接触区间来自实际margin；递推穿透验收5 mm与位置RMS容差4 cm仅是任务验收配置。
投影、QP教师、AL及形状cap替换等研究入口不属于该基线训练链路。

当前 Generator scratch 默认使用深度图、实际端点接触点目标，独立
`keep_patch_weight` 为 0；它与上述冻结配方分别记录，见
[Generator 当前说明](../generator/README.md)。本包提供几何与约束能力，调用方
显式选择损失及配置，不据旧基线覆盖新训练参数。

- `newton_witness_loss.py`：冻结当前 Newton witness 后经 FK 计算距离梯度。
- `contact_constrained_projector.py`、`heightmap_contact_projector.py`：姿态投影。
- `trajectory_projection.py`、`mechanical.py`：轨迹约束与机械可行性检查。
- `collision_geometry.py`、`part_collision_geometry.py`：权威资产碰撞几何。
- `device_contact_objective.py`、`newton_plan_realization.py`：接触约束与实际实现检查。
- `constraint_learning.py`、`constraint_residuals.py`：通用有符号约束、固定罚项与 AL；任务和机器人几何由外部提供，[设计与实验](docs/constraint-learning.md)。
- `feasibility_filter.py`：可配置的逐项保护与实际接触证据筛选，[改进对照与边界](docs/feasibility-filter.md)。
- `research/`：固定场景的研究代码；[第14帧研究](docs/step14.md)、[SDF/共享面梯度对照](docs/step14-gradient-fields.md)。

```python
from contact_solver.newton_witness_loss import full_body_violation, query_local_distances
from contact_solver.trajectory_projection import project_interaction_q_trajectory
```

本包不依赖 generator。共享 FK 在 `somaforge_core.g1_kinematics`；
共享结果结构在 `somaforge_core.prediction_contracts`。
共享 Newton 碰撞查询与过滤在 `somaforge_core.newton_collision` 和
`somaforge_core.newton_collision_filter`；本包也不依赖 Motion Edit。

几何距离、退出面及穿透损失只用于优化／诊断，不是接触真值。
实际接触继续使用当前 Newton/MJWarp 的 CONSTRAINT、dist < includemargin、
efc_address 分配状态，并经 `somaforge_core.contact_face_selection` 选择真实水平主表面。
原始点对保留审计；模型预测接触仍仅为意图。

共享职责见 [Core](../somaforge_core/README.md)，接触、载荷和滑移口径见
[支撑证据](../../docs/support-semantics.md)。WBT 的稀疏／带力 expert 与蒸馏使用
[原生模块配置](../../docs/wbt-modular-experiments.md)；本包不会自动启动这些训练。
全部入口见 [文档索引](../../docs/README.md)。

研究中的多面 SQP 尚未替换生产梯度，也不代表连续轨迹已满足承重或无滑动。
研究输出和大文件保留于根目录 `tmp/`。

- [Constraint-aware direction experiments](docs/constraint-direction.md): pose/network-space updates, unchanged Newton acceptance.

- [Training readiness and cost](docs/training-readiness.md): parameter-update integration and explicit separation evidence.

- [接触区间、完整几何与投影求解](docs/contact-geometry-projection.md)：验收一致的穿透梯度、小维度投影和数值可行性检查；独立实验配置，不改生产训练。

普通 batch 接入：[`docs/minibatch-training.md`](docs/minibatch-training.md)。
`ConstrainedOptimizerStep` 投影实际 optimizer 增量并支持完整拒绝回滚，
`PredictorConstraintAdapter` 提供当前 G1 box/ground 数据的几何与接触残差。
