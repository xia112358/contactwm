# Predictor v1 latest baseline

2026-10-10由用户指定为当前唯一训练及分析基线：`predictor.v1_latest.20261010`。
权威登记为[baseline.json](baseline.json)，项目当前指针为[../current.json](../current.json)。

固定分析checkpoint为
`runtime/current/models/generator/v1_shared_interval_scratch1100_20261009/step_1000.pt`，
SHA256为`b50df5dc0271717dc592875289fd54c93b0502073f5b61781a21ccc860ff1a8b`。
它是最新保存结果，未按单轮验证峰值选取。模型仍为v1共享观测编码器及root＋部位平均读出，
宽度192、3层、位置宽度32；v2独立编码器改版未采用。

## 训练配方

[training_config.json](training_config.json)记录新训练配置，
[training_run_config.json](training_run_config.json)保留实际历史运行配置。
默认数据为`predictor_loaded_material207_joint_v12_20261004`，来源为207条增广。
train1470行：高度因子0.95/1.00/1.05；validation780行：0.90；test855行：1.10。
所有增广共享同一原始源轨迹，不能声称独立源动作泛化。

scratch初始化，无predictor预训练权重或旧Adam；训练关节均值只用于输出bias初始化。
日程为100轮示范预热＋1000轮反馈，学习率3e-4按1100轮日程衰减至3e-5。
每轮遍历1470条示范；反馈阶段另有128槽×10次更新的持续池，合计2750次预测、
22次minibatch优化更新。示范外自主递推启用；失败输出参与loss，但输入及目标回退。
keep权重1，位置权重1，网络和其他loss权重保持本次实际配置。

正式入口：

```bash
source scripts/source_somaforge.sh
runtime/environments/newton-main-20260914/bin/python -m generator.train_full1000_position \
  --headless --device cuda:0 \
  --output runtime/current/models/generator/v1_shared_interval_new_run
```

每次指定未使用的输出目录；以上命令未执行。完整参数保存在
[reproduction_command.json](reproduction_command.json)，数据与仿真依赖的SHA256见baseline登记。
新训练从零初始化；固定分析checkpoint不作为scratch初始化权重。

## 目标与接触语义

| 目标 | 当前定义 |
| --- | --- |
| 接触距离 | 到目标有限表面实际margin邻域上半侧的区间距离，内部不追精确点或中点 |
| 完整避碰 | 当前初始化场景的机器人—地形及自碰撞实体有符号距离，材料点经FK反传 |
| 缺失区域恢复 | 训练示范真实Newton接触材料点库，候选出现后仍保留场，不提供教师姿态 |
| 统一查询 | `coherent_shape_target_material_query_interval_v2`，上下界来自同一次材料查询；其他实体、自碰撞及释放独立覆盖 |
| 端点位置 | 真实部位形状到预测XY位置的法向允许段；原有全局4cm RMS容差内位置项梯度为零 |
| keep保持 | 网络自身keep意图，当前真实材料区域的最小切向运动，允许绕内部支点转动 |
| 罚函数 | 接触区间、实体避碰及keep采用`log1p(r²)`，按固定解剖区域mean＋max汇总 |
| 无效原生法线 | 审计并拒绝递推；只屏蔽对应无效距离行，完整实体和其他有效损失继续反传 |

实际接触由当前Newton CONSTRAINT激活、dist小于实际includemargin、约束分配及
真实主向上水平表面筛选决定。几何区间和零loss不代替激活判定。
原始点对保留审计；侧面不能提升为顶面。机器人唯一权威资产为G1 spherehand URDF。

递推接纳使用实际接触、完整避碰、关节限制、自身位置及keep角色一致性。
区域witness分布仅作诊断；区域筛选的位置完整性仍保留。
运行穿透验收5mm不改变优化的穿透目标或仿真margin。
端点预测没有逐帧实际载荷与执行轨迹，支撑及累计滑移均为unknown，不施加伪造的6cm端点滑移gate。
训练链路没有QP教师、推理投影、自定义修正梯度或C1退出场。

## 实际训练记录和比较边界

原始运行计划1100轮，performance记录到1002轮，最后保存1000轮；没有完成报告，
退出原因未知。用户将该最新版本指定为后续研究起点，不表示完整训练或物理执行验收通过。
[training_summary.json](training_summary.json)保存整段统计，而非选取单轮峰值。

| 验证区间 | 接触＋避碰平均通过率 | 平均穿透mm | 平均身体位置误差cm |
| --- | ---: | ---: | ---: |
| 101–300 | 52.23% | 2.242 | 3.487 |
| 301–550 | 57.82% | 1.210 | 2.677 |
| 551–800 | 64.67% | 0.718 | 2.032 |
| 801–1000 | 71.54% | 0.617 | 1.844 |

均为相同780个验证端点。穿透先取每个姿态的Newton/完整实体最大深度，再聚合。
旧source运行只到550轮，且旧区域分布递推gate与当前不同，不能当作纯loss单变量对照。
最新查询身份重构在50次匹配更新上模型、Adam及3105端点评估完全一致，未证明独立性能增益。

950轮motion4的30步递推已有个例证据，不能归属于固定1000轮checkpoint。
本次登记未新跑1000轮递推或完整留出递推集；
[evaluation_command.json](evaluation_command.json)仅记录后续诊断命令。
端点几何通过不等于实际承重、无滑移或底层policy成功执行。

## 历史封存

旧scratch_recipe.20261002、source550及control500已[原地封存](../README.md)，
保留权重、数据、配置、代码快照和失败证据。项目默认配置、资产清单和训练清单指向本版本。
原训练源码哈希与登记时源码哈希分别保存在[source_provenance.json](source_provenance.json)，
训练材料库、校准和checkpoint均绑定外部文件哈希。

2026-10-10 的路径迁移将训练启动记录和源码快照移入正式 `runtime/current/archives/`，
[source_provenance.json](source_provenance.json) 已更新引用。源码快照与 checkpoint 字节不变；
JSON 内部路径更新后的哈希另行登记，原哈希保留用于审计。迁移不改变训练结果或数据准入。
