# 支撑证据：采集、编辑、训练与预测

统一版本：`newton_contact_load_slip_assessment_v1`，实现于
`somaforge_core.support_semantics`，原始求解器证据读取于
`somaforge_core.newton_support`。

有效接触的表面归属版本为 `newton_source_triangle_witness_normal_fan_v2`：
实际地形 witness 在其来源三角形上的几何特征必须属于被选主表面。
侧面内部点不能因三角形含有顶面顶点就被提升为顶面接触；真实棱边仍可按
Newton 法线归属相邻面。CPU/GPU 共用该几何规则，保留原始点对、激活与分配。

## 分开判断的四件事

| 判断 | 所需证据 | 不能替代它的东西 |
| --- | --- | --- |
| 有效接触 | 当前 Newton/MJWarp 的 `CONSTRAINT` 类型、`dist < includemargin`、实际约束行分配，以及共享主表面筛选 | 裸距离、预测意图、接触力阈值 |
| 承重贡献 | 上述有效接触中的同次求解法向载荷 | 持续接触标签、部位质心速度、编辑后继承的力 |
| 承重接触的滑移运动 | 实际受力作用点相对环境的切向速度；按载荷统计 RMS，保留区域速度分布 | 单个最低运动点、虚拟压力中心、不同 witness 之间的距离 |
| 接触保持意图 | 事件显式声明的 keep 角色 | 对真实承重或固定支撑的断言 |

`load_known` 表示观测完整，`load_bearing` 表示当前有正法向载荷贡献。
完整观测中的零载荷是卸载，缺少观测是未知，两者不能混用。
滑动的接触仍可承重，不能通过取消承重标签把滑动隐藏掉。
正载荷贡献本身也不证明动力学平衡或足够的支撑能力。

脚跟／脚尖切换在足部汇总内处理，但不同环境表面不能合成一个持续接触。
转动时，实际受力点相对环境速度小、未受力点移动大，可以是正常支点转动。
多个受力点的速度不能先做向量平均，避免正反方向运动相互抵消。

没有经过标定的速度预算时，只报告载荷和运动量，不输出“无滑移通过”。
几何 pivot 预算以及原来的端点 6 cm 位移容差均不是物理滑移验收阈值。

## 各阶段的职责

1. **采集**：full recording 保留原始点对、约束分配、同次求解的逐接触载荷、
   受力点局部坐标及相对速度，并通过统一选择器输出 `observed_support_*`。
   采集是被动读取，不改已训练 policy、资产、margin、gap 或运动。
2. **聚合与切分**：持续激活不能自动新增固定支撑角色。
   保留来源 execution、事件映射和原始载荷转移。几何运动诊断不代替承重证据。
   `prepare_predictor_event_paths.py` 仅从动作时间表读取起止帧，其余旧接触目标、
   建立／保持角色和排除标记全部忽略。标签来自该条示范自身的当前 Newton 观测。
   所有动作段保留，不再生成 accepted/rejected 分区，也不移动切分边界。
   新证书只校验示范、观测、动作时间表及缓存的对应关系，不判定示范质量。
   原 v3 筛选证书不能用于新的训练入口，必须重新物化。
3. **编辑与增广**：改姿态、IK、镜像或拼接后，原载荷只能作为 source reference。
   `reference_only_support` 保留审计数据，令输出实际支撑为未知。
   裁剪后的帧需要重新绑定原生录制；拼接和再次编辑保留 keep 意图，不能继承实际支撑。
   有意生成的力参考可以用于目标设计，但其 provenance 必须是 diagnostic，
   不能是输出轨迹实际执行的受力证据。重新静态查询只能验证接触，不能补出动力学载荷。
   增广与整段修正共用受力材料点切向 RMS，再按显式 keep 阶段累计。
   **6 cm 阶段累计运动预算**用于该几何参考的运动验收与超预算罚项；既不固定末端原点，
   也不惩罚受力点不动的转动。已知卸载区间不计入；未知证据不能自动通过。
   相对源轨迹新增运动只作软正则与诊断，不能单独判定验收失败。正增量逐区间累计，
   其余量为源步长中位数加三倍 MAD，软项权重为 0.1；不是接触阈值或物理无滑移标准。
   顺序 IK 目标为 `loaded_material_phase_budget_source_regularizer_v2`，整段修正为
   `source_relative_acceleration_loaded_material_budget_v10`，生成缓存为
   `native_observed_source_loaded_material_budget_edit_v8`。接触前软引导及目标偏移混合已移除，
   接触任务只作用于源观测区间；带引导的旧优化缓存不可直接复用。
4. **训练监督**：新数据的角色 2 来自该动作段内持续激活的同部位／同表面接触，
   角色 1 来自该段内开始并延续至终点的实际接触 episode。二者用于接触计划监督，
   不表示承重或静止支撑；帧间点对、脚跟／脚尖变化不另切动作段。
   `support_supervision.py` 只读取绑定 motion hash、原 recording hash、权威资产、
   当前 Newton runtime 和原生帧时序的独立观测文件。
   `reference_start/target_observed_*` 是示范观测；递推生成状态不能继承这些原始载荷。
   缺失证据使用 known mask 和 NaN，不能自动把持续接触填成支撑。
5. **运动指标与预测**：`somaforge_core.loaded_material_motion`统一计算原始受力材料点
   在相邻姿态下的切向运动，按原始载荷做RMS，再按阶段累计，版本为
   `source_loaded_tangent_material_motion_v1`。6 cm是几何参考路径预算，不是物理滑移真值。
   已知卸载不虚构静止点；缺失载荷或已删除区间的相邻求解证据必须标为未知。
   编辑和软拼接后按新姿态、新帧映射重新计算，不沿用完整轨迹的旧审计冒充裁剪结果。
   当前predictor只输出事件端点，路径及实际支撑均未知，不施加路径loss/gate。
   `keep_patch_loss`另作接触保持意图的端点几何正则：从当前真实、已分配且通过主表面
   筛选的Newton接触绑定材料点，经FK计算该材料区域的最小切向运动。
   使用材料区域凸包允许区域内支点转动，不固定任意单个witness或link原点。
   网络预测为keep才启用；建立／释放不套用。容许量来自对应动作示范自身的材料区域
   运动；预测角色与示范不同时，使用训练集keep样本的部位中位数，无样本则报告未知。
   该loss既不是实际受力支撑判定，也不是6cm路径预算或递推验收gate。
   旧首尾6 cm位移loss、训练池拒绝及递推gate已退役；实际Newton接触和安全检查保留。
   `endpoint_failure_reasons(..., require_observed_support=True)`仍拒绝未知或未经验证的实际支撑。
6. **可视化**：几何箭头和实际载荷／受力点速度分开显示。
   最低几何运动点不标成已确认支撑。支持独立观测文件的 hash 校验。

历史 WBT 的 `support_part_mask` 保留为 keep 意图兼容字段；新输入优先读
`contact_keep_intent_mask`。原 reward 中的 net-force／body-origin speed 仍是
历史奖励 shaping，不作为新标签或物理支撑判定。其值保持不变，避免改变原 policy 的执行；
监测项已明确命名为 body-origin speed proxy。

## 原始执行观测的物化

运行 `scripts/materialize_recorded_support.py --motion ... --original-state ... --output ...`。
工具检查原生连续帧、关节顺序、未改动的 root／关节姿态、采样率及原 recording hash，
从保存的真实求解器通道生成独立 support observations。
编辑／变速／事件重采样后的轨迹不能继承这份观测。

当前原始执行示例：
`runtime/current/motions/climb00_contact_dataset/versions/raw_execution_reference_20260930/support_observations.npz`。
932 帧载荷观测均完整。第 23–25 帧左脚无有效主表面接触且载荷为零，右脚仍承重；
第 26 帧左脚重新承重，受力点相对切向 RMS 速度约 10.17 cm/s。
第 125 帧没有有效主表面正载荷贡献，作为原始物理记录保留，不通过修改姿态消除。

观测只有每个记录间隔的最后一次物理求解。受力点速度对应约束求值几何，
保存位姿对应积分后状态。它们不能冒充严格同帧，也不能外推为未记录子步的全程无滑移证明。
旧数据集／训练 manifest 未自动切换；本次没有重训。

## 正式入口与退役边界

- 实际求解样本分析：`scripts/analyze_recorded_support.py`。
- 原始执行观测物化：`scripts/materialize_recorded_support.py`。
- 几何接触区域运动：`scripts/analyze_contact_region_motion.py`，仅作几何诊断。
- 示范动作标签与证据：`scripts/prepare_predictor_event_paths.py --motion ... --labels ...
  --intervals ... --output ...`。只使用动作时间切分，输出自身观测标签与完整性证书。
- 全量训练缓存：`scripts/prepare_predictor_training_data.py --collection ... --output ...`。
  当前默认数据为 `training/predictor_demo_observed207_geometry_v2_20261002`，包含 207 条示范的
  3,105 个动作样本；不存在逐段质量筛选或旧语义继承。
  箱顶与地面必须由共享 `ground_top_catalog` 选择，不能取第一个非地面表面。
  缓存生成及训练启动都校验箱顶高度和轮廓与保存的实际 Newton 主表面三角形一致。
  缓存身份包含编辑计划和表面目录内容；旧侧面几何缓存不得继续训练。
- 全量原始执行增广：`scripts/expand_source_preserving_augmentations.py`，要求显式
  提供原始 motion、Newton labels、support observations、original state、事件与编辑计划。
  不接受旧 dataset/archive 默认入口，也不继承旧增广姿态。完整及裁剪输出各自重新查询
  Newton；原受力材料点的运动对比只是几何审计，编辑后的实际支撑仍需重新执行验证。
  具体切口和生成参数见
  [全量编辑契约](../packages/motion_edit/docs/support-preserving-generation.md#native-execution-edit-collection-2026-10-01)。

旧几何支撑聚合优化器、专用聚合材料点模块及重复的逐帧审计入口已移到系统回收站。
不能再通过它们改写原始执行或把逐帧接触一致性当成支撑判断。
仓库审计拒绝这些入口和旧模块导入被重新引入。

WBT 历史奖励项及其配置名称保留用于原 checkpoint 的执行兼容，数值计算不变；
它们不能产生新支撑真值。新采集、编辑、切分及观测读取共用上述证据契约。
