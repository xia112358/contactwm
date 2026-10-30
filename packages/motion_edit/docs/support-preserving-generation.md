> Archived: trajectory editing/generation and Infiller were retired on 2026-10-30. This document is historical, not an active workflow. Next-contact Predictor remains supported.

# 编辑轨迹的支撑保持与站立段裁剪

Implementation note (2026-10-22): this document retains historical solver names
for experiment provenance. Maintained generation now uses `trajectory_problem.py`,
`trajectory_types.py` and the float32 `torch_newton_ik.py` backend in managed
`main`. The former PyRoki/JAX solver and dedicated IK environment are retired.
See [current generation setup](../README.md#torch--newton-generation-backend).

## 当前接触区域运动口径（2026-09-30）

**目标修正：还原原始物理执行，不用几何滑移代价创造新的执行。**
2026-09-30 的 `support_coverage_repair` 与 `support_budget_repair` 是已停用的
几何优化诊断，不能作为正式动力学参考。旧几何聚合优化器及其专用材料点传输模块
已移到系统回收站，不再提供可执行入口；原始执行代表不能通过优化姿态制造“物理还原”。
持续接触激活不意味着部位必须固定；原始载荷转移、卸载时的姿态调整应保留。
几何预算、接触激活与真实载荷速度是不同证据，不能互相冒充。

`versions/raw_execution_reference_20260930` 提供完整执行代表的对照：
用事件对齐后的全身距离选择原始执行，但导出时保留它的原始物理时钟及姿态，
不应用时间拉伸、逐帧均值、IK 或滑移优化。原始速度/控制通道保存在
`original_state.npz`，载荷通过原录制的环境/帧索引及哈希追溯；固定姿态重新查询
只用于该姿态的 Newton 接触标签，不冒充原轨迹同次求解的动力学载荷。

`native_contact_pivot_and_region_motion_v3` 统一聚合、Motion Edit 原生修复、事件几何运动验收和
`scripts/analyze_contact_region_motion.py` 的逐步统计。每步只读取当前 Newton 已激活、
已分配且经共享主表面筛选的点对，将同一个机器人局部材料点变换到下一帧，计算切向位移。
下一步可随真实前掌／后跟接触变化选取不同区域，不比较两个不同 witness 的世界位置。

几何运动项使用实际接触样本的最小切向位移 `pivot`，衡量是否存在低运动支点，
允许绕它转动。同时保留 q25、中位数、q75、最大值和样本数，不用最低点掩盖区域运动。
这些几何样本无候选受力分配，不能解释为面积、受力比例或整个足部无滑动。
`support` 兼容字段与 `pivot` 相同；q25 仅在 `lower_quartile` 诊断字段中保留。
旧 v2 报告必须按其原 schema 解释，不能把旧 `support=q25` 当新支点指标。

优化和验收共用同一支点实现；事件阶段内累加逐步 `pivot` 位移，不因前掌／后跟
样本变化重置。默认几何预算改为按同部位源支撑时段去重汇总每步运动，
取 `中位数 + 3×MAD` 后乘事件步数；源累计运动超预算会标记为质量问题，
不再用累计运动本身兜底，也不再用异常事件的 P95 放大预算。
初始化一次并保留统计依据；候选更新不重新放宽预算。不再隐式使用6 cm。
这是源轨迹的几何统计预算，不是从受力速度推出来的摩擦滑移容差，也没有覆盖所有示范。
预算不参与接触激活判断。无实际激活接触的步骤记为缺测，候选几何点对不能
补成支撑测量；优化可保留源样本作恢复引导，验收仍报告缺测并不通过。
下文历史实验采用的旧中位数／候选点测量保持历史含义，不能冒充当前口径的验收结果。

几何最低点不定位真实承重支撑点，不能作为承重支撑验收。`phase_support` 显式标记
`support_location_status=unknown_without_same_solve_per_constraint_force`；其 `passed`
只表示几何运动预算通过。真实支撑运动由 `newton_support.support_motion` 在已验证的
激活／分配／主表面点对上，按同次 MJWarp 求解的法向力加权计算作用点切向相对速度 RMS。
另报受力点最低／最高运动；即使一个受力点静止，其余受力点滑动仍反映在RMS中。
不以单个材料点、不以多个点的平均速度，也不以虚构的接触点包围区域代替真实支撑。
原始力、作用位置、速度保留独立审计；力不重新定义接触。零合力或缺失逐约束力时，
支撑位置明确未知，不认为静止。旧 raw Newton force 全零且仅有部位合力的记录无法补出
逐接触点受力，必须按新增 `solver_support_semantics` 重新采集。

当前 `newton_mjwarp_force_bearing_point_motion_v3` 同时保存求解前、求解后作用点速度，
以及由同次约束评估的 `xpos/xmat` 换算的 Newton 刚体局部作用点坐标；不能用积分后位姿反算。
`cvel` 是受力修正前的速度；主运动指标使用本次求解后的 `qvel`，经实际 `cdof` 和
刚体祖先链构造作用点速度，不运行额外物理步、不修改求解器状态。力与作用位置来自
同次求解，几何仍是该次约束评估的积分前几何；不得与记录的积分后位姿冒充严格同帧。
真实主表面通过同一原生场景有限三角面映射和共享主表面选择器归属；所有原始约束点保留。
正式分析入口为 `scripts/analyze_recorded_support.py`，缺少受力、速度、实际场景映射或
当前 Newton 版本证据时明确拒绝，不能用旧部位合力或裸距离重新生成支撑点。
它输出逐部位速度中位数／P95／P99；这些分布尚不是自动验收阈值。当前录制仅保存
每个控制周期最后一次物理求解，不能将速度样本冒充完整子步积分的滑移路程。

Motion Edit 的 `support_slip` 只读报告引用经过校验的 motion 和实际 Newton 标签，
显示每个接触样本从当前姿态到下一姿态的材料点切向位移箭头，统计最低／中位／最大。
绿色最低点只表示几何支点，橙色其他点不自动判为错误；箭头倍率不改变统计数值。
旧 `support_motion` 固定点曲线仍是历史材料点诊断，不作为新滑移标记。

已退役的几何聚合实验曾使用源受力证据限定支撑部位和表面，
候选可在同一末端的实际接触子体之间转移，不固定源受力的碰撞子体。
并定期重新查询候选的实际 Newton 接触点。其运动区间为 `[start,end)`；
终点姿态接触仍保留，但不误算完成事件后的下一步。每步对同一个候选材料点计算切向运动，
取真实接触样本中的最小值：允许绕实际支点转动，不插值构造虚拟支点，也不将
不同 witness 的位置变化记作滑动。缺少实际支点单独报告，由接触恢复约束处理，
不能当成零滑动通过。此项仅衡量几何上是否存在低运动支点，不能认证整个接触
区域无滑动或候选的实际受力分配。policy 只用于提取原始执行轨迹；不使用旧 policy
跟踪聚合结果作为此次验收。

以下 `force_support_v3_20260930` 为历史源材料点实验，其优化器和专用模块已退役。
当时每步跟踪同一个局部材料点，但允许后续步骤换支撑点；
逐点平方切向运动按实际源法向力加权，不先平均速度制造虚构静止支点。
每条执行先独立归一化受力，再等权聚合执行，接触点多的执行不会多占票数。
此运动项不包含接触距离目标；接触激活、约束分配、共享主表面和全身穿透仍由新 Newton
查询独立验证。聚合候选的材料点运动是源受力位置上的 FK 诊断，不能声称候选执行后
仍有相同受力分配。对照版只关闭这一运动项，其他目标一致。
正式数据集不会自动切换到实验结果。

### 承重点事件聚合对照（2026-09-30）

`force_support_v3_20260930` 补录原 `model_06000.pt` 的16环境完整执行，15条完整轨迹
参与拟合；末段仍裁剪到931帧，事件假设、双接触完成要求保持。两版均采用相同的
200步优化和原生接触／全身避碰要求；对照版仅关闭承重点运动项。
多接触完成要求按其真实部位分别提供几何恢复引导，使用实际 `includemargin` 上界，
不把整个脚已接触当作脚尖、脚跟同时成立，也不吸引到固定1 mm。
源材料点贴面引导在候选点缺失时仍提供梯度，但其本身不生成接触真值。

| 同一源受力位置／权重下的 FK 运动指标 | 聚合前 | 对照版 | 承重点版 |
| --- | ---: | ---: | ---: |
| 全部支撑阶段合计（cm） | 39.670 | 65.774 | 18.961 |
| 320–470左脚（cm） | 3.933 | 7.488 | 2.710 |
| 320–470右脚（cm） | 6.153 | 6.109 | 5.567 |

两版均通过33项事件检查，926–931帧脚尖／脚跟持续双接触，初始化 Newton 模型未检测到
穿透；逐帧接触170／171附近仍有两条差异，但事件级验收通过，不能把它们另立成事件。
承重点版最大关节步进12.237°（源12.179°），最大二阶差分6.536°（源7.266°）；
原有不连续性尚未完全消除。右脚320–470阶段改善有限。
这些数值不等同于候选接触部位的实际滑动，旧承重点在候选转动后可能已不再支撑。
正式实验目录包含 `source/`、新的固定姿态 Newton 标签、完整求解器补录证据、对照和
哈希 manifest；`training_ready=false`，训练入口和现有数据清单未变更。

### 实际接触支点聚合对照（2026-09-30）

`support_site_retry_20260930` 复用同一15条原始 rollout 受力证据和事件对齐，
两版各优化200步，只开关候选接触支点运动项，每10步重查候选实际接触。
完整导出后的独立 Newton 复查，源／对照／新约束累计几何支点运动分别约
24.57／58.16／3.30 cm。三版767个支撑步骤均无支点缺测，事件验收通过，
926–931帧均保持脚尖／脚跟双接触。两版候选未检测到穿透、无整帧失去全部接触；
仍有两条逐帧接触差异，按既有事件噪声规则处理。新约束版最大关节步进12.231°，
二阶差分6.529°，源既有的关节跳变未解决。

`site_audit.json` 保存阶段统计与事件结果，`*_candidate_sites.npz` 保存实际支点
局部坐标供审计。指标是实际接触样本中的低运动支点存在性，不代表整片接触区域
无滑动或承重成立；原始 rollout 的受力运动仍独立分析。此实验不执行 policy 跟踪，
不切换训练清单，`training_ready=false`。

生成应限制支撑部位自身的切向滑移，同时允许真实支点上的转动与支撑位置转换。
源材料点云的运动保留为诊断，不能直接解释为候选滑动。这是轨迹优化要求，
不是新的接触定义。

### 历史统一接口复查（遗漏开头支撑，不能作为当前验收）

`support_motion_v3_validation.json` 使用18个显式支撑事件和该事件目标表面的全部
实际接触样本，不再沿用旧实验的源受力子体限制或端点后多算一步的区间。
源／对照／新约束几何支点累计分别为21.32／55.57／3.28 cm；
当时源和新约束均在旧源统计预算内，对照13段超预算。旧预算范围约0.07–6.58 cm，
按各事件源每步P95和时长生成；当时 `SourcePhaseMotion` 对源轨迹的损失为0。
上述统计未覆盖0–103；当前已补齐实际持续接触时段，并改用同部位 median+3MAD，
旧“3.28 cm”和旧预算不能代替完整轨迹的新验收。
新约束版143–171左脚支点累计约0.059 cm，但区域样本中位累计约4.578 cm，
说明不能单凭支点统计给出整片无滑动结论。该分布也不代表受力比例。

`original_loaded_motion_episodes.json` 另保存原始执行的受力点最低、最高及加权RMS
速度和各部位P95/P99；只统计各环境首次不中断执行，重置后样本排除，失败执行标记。
候选无同次求解受力，始终明确未知，不把源的受力权重转成候选滑移真值。

## 编辑约束

- 第 0 帧参与与后续帧相同的 IK；不再用独立 `initial_qpos` 覆盖、锁死输出首帧。
- 接触编辑窗口不缩短源接触区间。
- 按末端合并连续接触段；足跟、足尖和碰撞形状切换不重置支撑。
- 同一段使用一致的刚性编辑与源材料点运动，保留示范中的滚动。
- 单点材料样本补姿态约束；线接触的滚动自由度保留。
- `support_residual_scale` 默认 100，是优化残差系数，不是接触距离阈值；有限权重不保证接触可行。
- `support_rotation_policy=episode_yaw` 保持整段编辑航向；`authored_surface` 是待验收的替代方案，不能因材料点误差更小就认为更好。
- `support_origin_policy=median` 使用整段参考平移中位数；`landing` 使用段首参考，防止后续参考改变首次落点。
- 接触任务只作用于源 Newton 观测对应的接触区间。非接触期由全身姿态参考、修正连续性和碰撞目标处理，不另造接触前的位置／朝向引导或从已求解姿态混合出的接近目标。
- 时间差分只引用真实历史：第 0 帧不施加速度／相邻帧平滑项，第 0、1 帧不施加加速度项，关节与 root 同步处理。源姿态可作求解初值，但不能冒充输出轨迹的负帧历史。
- `native_hard_release` 会按源帧号禁止提前接触。其优化约束不能代替真实接触时序的 Newton 查询。

## 验收

### 目标去重版本

计划 metadata 的 `augmentation_objective=consolidated_v1` 启用去重目标；
未指定时拒绝生成，不再隐式选择旧目标。去重版本尚需逐轨迹 Newton 验收。

最终 IK 按四类职责组织：

|职责|内容|去除的重叠|
|---|---|---|
|末端任务|接触期表面法向、切向支撑、有限表面边界、退化材料点的朝向补足|支撑期用 episode 权重替换弱表面参考，不再叠加另一份 XYZ 残差；不再优化额外逐帧落点坐标或接触前引导|
|姿态参考|Laplacian 全身参考、关节/root 偏移代价、非支撑期足部朝向|支撑期关闭重复的整足朝向先验|
|修正连续性|相对参考的关节速度/加速度、root 修正速度/加速度|不使用绝对相邻姿态静止项，也不再叠加落点时序项|
|碰撞|相对源参考的加深惩罚、自碰撞|关闭双向距离相似吸引；参考距离仍用于计算允许的源几何基准|

接近和支撑共享编译出的材料点目标，接近阶段不会修改真实接触 mask。
法向支撑强度保持原来的 episode 系数，避免去重时同时放松法向接触要求。
Laplacian 是目标传播阶段，末端按 episode 保持、材料点固定采样分别解决时间身份和空间身份，不能当作重复机制删除。
此版本仍是逐帧 IK，不代表整段联合优化或动力学执行。

### 实际检查

正式检查使用 `scripts/analyze_contact_region_motion.py` 的材料点区域统计、
`event_acceptance` 的事件级检查及原生查询的 `full_robot_separation`。
旧逐帧一致性审计入口已退役。完整对齐轨迹需分别检查：

1. 校验权威资产、motion/labels 哈希绑定和当前 Newton 接触语义。
2. 通过共享主表面选择器读取真实有效接触；保留原始点对审计。
3. 分开报告缺失／新增接触、未分配约束、无接触帧与原始穿透距离。
4. 在相邻帧跟踪同一个材料点，按末端报告相对源运动的额外位移。
5. 报告真实首步位移。IK 返回成功不能替代上述检查。
6. 扫描全部相邻帧的末端位移、关节增量及二阶差分，覆盖接触建立、释放和摆动阶段。持续支撑稳定不能掩盖接触切换时跳变。

没有固定 handle 时，旧 `fixed_contact_drift` 指标现在是 `null/not_applicable`，不再给出零漂移的假象。
这些检查不会自动设置 `training_ready=true`；接触激活不等于承重、无滑动或动力学可执行。

## 裁剪

`scripts/compact_edited_stationary.py` 接受已有证据支持的站立区间，移除 `(start,end]`。
默认沿用已验证过的16帧五次函数软连接与 root SLERP：将保留起点之前的窗口
逐渐混合到停滞区间末端对应窗口，保留后续原始动作和裁剪后的总帧数。
窗口取决于显式区间，不再硬编码旧1005帧版的481/681边界。
混合后的姿态属于编辑参考，完整保存两组源帧和权重，并重新通过权威资产执行 Newton FK。
`--blend-frames 0` 仅导出不连贯的精确帧选取诊断，不能作为默认训练连接。
每帧保留 `source_frame_indices`，记录切口 `seam_frames` 和连续片段范围。
混合帧不能声称等于某一原始录制帧；不继承原始受力和支撑速度。
软连接后重新计算连续速度；精确帧选取时速度只在各连续片段内部计算。
裁剪产物必须重新查询 Newton 接触并重建事件，不能沿用旧缓存或拼接旧标签。
未经单独验收的接缝不可作为跨段训练动作。

`scripts/soften_edit_collection.py` 为已有全量编辑补齐上述软连接，逐条核对
双侧持续部位的真实接触、关节单步尺度和相对源窗口的全身穿透，再连接同一事件边界。
`scripts/prepare_predictor_training_data.py` 用重新查询的实际接触证据生成独立清单和缓存。
动作时间表仅提供起止帧。训练标签、接触表面及建立／保持角色从该条编辑示范
自身的当前 Newton 接触提取，不能继承原源事件的接触要求或排除标记。
不再使用 ±3 帧完成边界调整和 accepted/rejected 事件筛选；所有时间段完整保留，
帧级断触不另生成动作段，接触缺失如实记录为缺失，不为迁就旧目标修改示范。
`observed_demonstration_events_v1` 与内容证书绑定同一 motion、Newton 标签和动作时间表。
当前训练默认使用 `training/predictor_demo_observed207_geometry_v2_20261002`：每条 15 段，
全量 3,105 样本。证书只检查观测身份和缓存对应，不证明编辑轨迹具有动力学支撑。
场景几何通过共享向上主表面选择器读取，并在生成缓存和训练启动时对照
实际 Newton 顶面三角形。表面目录顺序不能决定箱顶；原侧面几何缓存已退出默认入口。

正式实验产物放在 `runtime/current/motions/climb00_contact_dataset/repairs/`。
旧 207 条及当前训练 manifest 不因单个候选生成成功而自动替换。

## 统一入口与原生几何修复

### 旧路径退役（2026-09-28）

正式生成只保留 `rollout_authority.generate_contact_aware_pyroki_preview` 实现。
CLI、编辑器和批处理共用该后端及 `generation.contract` 校验：

- plan/taskspace 必须显式声明 `augmentation_objective=consolidated_v1` 和
  `free_surface_contacts=true`。缺失版本不再默认 `legacy`，旧计划必须按事件和真实表面重建。
- 源 motion 必须通过权威 G1 资产指纹校验；IK 不接受其他 URDF。
- IK 不再接受裸 LTE keypoints，不再忽略未解析的语义部位或接触任务。
  链接只按名称或显式别名匹配，不做子串猜测；不把缺失足部球体替换成踝关节。
- 旧 `apply_contact_edit_plan_to_motion` 实现、公开导出、旧导入别名、
  旧 subprocess 调用和重复接触提取实现已退役。
  LTE 模块保留当前接触图和几何阶段使用的内部函数，不保留旧轨迹生成入口。
- 旧格式拒绝测试替代旧入口成功测试。现存 207 个 `design_v2` 计划通过配置校验，
  这仅证明格式可用，不表示增广轨迹通过物理验收。

四个经确认的旧文件已移入系统回收站。旧数据、checkpoint、训练 manifest
未删除或替换；它们也不会因为入口清理而自动获得新的接触语义版本。

`generate-ref --plan ... --output-motion ...` 和
`generate-ref --plan-manifest ... --output-motion-dir ...` 共用
`generate_contact_aware_pyroki_preview`。旧 LTE CLI 与旧增广队列已移除；
不能用 proxy 误差、旧硬锚点元数据或源接触图把候选直接标为合格。
每个候选的 `.generation.json` 保存实际求解目标与 `training_ready=false`。

`native_contact_refinement.refine_trajectory` 对候选进行软轨迹修复：

- 保持接触按事件区间表达，足部形状切换及帧噪声不重新切事件。
- 每次更新重查当前政策配置初始化的 Newton 场景，验证 fingerprint。
- 全身环境碰撞、自碰撞及接触激活均取实际 witness；未分配接触报错。
- 接触／避碰统一复用 `contact_solver.contact_regions.unified_region_objective`，不再调用激活后归零的 `selected_contact_activation_deficit`。
- 通过完整 Newton witness 张量读取全身点对，真实主表面使用 source-triangle normal-fan；不再用旧 HTTP 精简点对训练。
- 缺少候选点对时，用权威碰撞区域的完整表面采样向事件绑定的有限主表面提供梯度；缺少源目标则报错，不把无候选记为成功。
- 接近目标按真实主表面绑定。不同固定子链接的材料点先经 FK 转到共同末端坐标，不能直接混用局部坐标。
- 保留相对原候选的支撑材料点运动与二阶连续性。毫米尺度仅归一化损失，不定义接触阈值。
- 周期性复查全部帧；只有无全身穿透、所有事件接触实现且无缺失候选才通过。

这仍是运动学修复。后续须重新写出 canonical motion、执行实际接触重标注，
再验证轨迹连续性与动力学执行；不能把静态复查成功等同于可承重或可执行。

### 共享区域损失接入（2026-09-28）

`regional_contact_refinement.EventContactRegions` 只把训练中的单箱接近几何
适配为事件绑定的任意有限水平面；区域划分、区间上下界、固定区域归一化、
稳健惩罚及双侧梯度聚合直接调用共享实现，不复制第二份训练损失。
当前共享优化区间上界为实际 `includemargin`，下界为零；此前区域路径仍使用
0.05 倍 margin，已在 2026-09-29 改正。区间仅用于优化，真实接触仍由
`CONSTRAINT`、`dist < includemargin` 与约束分配决定。边界处零残差不等于激活。
原有支撑材料点运动、相对参考连续性和姿态偏移正则保留。

正式复现入口为 `scripts/refine_motion_contacts.py`，需要显式传入 motion、
events、support-phases、event-reference-contacts、taskspace、checkpoint、manifest、
binding、continuity-reference 与新的 output 目录。
它启动专用 Newton 查询 worker，不启动训练；`--audit-frames` 保存同一次查询的
CPU 与设备接触证据交叉核验。历史 tmp 驱动所用的精简 HTTP query 已不满足
新接口，不能作为备用修复路径。所有输出保持 `training_ready=false`。

验证记录：

- h110 v4 成品经当前完整 witness 查询，初始即为零缺接触、零穿透，
  因而未执行优化；这不是新损失的修复收益，旧报告不能直接作为当前失败真值。
- 从 `motions_full` 未修复 h110 候选重跑 240 步：最大穿透
  12.668 → 1.203 mm；缺接触部位帧 16 → 0（第 32 步后始终为零）；
  最终仍有 32 帧存在负距离，`passed=false`、`training_ready=false`。
- 第 282、283 帧同一次 Newton 查询的 CPU 与设备主表面接触结果一致；
  全部原始查询结果保存在输出目录的 `*_same_pass_audit.json`。
- 结果目录：`runtime/current/motions/climb00_contact_dataset/versions/`
  `event_consensus_207_20260928/native_refinement_v7raw/`
  `pair48_10_h110_dp0p000_ap20_ru_m050/`。未替换训练 manifest，未启动剩余 204 条。
- 完整 Motion Edit 与共享区域目标测试共 423 项、3 项子测试通过；随后新增的
  通用有限面／训练单箱接近代价等价测试也通过。predictor 文件未修改。

### 事件三态与源轨迹连续性修复

增广事件是稀疏意图，不能把所有非 required 的部位当作 release。适配层显式
传入 release mask；未约束的部位只参与全身避碰，不被强制离面。共享训练目标
不传该参数时保持原有完整接触计划语义，predictor 未改动。

显式释放采用 `release_events`，每项为 `part_index`、`start_frame`、
`end_frame_exclusive`、`scope: whole_endpoint`。与 required 重叠时报错。
旧 touchdown 的 `release_frame` 不用于推导整个末端释放：真实数据中它可跨越
该部位在其他表面的有效支撑。没有显式释放证据就保持未约束，不补造离面要求。
显式释放的优化下界为实际 pair `includemargin`；验收仍读取实际有效接触，
任一向上主表面的已实现接触都会计为 release violation。

`--continuity-reference` 必须指向未经过增广 IK、逐帧对齐的聚合源 motion。
校验权威资产、帧数、fps 和关节名称，重排到 canonical joint order。
关节一阶、二阶差分对齐源轨迹；静态偏移允许存在。编辑后的候选继续作为
姿态偏移及世界空间材料点参考，不把未变换的源 root/世界坐标套到新地形。
配置源连续性时执行完整优化步数，不因静态接触提前通过就保留跳变。
报告独立记录真实接触、穿透、释放违例和连续性残差；`passed` 仅表示静态
几何验收，所有候选仍为 `training_ready=false`。

h110 同一未修复候选重跑 240 步，结果保存于
`native_refinement_v8/pair48_10_h110_dp0p000_ap20_ru_m050/`：

| 指标 | 初始 | 旧版 240 步 | 三态＋源连续性 240 步 |
| --- | ---: | ---: | ---: |
| 最大穿透 mm | 12.668 | 1.203 | 0.634 |
| 存在穿透的帧数 | 34 | 32 | 45 |
| 缺接触部位帧数 | 16 | 0 | 1 |
| 最大单帧关节变化 ° | 22.384 | 22.368 | 16.292 |

新版缺接触为第 171 帧左手，仍未通过静态验收。关节速度残差
0.9060 → 0.6628，二阶差分残差 0.2405 → 0.1082；改善不等于跳变已消除。
旧版延长到 640 步仍为 22.383°，因此仅延长旧优化不能解决参考错误。
当前事件文件没有明确整个末端释放事件，5112 个未约束部位帧不再被误当作释放。
`comparison.json` 保存全部过渡的关节和末端连续性审计及旧版对比。
426 项回归测试、3 项子测试通过；静态仓库检查无错误。未替换训练数据，
未启动剩余 204 条增广。

### 缺失点对材料点引导与输出平滑

`event_material_recovery_output_smoothness_v1` 在没有匹配目标主表面的 Newton
点对时，使用事件已有的局部材料点及有限面内部目标进行软恢复。没有事件材料点
则报错，不虚构接触目标。恢复点对后关闭该补充项，继续共享区域损失；引导不会
修改真实接触标签或放宽激活条件。归一化读取本次查询的实际 configured margin。

末端平滑从“相对坏候选的二阶差分”改为“输出材料点的二阶差分”，避免把原 IK
跳变当作必须保持的参考运动。支撑运动项、源关节差分和姿态先验保留。
所有修改作用于通用增广修复，不包含特定帧号或部位的例外规则。

h110 从同一未修复候选运行 240 步，产物保存在
`native_refinement_v9/pair48_10_h110_dp0p000_ap20_ru_m050/`。

| 指标 | v8 | v9 |
| --- | ---: | ---: |
| 缺接触部位帧数 | 1 | 0 |
| 最大穿透 mm | 0.634 | 0.691 |
| 存在穿透的帧数 | 45 | 49 |
| 第 29 帧最大关节变化 ° | 16.292 | 16.161 |

第 171 帧左手接触恢复；29、171、282、283 帧同一次查询的 CPU 与设备
主表面筛选结果一致。源关节二阶差分残差为 0.1006（v8 为 0.1082），
但速度残差为 0.6709（v8 为 0.6628），不能概括为连续性全面改善。
本轮解决缺接触，跳变改善很小，残余穿透略差；仍为 `passed=false`，
不能发布为可用训练增广。`comparison.json` 保存完整连续性及接触审计。
428 项测试、3 项子测试通过，仓库静态检查无错误。正式数据与 predictor 未改。

### 当前验收：动作的建立／保持／释放

`action_contact_acceptance_v1` 取代未采用的源接触阶段模板方案。只根据每个动作的
source/target surface 与 touchdown 意图检查建立、保持、释放；不比较源轨迹的
阶段数量、各阶段长度或精确触地帧。开始和结束都在同一表面的非活动末端属于保持，
因此不会因旧 `persistent_parts` 遗漏含噪支撑而完全不检查这段。

源示范仅标定统一时间容差：已声明保持的动作中，同一末端／表面两次实际接触间
的有界空缺；当前数据最大为 6 帧（50 Hz 下 0.12 秒）。不同表面的接触不能当作
空缺填补，裁剪缺口不跨越。稳定建立／释放沿用已有 3 帧证据要求，在动作完成
附近的容差窗口检查，不硬卡单个终点。保持检查允许动作交接边界的同一时间容差，
但要求中间持续支撑，不能把长期断触当噪声。全部原始 Newton 激活、分配、点对保留。

优化与验收使用同一动作要求，并抑制已被事件容差接受的边界偏移／短断点修正。
正式修复入口新增必填 `--event-reference-contacts`；`--audit-only` 仅重新查询并
验收，不更新姿态。旧阶段模板 schema 直接拒绝，没有备用入口。接触事件、全身
穿透分别报告；通过接触不代表滑动、跳变或动力学已合格，仍不自动发布训练数据。

`event_acceptance_v11/` 保存当前合同和复核报告。对已保存的完整、同姿态 Newton
点对重新验算（校验激活、分配、q 与文件哈希），不复用旧通过标志：

- 源示范：44 项动作接触检查全部通过。
- v8：43/44，通过建立左手接触；失败为动作 0004 右手保持，183–193 连续 11 帧断触。
- v9：41/44；失败为动作 0000 左脚保持和动作 0004 双手保持。
- 旧版“第 171 帧左手失败”不再作为单帧否决项；穿透和连续性数值没有改变。

这些是新验收结果，不是新一轮轨迹优化结果。旧 `event_acceptance_v10` 阶段匹配
报告仅为已弃用实验记录，不是当前验收基线。

### 冻结旧 207 的同条件基线

基线固定为归档 `temporal207_training_v2_ready` 的原始姿态，207 个 motion 文件
逐一核对内容哈希。`baseline207_comparison/baseline_lock.json` 记录完整清单；
这是比较基线，不是重新声明全部旧数据已通过物理验收。

首个对照为完全同名 h110：`pair48_10_h110_dp0p000_ap20_ru_m050`。
旧轨迹在与 v8/v9 相同的 terrain、canonical robot、Newton model fingerprint
中重新查询。`protocol.json` 冻结相同动作合同及验收代码哈希；全部结果使用当前
同一验收重新计算，不比较不同版本报告中的旧通过标志，也不修改原姿态。

| 指标 | 旧 207 | v8 | v9 |
| --- | ---: | ---: | ---: |
| 动作接触检查通过 | 43/44 | 43/44 | 41/44 |
| 最大穿透 mm | 3.265 | 0.634 | 0.691 |
| 最大单帧关节变化 ° | 11.579 | 16.292 | 16.161 |
| 第 29 帧右髋 yaw 变化 ° | 0.806 | 16.292 | 16.161 |
| 第 0→1 帧 root 位移 mm | 52.209 | 6.181 | 13.170 |

新版减少最大穿透及开头 root 跳动，但新增严重关节跳变；v9 还减少了通过的
接触要求。新生成器未经 refinement 的候选在第 29 帧已跳变 22.384°，所以退化
起于生成阶段，后续修补仅部分减小了它。不能以局部改善认定整条轨迹更好。

本轮决策：v8、v9 不晋级。实验修复仍只由独立脚本调用，不接入正式生成或替换
训练 manifest。后续只针对旧基线可复现的缺陷做单项修复；固定同一比较协议，
出现新增跳变或支撑退化的候选不采用。此结论只覆盖本条同条件对照，不能推及
全部 207 条或动力学执行。完整证据及决策见 `baseline207_comparison/comparison.json`。

### 从旧基线直接修复的首轮结果

`baseline207_repair_v1/h110/` 从旧 207 原 h110 姿态直接运行 240 步，未经过
引入第 29 帧跳变的新生成器。场景、动作合同、验收代码与冻结协议一致，未改
通过条件。本轮没有再修改求解器代码。

| 指标 | 旧 207 | 旧基线修复 240 步 |
| --- | ---: | ---: |
| 动作接触检查通过 | 43/44 | 44/44 |
| 最大穿透 mm | 3.265 | 1.390 |
| 第 29 帧右髋 yaw 变化 ° | 0.806 | 0.689 |
| 第 0→1 帧 root 位移 mm | 52.209 | 25.909 |
| 最大单帧关节变化 °（均在第 891 帧） | 11.579 | 11.827 |
| 原始负距离帧数（诊断） | 13 | 69 |

右手保持事件修复，且未引入此前第 29 帧的大跳变；但最大关节步长略增，负距离
分布仍需检查，不能只用最大穿透下降宣称全面改善。`passed=false`、
`training_ready=false`，不替换原数据、不批量发布。`comparison.json` 保存与旧基线
的完整逐项比较，`motion.npz` 为本轮候选，原始旧 207 文件保持不变。

### Historical source-relative expansion (invalidated, 2026-09-28)

The old runner initialized from archived `temporal207_training_v2_ready` poses
and measured additional motion relative to the new source. That failed to inherit
the rebuilt source trajectory. This initialization path has been removed; the
archive index now supplies only edit IDs, never initial poses.

Support motion compares the same local material point at adjacent frames with
its source displacement, transformed by the authored surface rotation. Only the
tangential difference is penalized. The cumulative residual resets on an
endpoint/surface episode boundary, not a change of foot contact sample. This
preserves source rolling while exposing slow accumulated augmentation drift.
The source pose sequence is not edited. Historical labels are not reused as
current contact truth.

The old relative-motion acceptance could pass an already sliding candidate and
its inferred keep roles did not match the v5 phases. Those reports do not certify
support supervision. Current acceptance checks absolute material motion over
explicit v5 phases and uses the accepted h110 residual-depth allowance; it does
not expand that allowance to match a poor initializer. Neither metric changes
the native contact activation rule. Refinement retains its best audited iterate.

Historical batches live under `source_support_expansion*`; their reports cannot
be reused by the corrected objective. Missing terrain bindings are
created by the current Newton worker before refinement.
Full poses and cropped poses are retained (remove 472–699, retain 471), and the
crop contract forbids supervision across the seam. Cropped contact labels remain
explicitly pending fresh native relabeling; `training_ready` stays false.

### Concurrent expansion

`expand_source_preserving_augmentations.py --workers 3 --query-worlds 32`
uses three trajectory jobs, each with a 32-world native query worker. Scene
manifest/binding preparation and taskspace compilation are serialized to avoid
shared-cache races; optimization jobs overlap. Completed reports are reused
without recompilation. Each optimizer uses one CPU thread; the launcher limits
OMP/OpenBLAS/MKL threads to one. `status.json` lists all active jobs, and new
acceptance reports record wall-clock seconds (including preparation/waiting).
The historical run was `source_support_expansion_parallel_v4`, reusing v3 results.
Concurrency itself did not change acceptance thresholds or optimization losses.
Its support contract was subsequently invalidated as described below.

### Correction: inherit the explicit v5 support phases

The old expansion reports above do **not** certify support-preserving training
supervision. They reconstructed `keep` from endpoint surfaces and initialized
from old augmented poses. In particular, the source's first 0–103 action has no
persistent support role; interpreting its left foot as `keep` was incorrect.

The corrected runner requires `--support-phases` and verifies exact agreement
with the source event roles (18 keep phases and 15 establishment events here).
The contact contract is now `explicit_action_contact_acceptance_v2`: identical
endpoint surfaces cannot invent a keep or release role. The initializer is a
fresh authored IK edit of `source/motion.npz`; source hashes and edit provenance
are stored. Old augmented candidates cannot be reused under this contract.

Optimization additionally uses an absolute material-motion budget on source
contact samples for explicit keep phases. Acceptance independently measures
same-material tangential motion from fresh Newton witnesses, grouped per endpoint
and frame, then accumulated within each declared phase. The 6 cm path budget is
an augmentation motion criterion, not a contact threshold and not the old
endpoint-only 6 cm statistic. Native geometry witnesses can measure motion during
short activation holes without changing the event-level contact evidence. Missing
motion measurements are reported rather than credited as zero motion.

The aggregate source has zero loss under the new motion-budget term. The initial
rigid-transform-only diagnostic did not adapt sufficiently to edited terrain;
The complete source IK initializer was first tested in
`explicit_v5_source_pilot_v3`. Its three candidates passed contact events, but
none passed the combined geometry/support acceptance. These remain diagnostics.
`explicit_v5_source_pilot_v4` additionally normalizes the phase-motion violation
by the same 1 mm optimization scale used for geometric residuals (not by the
6 cm motion budget), and selects lower penetration among otherwise equally
passing candidates. This is objective version `explicit_phase_source_recovery_v3`;
older objective reports cannot be silently reused. Initializer source/plan hashes
are checked before refinement. The editor's object-typed joint names are converted
to Unicode before refinement so that its inputs require no pickle loading.
The subsequent smaller-step diagnostic exposed another mismatch: source material
samples can cease to represent the edited trajectory's contact location. Objective
`explicit_phase_source_recovery_v4` refreshes these samples from each native audit,
using exactly the same material-motion calculation for optimization and acceptance.
Missing native groups retain source samples only as guidance, never as validation.
Every complete phase is checked for numerical agreement of both metrics.
Cached results must also match hashes of the source, native labels, event file,
support phases and edit plan, as well as optimization steps and learning rate;
matching an old motion ID alone is insufficient.

Validation completed in `explicit_v5_source_pilot_v6` (three concurrent native
32-world queries, 1200 optimization steps, learning rate 2e-5). All exported
candidates passed 33 contact-event checks and 18 explicit support-motion phases.
The maximum discrepancy between optimization and audit motion metrics was below
4e-9 m. The selected iterate is re-queried before export.

| Terrain height | Maximum phase path (cm) | Maximum penetration (mm) | Reference accepted |
| --- | ---: | ---: | --- |
| 90 cm | 5.923 | 0.518 | yes |
| 100 cm | 5.482 | 2.608 | no: residual geometry |
| 110 cm | 5.984 | 4.221 | no: residual geometry |

This validates the support inheritance/measurement correction, not universal
edit feasibility. No 207-case expansion was launched from these pilots. The
accepted reference still requires fresh cropped contact labels and downstream
execution validation; no training manifest was replaced. Joint-step acceptance
only means no increase relative to the allowed reference, not dynamics safety.
The previous 207 trajectories remain historical diagnostic artifacts, not newly
certified predictor supervision. No training manifest has been replaced.

### Actual activation interval experiment (2026-09-29)

The regional objective and both missing-region geometry proxies now use the
actual pair/configured `includemargin` as their upper approach boundary. This
matches the existing non-regional execution interval. Activated contacts with
positive gaps inside this boundary have no extra attraction toward 1 mm.
Full-body penetration and explicit release penalties remain unchanged. Native
activation/allocation are still mandatory; zero loss at the exact boundary does
not certify contact. Frozen witness gradients remain active if FK exits the band.

`activation_interval_pilot_v1_20260929` repeats the same three v5 IK initializers,
1200 steps and 2e-5 learning rate against `explicit_v5_source_pilot_v6`. Only the
contact approach upper bound changed; the support budget and regularizers did
not change. Reports use `explicit_phase_source_recovery_v5` and
`native_activation_upper_gap_v1`, preventing reuse of the old 1 mm objective.

| Height | Last-phase left-foot material path, old → new (cm) | Max penetration, old → new (mm) |
| --- | ---: | ---: |
| 90 cm | 5.026 → 4.168 | 0.518 → 0.000 |
| 100 cm | 4.847 → 4.094 | 2.608 → 0.009 |
| 110 cm | 5.984 → 4.138 | 4.221 → 0.030 |

The source left-foot phase path is 4.124 cm. All three selected candidates pass
33 native contact-event checks, 18 support-motion phases and the existing
reference checks. Maximum phase paths are 5.501, 5.394 and 5.726 cm respectively.
Joint velocity/acceleration residuals improve in all three cases. Maximum joint
steps rise by about 0.056° in the 90/100 cm cases and decrease in the 110 cm case;
the latter still has an 18.746° step, so this result does not resolve all
trajectory-continuity concerns. No training manifest was replaced or full 207
expansion launched. `comparison.json` records the complete per-case comparison.

### Production replacement

Historical pivot/minimum-point experiments below do not define current loaded
material motion. The native-source collection uses the shared
`somaforge_core.loaded_material_motion` implementation: load-weighted RMS of
same-body-local material-point tangential steps, accumulated by declared phase.
Known unloaded samples do not become fixed support; missing load evidence is
unknown. Soft-joined output is recomputed on its own pose clock. A join crossing
a deleted source span lacks adjacent original solve evidence and cannot inherit
a passing source motion verdict. These are original-load geometric references,
not edited execution forces or a physical no-slip certificate.

The actual-margin interval is the sole regional implementation, rather than an
optional pilot mode. Both `scripts/refine_motion_contacts.py` and
`scripts/expand_source_preserving_augmentations.py` take their defaults from
`RefinementConfig`: 1200 steps and learning rate 2e-5. The objective schema is
shared by both entry points. Reuse requires the current interval schema on the
audit and acceptance report, as well as identical source/plan hashes and optimizer
parameters; old 1 mm results cannot be resumed as current candidates.

The full expansion is stored separately as `activation_interval_207_20260929`,
reusing only the three verified current-interval pilots. Fresh initialization
comes from the event-aligned v5 source. Cropped native contact labels and downstream
validation are still required before publication as training data. Existing
predictor data and training entry points are not changed by this replacement.
# Predictor 参考路径验收

`scripts/prepare_predictor_event_paths.py` 从源事件与当前 Newton 标签生成
整段参考路径证书及接受/拒绝事件表。明确 keep 约束必须通过；未声明 keep
的源实际接触时段只提供候选支撑，允许多个区域在事件内部接替覆盖。
材料点、脚尖/脚跟切换不会另起事件；拒绝段和裁剪间隙不会被跨接。

`generator.training_data.prepare` 读取新 manifest 的 `support_path_acceptance`，
核验证书、motion/contact/事件内容和缓存帧边界。旧 manifest 会明确报告
`legacy_unchecked`。这个证书验证示范整段，不验证网络尚未输出的中间轨迹。

当前正式切分位于 `versions/predictor_event_paths_20260930_final`：14 个接受
事件，917–931 完成/调整段排除。`training_ready=false`，没有替换训练缓存。
源相对预算不能证明源质量，0–103 的候选左脚几何 pivot 路程仍为 12.57 cm；
候选没有同次求解载荷数据，承重滑移依旧未知。

### Current support evidence contract (2026-09-30)

The preceding absolute keep-phase budget descriptions document historical v6
experiments. Current v7 preserves original phase motion and penalizes added
geometric motion; contact keep intent does not imply stationary measured support.
Collection, source load observations, edit invalidation, predictor supervision
and endpoint diagnostics share [the support evidence contract](../../../docs/support-semantics.md).
Edited candidates require actual execution to establish loads. Neither a geometric
pivot budget nor endpoint retention certifies loaded patch slip. Existing corpora
and training manifests have not been automatically migrated.

### Native execution edit collection (2026-10-01)

`scripts/expand_source_preserving_augmentations.py` now rebuilds a collection from
explicit source evidence. The historical dataset/archive invocation and implicit
1200-step refinement defaults described above no longer apply to this entry.
It requires the native source motion, current Newton labels, original support
observations and recording provenance, source events, previous authoring events,
authored plans, scene-initialization checkpoint and a fresh output directory.
`--resume` verifies the complete input signature; old augmented poses are not
initializers. `scripts/refine_motion_contacts.py` remains a separate explicit
refinement tool.

The collection `native_support_edit207_20261001` uses the unchanged 932-frame
`raw_execution_reference_20260930` source. All 207 surface edits retain their
authored height, approach and heading parameters. Event correspondence maps edit
metadata onto the native clock; it does not resample or optimize source poses.
Shape fragments and surface bindings are rebuilt from the source's current
Newton labels. Generation uses the consolidated contact-aware IK, 25 iterations
and joint acceleration weight 8. Dedicated persistent workers cache compilation
only. `--workers` controls IK generation (default 3); `--label-workers` controls
independent terrain queries (default 5). Each query model has one Newton world.

The approved middle hold is specified as `--hold 467 699`: source frame 467 is
retained and frames 468–699 are removed. Each full 932-frame edit has a 700-frame
crop with two independent clips `[0,468)` and `[468,700)`. Positions are copied
without seam interpolation, velocities are differentiated separately, and events
cannot cross the cut. The original completion/adjustment event 917–931 is retained
for playback but remains excluded from predictor action supervision.

Full and cropped edits receive separate fresh single-world Newton queries for
their exact terrain. The batch retains actual activation/allocation labels and
full-robot terrain/self separation. `generation_complete` means generation and
these audits finished; event failures and geometry residuals remain visible in
each `acceptance.json` and the collection `index.json`.
Full and cropped event checks are reported separately. Cropped completion windows
are bounded by the continuous clips, so stable evidence after the cut cannot
certify an event before it. Dropout tolerance is inherited from the native source
contract; a candidate's own activation holes cannot enlarge it. Resume validates motion hashes and native model
fingerprints of existing labels, including cropped labels.

The audit also evaluates original solver application sites through source and
edited FK. It compares same-material tangential displacement using original
normal-load-weighted RMS of point magnitudes, without cancellation between points.
Unloaded or unobserved source samples are not invented fixed support targets.
These are geometric editing references, not measured forces or slip of the edited
execution. The current `native_observed_source_loaded_material_edit_v3` generator also
uses these sites as a soft IK residual. Each candidate adjacent-pose interval
follows the same source-loaded body-local material points. Positive RMS step
excess accumulates over each declared phase; less movement elsewhere cannot pay
for it. The allowance is one source-step median plus three MADs, frozen from
the original motion. The sequential solver tracks already committed excess;
the full-trajectory refinement tool uses the same phase sum and can update all
poses together. Neither penalty is a hard guarantee: the independent final
audit remains required. The unchanged source has zero added-motion loss.

`scripts/refine_motion_contacts.py` requires `--source-loaded-sites` and
`--source-observations` as well as the exact source phases. Static Newton witnesses
remain geometric diagnostics and cannot replace these original force references.
Cache resume rejects the previous generation schema and changed evidence hashes.
Full and cropped reference audits use their own pose clocks; a deleted-span join
has unknown original solve evidence rather than a passing zero motion value. All edited
physical support stays `unknown_without_new_execution`, and `training_ready` is
false until downstream validation. The batch does not replace a predictor
manifest or start training.

### Loaded material objective validation (2026-10-02)

The production generator and explicit trajectory-refinement loss now share the
original-load RMS definition and positive interval-excess accounting. The old
minimum-point phase acceptance is retained only as a named geometric diagnostic;
static witnesses cannot refresh the load reference. The default residual
normalization is 0.01 m, independent of activation margins and phase allowances.
Known unloaded phases have no motion target; unknown evidence cannot certify a
pass. The unchanged 932-frame source has zero loss and 18 reference phases below
the 6 cm geometric budget (maximum 5.5297 cm).

Two production edits, followed by fresh Newton queries, give:

| Case | Maximum added phase motion, old → new (cm) | Terrain penetration, old → new (mm) | Self penetration (mm) |
| --- | ---: | ---: | ---: |
| `cover144_05_h100_dp50p000_ap10_center` | 1.3729 → 0.1074 | 2.1116 → 2.1162 | 0 → 0 |
| `cover144_08_h110_dp18p750_ap0_center` | 1.1268 → 0.1651 | 3.4106 → 3.4111 | 0 → 0 |

Maximum total phase motion remains about 5.53 cm because original motion is
preserved. Optimizer-side and independent output-FK phase paths agree within
0.001 mm. Both cases still fail two pre-existing event checks. The stricter
source-added-motion allowance passes 15/18 and 16/18 phases respectively; passing
the separate 6 cm budget does not imply all requirements pass. These candidates
do not replace formal training data, and their actual executed support is unknown.

The whole-trajectory loss also has finite gradients on the real generated motion:
one gradient step lowers loss from 9.2293e-5 to 9.0006e-5. The separate full native
refinement run stops because the event request lacks matching authored material
targets for left-hand frames 170–172. Its preflight now reports this before
starting the query worker; it does not invent a face or drop the requested role.
This is not a completed native trajectory-recovery experiment. Detailed evidence
is in `tmp/loaded_material_fix_20261002/validation.json`; the initial stronger
normalization trial is retained and rejected for introducing self penetration.

### Source event binding repair (2026-10-02)

The two failed event checks above also failed on the unchanged source itself.
The authored clock requested right-foot establishment at frame 142 while its own
Newton labels were off at 141–146. The left-hand request at 170–172 likewise
contradicted the source labels; its next observed contact episode starts at 180.
These are inherited role errors, not evidence that IK caused those failures.

Generation and standalone refinement now use
`somaforge_core.demonstration_events.materialize` on each retained source block,
the same materializer used for predictor demonstrations. Input events contribute
only action intervals and IDs for edit correspondence. The original 15 actions,
poses and middle cut remain unchanged. Establishment, keep, observed release and
surface identities come from the source's own current Newton labels. An observed
release is not invented as an additional hard release command. The resulting 14
keep phases bind the original loaded material sites; old 18-phase references are
rejected. The 3-frame stability rule and simulation parameters are unchanged.
Source self-audit must pass before generation. Candidate audits retain these
source requirements and policy; candidates cannot recalibrate them from their
own contacts. Production generation refuses an unmaterialized old event bundle,
and cache schema v3 rejects previous completed edits.

Two edits were regenerated through the production path. Both full and cropped
700-frame motions pass the source-bound event checks (zero failures). Maximum
added material motion is 0.0662 cm and 0.0588 cm; both retain all 14 phases under
the separate 6 cm geometric budget. The stricter source-added-motion allowance
passes 13/14 and 14/14 phases. Terrain penetration is 2.1170 mm and 3.4112 mm,
and self penetration is zero. Original source bytes are unchanged.

The previously blocked whole-trajectory experiment now completes 32 updates.
Its in-process audit reduces maximum penetration from 2.1169 to 0.7014 mm while
retaining event acceptance, but increases maximum added material motion from
0.0662 to 0.2023 cm and reduces source-allowance passes from 13/14 to 5/14.
This candidate is not a successful simultaneous constraint repair. It is kept
as a diagnostic rather than replacing the generated motions or training data.
An independent single-world Newton query of the exported result confirms
0.7014 mm terrain penetration, zero self penetration, zero event failures and
5/14 source-allowance passes. All 14 phases remain under the separate 6 cm
budget; failing the much smaller source allowance is not a 6 cm sliding failure.
Evidence is under `tmp/loaded_material_fix_20261002/observed_events/`.

### Source-relative motion regularization (2026-10-02)

The full-trajectory refiner still penalized absolute endpoint-marker acceleration.
On the unchanged physical source, that weighted term was 0.1651 rather than zero:
it could flatten demonstrated motion. Its support term also measured changes of
four arbitrary endpoint markers, in addition to the original loaded-site term.
A rotation about a stationary loaded material point can move those markers, so
this extra term must not act as another support criterion.

The v9 objective uses acceleration residuals relative to the immutable aligned
source. The source itself has zero loss and gradient; constant position or
velocity offsets remain free, while introduced jumps have restoring gradients.
With original load evidence, `trajectory_support_regularizer` evaluates only the
original loaded material metric. Whole-endpoint markers remain a continuity
prior and a named geometric fallback when original load evidence is absent;
they do not certify physical support. No per-frame reference site refresh, new
contact threshold, budget relaxation, or QP teacher is introduced.

The following runs start from the identical h100 edit and each perform 32 Adam
updates. Independent single-world Newton queries audit their exported poses.

| Objective | Learning rate | Terrain penetration (mm) | Maximum added motion (cm) | Strict source allowance passes |
| --- | ---: | ---: | ---: | ---: |
| Previous absolute acceleration + marker support | 2e-5 | 0.7014 | 0.2023 | 5/14 |
| Source-relative acceleration, marker support retained (ablation only) | 2e-5 | 0.7408 | 0.0737 | 8/14 |
| Source-relative acceleration, loaded material support only | 2e-5 | 0.7408 | 0.2435 | 7/14 |
| Same new objective, smaller step | 2e-6 | 1.7687 | 0.1195 | 8/14 |

All four runs pass event acceptance, have zero self penetration and retain all
14 phases below the separate 6 cm geometric budget. Smaller steps reduce the
new weighted material loss from 8.8013e-5 to 2.7442e-6 (96.9%), but that does not
mean every strict source allowance passes or that maximum added motion decreased
from the initial edit's 0.0662 cm. Removing the duplicate support term repairs
its semantics; the same-step ablation does not establish a numerical improvement
over retaining it. Step size and the remaining soft objectives still permit
tradeoffs. The production default learning rate remains unchanged; the smaller
step is an explicit experiment rather than a general calibration result.

Tests verify source-zero gradients, jump recovery and free rotation about a
stationary loaded point, with 100 relevant tests passing. Original source poses,
contact rules, simulation parameters and formal training data are unchanged.
No predictor training was started, and edited executed support remains unknown.
Snapshots and full evidence are in
`tmp/loaded_material_fix_20261002/regularization_ablation/comparison.json`.

### 2026-10-03: phase budget governs motion acceptance

The approved rebuild uses `native_observed_source_loaded_material_budget_edit_v6`.
Sequential IK and whole-trajectory refinement penalize cumulative loaded material
motion above the shared 0.06 m phase budget. Positive source-relative excess
remains a soft preservation term with weight 0.1, with its median + 3 MAD
allowance. Exceeding that allowance is diagnostic, not a motion acceptance failure.
Unknown loaded evidence still cannot certify a budget pass. Known unloaded
intervals remain unpinned, and endpoint-only predictions still cannot supply a
trajectory path or executed support certificate.

Fresh original policy recording, relabeling, complete augmentation and scratch
retraining are in progress under the separate formal version
`runtime/current/motions/climb00_contact_dataset/versions/loaded_material_rebuild_20261003`.
The recording preserves native poses and actual same-solve loads. Static queries
of edits establish contact geometry; edited execution loads remain unknown.

### 2026-10-04: geometric surface witnesses and production preflight

The fresh recording has 16 keep phases, all within the 6 cm reference budget
(maximum 5.6852 cm). The action clock retains all 15 actions. A transient hand
contact at frame 170 is not treated as completed establishment: the same-surface
stable endpoint is frame 183 in this recording. This changes the action boundary,
not the recorded poses, solver parameters, or contact activation rule.

The geometry loss previously interpreted Newton sphere support points as points
on the physical sphere surface. The current witness conversion reads the actual
contact thickness and subtracts the actual model shape margin to recover only
the geometric radius/offset. It preserves the separate inflated constraint
distance. Subtracting the complete contact thickness had incorrectly penalized
the inflated margin as physical penetration and displaced a valid hand contact;
that failed v5 output is retained for diagnosis and is excluded from the rebuild.
Neither the robot assets nor the simulation margins are changed.

The current witness schema is `newton_geometry_surface_witness_v3`, and collision
reference caches use `newton_collision_reference_geometry_surfaces_v3`. Generation
records and verifies the effective 0.01 m depth residual normalization, four
collision refinements, loaded-material objective, and source/plan hashes before
allowing a completed edit to be reused.

The two production pilot edits pass all 34 source-bound event checks and all 16
material-motion phase budgets. Their maximum paths are 5.6850 and 5.6860 cm,
maximum terrain penetrations are 1.3227 and 1.3255 mm, and self penetration is
zero. Source-added motion remains diagnostic rather than a rejection rule.
These are static geometry and reference-motion checks; edited executed support
is still unknown. The complete v6 generation subsequently finished: all 207
edits passed the material budget, but 80 failed source-bound contact events.
The soft-join stage stopped at its event-preservation check, and formal scratch
training did not start. The two pilots did not establish full-collection quality.

Whole-trajectory candidate selection also uses the new phase metric. It ranks
event failures, unknown phase evidence, phase-budget failures, penetration and
total measured material path. Arbitrary single-point accumulated drift is only
a named geometric diagnostic and cannot select or accept a candidate. Geometric
pivot/region distribution tools remain diagnostics rather than motion acceptance.

### 2026-10-04: remove the added pre-contact guidance

The optional 0.3 s position/orientation approach task, its start-offset mixing,
and its configuration forwarding have been removed from augmentation IK.
Rebinding old plans drops those retired parameters. The generation schema is
`native_observed_source_loaded_material_budget_edit_v8`; v7 results that used
approach guidance cannot be resumed as the new objective. Historical source and
diagnostic outputs remain intact.

Only the existing contact-period task, pose reference, correction continuity
and collision responsibilities remain. The observed material-basis sampling
repair and consistent float64 solve are retained. No new residual or acceptance
threshold is added. Removing guidance alone is not a claim of smooth trajectories:
the preceding no-guidance ablation had larger inter-frame jumps, so the original
continuity solve still needs to account for impending contact constraints.

The production removal passed 58 relevant tests and four fresh native-query
pilots. All four preserved the source-bound contact events and all 16 material
phase budgets. This is not full trajectory acceptance: three pilots have maximum
single-frame body steps of 21.37, 25.32 and 26.27 cm, and one has 2.24 mm maximum
self penetration. For the latter two body steps, the jump is at 318 -> 319,
exactly when the new left-foot task begins. The solver fixes preceding poses
rather than jointly adjusting them with the new contact. The four-pilot report
and comparison are retained in
`tmp/support_rebuild_20261003/contact_only_cleanup_v8.json`.
All 207 new plans exclude the retired approach parameters; the source motion,
contact labels, events, loaded sites and observations are byte-identical to the
v6 source. Formal training has not started from these candidates.

### 2026-10-04: joint trajectory derivatives and solve time

The continuous IK diagnostic assembles the existing residuals without running
any per-frame optimizations. Its history poses and loaded-material phase
prefixes depend on the current full trajectory. No approach target, additional
loss, changed coefficient or revised acceptance tolerance is introduced.
`continuous_ik.py` and `continuous_sparse_ik.py` are experimental solver modules;
the production generation entry has not been switched while broader checks
remain incomplete.

Exact sparse frame/history blocks and material-prefix derivatives replace
repeated full-trajectory autodiff inside LSMR. Two Jacobian-product checks on
the real 932-frame, 30,756-variable trajectory agree with complete JAX autodiff
to relative L2 error below 6e-15. Five focused tests cover temporal coupling,
the loaded phase budget, changing sites/root yaw, bounded optimization and
restoring the correct native geometry after a rejected candidate.

For `cover144_122_h100_dm42p650_ap20_rxy_121`, the same five rounds of 25
evaluations took 2,571.8 s with matrix-free CPU products, 722.4 s with sparse
CPU derivatives, and 329.4 s with sparse CUDA derivatives. CPU/GPU cases had
different competing diagnostic workloads; this is an observed local timing,
not an isolated hardware benchmark. Their costs and optimization paths differ
with numerical column scaling and approximate LSMR steps, although the residual
formula and weights are unchanged. None reached the configured stationarity
tolerance within this evaluation budget.

The sparse CUDA candidate's 318 -> 319 body step was 8.38 cm versus the source's
7.98 cm and the sequential contact-only candidate's 25.32 cm. Independent
Newton/MJWarp queries passed all 34 source event checks and all 16 original
loaded-material geometric phase budgets, with maximum phase path 5.685 cm.
Its final native self penetration was still 5.79 mm because the witness planes
were held for each round. Re-querying geometry at each trial, continuing the
same complete trajectory for 80 evaluations, took another 185.0 s and reduced
self penetration to 0.0273 mm. The independent event/material checks still
passed; edited execution loads remain unknown.

This is not yet general acceptance. The h110 cold-start diagnostic stalls with
large stationarity error. At frame 908, perturbing q by at most 1.44e-6 changes
the same terrain/left-sphere-hand pair's reported geometric penetration from
11.38 to 33.62 mm and switches its normal from vertical to horizontal. A
Jacobian match against the fixed-witness graph does not validate this queried
field's continuity. The native query issue needs investigation before production
adoption, rather than extra residuals or weaker acceptance. Timing reports,
full pose checkpoints, independent labels and the small-perturbation evidence
are retained under `tmp/support_rebuild_20261003/continuous_joint_diagnostic/`.

The additional cold-start h095 and h110 cases took 300.0 and 257.8 s with live
native queries. Both independently passed the 34 event checks and 16 material
budgets. h095's maximum body step fell from 21.37 to 10.21 cm with zero queried
self penetration. h110's contact transition step fell from 26.27 to 7.93 cm,
but its remaining 21.52 mm terrain penetration makes it unsuitable for use.
Repeating frame 908's identical poses gives repeatable results; switching poses
moves the hand centre only 1.14 micrometres but changes the shape pair's depth
by 22.23 mm. Disabling contact reduction in a separate diagnostic preserves
the same discontinuity, so contact reduction is not its cause. Neither the
physics model nor the contact-labeling worker was changed in this ablation.

### 2026-10-04: shared FK timing and the triangle-culling discontinuity

The experimental factored derivative evaluates each pose/FK Jacobian once,
shares it with the frame tasks and adjacent material intervals, and assembles
linear history blocks and phase prefixes analytically. It retains float64 and
the existing objective. Nine focused tests passed, including restoring witness
arguments and activity masks after a rejected trial. Two products on the real
932-frame graph agree with full JAX autodiff within 2.4e-16 relative L2 error.
The same h095 cold-start budget of five rounds of 25 evaluations took 249.0 s,
versus 300.0 s with the preceding sparse derivative. Independent native queries
passed all 34 source event checks and 16 original material budgets (maximum
5.6854 cm), with 1.4099 mm terrain and zero self penetration. Stationarity was
not reached. The shared-FK implementation remains experimental.

The remaining native query overhead includes scalar Python witness transforms.
Vectorizing those transforms and pair summaries preserves every returned field
exactly on all 932 poses of that candidate. Two alternating passes took
1.232/1.268 s with scalar extraction and 0.954/0.909 s with vector extraction,
including the same collision and FK queries (median ratio 1.34). This changes
neither native candidate generation, margins, asset geometry nor contact truth.
With shared FK and vector extraction together, the full h095 diagnostic took
214.8 s (3.58 min). Its final solver states and all exported pose/velocity arrays
are bit-for-bit identical to the 249.0 s shared-FK candidate; only the recorded
source file path differs in provenance. That candidate's
independent native contact/material audit applies to the same poses. The
equivalence proof is retained in `ninth_global_h095_vectorized_query_cuda/`.
The preceding sparse h095 case took 300.0 s; this local comparison therefore
reduces the full observed run time by 28%, without changing residual weights,
precision, iteration count or acceptance thresholds.

The frame-908 discontinuity is now localized before MPR. Newton's mesh-convex
midphase rejects triangles when the **shape origin** lies behind their winding
plane. The left-hand origin's signed distance to triangles 1/5 changes from
-4.47e-8 to +5.36e-7 m; the candidate list changes from [6] to [1, 5, 6]. Calling
the same MPR core on triangle 5 without that cull returns 33.6148 and 33.6154 mm
on both poses, so this particular jump is not caused by MPR iteration or
contact reduction. The support-acceleration experiment found that acceleration
was already disabled in the CPU pipeline; its two runs are therefore not an
independent acceleration ablation.

There is also a signed-direction issue: top triangle 6 has an upward winding
normal, but its MPR normal is downward when the realized hand hull straddles
the triangle with its geometric centre below the top. A zero-thickness triangle
overlap does not define penetration into the closed obstacle volume. A separate
diagnostic using SAT on the actual verified closed convex native meshes returns
solid gaps -43.9130/-43.9124 mm with the same outward direction, changing only
0.634 micrometres. Moving the hand 1 mm along that direction increases the solid
gap by 1 mm, while the native triangle depth increases. These are geometric
diagnostics, not alternative contact labels or production optimization rows.
The required next correction concerns the penetration distance and direction
used by optimization; native contact activation and all original loss weights
remain authoritative. Evidence is under the h110 continuous diagnostic folder
in `native_triangle_core.json` and `closed_geometry_distance_diagnostic.json`.

An additional private query experiment represents only verified closed convex
terrain meshes as Newton convex solids, retaining their exact vertices,
transforms, robot shapes and margins. It does not alter the policy simulation
or labeling scene. The source's geometry allowances are queried again with
the same solid semantics; the old triangle reference cache is not reused.
This is a distance/direction experiment within the existing collision cost,
not an additional cost or a new contact definition, and the numerical field
differs from the preceding triangle-based objective.

The h110 whole-solid cold-start diagnostic took 143.6 s for the same 125-value
budget, passing all 34 independently checked source contact events and all
16 material budgets (maximum 5.6850 cm). Its native terrain penetration fell
from the earlier triangle-query candidate's 21.5188 to 4.8341 mm, with 0.0630 mm
self penetration. Its 318 -> 319 maximum body step was 8.1865 cm, compared with
7.9847 cm in source and 26.2683 cm in sequential IK. It still did not reach
stationarity, with reported optimality 1033.1; this result is not production
acceptance or evidence of executed edited support. Whole queried-value finite
differences must be checked, rather than only frozen-witness autodiff products.

That additional check fails. At the whole-solid endpoint, the frozen-witness
gradient predicts directional derivative -942.5, but the fully re-queried
central difference is +949889.4 for step 1e-6. Almost all of that jump comes
from frame 883's environment-collision row: a 1.905 cost contribution appears
on one side of the perturbation while the row is inactive on the other. Whole
convex terrain queries therefore remove the demonstrated triangle back-face
jump but do not yet give a consistent differentiable penetration field.
Evidence, current witness arrays and component/frame differences are retained
in `eleventh_h110_queried_gradient_audit/` and
`twelfth_h110_queried_gradient_frames/`. Production adoption remains blocked
on fixing the geometry value/derivative semantics, not on weakening acceptance
or introducing additional pose guidance.

### 2026-10-04: exact sphere field and normal joint-trajectory entry

The frame-883 jump is caused by a zero narrow-phase normal for physical sphere
`left_ankle_roll_sphere_1_link`. The witness-dot-normal geometry value is zero
at that pose, although the sphere overlaps the obstacle by 4.834 mm. A tiny
perturbation produces a unit normal and restores the omitted collision cost.
The exact sphere distance to the complete actual convex terrain stays near
-4.834 mm on both sides. `native_sphere_distance.py` now measures every physical
environment sphere using its native centre/radius and exact terrain vertices,
including the original collision filters. This field supplies the existing
collision residual only; it supplies no contact labels or support forces.

At the preceding endpoint, the fully queried directional derivative changes
from +949889 to -82.22, versus the corrected graph's prediction -84.86 at step
1e-6. At the new endpoint it is -28.63 versus -27.13. This fixes the demonstrated
missing sphere term. Float32 non-sphere narrow-phase values still vary under
small perturbations; these figures do not prove an exact derivative for the
entire refreshed geometric field.

Four cold-start 932-frame trials retain the original four objective families,
weights, poses and 6 cm material budgets. Each independently passes all 34
source event checks and all 16 original loaded-material phase budgets:

| plan | time (s) | maximum material path (cm) | 318 -> 319 step (cm) | terrain / self penetration (mm) |
| --- | ---: | ---: | ---: | ---: |
| h110, dm49.670, yaw20 | 163.4 | 5.6850 | 8.236 | 4.827 / 0.036 |
| h095, dp50, yaw-10 | 159.6 | 5.6854 | 7.509 | 1.339 / 0 |
| h100, dm42.650, yaw20 | 156.2 | 5.6846 | 8.382 | 4.824 / 0.533 |
| h100, dp50, yaw10 | 156.4 | 5.6850 | 8.051 | 1.323 / 0 |

The source transition is 7.985 cm. The preceding sequential edits produced
21–26 cm maximum steps in the first three cases; the fourth had a 10.315 cm
maximum and remains near that value (10.353 cm). These native-label audits establish
kinematic contact evidence and source-material motion, not dynamically executed
edited support. The trials exhaust their evaluation budgets without reaching
stationarity. The remaining 4.83 mm sphere penetration is **new excess**, not a
source allowance. Doubling h110's budget reduces cost from 55.241 to 50.877,
but leaves penetration at 4.826 mm; additional iterations alone do not resolve it.

`solve_pyroki_fullbody_ik` now compiles frame targets with
`assemble_pyroki_fullbody_problem` and solves the entire pose trajectory with
shared FK and exact sparse history/material-prefix blocks. Previous poses are
live variables. The per-frame least-squares/retry path is removed; retired hard
per-frame release metadata raises an explicit error. The diagnostic uses the
same assembler, without source inspection or an injected local solver. Original
loaded motion is recomputed after the solve for reporting only.

The official h110 entry takes 151.7 s and reproduces all five diagnostic solve
costs exactly. All root translations and actuated angles are byte-identical;
the established exporter normalizes quaternion signs, so complete pose arrays
are not byte-identical. Sign-aligned quaternion error is at most 1.5e-7 and body
position error at most 1.2e-7 m. Its own native relabel audit again passes all
34 events and 16 budgets. Sixty-four focused solver, geometry, material and
generation-contract tests pass. The stable last round spends 15.7 of 25.0 s
refreshing geometry; that is now the dominant measured cost.

The private optimizer scene uses verified closed convex obstacle meshes; the
policy/labeling scene, canonical asset, simulation settings and contact
activation rule remain unchanged. Source distance caches carry the new geometry
schema, and collection schema v9 verifies the effective solver/geometry schemas
and zero per-frame optimization calls. Old v8 sequential results cannot resume
as v9 outputs. Full 207-edit regeneration and training have not been started
with this candidate. Remaining collision tradeoffs and non-sphere query
variation require further diagnosis, without adding guidance or weakening
native event/material acceptance.

### 2026-10-04: measured runtime, task competition and body-name resolution

The official 151.669 s h110 run spends 77.850 s refreshing geometry, 21.023 s
computing Jacobians and 2.337 s evaluating residual values. Another 34.588 s is
unattributed time inside the optimizer, including its linear/trust-region
operations; this is not a separately measured LSMR profile. Assembly, export
and outer refreshes take 15.871 s. There are 125 function evaluations and
103 Jacobian evaluations for 30,756 pose variables. The complete breakdown is
retained in `continuous_joint_diagnostic/official_performance_breakdown.json`.

At the previous h110 endpoint, frame 883's endpoint and collision gradients
have cosine -0.999994 and norms approximately 2120. No variable there is at a
bound. More iterations cannot remove the collision by themselves while these
objectives oppose each other. Removing the strong absolute XY support target
is only a diagnostic candidate: the cold solve fails six native event checks
and reaches 22.752 cm material motion. Starting from the preceding valid joint
solve removes the new sphere overlap and keeps material motion at 5.520 cm,
but fails one left-knee keep event, with an interruption at frames 263–271.
Neither variant is accepted for collection generation.

That diagnosis uncovered a separate production interface error. All 299
contact records in this h110 task use full native USD body paths. The part
resolver previously matched only strings beginning with `left_` or `right_`,
so all 9056 record-frame occurrences were omitted from the optimizer's
requested-contact surface table. Native solver labels were not affected.
The resolver now uses the actual path leaf and the same normalization for
reference-body matching. It recognizes LF/RF/LK/LH/RH in this real task;
ancestor names cannot misclassify the endpoint. Thirty-three focused tests
pass, including hierarchical labels and cache invalidation.

Collection schema v10 rejects v9 output produced before this interface fix.
Source geometry caches, source poses, policy settings, weights and acceptance
criteria are unchanged. The first 25-evaluation cold surface-placement probe
after the fix takes 43.6 s and still fails to resolve geometry (65.458 mm
terrain and 9.904 mm self overlap). This short diagnostic is not a successful
new production baseline; earlier native-label audits do not validate the
changed optimizer. Full regeneration and training remain unstarted.

### 2026-10-04: signed constraints and whole-trajectory solver diagnostics

The source-relative collision floor is now `min(source_signed_distance, 0)`
for an eligible requested reference body. Preserving a positive source gap as
a collision lower bound incorrectly penalized approaching a surface without
penetration. The negative observed allowance, target residuals, weights, source
poses, material sites and policy simulation settings are preserved.

`NativeCollisionRows` can expose the same queried signed gaps and their
unshifted witnesses on both sides of the boundary. A zero collision hinge must
not erase a positive-clearance constraint's derivative. The optional multiplier
residuals and full signed-constraint linearizations do not change native contact
activation or the physical penetration audit. Fifty-six focused tests pass.

The following are **diagnostic solvers, not the production default**:

| h110 trial | initialization | time (s) | 34 events | 16 material budgets | max material path (cm) | new environment overlap (mm) |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 28: fixed-penalty augmented Lagrangian | preceding whole-trajectory solution | 160.7 | pass | pass | 5.68494 | 0.6371 |
| 35: sparse SQP with step damping | preceding whole-trajectory solution | 36.4 | pass | pass | 5.68494 | 0.001274 |
| 36: SQP with all queried signed gaps | original source configuration and edited root | 128.4 | pass | pass | 5.77757 | 0.000889 |
| 37: tighter linear subproblem precision | original source configuration and edited root | 145.4 | pass | pass | 5.77839 | numerical zero at iteration 15 |

Times are total diagnostic assembly/solve/export times. Trials 36 and 37 use the
normal assembler after the body-name/floor fixes; the preceding 151.7 s official
measurement is from before these fixes and is not an identical-objective speed
benchmark. Trial 35 is recovery from an existing solution, not a cold-start
generation result. Native independent audits of all four trials pass the
unchanged event and 6 cm material checks. Trial 37's actual native audit reports
0.40434 mm maximum terrain penetration, within the original allowed support
penetration, and 0.000120 mm self overlap; private optimizer numerical zero does
not establish exact zero in the policy scene. Edited load-bearing dynamics
remain unknown without execution.

The SQP factors the original sparse pose metric and solves its small active
dual. Redundant geometry/bound rows make this dual semidefinite: a generic
objective convergence report did not certify primal bounds. A standard cone
solver resolves that numerical subproblem, typically around 0.4 s for warm
geometry; cold material-prefix coupling makes the first subproblems slower.
Diagonal damping controls the step model, while every trial evaluates the
unchanged objective and re-queries geometry. No pose guide is introduced.

The complete signed-constraint Jacobian has 32,275 rows and 322,654 nonzeros at
the cold h110 start. Two independent JAX directional checks agree to 1.78e-15
maximum absolute error. This checks frozen-witness derivatives, not derivatives
of the re-queried native narrow phase. Boundary feasibility tolerances must not
be used to deactivate constraints crossed by the free objective step; likewise
absolute dual gap tolerances require objective/variable scaling near zero.

The cold candidates reduce the former 318 -> 319 discontinuity from 26.268 cm
to 8.097 cm (source 7.985 cm), with 11.030 cm maximum body step versus source
10.318 cm. They are not stationary objective minima: very small late steps and
rejected model trials still occur. Re-queried geometry derivative consistency
is being diagnosed before selecting a production solver or starting the full
207-edit generation and training.

That re-query diagnosis confirms a non-sphere failure as well. At h110 frames
272, 278 and 282 the native private narrow phase returns a zero normal for the
right-knee/obstacle pair. Dotting witness separation with that normal yields a
false zero distance and a false zero Jacobian. Querying the same actual native
convex vertices with a double-precision GJK/EPA geometry diagnostic gives
positive gaps of 0.05059, 0.06692 and 0.07659 mm, respectively. At frame 858 the
left-knee pair similarly has an undefined native normal. Re-queried directional
derivatives differ from the frozen-witness Jacobian, including larger jumps in
other pairs; passing the frozen JAX audit cannot certify the queried field.

`NativeCollisionRows` now rejects undefined non-sphere environment/self normals
explicitly and marks geometry as unknown. Exact native-sphere environment
queries still bypass their known failed native normals. An undefined witness
cannot be called a constant satisfied constraint or certify zero penetration.
This also means earlier candidates' native event and material passes remain
valid, but their private zero-overlap reports are not certificates of reliable
geometry for every body. A separate double-precision convex query is being
tested only as an optimizer geometry diagnostic; it supplies no policy contact
truth, labels, forces or acceptance thresholds, and is not a production default.

The complete defined-distance probe subsequently resolves the late stall. With
the same native shape vertices, primitive sizes and physical transforms,
double-precision geometry keeps step damping near 1e-5 instead of 1e11 and
reduces the cold h110 objective from the stalled 204.59 to 95.46 in 20 coupled
iterations. It takes 140.7 s, passes all 34 native events and all 16 original
material budgets, and reaches 5.68471 cm maximum material motion. The former
26.268 cm discontinuity is 8.306 cm. Independent policy-scene geometry reports
0.40407 mm allowed support penetration and 0.001203 mm self overlap.

Re-queried derivative checks on 30,650 signed constraints now have relative
errors 4.10e-9, 3.15e-9, 2.96e-8 and 2.14e-7 at perturbations 1e-3 through 1e-6.
The preceding native-witness field errors at those perturbations were 0.163,
0.816, 8.83 and 4.07. These are queried-value checks, rather than only a frozen
witness JVP comparison. The tiny-step slowdown was therefore not resolved by
changing loss weights or adding pose guidance.

The defined convex distance implementation is now connected to the normal
assembler and source-reference measurement. It reads the existing native
model's geometry and candidates, uses the optimizer's exact FK, and leaves
policy contact activation, margins, assets and labels unchanged. Source and
edited geometry share schema `native_model_defined_convex_and_exact_sphere_distance_v2`;
source cache metadata includes the convex witness schema instead of the old
hardcoded geometry version. Collection v11 invalidates v10 results. Coal and
Clarabel are generation extras; importing the editor does not require them.
Sixty-one focused geometry, solver and material tests pass.

Using that normal assembler (without the diagnostic geometry override)
reproduces all 20 coupled probe costs to approximately 1e-10, including fresh
source distances, in 150.1 s including export. This is still the diagnostic SQP: the main solve
entry has not yet switched from its coupled least-squares algorithm. Broader
cold-plan checks, solver selection and full regeneration remain outstanding;
none of these results establishes executed edited load-bearing dynamics.

The other three normal-assembler cold plans have now also completed native
audits. All four pass the same 34 events and 16 material budgets:

| plan | max material path (cm) | 318 -> 319 step (cm) | native self overlap (mm) |
| --- | ---: | ---: | ---: |
| h110, dm49.670, yaw20 | 5.68471 | 8.306 | 0.001203 |
| h100, dm42.650, yaw20 | 5.68457 | 8.346 | 0.001484 |
| h095, dp50, yaw-10 | 5.68561 | 7.941 | 0 |
| h100, dp50, yaw10 | 5.68514 | 8.038 | 0 |

Maximum native terrain overlap is 0.4040–0.4043 mm at allowed original support.
The last three probes ran concurrently for 30 iterations, so their 161–203 s
wall times are not single-worker speed comparisons. This verifies geometry and
trajectory recovery, not a completed speed optimization. The consolidated
evidence is `continuous_joint_diagnostic/defined_convex_geometry_4cases.json`.
At the time of these four probes, main solver replacement and full generation
remained pending. Future internal
trajectory checkpoints also carry and validate `robot_asset_json`; old
unbound state snapshots cannot silently resume as a new baseline.

### 2026-10-04: duplicate query removal and the formal constrained solver

The 20-iteration h110 probe unnecessarily queried each iteration's accepted
state twice, and queried it again when entering the next iteration. Keeping the
problem and its exact-state witness/value cache across SQP iterations reduces
whole-trajectory queries from 65 to 26. A cold repeat takes 112.8 s instead of
150.1 s (24.8% less). Every exported joint position and body pose is identical
in this diagnostic repeat; maximum per-iteration objective difference is
5.04e-9. Velocity rounding and export provenance differ, so this is not a claim
that the entire NPZ is byte-identical.

One 932-frame refresh at the validated h100 result takes about 1.0 s. Profiling
attributes most refresh work to native collision dispatch and geometry witness
processing. Repeated native records for the same shape pair now reuse the
full-shape query and transforms within that one pose; a changed pose always
queries them afresh. This second change only measures about 2% refresh savings,
so the demonstrated overall speed gain comes mainly from removing duplicate
trajectory queries.

The formal `solve_pyroki_fullbody_ik` entry now calls the same sparse constrained
whole-trajectory solver, using the original residual/Jacobian and pose bounds.
All queried signed geometry rows survive even when their hinge residual is
zero. Solver scaling, damping and merit decisions introduce no new task loss,
pose guidance, contact criterion or support budget. The nonlinear merit uses
the same full record set as its linear model. Numerical KKT checks distinguish
stationarity from exhausting the configured evaluation budget.

The formal cold h110 probe takes 110.7 s with **25 total value evaluations**.
It independently passes all 34 original native event checks and all 16 original
6 cm material budgets (maximum 5.68471 cm). Native terrain/self overlap remains
0.404067 / 0.001203 mm, and the 318 -> 319 body step is 8.306 cm. The budget is
exhausted, so `ik_solver_converged=false`; contact/material acceptance is a
separate, independently verified result. This 25-evaluation timing is not a
measurement of the normal 125-evaluation maximum.

The production solver schema is
`joint_trajectory_constrained_shared_fk_live_geometry_v3`, and generated
collections use v12 to reject cached v11/v10 outputs. Geometry queries and
solver iterations no longer count as independent outer refinement solves.
The production generation extras include Coal and Clarabel. Float64 remains
the authoritative residual/Jacobian precision; the optional float32 derivative
test compares against a complete graph at the same precision, since the
material reduction itself can differ between precisions.

Evidence: `continuous_joint_diagnostic/fortyeighth_no_duplicate_queries/`,
`continuous_joint_diagnostic/refresh_profile/`, and
`continuous_joint_diagnostic/fortyninth_official_constrained_entry/`.
Full207 generation, stand-cut/soft-join relabeling, edited dynamics execution
and predictor retraining remain outstanding.

The normal 125-evaluation maximum has now also been measured cold on h110:
251.0 s, 64 SQP iterations and 100 actual value evaluations, ending with
`step_not_accepted` rather than stationarity. All 34 native events and 16
material budgets still pass; maximum material path is 5.68488 cm, and native
terrain/self overlap is 0.404168 / 0.000047 mm. The original cost decreases from
95.46 at iteration 20 to 86.43, but the final retries reduce pose changes to
approximately 1e-16 and compare model reductions near 1e-13. These last ratios
can mistake rounding for progress. The solver now reports numerical stagnation
when a candidate is unchanged or its predicted merit reduction is unresolved
at float64 precision. This termination is explicitly not a convergence or
native feasibility certificate. A cold repeat takes 239.7 s, with the same
native event/material acceptance. The default-budget evidence is
`continuous_joint_diagnostic/fiftieth_official_default_budget/`.

A fresh check reconstructed the same formal endpoint from its float32 export
(not the lost original internal state). Along a constrained SQP direction,
the predicted original-objective derivative is -0.0714545823; fully re-queried
central differences give -0.0714545890 at 1e-4 and -0.0714545855 at 1e-5.
This supports the defined field/Jacobian at this quantized endpoint; it does
not prove the remaining late merit/trust-region behavior is resolved. Evidence:
`continuous_joint_diagnostic/fiftyfirst_late_queried_gradient/` and
`continuous_joint_diagnostic/fiftysecond_official_stagnation_guard/`.

The saved float64, asset-bound internal states expose a second-order FK
geometry error: a proposed 0.00970-norm tangent step decreases the original
cost, but nonlinear geometry acquires 6.51 micrometres of violation that the
linear model did not predict. A normal correction of norm 0.0000416 in the
same original metric reduces that error to 3.26e-11 m and decreases the exact
original merit by 0.000624. This is a standard SQP second-order correction;
see the [SciPy implementation](https://github.com/scipy/scipy/blob/v1.17.0/scipy/optimize/_trustregion_constr/equality_constrained_sqp.py).
It adds no task residual, guidance target, contact criterion or physical
tolerance. Only the unchanged, fully re-queried exact merit can accept it.
Optional infeasible normal corrections reject that trial and reduce the
tangent step; primary subproblem or undefined-geometry failures still surface.

The cold formal repeat with second-order correction takes **310.2 s** and
reaches cost 85.3074 before the 125-value budget is exhausted. It improves
the endpoint objective and avoids the previous infinitesimal-step stall,
but is **not a speed improvement or a convergence certificate**. Its own
native audit passes 34/34 events and 16/16 material budgets, maximum
5.68492 cm; terrain/self overlap is 0.403993 / 0.000148 mm. Evidence:
`continuous_joint_diagnostic/fiftyfifth_second_order_correction/` and
`continuous_joint_diagnostic/fiftysixth_official_second_order_correction/`.
A 1 nm deadband in the nonlinear merit was separately tested and rejected:
300.7 s without resolving the stall. The production merit remains the
original strict L1 violation.

Private optimizer scenes now opt in to a captured replay of their identical
Newton collision kernels. Broad-phase candidate generation, counter resets
and changed-pose queries still run every time. No contact or pose is cached;
the actual policy's contact-label path is untouched. This uses Warp's
[CPU graph replay](https://nvidia.github.io/warp/v1.17/user_guide/runtime.html).
At all 932 original source poses, collision fields match ordinary dispatch
bit for bit; repeated earlier poses also match. The isolated query sequence
measures 0.416 -> 0.199 s. This is an isolated collision-query speedup, not
yet an overall trajectory timing. Evidence:
`continuous_joint_diagnostic/collision_graph.json`.

The formal 125-value repeat completes in 282.0 s versus 310.2 s, with all
internal states and all iteration costs bitwise identical. The median
one-evaluation refresh falls from 1.136 to 0.863 s. The captured run briefly
overlapped a separate sparse-primal diagnostic, so the observed full runtime
is not an isolated, resource-controlled benchmark. Its fresh native audit
again passes 34/34 events and 16/16 material budgets, with the same geometry
and maximum path. These savings do not establish numerical stationarity;
both solves stop at the configured budget. Evidence:
`continuous_joint_diagnostic/query_graph_speed_and_native_acceptance.json`.

An alternative direct sparse primal QP was tested on the same asset-bound
late state: 20.39 s versus 0.732 s for the existing active dual. It was not
adopted. Evidence: `continuous_joint_diagnostic/sparse_primal_qp/`.

Fresh formal collection preparation additionally found two stale entry
issues: the source-layer builder still imported removed `climb00_pipeline`,
and the persistent IK worker invokes its entry as a file while its newly
added imports assumed a package launch. Both now resolve through current
absolute package imports. The failed attempt is retained under
`edits207_joint_v12/failed_attempts/`; no source or failure evidence was deleted.
The new collection retains all 207 plans and unchanged native source poses.
Two fresh production pilots, h100 centre/+50 cm and h110 centre/+18.75 cm,
now pass all 34 native events before and after the 932 -> 700 frame standing
removal/16-frame soft join, and all 16 full-trajectory material budgets.
Maximum material paths are 5.68514 and 5.68558 cm, native terrain overlap
0.40419 mm and self overlap zero. The h100 solver reaches KKT stationarity;
h110 exhausts its value budget. CPU-worker generation takes 267.7 and 308.5 s
(308.6 and 349.8 s including separate native exports/relabels). These are
formal production timings, distinct from the GPU-backed diagnostic above.
Evidence: `edits207_joint_v12/pilot_report.json`.

The formal worker's shared FK derivatives consume about 0.70 s on CPU,
versus about 0.08 s in the GPU-backed diagnostic. Full collection generation
has therefore started with three dedicated GPU FK workers and demand-driven
JAX allocation; optimizer Newton geometry still uses the same CPU scene.
This changes numerical execution placement, not original losses, bounds,
physics configuration, native contact rules or gates. Complete regeneration,
independent full/cropped relabeling, join auditing, data preparation and the
existing scratch recipe run in order, preserving all 207 plans. At this point
the collection is running; full207 success and retraining are not yet claimed.

The same full-trajectory metric now uses exact banded Cholesky when its
actual bandwidth fits the temporal pose blocks; wide material-phase coupling
retains the original sparse factorization. No coefficient is truncated.
Expanding active sets reuse previously solved inverse columns. The canonical,
asset-bound late-state comparison preserves the correction to 5.6e-13 and
all original geometry inequalities. Banded LAPACK's default 16 BLAS threads
proved pathological: the identical subproblem takes 16.59 s, of which 16.34 s
is factorization, versus 0.123 s with one thread. Banded operations now scope
BLAS to one thread and restore the caller's configuration afterward.

The scoped production subproblem repeats in 0.117--0.189 s. A cold full
125-value repeat takes 277.0 s; the unscoped banded attempt takes 879.1 s.
Their 88 iteration costs and final internal states are unchanged. Against
the earlier sparse run's 282.0 s, however, this is not evidence of a stable
overall speedup: concurrent full collection generation competes for CPU.
Recorded QP time falls from 110.0 to 64.9 s and the late QP median from
0.780 to 0.190 s, while refresh and Jacobian times rise. Whole-trajectory
geometry querying now accounts for about 123 s of the observed full repeat.
Evidence: `continuous_joint_diagnostic/banded_threading/` and
`continuous_joint_diagnostic/banded_full_runtime_breakdown.json`.

The scoped repeat's own unchanged native audit passes 34/34 events and all
16 material budgets, maximum 5.68492 cm; terrain/self overlap is
0.403993 / 0.000148 mm. Its final internal states are bitwise identical to
the unscoped banded repeat, but it still stops at the 125-value budget,
without a stationarity certificate. All 29 continuous solver, exact
derivative and geometry-constraint tests pass after the thread-scope change.
Evidence: `continuous_joint_diagnostic/banded_scoped_speed_and_native_acceptance.json`.

Early independent checks of eight generated formal trajectories cover h090,
h095, h100, h105 and h110, including translated targets. Each passes 34/34 full
and cropped native event checks and 16/16 original material budgets;
maximum paths range from 5.68442 to 5.68581 cm. These checks do not establish
dynamic execution or acceptance of the remaining trajectories. Evidence:
`edits207_joint_v12/early_native_audit.json`.

A subsequent fixed snapshot of 117 complete formal edits was relabeled in all
five exact native scenes. All 117 pass the original 34-event full-trajectory
checks and all 16 source-loaded material geometry budgets. The largest
reference cumulative tangent path is 5.68671 cm; maximum native terrain/self
penetration is 0.404686 / 0.00028954 mm, with no invalid penetrating witnesses.
No edit was filtered. This is a source-load geometry reference audit, not
evidence of new executed support forces, and does not certify the remaining
90 plans. Evidence: `tmp/support_rebuild_20261003/completed_snapshot_native_audit/summary.json`.

The new soft-joined data path also passes an entry preflight across four edits:
each preserves 34 native event checks and its 700-frame soft connection. The
complete action clocks produce 60 samples (30 training, 15 validation, 15 test).
One scratch warmup interval performs 30 predictions and updates 148 model state
tensors; a separate one-step raw-recursion preflight executes the current
evaluator. These are interface checks, not trained-model quality results.
The runtime must install both local `somaforge-generator` and `contact-solver`
packages; relying on the generation-only import path leaves data preparation
unable to import `generator`. Evidence: `tmp/support_rebuild_20261003/joint_pipeline_import_preflight.json`,
`joint_v12_training_entry_preflight/entry_preflight_summary.json` and
`joint_v12_recursive_entry_preflight/entry_preflight_summary.json` in that directory.

Fresh refresh profiling identifies redundant argument transfer: the 54
arguments occupy 31.22 MB, but only 12 witness arguments (8.89 MB) change
between trials. NativeCollisionRows now refreshes every pose and reuses the
existing device buffers for the immutable authored targets and source material
sites. Complete argument comparisons are bitwise identical; stack/upload
time changes from 0.088--0.100 s to 0.006--0.009 s. The cold formal repeat
takes 254.2 s versus 277.0 s with the earlier scoped factor. Its final internal
states are bitwise identical and its own native 34-event/16-budget audit passes.
Evidence: `continuous_joint_diagnostic/geometry_argument_upload/` and
`continuous_joint_diagnostic/sixtieth_official_fixed_argument_buffers/`.

Private joint-pose queries additionally capture the identical FK kernel with
the collision kernels; body-pose queries still replay collision alone, preserving
the caller's transforms. Every contact field and every body transform matches
at all 932 source poses, including revisited poses. One isolated comparison was
slower under concurrent generation; alternating repeats give medians 0.235 and
0.207 s for the complete source query sequence. The next cold formal repeat
takes 240.6 s with the same final internal states and unchanged native audit.
All these full timings overlap other work and are observed runtimes, not
resource-controlled speed guarantees. The related 54 tests pass. Evidence:
`continuous_joint_diagnostic/fk_collision_graph*.json` and
`continuous_joint_diagnostic/geometry_execution_speed_and_native_acceptance.json`.

Signed-constraint differentiation no longer evaluates a second moving-body
derivative for static terrain rows. All 30,542 signed clearances and every
scaled sparse Jacobian coefficient match the previously saved physical
subproblem bitwise; only 1,952 rows need the second derivative. The independent
yaw/self-cancellation derivative and coupled-solver tests still pass (14 tests).
Evidence: `continuous_joint_diagnostic/moving_witness_derivative.json`.

The independent reload monitor was retired after its termination was followed
by termination of its three worker children, interrupting generation. All 64
completed trajectories were validated and retained with unchanged hashes;
four incomplete attempts, worker logs and the retired helper text are retained
under `edits207_joint_v12/failed_attempts/worker_monitor_exit_*/`. The complete
207-plan pipeline resumed through its original cache owner, which owns the
worker lifecycle until editing completes. No plan was removed and training had
not started. Evidence: `edits207_joint_v12/worker_monitor_recovery.json`.

The completed formal collection now has 207/207 native event passes and
3312/3312 full-trajectory source-material budget passes (maximum 5.68671 cm).
All 207 soft joins preserve the native events; their maximum seam root/body
steps are 0.15123/0.55933 cm. The 700-frame clips yield all 3105 action samples.
The first complete 100+1000 scratch run passes prefixes 3/5/3 in the h090/h100/h110
recursive cases. Reference-input evaluation first fails at the same events,
so recursive input drift alone does not explain the execution errors.

A separate feedback bug was verified across all 6210 current/target dataset
poses. `CanonicalBoxGeometry`'s support-plane lower bound falsely rejects the
260-frame knee pose in all 207 motions (414 current/target occurrences).
Every associated convex outer hull is actually separated from the box by
13.885--14.409 mm; the proxy reports up to 51.266 mm penetration. A negative
separating-axis lower bound does not establish penetration. Continuation now
uses the existing native penetration, allocation, invalid-witness and contact
checks only, with the unchanged 3 cm episode boundary. All 6210 demonstrated
poses pass; the 17 continuation/pool/endpoint tests pass. Optimization losses,
native contact activation and dataset labels are unchanged. Evidence is in
`tmp/support_rebuild_20261003/rollout_geometry_guard_audit/`; the same scratch
recipe is repeated in `loaded_material_scratch1100_native_boundary_20261005`
with independent final-checkpoint recursion against the prior run.

The native-boundary repeat completed all 1100 intervals. Initialization has
157 bitwise-identical tensors to the prior run; data, seed, schedule, losses
and acceptance thresholds are unchanged. Final validation endpoint acceptance
is 69.1026%, with mean native penetration 0.4779 mm. Final-checkpoint recursion
passes prefixes 3/15/3: the h100 case completes all 15 demonstrated actions
with matching actual contacts and zero penetration, versus 5 actions before
the continuation correction. The h090 fourth action misses the right hand
and penetrates the right foot by 1.9782 mm; h110 has the intended contacts but
penetrates the right hand by 19.2366 mm. These are remaining execution errors,
not proof that the complete learning problem is solved. The h100 diagnostic
30-step rollout is valid for its first 15 demonstrated actions; unsupported
post-demonstration continuation has six penetrating poses and a maximum
55.3012 mm penetration, and is not a successful 30-action chain. Evidence:
`runtime/current/models/generator/loaded_material_scratch1100_native_boundary_20261005/recursive_evaluation/comparison.json`.

Independent reference-input evaluations confirm residual single-step errors:
h090's fourth action has the correct contact topology but 2.6449 mm right-foot
penetration; h110's fourth has correct topology but 10.1454 mm right-hand
penetration. Recursive drift increases the latter error but cannot explain it
alone. These evaluations reset only the diagnostic input to each demonstrated
pose, without teacher training, solver correction or projection. Evidence:
`tmp/support_rebuild_20261003/native_boundary_reference_input_audit/`.

The remaining h090/h110 failures are held-out heights: only 0.95/1.00/1.05
enter training; 0.90 is validation and 1.10 is test. A frozen production-loss
audit at the exact reference-input failure outputs finds valid nonzero
penetration gradients. Total q-space gradients align with the execution
gradients by 0.99934/0.99820. A diagnostic update bounded to 1 mm root movement
reduces freshly queried penetration from 2.6449 to 1.5446 mm and 10.1454 to
8.2577 mm, respectively. This is a derivative check, not an inference
projector. Demonstrated target controls have zero execution penalty and zero
penetration, with imitation residual below 1e-12. Execution gradients reach
the current network parameters; cached versus fresh native contact inputs
have identical masks and active-anchor differences below 0.00006 mm in these
three cases. Checkpoint tensors remain bitwise unchanged.

The frozen all-3105 reference-input audit uses the existing combined production
feedback gate, not endpoint-layout success alone. It passes 1260/1470 training
endpoints (85.7143%), 416/780 validation endpoints (53.3333%) and 468/855 test
endpoints (54.7368%). Mean native penetration is 0.0494/0.4779/1.8538 mm,
respectively; maxima are 4.8428/8.1398/20.6231 mm. Thus the remaining model
errors are not explained by missing geometry derivatives or inconsistent
contact representative inputs in the audited cases. These single-endpoint
counts do not certify full recursive chains. No held-out sample enters
parameter training. Evidence: `tmp/support_rebuild_20261003/native_boundary_execution_gradient_audit_v4/`.

Frozen parameter-adjoint checks further find positive clearance derivatives
for the total raw and production-clipped gradients at both failed outputs.
Fresh native queries of single-example counterfactual parameter steps show
that this local fact is not a finite-step feasibility guarantee. Clipped SGD
removes h090's 2.6333 mm foot penetration but loses required hand contact;
saved-moment AdamW reduces it to 1.2656 mm and preserves the contact topology.
For the initially feasible h100 output, saved-moment AdamW introduces 1.3251 mm
penetration, whereas clipped SGD stays nonpenetrating. At h110, clipped SGD
reduces penetration from 10.1507 to 5.2823 mm and saved-moment AdamW to 8.7403 mm.
All queries use the original native scene, allocation checks and margins.
These probes do not simulate the actual 128-example training batch and must
not be interpreted as evidence that every training update behaves this way.
Weights remain bitwise unchanged; no optimizer or inference correction is
installed. They expose finite-step contact-boundary crossing and history
effects as additional mechanisms to examine before changing the objective.
Evidence: `tmp/support_rebuild_20261003/native_boundary_parameter_finite_steps.json`.

The saved 128-slot production pool was also evaluated with its exact original
loss, checkpoint, clipping groups and AdamW state. This is a counterfactual
next batch after interval 1100, not a reconstruction of a historical update.
Mean loss decreases from 4.355617 to 4.348495, while maximum freshly queried
native penetration increases from 0.5218 to 2.6972 mm. One previously safe
sample becomes generation-invalid; combined completion improves from 40 to
43 samples but two previously complete samples become incomplete. Clipped SGD
removes this batch's penetration but reduces combined completion to 37. Thus
neither average loss descent nor an optimizer substitution establishes
per-sample feasibility. No training weights or optimizer state were changed.
Evidence: `tmp/support_rebuild_20261003/native_boundary_real_batch_audit/real_batch_update_audit.json`.

The final delivery audit verifies all 207 joint-trajectory solver records,
zero independent per-frame optimization calls, unchanged source joint poses,
canonical robot assets, current plan hashes, all native event checks and all
3312 full-trajectory source-material budgets. It also verifies all 207
700-frame soft joins, all 3105 retained action samples, every event certificate
hash and every original training dependency hash. The native-boundary scratch
repeat completes 100 warmup plus 1000 feedback intervals; its configuration
differs from the preceding run only in output path. These checks complete the
continuous data rebuild and requested training/evaluation comparison; they
do not establish perfect predictor generalization.

The scope of the support evidence remains explicit. Full pre-join trajectories
have a maximum source-loaded material tangent path of 5.68671 cm. After soft
joining, 3105 phase budgets are measured and pass; 207 phases each contain one
unknown seam interval, and the largest observed partial path is 5.89399 cm.
The connection passes native contact and continuity checks, but no new
execution loads are available at that seam. Therefore `training_ready=false`
is retained with the explicit self-observed demonstration training scope;
it must not be presented as a certification of edited physical support.
Policy reexecution is outside the user-authorized scope. Of the 207 solves,
14 report KKT stationarity; native event/budget acceptance of the remaining
trajectories does not imply optimizer convergence. Evidence:
`tmp/support_rebuild_20261003/whole_trajectory_delivery_audit.json`.
