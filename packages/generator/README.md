# Generator

Predictor 的独立实现，以及数据、训练、递推评估工具。
Infiller 模块已退役，历史数据和权重保留归档。
WBT policy、环境、仿真训练与执行仍位于 `src/holosoma/`。

入口导航：[环境配置](../../docs/environments.md)、
[共享资产与接触契约](../somaforge_core/README.md)、
[WBT expert 与蒸馏模块](../../docs/wbt-modular-experiments.md)、
[文档索引](../../docs/README.md)。各阶段独立调用，本文不自动启动 WBT 或选择 checkpoint。

新 scratch 训练默认启用 `depth_observation` 和 `world_surface_scan`：环境输入是
按相机像素排列的二维光轴深度图，使用 2D CNN 编码深度、有效像素和标定射线坐标，
不使用点云网络或体素。Role、位置选择和姿态执行都读取深度图特征；部位区域特征
将身体 landmark 投影回原始深度图，使用光轴距离残差与显式有效掩码，不查询高度图。
位置评分融合部位特征、对应图像特征及候选 XYZ 的位置编码。
未缩放的标定光轴深度通过完整相机姿态反投影为世界 XYZ，交互 head 直接选择
原始像素对应的观察网格顶点并保留实际表面身份。CNN 的特征下采样不修改几何深度。
不同表面之间禁止连接三角形，地面和箱顶保持原水平面，箱沿不插值为斜坡。
新深度模型拒绝 heightmap 输入；数据准备、训练和递推不生成高度图。遮挡中的当前接触由实际
Newton 分配约束的地形见证点补充；不可见的未来目标显式标记为无覆盖。
此受控深度前端使用实际初始化地形三角形，尚不模拟机器人自身遮挡和传感器噪声。
接触真值仍使用 Newton 和共享主表面筛选；深度不定义接触。
checkpoint 保存独立的 `calibrated_optical_depth_image_v1` 输入契约。
历史 `depth_observation=False` 权重（包括 2026-10-12 的第 500 轮）仍按原高度图
编码器加载；仅三维化候选点的旧模型不能称为深度图编码模型。新结构必须从零训练。

新深度训练的接触点契约为 `newton_matched_endpoint_contact_points_xyz_v2`：所有有效部位
共用一个三维端点接触点 loss。keep 比较同部位／同表面起终点有效接触点集的最小距离；
其他有效角色比较终点有效点集与计划目标点。保留完整实际点集，不再分别选择每帧
最深点，避免脚跟／前掌代表点切换造成假位移。点集不相交且真实偏移时仍受惩罚，
不要求整个材料区域静止。旧 `newton_endpoint_contact_points_xyz_v1` 仅保留作历史评估。
keep 位置监督从起点实际点集中选择离示范终点点集最近的起点；不把未来点写入观测。
各角色共用原 4 cm RMS 空间任务容差与权重，
不另加材料点移动距离或累计轨迹 loss；这不是接触激活阈值。
终点没有有效的同部位／同表面接触时标为缺失。此时仅在统一接触 loss 中使用原生身体
表面到计划目标法向区间的三维距离，选择距离最近的目标身体区域替换原有趋近项。
一个代表点只约束一个最近区域；其他选中区域保留形状—表面区间，避免把脚跟／前掌
同时拉到同一点。
端点位置 loss 只比较实际接触点，不再叠加几何恢复距离。法向区间读取场景实际 margin，
切向定位、超出法向区间的距离及完整实体避碰仍有梯度；几何接近不能替代实际接触验收。
训练启动保存 `reference_contact_point_audit.json`；示范 keep 点集仍超出 4 cm 的样本
显式屏蔽冲突的姿态模仿项并计数，保留角色标签、位置监督及接触／安全损失。
连续未断触是 keep 示范标签的前提；
仅凭两个端点不能验证生成运动中间未断触，仍须由实际轨迹检查。

2026-10-10 登记的固定比较基线为
[`predictor.v1_latest.20261010`](../../baselines/predictor_v1_latest_20261010/README.md)。
其训练入口使用 `full1000_position_predictor_v1`，共享观测
编码器梯度，保留原 root＋部位平均读出。新的形状—目标表面统一区间 loss 默认启用；
网络架构版本与区间 loss 语义版本是独立的。
训练从零初始化，采用100轮预热＋1000轮反馈日程，使用
loaded-material207_joint_v12 数据；该历史基线的keep材料区域保持权重为1。
新深度训练禁止启用该独立材料位移正则，keep参与上面的统一端点接触目标。
基线checkpoint固定为`v1_shared_interval_scratch1100_20261009/step_1000.pt`，
原始运行位于`runtime/current/models/generator/`。该运行记录到1002轮、最后保存1000轮，
没有1100轮完成报告；登记为后续比较起点，不声称全程完成或物理执行通过。
新训练输出默认`runtime/current/models/generator/v1_shared_interval_new_run`，
每次运行应显式传入未使用的`--output`目录。checkpoint不作为scratch初始化权重。
v2 整体改版未被采用为基线；代码保留作显式实验，通过
`--architecture full1000_position_predictor_v2 --no-execution-observation-gradients` 选择。
架构记录见 [structured-predictor](docs/structured-predictor.md)。
正式递推入口为 `python -m generator.evaluate_position_rollout`；旧tmp启动器只转发此实现。
完整checkpoint仅能由评估或显式只读梯度审计加载，不能作为训练初始化。

旧scratch配方、source550及control500已[封存](../../baselines/README.md)，
不再作为当前训练默认值。保存的历史配置、原始数据、权重和源代码快照用于对照；
历史v1的编码器、token与checkpoint不能直接当作v2。
原始启动记录和925 checkpoint的30次递推保留用于同条件比较，不重写历史架构。
该递推来自训练集motion4，前15次有示范，第6次首次失败，后15次继续自主预测；
不能作为留出泛化或实际承重、无滑移结论。

```python
from generator.structured_position_predictor import StructuredPositionPredictor
```

从仓库根目录 `source scripts/source_somaforge.sh` 后使用新入口，例如
`python -m generator.train_full1000_position --help`。

依赖方向为 `generator → contact_solver → somaforge_core`；共享 FK、预测结果结构、
高度图坐标约定在 core。网络中的几何损失调用 contact_solver。
旧 `climb00_pipeline` 模块仅供显式历史复现，不再进入默认环境或安装清单；
使用方式见[兼容包说明](../climb00_pipeline/README.md)。别名仍指向同一实现模块；
历史兼容模型保留原参数名与forward；新的v2使用显式版本校验。旧错误机器人资产产物仍无效。
回归测试按实现归入 `generator/tests`、`contact_solver/tests` 和
`somaforge_core/tests`；`test_package_boundaries.py` 单独验证历史兼容入口。

普通 predictor batch 的可选约束训练入口为
`python -m generator.train_full1000_position --constraint-training ...`。
使用同一 predictor、数据划分和 Newton 查询，每个 batch 一次受约束 AdamW 更新。
默认分支不变；当前先支持固定接触意图的 reference batch，未接入递推 rollout。
使用说明和边界见 [minibatch-training](../contact_solver/docs/minibatch-training.md)。

大批量持续池（`train_full1000_position --parallel-rollouts --gpu-pipeline`）采用验收后提交：
A → B′ 通过现有任务、几何、端点接触保持及已启用的进度验证，才将 B′ 放回输入并推进目标。
失败输出仍参与本轮 loss/backprop，但下一轮保留本次预测前的完整 A 和目标 B；
即使 A 来自之前成功的递推，窗口内也原样保留。
严重穿透、无有效接触等失败同样回退，不把失败 B′ 用于 B′ → B 训练。

当前训练默认启用 `continuous_demonstration_training`：反馈阶段每个统计轮次仍遍历一次
原始训练示范，按示范接触条件训练同一个网络，使用现有完整损失。
持续池的自主条件调度和失败回退保持原样；示范分支不向池提交姿态，
也不把示范外状态配上旧示范目标。日志分别记录 `demonstration_predictions`
和持续池统计，因此与旧结果比较时需同时报告总预测数、更新数和训练耗时。
教师条件同时覆盖角色、部位、可见位置及新鲜 Newton 端点区域；区域分类 logits
仍由网络独立预测和监督。缺失真实区域证据明确标记为未知，不能暗用预测区域冒充教师条件。
历史配方的纯持续池阶段可用 `--no-continuous-demonstration-training` 对照，
该开关不恢复遗漏区域的旧教师语义；既有 checkpoint 的自主推理参数和接口不变。

年龄与前进历史只累计成功提交；连续失败累计 retries/rollbacks。
每槽每次预测均累计 attempts，成功、失败都计数，跨统计轮次保留。
达到 `parallel_episode_limit`（默认64次预测更新）后从训练集重新采样；
完整观测、目标索引一起替换，清空 attempts、重试、成功深度及前进历史，恢复示范监督。
第64次输出仍参与本轮反传，下次输入才切换；成功不会延长窗口。
日志中的 *_rejections 表示候选失败；periodic_resamples 单独统计周期刷新，
resets/timeout_resets 包含该刷新，rollbacks 只统计未到刷新边界的失败保留。
状态 schema 为 bounded_attempt_rollout_v5，保存 attempts、resample_interval 和 allow_post_demo。
完成最后示范目标后仍关闭示范监督并继续自主递推；后继不得进入验证集。
使用 `--no-parallel-allow-post-demo` 时，完成最后示范目标后立即重新采样，
不产生示范外输入；最后目标未通过验收时仍保留A，直至成功或达到64次刷新。
terminal_resets 与 periodic_resamples 分开计数，同时到期只记一次末尾重置。
`parallel_retry_limit` 仍为兼容参数，不单独截断；自主状态同样受64次窗口限制。
示范外默认开启，提交仅要求自身计划的实际接触、区域、位置、保持角色一致性和安全；
末帧的示范接触、位置和姿态不再作为目标。示范监督loss归零，几何、位置与保持loss继续反传。
新深度模型自主keep同样比较实际起点与终点接触点；历史材料位移实验才使用训练集统计容许量。
`autonomous_inputs`记录实际用于优化的示范外输入数，区别于提交后处于自主状态的槽数。
此修改不解除 `--constraint-training` 暂只支持 reference batch 的限制。

`--acceptance-penetration-m`（当前默认0.03m）仅控制递推提交和任务验收。
原优化默认限值、Newton margin/gap、接触激活和全部loss不随它改变。
checkpoint分别保存运行验收与优化限值；递推评估继承checkpoint设置，显式覆盖可用于同容差对比。
缺少该设置的历史checkpoint保留原1mm验收。示范外评估同样检查自身接触、区域、位置和安全，
无未来示范时`task_acceptance=None`，不能将未知当作通过。

支撑运动统一使用 `source_loaded_tangent_material_motion_v1`：按原始实际受力点的
局部坐标跟踪相邻姿态的同一材料点，投影到所在表面的切平面，以原始载荷做RMS汇总，
再累计阶段运动。允许转动、足跟/前掌切换和卸载；不取单个最小位移点。
6 cm仅是该几何参考累计运动的预算，不是Newton接触阈值，也不证明新执行无滑移。
完整轨迹和独立原始载荷观测具备时，共用 `somaforge_core.loaded_material_motion`。
裁剪/软拼接后必须按新姿态和帧映射重算；跨删除区间的接缝缺少相邻原始求解证据，标为未知。

当前模型仅输出下一事件端点，缺少完整执行轨迹和逐帧实际载荷，
支撑运动报告为unknown，不施加路径运动loss或gate。
原 `predicted_contact_retention_weight`、`require_endpoint_contact_retention`、
`endpoint_contact_tolerance_m`配置已移除，旧首尾6 cm规则不能再启用。
实际Newton接触、穿透、关节限位、端点布局和保持意图一致性检查仍生效。
有完整执行观测时，可通过 `require_observed_support=True`严格要求实际支撑验证。

端点位置loss与验收共用默认4 cm RMS空间容差（历史高度图为XY，新深度模型为XYZ）：
先对目标部位的平方误差求均值，再惩罚超出容差的RMS部分；进入范围后位置loss与
梯度均为零，不继续追精确点。新深度训练使用GPU实际点对，读数评估使用相同选点规则。
缺失点对、接触激活、部位区域和安全检查独立进行，零位置loss不表示任务通过。

新 Predictor 总 loss 统一由 `generator.endpoint_objective` 组合为五组：规划分类、
姿态模仿、统一接触／避碰、实际端点位置、关节限位。姿态模仿保留根位姿和关节均方
误差，以及权重0.5的肢体FK位置／朝向均值＋最大误差；结构项不再重复计算最大关节误差。
日志 `loss_contribution_*` 保存实际加权后的五组，逐样本相加等于总 loss；旧实现的
未优化接触／位置／姿态先验损失使用 `diagnostic_*` 名称，不能当作额外优化项。
checkpoint 保存 `endpoint_predictor_single_recovery_objective_v1`；旧目标的梯度审计须
使用归档源代码，不能用当前代码冒充原 loss。网络结构及实际接触点契约没有因此改变。

Predictor 的默认任务容差与优化尺度集中于 `contact_solver.endpoint_task`：穿透拒绝3 cm、
端点位置RMS 4 cm、关节越界0.15 rad；关节限位loss尺度0.1 rad、穿透loss尺度5 mm。
`--endpoint-position-tolerance-m` 和 `--joint-excess-rad` 与穿透配置一同保存，训练及递推
评估共用 `configured_acceptance`；位置容差不再依赖高度图／深度图分辨率。
旧 checkpoint 继承其已保存的穿透门槛；缺少空间／关节字段时保留历史4 cm／0.15 rad。
连穿透字段也缺失的旧 checkpoint 保留显式标记的1 mm库级历史默认，不自动放宽。
递推报告记录使用了哪些历史缺省字段。示范监督阶段仍同时检查自身计划和示范任务，
两种验收目标分别报告，不重复添加位置loss，也不改变既有阶段推进规则。

统一避碰项从当前初始化的 Newton 场景读取完整碰撞实体及实际碰撞过滤，
对所有允许的地形／自碰撞形状对计算有符号 GJK/EPA 距离，使用其材料点经 FK
反传。地形 mesh 的独立封闭组件分别处理，完整包围也不会因缺少三角面候选而漏检。
不支持的非封闭／非凸实体明确报错。接触激活判定不变。
Predictor 训练统计、验证和递推提交统一只使用完整实体深度，默认最大穿透超过
3 cm 拒绝，等于或小于3 cm通过穿透检查，仍须满足其他接触和布局条件。
Newton 原生深度仅保留诊断；严重程度分类不增加第二个拒绝门槛。
缺失／无效实体距离不能退回 Newton 深度或记为零。
checkpoint 和递推报告保存 `complete_solid_endpoint_penetration_v1`、几何语义和
场景 fingerprint；旧 checkpoint 的训练口径单独记录，重新评估不得冒充旧口径。
新训练以同一通过率和实体穿透深度计算 `validation_score`，保存
`best_validation_score.pt`；旧的 Newton 零穿透指标仅作为带 diagnostic 名称的诊断，
不参与 checkpoint 排序。

已存在目标部位／表面接触时，统一接触训练保留选定区域的形状—表面区间监督。
其材料点库启动时复用训练集
示范端点的实际Newton点对，将已激活、已分配、主向上表面上的机器人材料点绑定到
各自link局部坐标。验证集不进入材料点库；点库不提供目标姿态，也不是网络输入。
材料点库保存在`contact_material_skin.pt`，未观测区域的资产趋近代理显式报为未知。
它只是loss几何，接触实现仍由每次预测上的实际Newton点对判断。
所选材料点的两侧区间残差在碰撞候选出现后仍保留，深嵌入不会因为候选存在而关闭
向目标表面的恢复方向。有限表面按真实margin的边缘邻域计算距离，不要求投影
严格落在矩形内；示范中距顶面边缘仍在margin内的接触不会被额外处罚。

新 Predictor 训练默认使用 `--penetration-penalty huber --penetration-scale-m 0.005`：
完整实体穿透以及已合并到同一目标材料查询的穿透下界使用 Huber(p/s, delta=1)，
目标是零穿透，3 cm 仅为验收容差；5 mm 为 Huber 残差尺度，并非拒绝阈值。
接触趋近／释放项保留 `log1p(r²)`，只在同一
材料查询完全一致时去重；深穿透梯度不会随深度减小。固定身体区域的平均项加最大项
聚合保持不变。`--penetration-penalty log1p` 保留原标量 loss 供 A/B 对照，但仍使用
相同的完整实体验收。其他调用者默认保留历史 log1p。
此前C1任务退出场在浅侧穿透中会放大错误方向，已
从训练目标移除。这里反传的是标量loss的真实FK导数，不求QP或注入自定义修正梯度。
这些改动不保证每次参数更新同时改善所有约束；完整学习和递推效果须通过新训练验证。
原生点对的不可用法线保留审计及验收拒绝，但不再屏蔽整条姿态的所有损失。
不可用的原生距离行不参与距离梯度；已验证完整实体距离和其他正常FK损失继续反传。
完整实体几何本身未知时仍明确报错，不能用零loss或替代接触标签掩盖。

递推日志新增 `penetration_exceeded`、`penetration_unknown`、`penetration_only` 和
`penetration_and_contact`：分别统计超限、未知、仅穿透不合格以及穿透与自身接触
同时失败的次数；这些类别有重叠。`penetration_only` 排除布局、示范接触、关节限位、
无效约束分配等其他拒绝原因。比较 A/B 时使用相同种子、更新预算和30步起点，重点
报告连续成功长度、整链成功率、穿透深度分布及接触实现率。实现修改不会热更新或
自动重启正在运行的训练；A/B 使用独立新输出目录。

历史高度图训练入口可用`--no-depth-observation --keep-patch-weight 1.0`：仅对网络的keep意图增加材料区域几何保持
正则。当前材料点取自实际Newton的已分配有效接触，FK保持材料身份；在整个区域中
寻找最小切向运动，允许绕区域内支点转动。对应示范动作提供运动容许量，其他预测
keep使用训练集部位中位数，未观测部位标为未知。标定保存在`keep_patch_calibration.pt`；
日志记录loss、运动、超额及缺失项。新训练默认关闭（权重0），显式权重1仅供历史复现；
以下200轮实验是历史诊断，不能作为当前整段训练效果的结论。
这是端点意图正则，不提供实际载荷或累计滑移证据，也不改变接触真值和递推验收。

2026-10-05同种子、同数据、同日程的200轮实验（100示范＋100反馈）中，h100连续
30步的前15步保持部位材料区域平均运动由6.22cm降到2.98cm，30步最大穿透由
75.9mm降到25.8mm；但验证端点通过率由40.0%降到23.6%，递推通过前缀由1步降到0步。
这轮训练仍处于完整teacher计划阶段，后15步无示范目标。冻结真实训练池诊断中，
69个可比较样本有34个保持／接触实现姿态梯度反向，不能把减少运动等同于约束协调成功。
完整结果为`runtime/current/models/generator/keep_patch_scratch200_20261005/experiment_result.json`。
示范载荷只作reference，递推输出不能继承；缺少观测保留known mask和NaN。

全链路接触、承重、滑移与保持意图的统一口径见
[支撑证据](../../docs/support-semantics.md)。网络结构和示范数据不会因附加观测自动改变。
