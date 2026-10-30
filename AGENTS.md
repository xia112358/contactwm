新添加/删除/改动代码前永远先计划好问我，除非要求连续诊断。

除非用户明确要求停止/重启训练，禁止擅自停止正在运行的训练进程；需要释放 GPU 做 eval/visualization 时必须先说明原因并等待用户确认。

训练保存规则（按用户最新要求）：各类 checkpoint 沿用各自原有的常规格式，支持正常续训；WBT 保存模型、优化器、归一化、训练配置与进度及已有采样器／课程状态，禁止打包整个仿真器／原生求解器快照。按原有保存间隔保留里程碑和最终 checkpoint，不限制只留最后一份。PHP 五组独立 WBT 训练使用 4096 环境。其他模块沿用其原有格式，不额外扩展保存内容。已有文件不得擅自删除。

Isaac Sim / CUDA 初始化失败通常是 sandbox 限制，到本机环境可以跑。

SomaForge 的 G1 机器人唯一权威资产是
`src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf`。
禁止绕过 `somaforge_core.robot_assets` 使用其他 G1 URDF/XML；缺少
`robot_asset_json` 的 motion、GMVQ 数据和 checkpoint 均视为旧错误资产产物。

接触判定统一规则（当前 Newton/MJWarp 仿真）：

- 数据标注、touchdown 切分、predictor/DAgger 监督、闭环接触观测及评估，必须使用当前 Newton/MJWarp 的实际接触判定。不得另建一套裸几何距离阈值来定义接触，不做其他后端的替代判定。
- 区分碰撞候选、约束激活、约束分配和接触力。以求解器实际 `type` 含 `CONSTRAINT` 且 `dist < includemargin` 为激活条件，并检查 `efc_address` 的约束分配状态；未分配的激活接触须单独报错/标记，不能默认为正常接触。接触力用于受力/支撑分析，不能代替接触激活判定。
- 读取当前求解器实际字段及形状映射，禁止硬编码 5 mm、2 cm 等阈值或猜测 margin。不得为迁就离线标签擅自修改已训练 policy 所用的仿真 margin、gap 或机器人碰撞资产。
- 缺失求解器接触字段时，明确报缺失/未知；不得静默回退到距离、预测接触意图或力阈值。旧几何标注只能作为诊断产物，未经 Newton 验证不得作为新训练基线或接触真值。
- 保存接触语义版本、环境/部位/形状映射及采样时序；求解器接触与积分后位姿不得冒充严格同帧。必须使用权威机器人资产。部位汇总不能把自碰撞或跨机器人碰撞当作地形接触。
- 预测的接触是意图，不是已实现接触；距离、穿透、贴面及滑动指标可用于优化和诊断，但不能自行重新定义仿真接触。激活接触也不等于已满足承重、无滑动等轨迹要求。
- 本任务有效接触统一经 `somaforge_core.contact_face_selection` 按真实主表面筛选：只保留地面／水平顶面，侧面及其他非向上水平面作为异常排除。聚合证据、标签、切分、增广约束和有效接触可视化必须共用该选择；原始 Newton 点对完整保留作审计，不修改位姿、不把侧面的候选顶面提升为主表面。旧任务标签必须显式重新物化，不能沿用旧缓存冒充新规则。

读文件、查看配置、搜索文本、查看数据/日志/CSV/YAML/JSON、复述文件内容、汇总简单指标等基本任务，如果不需要明显推理，优先调用使用 spark 模型的 subagent 来完成。

像更新 AGENTS.md、补一条简单说明、调整少量文案这类低风险、低推理的简单修改任务，也优先调用使用 spark 模型的 subagent 来完成；仍需遵守“改代码/文件前先计划并问我”的规则。

项目相关临时脚本、图、CSV、分析产物统一放到项目根目录 `tmp/`，不要放系统 `/tmp`。

CLI 标准：

- train/eval/replay/run_sim 主入口统一先由 IsaacLab AppLauncher 解析官方参数，再由 tyro 解析项目配置。
- 可视化只使用一个开关：`--visualizer kit`。
- 不写 `--visualizer` 时默认 headless。
- `--headless` 优先级最高，强制无窗口。
- `training.headless=False` 不再作为打开 Kit 窗口的 CLI 入口，只接受最终 AppLauncher/CLI 状态同步。
- `--enable_cameras`、`--device`、`--experience`、`--kit_args` 等官方 AppLauncher 参数归 AppLauncher 解析，不让 tyro 当成项目配置吞掉。
