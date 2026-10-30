# baseline1000 第14步静态恢复研究

## 结论

初始左脚点对缺失定位到mesh–convex midphase的front-facing剔除，而不是全身损失聚合遗漏或MJWarp转换丢失。实体退出约束能跨过该检测盲区，向箱顶恢复比本次侧向候选更合适。所有结果只针对这个固定姿态。

## 点对为何缺失

shape69/70是left_ankle_roll_link。原姿态粗筛包含(0,69)/(0,70)；BVH查询得到三角形[1,4,5,6,7,9]，全部在front-facing过滤被剔除，triangle_pairs、raw contacts和solver rows均为0。
上游中心实际取shape transform原点（不是几何体质心或最近点）。原点z=0.626862586 m，箱顶z=0.634044945 m，位于顶面内侧7.18236 mm。顶面4/6的center signed distance=-0.00718236 m。
之前global_single_0.25姿态的原点z=0.634057283 m，顶面符号距离+0.0000123382 m；三角形4/6开始通过过滤，随后出现向下法线的局部穿透分支。
代码：[collision_core.py:1014](/home/xiaz/somaforge/runtime/environments/newton-main-20260914/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1014) 与 [:1115](/home/xiaz/somaforge/runtime/environments/newton-main-20260914/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:1115)。midphase_audit.json调用同版本上游函数做只读复核，没有改碰撞流水线。

## 实验方法

权威URDF全部47个collision元素（29 mesh、14 sphere、4 cylinder）参与实体退出约束：mesh用凸包顶点求支撑面最小值，球/圆柱用解析支撑函数；地形使用当前场景真实闭合凸箱体，地面z=0。对每个形状选择一个箱体分离面，左脚枚举顶面及4个侧面，求35维姿态增量的软约束QP。
每轮使用可微FK的一阶Jacobian、关节限位、平移每轴最多2 cm/角度每轴最多0.1 rad的信赖域、松弛变量和线搜索。实际自碰撞点对也进入约束。每个提案均重新查询Newton；优化接受不要求原始三角形最大负距离单调下降。
这是所选箱体分离面的保守约束，不是一般网格MTD/EPA；不能据此宣称非凸场景或其他机器人上同样有效。

## 接触保持对照

A：固定原材料点的XYZ目标（将原始嵌入的法向位置投到表面附近）。向上恢复消除了深嵌入，但材料点误差约1.62 cm，主要在膝盖法向。
B：法向使用当前最低支撑点并固定到margin内某个目标高度；仍有约1.58 cm法向目标误差。
C：保留原材料点切向XY，法向允许[0,0.9×原始实际includemargin]区间；接触是否保留最终由Newton active+allocated及主表面决定。该区间仅是优化约束，不是新的接触标签规则。

## 最终区间恢复结果

- 结果姿态：face3_iter5；研究预设静态检查通过：True。
- Newton全身最大负距离：0.000000 cm；自碰撞：0.000000 cm。
- 全部形状所选箱体分离面/地面最大违反量：0.000000 cm。
- 左脚原始网格逐顶点独立检查：箱内顶点0个（初始23328个），最大顶点最近出界距离0.000000 cm。
- 原左手、左膝接触全部保留：True；实际六部位mask=[True, False, True, False, True, False]。
- 原接触代表点XY位移cm：[None, None, 0.021205366638827654, None, 0.0598754634217675, None]。
- 根平移变化：3.125 cm；最大关节变化：19.067度；关节超限：0 rad。

检查口径：Newton负距离≤项目既有1 mm验收限，实体分离面违反≤0.2 mm、关节限位、原接触身份保留、优化接触残差≤1 cm。后两项数值属于本静态研究的容差，不改变接触判定。
原始姿态已有严重穿透，恢复中间步允许原接触暂时丢失；恢复终点保留原接触，但不是已验证的连续无滑动、承重或动力学运动。新增左脚接触不等于原预测任务计划全部实现。

## 实验候选

| 材料点固定的退出候选 | Newton最大负距离cm | 分离约束违反cm | 原接触全保留 | 材料点误差cm | 根位移cm |
|---|---:|---:|---|---:|---:|
| face0 [-0.9103364821118746, 0.4138689277249223, 0.0] | 3.9226 | 8.5992 | True | 13.808 | 3.464 |
| face1 [-0.4138689490073437, -0.9103364724361848, -0.0] | 0.0000 | 0.2967 | False | 28.609 | 25.459 |
| face3 [0.0, -0.0, 1.0] | 0.0155 | 0.0158 | True | 1.616 | 2.404 |
| face4 [0.41386894583199896, 0.9103364738798011, 0.0] | 0.0051 | 0.0437 | True | 15.448 | 18.779 |
| face5 [0.910336473352303, -0.4138689469922715, 0.0] | 0.0276 | 0.4026 | False | 55.809 | 25.913 |

## 产物

diagnostic.json：区间保持实验完整记录；queries/*.pt：每个提案原始Newton点对与midphase/broadphase/raw映射；midphase_audit.json：背面剔除复核；viewer.json：原姿态、梯度失败与SQP候选对比。
多方向实验位于../baseline1000_step14_static_research_20260925；定高对照位于../baseline1000_step14_static_surface_20260925。所有实验未修改checkpoint、机器人资产、margin/gap或生产求解代码。

## 新的代码位置与复现边界

- `contact_solver.research.step14_static.run(query)`：固定第14帧多面软约束 SQP。
- `contact_solver.research.step14_midphase.inspect(query, shape)`：只读 midphase 审计。
- `tmp/research_step14_static.py` 等旧脚本保留，历史命令仍可追溯。

研究代码依赖已初始化的原场景 query，不能直接当通用 CLI 执行。
在项目根目录运行，使用当前 Newton 环境（`source scripts/source_isaaclab3_newton_setup.sh`），
设置 `STEP14_RESEARCH_OUTPUT=tmp/<新的实验目录>`，然后将同一场景 query 传给 `run`。
额外依赖为 contact-solver 的 research extra；Newton 私有 midphase API 固定于
`runtime/environments/newton-main-20260914` 版本，不宣称兼容其他 Newton 版本。
研究输入是 `tmp/baseline1000_step14_gradient_20260925_v2/diagnostic.json`，
输出审计在 `tmp/baseline1000_step14_static_interval_20260925/`。
本次只归档研究实现，没有重跑仿真，也未把研究约束接入生产梯度。
