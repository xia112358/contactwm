# 第14帧：SDF / 共享面梯度研究

固定原始姿态；实际 Newton 模型指纹与前一轮 SQP 研究一致。每个试探姿态重新查询 Newton，不做训练或积分。


## 本轮结论

SDF 补全了 Newton 缺失的左踝 roll 几何梯度，但本轮所有梯度方法均未通过静态恢复验收。
最好的 SDF 求和版本在 180 步内把足部网格穿透由 8.9981 cm 降至 0.7010 cm，原左手/左膝接触保留，
但仍有 1003 个足部顶点在箱内，根平移 13.6703 cm、最大关节变化 21.9299°。
此前 QP 参照能消除足部嵌入，根位移约 3.125 cm，因此当前梯度更新尚未复现该恢复效果。

初始所有嵌入足部样本的 SDF 出口均为箱顶，不能将这帧的停滞归因于初始点场的不同出口抵消。
已观察到普通下降缓慢、关节限位附近停步，以及 Newton 非法法线试探。最大值/支撑点切换和接触耦合是需要继续隔离的因素。
求和版拒绝并审计了 54 次非法法线试探。对具体停步原因仍需分项方向导数分析，不能仅凭这些对照确定单一根因。
后续方向已按训练目标修正：直接验证标量损失反传到预测网络后的实际效果，见 [网络单步梯度验证](step14-loss-direction.md)。不再将外部约束修正器的成功当成训练损失成功。

## 结果

| 梯度更新 | 碰撞损失 | 步数 | 全足网格箱内深度 cm | 箱内顶点 | Newton负距离 cm | 原接触保持 | 静态通过 |
|---|---|---:|---:|---:|---:|---|---|
| steepest | newton | 32 | 5.85005 | 9387 | 3.90029 | True | False |
| steepest | sdf | 180 | 3.81740 | 6080 | 3.69425 | True | False |
| steepest | shared_top | 180 | 3.63019 | 5883 | 3.51786 | True | False |
| steepest | shared_auto | 180 | 3.63019 | 5883 | 3.51786 | True | False |
| metric | newton | 15 | 1.05841 | 1069 | 0.42208 | True | False |
| metric | sdf | 19 | 1.98208 | 6695 | 1.98196 | True | False |
| metric | shared_auto | 21 | 1.96972 | 6641 | 1.96283 | True | False |
| metric_sum | sdf_sum | 180 | 0.70102 | 1003 | 0.70104 | True | False |
| metric_sum | shared_sum | 33 | 0.93903 | 1780 | 0.91917 | True | False |

## 已验证的事实

- 初始足部箱内深度 8.99810 cm；原始网格箱内顶点 23328。
- 3246 个箱内足部表面样本全部指向箱顶，未见初始出口方向抵消。自动选面与指定箱顶一致；普通下降两组结果逐位相同。
- SDF/共享面补出了缺失的左踝 roll 梯度。Newton 的其他腿部梯度仍然存在，不能把整个左腿说成没有梯度。
- 初始完整损失的自动微分与中心差分相对 L2 误差约 4.6e-10。Newton 这一数值检验针对冻结 witness 的局部代理，不宣称跨点对切换也可微。

## 方法与边界

- 新增 sdf_sum 为每形状表面样本的平方穿透均值再对形状求和；shared_sum 为每形状支撑穿透平方求和。它们取消全局最大点项，同时也改变了残差权重分布，不能把差异全部归因于平滑性。
- sum 版本遇到非法 Newton 法线试探时保存原始点对、明确拒绝并缩步；不替换法线、不回退几何接触。
- 解析 SDF 来自真实场景闭合 OBB，并核对实际地形顶点；不是高度差替代。权威机器人 47 个 collision 元素，每形状 512 个表面积采样点。
- 共享面使用网格凸包顶点、球/圆柱解析支撑值。最终足部检查使用全部原始 collision mesh 顶点，与采样集独立。
- 碰撞项为平方深度的 mean+max；SDF 是所有形状等数采样点，共享面是各形状支撑深度，Newton 是实际点对。因此数值权重/聚合不完全等价，不能据此排名通用优劣。
- 所有组接触保持项相同：原材料点 XY + 实际 margin 内的法向区间，contact 权重10。关节尺度为根平移 .02m、转角/关节 .1rad。
- metric 更新在同一损失梯度上使用 (I+20 J_contact^T J_contact)^-1，并在活动关节限位处限制可行方向。没有使用此前的全身碰撞约束 QP，也没有用其解作为监督方向。
- 原接触是否保留由当前 Newton 实际激活、分配和真实水平主表面确定。SDF/分离面只是优化和诊断，不能定义接触真值。
- 早期 metric、metric_v2 结果保留；两轮暴露的限位试探步/方向问题在 metric_v3 中修正。
- 本次仅静态单场景，允许恢复途中暂时丢接触；不证明全局可恢复、承重、无滑动或动力学可执行。

## 产物

- 普通下降：`tmp/step14_gradient_fields_20260925_v1`
- 预条件梯度：`tmp/step14_gradient_fields_20260925_metric_v3`
- 求和版本：`tmp/step14_gradient_fields_20260925_sum_v2`
- 图表/CSV/合并可视化：`tmp/step14_gradient_fields_20260925_report`
- 所有原始 Newton 点对和实际约束字段在对应 `queries/*.pt`；终点完整审计为 `*_newton_audit.json`。

## 运行

在仓库根目录使用已固定的 Newton 环境。下面的输出目录必须是新的：

```bash
source scripts/source_isaaclab3_newton_setup.sh
export STEP14_FIELD_OUTPUT=tmp/step14_gradient_fields_new
export STEP14_FIELD_OPTIMIZER=contact_metric
export STEP14_FIELD_METHODS=sdf_sum,shared_sum
export STEP14_FIELD_STEPS=180
mkdir -p "$STEP14_FIELD_OUTPUT"
python -m contact_solver.research.step14_gradient_field \
  --checkpoint runtime/current/holosoma/logs/WholeBodyTracking/20260727_085520-g1_29dof_wbt_single_climb00_completionema_horizon50_ncon160_from4k_to10k-locomotion/model_06000.pt \
  --motion-manifest tmp/newton_contact_sources_v1/1789326506727043514/scene_1/manifest.json \
  --binding "$STEP14_FIELD_OUTPUT/binding.json" --create-native-binding \
  --inspection-output "$STEP14_FIELD_OUTPUT/model.json" \
  --query-nconmax 2048 --query-njmax 16384 --device cuda:0 \
  --capture-contact-sources --tensor-auth tmp/baseline_step14_gradient_auth.bin
```

这里复用标准 worker 的 AppLauncher/tyro 初始化与原有场景指纹。
研究入口将 tensor-server 回调替换为本进程实验，不启动监听；`--tensor-auth` 仅用于进入该 worker 分支，
不会向外发送文件内容。输入诊断、原场景 manifest 和策略 checkpoint 均为本地研究依赖。
机器人资产通过 core 获取；没有改 policy、collision margin/gap 或原始 contact semantics。

可视化：

```bash
./motion-edit contact-editor \
  --review tmp/step14_gradient_fields_20260925_report/viewer.json \
  --port 8094
```

左侧原姿态、右侧试验姿态，绿色箭头为同一足部材料点的真实位移；显示倍率单独标明。
在同一 Motion Edit 页面按方法和保存迭代选择；旧接触标记只作为历史诊断。
可选择此前 QP 参照。它仅供比较，没有用于构造本轮损失、梯度或监督目标。
