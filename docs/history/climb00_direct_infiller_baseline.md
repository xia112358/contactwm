# Climb00 历史实验：direct_infiller_v1

> 历史实验配方：下文 `tmp/` 是当时的路径记录，不能作为当前入口。已保留的语料和源码记录迁入 `runtime/current/archives/preserved_inputs_20261010/`；回归函数归入 `generator.research.archived_experiments`。迁移不代表旧语料满足当前接触语义。
> 保留用于复现，不作为最新训练入口；不得靠更改版本标签绕过重新物化与验收。

基线是结构、数据、优化和评估的约定，不是某个 `.pt`。后续实验以此为
对照；旧共享修正器只作为历史对照，不删除。修改本约定应另立版本。

## 当时的启动与复现（入口已退休）

以下保留当时的命令记录。`scripts/train_climb00_direct_baseline.py` 已退休并
移到系统回收站，不是当前可运行入口。当前入口见[维护文档](../repository-maintenance.md)。
`configs/climb00/direct_infiller_v1.json` 和被引用的历史语料继续保留。

```bash
env OMP_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n env_somaforge python \
  scripts/train_climb00_direct_baseline.py --headless --device cuda:0 \
  --output tmp/climb00_direct_baseline_run
```

配置来源为 `configs/climb00/direct_infiller_v1.json`。官方参数由 AppLauncher
先解析，训练参数再由 tyro 解析。可覆盖参数，但应将其作为显式实验变体；
实际配置、数据划分和初始化状态写入运行目录的 config.json。

- 随机初始化，seed=20260908；不加载 infiller 权重，不多轮微调。
- Adam，初始 LR=0.001，按墙钟时间余弦下降至 1e-5，梯度范数裁剪 10。
- 7800 秒时间预算，4000 epoch 上限，先到即停止；**不是固定 713 轮**。
- 等长分桶，不补帧：frame_budget=8192，batch 上限 512；当前训练集每轮
  67 批。固定批成员、缓存固定量，每轮打乱批顺序。
- 每 50 轮、首轮和最后一轮完整评估。best 按验证总损失选择；last 是预算
  结束结果。不得按测试结果挑 checkpoint。墙钟调度受硬件速度影响，不能
  保证不同机器得到相同轮数或数值；改固定轮数调度属于新变体。

## 结构和推理边界

`tmp/direct_contact_infiller.py`：182→256→256→256→36，隐藏层 Tanh，
输出层零初始化，其余层随机初始化。输入为起终点全身 q、预测接触锚点/
局部偏移/拓扑/表面、箱体几何、地面高度、时长与相位编码。

单次前向输出整段。内部使用三次端点插值与端点消失的可学习项；这不是
第二个生成器。四元数归一化，关节限位处理，首末帧 q 精确固定，邻接帧
不固定。不输入旧轨迹中间帧、参考朝向、速度或事件编号；无后置修正、
DLS、投影。跨段切线匹配没有硬保证。

## 数据与资产

- 权威资产由 somaforge_core.robot_assets 解析；数据/checkpoint 必须校验
  robot_asset_json。历史模型仅使用已有审计迁移副本，不绕过资产检查。
- `tmp/build_shared_full_corpus.py` 构建 `tmp/contact_shared_full_v1/corpus.pt`；
  原脚本目标目录固定且拒绝已存在目录，不要为重建而删除现有数据。
- 冻结 predictor/旧 infiller 为缓存构建依赖，位于
  `tmp/contact_pipeline_migrated_v1/`；它们不是新 infiller 的初始化权重。
- 207 条完整轨迹、4761 段：训练 h095/100/105 共2254段，验证 h090共1196段，
  测试 h110共1311段。保持原 source→row→event 映射。
- 起终点与接触条件来自冻结流水线递推缓存；旧轨迹中间帧只作弱训练监督，
  原始参考接触部位朝向只作训练损失。不是在线重新生成缓存。

## 固定损失

实现复用 `tmp/full_shared_contact_model.py` 的 objective/prepare；保留该文件
是复用损失，不代表运行后置修正器。基线不得清理掉这些代码依赖。

- 50 Hz 原生帧，每间隔10份密集检查（500 Hz），全身权威碰撞几何，每mesh
  最多24采样点；箱体和地面都检查，报告不扣接触容差。
- 穿透训练代价对超过2 mm的部分按1 cm归一化，使用现有 exponential
  实现；箱体/地面各权重10，保留其平均、峰值和占用率聚合。
- 接触位置按5 mm归一化；persistent/touchdown/liftoff 权重10/12/6，
  touchdown在后20%，liftoff在前20%；接触朝向权重10，相对角步长权重2。
- 旧轨迹归一化偏离权重0.01，归一化二阶差分权重0.1。
- 18项根部/关节/旋转矩阵的一至三阶 peak/RMS 预算，惩罚权重2000。
  预算精确值已固定在基线JSON，不再依赖旧单段实验报告文件。

## 评估与参考运行

必须报告接触锚点误差、各段峰值穿透的均值/P95/全局最大值、完整轨迹
根部二阶差分最大值，不能用平均值替代最坏值。现训练器自动输出均值和
完整轨迹根部指标；P95/最坏穿透需对保存的全部片段以同样500 Hz口径补查。
接触锚点误差不等于真实表面贴合，根部平滑也不等于全身连续性。

参考运行 `tmp/contact_direct_infiller_v1`，713轮last，700轮best。以下仅是
713轮运行记录，不是挑选该权重作为基线定义：

| 指标 cm | 验证 h090 | 测试 h110 |
|---|---:|---:|
| 接触锚点误差 | 0.608115 | 0.654893 |
| 片段峰值穿透均值 | 0.458738 | 0.916292 |
| 片段峰值穿透P95 | 2.184877 | 2.639118 |
| 最坏穿透 | 2.946396 | 3.587109 |

完整轨迹根部二阶差分最坏1.732441 cm/frame²。仍有局部厘米级穿透。
固定23段、上游使用预设事件进度、理想几何输入；仅验证高度留出，未独立
验证角度留出、感知噪声、在线闭环和物理执行。现评估用缓存边界拼接，
实时生成器适配不是此次基线整理已完成的功能。

### 后续集成验证（不改变训练基线）

已新增 `tmp/generate_direct_infiller_closed_loop.py`，从初始状态重新调用
predictor和单段式infiller递推，不读取预测缓存作为输入、不加载旧infiller。
h090/h110各一条完整23段轨迹已验证，与同源缓存结果基本一致，见
`tmp/contact_direct_closed_loop_v1/README.md`。这只补充模型递推验证，
不代表已验证全部207条、新场景或物理状态反馈闭环。
