# 目标接触位置的观测覆盖

`packages/contact_observability/` 与其他研究包同级，集中处理目标接触位置是否出现在传感器观测中的审计和独立实验。已有 Predictor 和仿真包继续提供机器人资产、观测几何等公共能力。

代码和说明放在本目录；导入的历史记录、运行结果、图表和汇总统一放在 `tmp/contact_observability/`。

相关入口：[环境配置](../../docs/environments.md)、
[共享资产与接触契约](../somaforge_core/README.md)、[Generator](../generator/README.md)、
[WBT expert 与蒸馏](../../docs/wbt-modular-experiments.md)、
[文档索引](../../docs/README.md)。本目录是独立观测诊断工具，不是训练启动器或接触标签生产入口。
下面的 `tmp/` 报告只存在于本机生成结果中，不随 Git 仓库分发；新机器需运行对应命令。

## 历史实验与配置诊断

### 起始帧审计

来源：`tmp/predictor_establish_visibility_20261013/report.json` 与 `targets.csv`。

审计原示范中建立接触的目标：在每个区间的起始帧检查目标所属主表面上的投影点是否位于相机视场和量程内，并验证地形 first-hit 遮挡；接触保持目标不计入这项审计。该投影只用于可见性诊断，不改变原接触标签。

207 条动作共有 3,840 个目标，其中 2,852 个可见，988 个在视场外。候选覆盖不等于目标可见：170 个不可见目标仍被深度候选标签覆盖，527 个仍被完整候选标签覆盖。候选容差采用历史训练标签规则 `2 × scan_spacing_m`，不是接触判定阈值。

### 单条动作的全程累计覆盖

来源：`tmp/legacy_camera_route_coverage_20261013/` 下的 `report.json`、`targets.csv` 和 `run.py`。动作是 `guard01_h110_dp50p000_am10_ru_p050`，共 700 帧、50 Hz、13.98 秒。

逐目标记录区分以下时序：

| 统计 | 含义 | 历史及复跑结果 |
| --- | --- | --- |
| `covered_at_start` | 当前区间起始帧的同表面 4 cm 像素覆盖 | 15 / 19 |
| `seen_by_start` | 从动作开始到该区间起始帧，是否曾被覆盖 | 18 / 19 |
| `seen_before_target` | 严格在目标接触帧之前，是否曾被覆盖；不含目标帧 | 18 / 19 |
| `seen_anytime` | 整条动作任一帧是否曾被覆盖 | 19 / 19 |
| `first_seen_frame` | 首次覆盖发生的帧 | 逐目标保存 |

唯一例外是 sample 2160 的左脚地面目标：区间从第 0 帧开始，目标接触帧为 104，首次覆盖却在第 219 帧。全程最终覆盖不能作为决策时已经观察到目标的证据。上述结果也不表示整段始终可见。

这里的目标覆盖采用同一表面上真实 first-hit 深度像素与目标的最近距离，4 cm 仅是几何覆盖诊断容差，不能定义或替代实际接触。

面积覆盖采用 2 cm 表面局部网格：任一帧的实际 first-hit 深度像素命中某网格单元后，该单元累计计一次。这是近似采样面积覆盖。地面范围限定为路线 root XY 周围 3 m，并排除箱体下方；历史累计覆盖约为 72.976%，箱顶为 100%。箱顶只有 676 / 700 帧出现射线命中，累计面积全覆盖不代表每帧完整覆盖。

迁移后的独立入口已在 CPU 上完整复跑上述 700 帧，累计时序和面积覆盖结果与历史记录一致，并显式补充了当前起始帧覆盖 `covered_at_start = 15 / 19`。

### 根相机配置诊断

来源：`tmp/predictor_root_depth_20261013/final_report.json`。最终根相机配置报告的 3,840 / 3,840 个目标位于观测范围内，且达到同表面 4 cm 像素覆盖容差。该结果仍是几何观测诊断，不含身体遮挡，不能扩展为真实接触或实际传感器全覆盖结论。

独立入口也已用 `root_depth_87x58_fov89p5_pitch85_yaw_v1` 完整扫描上述同一条 700 帧动作：当前起始帧、截至起始帧累计、接触前和全程覆盖均为 19 / 19。本机产物位于 `tmp/contact_observability/route_root87/index.html`；这项单条动作复跑与历史 207 条动作起始帧审计分别保存。

## 观测与接触约定

历史全程实验使用原 torso 相机：87 × 58、水平视场 89.5°、向下俯仰 27°、光轴深度量程 0.3–3 m、安装偏移 `[0.01, 0.01, 0.44]`。历史结果是理想地形射线观测，不含机器人自遮挡、噪声或真实传感器扫描时序。

观测覆盖与接触真值分开记录。当前 Newton/MJWarp 的约束激活、约束分配及其原始接触字段仍是实际接触判定依据；预测目标是意图，几何接近、深度像素覆盖、面积覆盖或接触力不能自行重新定义接触。

G1 资产必须经 `somaforge_core.robot_assets` 使用唯一权威资产 `src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf`，并保留、核验 `robot_asset_json`。有效接触继续统一通过 `somaforge_core.contact_face_selection` 按真实主表面筛选，只保留地面与向上水平顶面；侧面观测可保留用于遮挡和审计，不作为有效接触标签。

历史记录保留来源与旧实验语义，不能自动成为经过当前 Newton 及主表面规则验证的新接触真值。沿用旧任务标签前必须显式重新物化；本目录的导入、汇总和可见性扫描不承担这一验证。

## 工具职责

`run.py` 提供历史导入、汇总和独立全程扫描入口：

- `import-history`：复制历史记录到 `tmp/contact_observability/history/`，生成来源清单与摘要，保留原文件。
- `summarize`：只读汇总导入结果；使用 `--write` 时将汇总写入 `tmp/contact_observability/summary.json`。
- `route`：重新运行独立的全程扫描，并将实验产物写入 `tmp/contact_observability/`。

历史 `run.py` 等脚本作为来源快照保存在 `history/` 中供审计。正式入口不执行这些含硬编码路径和配置的历史脚本。

本机历史导入记录包含 6 个实验的 98 个文件，约 158 MB，复制结果全部经过 SHA256 核验。生成的总索引为 `tmp/contact_observability/index.html`；独立复跑的查看器为 `tmp/contact_observability/route/index.html`。这些是本地报告路径，不是远程仓库可下载的资产。

```bash
# 导入历史记录，保留来源快照、清单和摘要。
python3 packages/contact_observability/run.py import-history

# 只读汇总；添加 --write 时保存汇总 JSON。
python3 packages/contact_observability/run.py summarize
python3 packages/contact_observability/run.py summarize --write

# 使用原相机重新扫描整条动作。
python3 scripts/manage_environment.py run --profile main -- python packages/contact_observability/run.py route

# 指定根相机配置和独立输出目录。
python3 scripts/manage_environment.py run --profile main -- python packages/contact_observability/run.py route \
  --profile root_depth_87x58_fov89p5_pitch85_yaw_v1 \
  --output tmp/contact_observability/route_root87

# 按需打开总 index.html；8108 仅为示例端口。
python3 -m http.server 8108 --directory tmp/contact_observability
```

启动示例 HTTP 服务后，在浏览器访问 `http://localhost:8108/` 查看总索引。上面的文档示例不会自动启动服务。

运行接口及参数以 `run.py --help` 和各子命令的 `--help` 为准。全程累计、截至决策时的历史累计、当前帧可见性应分别解释，不能相互替代。

`route` 要求输出目录为空；再次运行请用 `--output tmp/contact_observability/新的运行目录`。`summarize --json` 输出完整摘要；默认终端输出简短统计。
