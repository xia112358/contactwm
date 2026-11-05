# 接触参与与子步切分进度（2026-11-04）

历史快照；最新进度见 [2026-11-05](../contact-20261105/README.md)。

最新离线版本为 `partition-v8`：12 条来源、3789 个控制帧，审核排除 15 帧后保留
3774 帧、161 段。它与此前 `partition-v7` 的全部记录张量、目录及切点完全一致，
本版完成核心逻辑整理。统计见 [summary.json](summary.json)，逐段见 [cases.csv](cases.csv)。
[2026-11-03](../contact-20261103/README.md) 的 197 段保留为旧采集快照。

## 本轮实现

- **接触与支撑按实际物理子步记录。** 新采集在每次原生求解后读取分配约束、
  接触面和约束求值时刻的部位位姿。当前物理步长 2 ms、控制周期 20 ms；
  支撑快速确认的 5 次观测为 5 次物理求解，确认窗口跨度 8 ms。
  独立支撑事件还需累计 40 ms 实际承重，断触等待不计入。
  控制帧保留所有建立、确认、退出事件及原始子步状态，不能只取帧末接触掩码。
- **支撑参与统一为进入、退出或重新分配。** T 表示“参与变化”，不局限于
  从旧接触飞到新接触。此前没有接触的手开始参与也能形成支撑建立事件。
  实际观测到主形式后才记录脚/膝；腾空过程中不提前标注目标形式。
- **同一落点的卸载与恢复关联。** 对两端都有原生支撑证据的同部位、同面恢复，
  使用采集时序列化的真实 `surface_v2` 碰撞网格判断是否仍属原支撑过程。
  间隔不超过 300 ms、整个间隔保留共同投影区域、垂直偏离不超过 2 cm，
  并且没有插入另一主接触形式，才关联。共同区域逐次取交集，不能移动锚点跟随漂移。
  这只是已观测支撑事件的关联，不补写间隔中的原生激活、力、实际支撑或点 loss。
- **切分以支撑建立完成和独立姿态阶段为核心。** 临近完成事件可合并，延续的旧参与
  和静止收尾可吸收；完成后新开始的参与保留前一完成边界。单独撤出不制造新切点。
  记录结束时新开始但尚未完成的转移单独标注 `incomplete_transfer`。
  没有按段号修补，没有统一最短长度过滤或最大长度强制切段，也不沿重叠关系无限回溯。

支撑仍以原生激活和向上承重力为前提，30 N 进入、15 N 退出，退出持续时间及
间隔运动条件按采集契约解释。点 loss 继续使用独立的稳定性掩码。几何范围
仅用于原生支撑事件的同落点关联，不能替代实际接触判定。

## 数据变化和统计

| 指标 | 上次发布 `native-release-v5` | 最新 `partition-v8` |
|---|---:|---:|
| 来源条数 | 12 | 12 |
| 采集控制帧 | 4146 | 3789 |
| 保留控制帧 | 4146 | 3774 |
| 段数 | 197 | 161 |
| ≤10 帧短段 | 32 | 11 |
| >40 帧长段 | 5 | 6 |
| 最短 / 中位数 / 最长 | 2 / 23 / 49 | 6 / 24 / 49 |
| 旧端点目标子集 | 20 | 4 |

新采集中 `climb_76_original` 提前在 111 帧结束，上次为 468 帧，差额 357 帧
属于来源记录长度变化，不能归因于切分。该采集的失败标记仅作诊断，其有效区间
仍被保留，没有以整条 tracking 成功作为数据准入。

另外三段经人工审核的首尾排除共 15 帧，绑定源文件 SHA256：

- `climb_76_mirror [0,5)`：起始残片；
- `walk_05400_original [394,399)`：被记录终点截断的出发；
- `walk_07800_mirror [394,399)`：被记录终点截断的出发。

审核清单见 [reviewed-edge-exclusions.json](reviewed-edge-exclusions.json)。仅去掉完整首尾
段，其他段不因这次排除重切，不删除或改写原始采集。

新采集原始点掩码有 1299 个有效部位帧，保留区间内为 1294；159 段有逐帧点监督，
2 段只有轨迹。这个数量与旧采集的 1492 不宜直接解释为 loss 质量变化。
新数据的旧端点子集只有 4 段，也不等于其余动作无效或不能训练。
完整消费逐帧接触目标的训练入口仍待接通，本次没有启动训练。

## 验证与复现

最新源码的接触、子步、落点关联、切分及工作流回归为 410 项通过，另有 28 个
pytest subtests。`partition-v8` 已有与 v7 完全等价的整理检查
[consolidation-check.json](consolidation-check.json)；同步时额外重算全部切点、
检查保留帧恰好覆盖一次、原生主形式证据及产物 SHA256，见 [verification.json](verification.json)。

重建使用原库提供的历史身份和本次显式采集目录，不继承旧采集的 tracking/restart 结果：

```bash
python packages/supermimic/tools/rebuild_contact_topology_library.py \
  --previous /path/to/original-native-library \
  --capture_root /path/to/fresh-substep-captures \
  --edge_exclusions docs/research/contact-20261104/reviewed-edge-exclusions.json \
  --output tmp/contact-slices
python packages/supermimic/tools/export_contact_role_review.py \
  --dataset tmp/contact-slices \
  --task-manifest /path/to/original/task.json \
  --output tmp/contact-role-review
python -m http.server 18765 --bind 127.0.0.1 --directory tmp/contact-role-review
```

`--previous` 的 recording 元数据须指向实际可用的 teacher checkpoint；新采集需包含
`native-rows.pt`、对应报告、控制帧 recording、所有子步证据和配套碰撞模型。
本次提交提供源码及审计，不发布新的原始采集/模型 release。
`shapely` 和导出用 `trimesh` 已纳入 workflow 依赖锁定。

浏览器 `/?review=length` 查看 [11 个短段与 6 个长段](length-review.json)，按源帧、
角色、阶段和切点原因逐段播放；子步采集还展示实际建立/退出与承重采样计数，
紫色表示同一支撑过程中的暂时卸载。旧控制帧记录仍可导出，但不会虚构子步数据。

本次同步还通过全仓库静态检查（解析 1252 个 Python 文件）、提交版资源配置的
source-only 检查和浏览器回归。仓库版导出器已实际生成新 161 段子步记录及旧 197 段
控制帧记录。浏览器通过 11 个短段、6 个长段清单、子步统计、播放/暂停、帧定位和手机布局检查。
新增的导出/显示分支只区分这两种实际存在的采样格式，不为旧记录伪造子步证据；
本次同步未加入额外训练准入限制或生产兜底分支。
