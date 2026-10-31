# 接触实现与诊断记录（2026-10-31）

## 当前结论

当前训练采用 `surface_v2` 机器人与 `newton_contacts_v2`：Newton 生成碰撞，
MJWarp 解动力学；每个 2 ms 物理子步重新生成接触。控制周期为 20 ms。
这不是 Hydroelastic 整机训练，也不是 PhysX。

`newton_contacts_v1` 每 20 ms 才生成一次接触，不能与 v2 混作同一个物理契约。
`manifold_v3` 使用 MJWarp 内部碰撞，并保留 EPA 面投影法向修复。
法向修复没有修复所有异常接触位置。

真实接触仍要求原生 CONSTRAINT 激活、dist < includemargin 和有效约束分配。
几何穿透用于诊断，不能替代接触真值。

## 穿透原因及证据边界

- 大地板的 MJWarp 异常案例出现远离机器人约百米的接触位置。
  EPA witness 重建出现约 +6358/-6358 的外插系数；法向正常并不保证位置正常。
  缩小地板只能绕开触发条件，不作为根本修复。
- 另一失败快照中，脚在当前控制步开始前已经深入地面约 13.1 cm。
  对应源参考帧的脚约在地面上方 33.1 cm，不能把这 13 cm 归因于源帧本身。
  同时机器人根部比参考低约 24 cm。没有前 29 步完整历史，无法证明最初触发链。
- 源轨迹配当前权威碰撞形状确有毫米到厘米级穿透。step 的箱体穿透最大约
  28.7 mm；原始骨架与当前足部形状组合也有约 20 mm 重叠。
  这不是 CSV 精度或箱体缩放导致，也不足以断言原生成器使用相同碰撞形状。
  源轨迹未被直接修改。
- 冻结策略、相同随机种子的三组诊断中，每 20 ms 生成接触时最大足部地面穿透
  约为 64.72/43.51/47.69 mm；每 2 ms 约为 6.43/2.69/2.11 mm。
  这是重建条件下的配对实验，不是原 13 cm 失败的精确历史重放。

## 整机与训练进展

12 条动作的有限批次诊断累计 8400 个 world-control steps，无记录到的
非有限解算或非法接触法向。每控制步最后子步采样的最大激活穿透约 12 mm，
不代表所有子步的峰值。旧策略未通过动作完成测试。

随后 4096 训练环境加 36 个 probe 环境，从旧 1k 权重显式迁移到 v2，
使用全新采样重新拟合 WM 至 1.5k。日志推进到 1.65k；保存 checkpoint 为 1.5k。
停止原因是共享 GPU 资源，停止前未记录到接触解算失败。
这段短跑不证明长期稳定或所有动作都已学会。

## 统一 2 ms 单脚对照

相同权威足部凸包、17.5266 kg 载荷、0.6×0.6×0.1 m 地板，
从上方 5 cm 落下；每 2 ms 重建接触并采样。
共同 ShapeConfig 为 ke=1e7、kd=1e4、mu=0.7，开启 multi-contact。
Hydroelastic 的 kh=1e11，目标 SDF 分辨率 2 mm。
这些是统一单脚参数，不是整机训练 solref/solimp 参数复现。

| 指标 | MJWarp 内部接触（法向修复） | Newton 普通接触 | Hydroelastic |
|---|---:|---:|---:|
| 落脚峰值穿透 mm | 2.809 | 0.031 | 1.501 |
| 稳态承重点数 | 4 | 12–15 | 41 |
| 点均值额外跳变上限 mm | <0.001 | 10.742 | 0.581 |
| 受力中心额外跳变上限 mm | <0.001 | 0.014 | 0.002 |
| 非有限状态、反向法向、分配报错 | 无 | 无 | 无 |

跳变指标来自固定深度下每次平移 0.1 mm、扣除该已知位移后的变化。
完整数值及实际 solver 参数见 [threeway-2ms.json](threeway-2ms.json)。
Hydroelastic 有更多承重点，但本实验不支持它全面优于其他方法。
小平面结果不能排除原大场景故障，也不是长期训练保证。

**更正：**早期单脚脚本只重设 body pose，未同步 joint state，随后 reset 又覆盖
关节状态。旧静态扫描和滑动结果撤回。上述表格只使用同步修复后的重跑结果。
新脚本显式设置 joint_q/joint_qd、eval_fk，并以 flags=0 清空求解器历史而不覆盖姿态。

## 位置 loss 的含义

当前单段默认 `surface_region`：对固定碰撞表面节点分别计算距离代价，
然后按面积权重做标量 log-sum-exp 平滑最小值，不平均接触点坐标。
实测 direct reader 有目标时选最近的真实接触点，无目标时选最深点。
旧 classified 采集路径仍有均值汇总，不能与当前训练路径混为一谈。
点均值跳变不是当前位置 loss 跳变的直接证据。

稀疏表面计算只变换参与事件的部位，保持 loss 和梯度；CPU 合成测试并非
所有配置都加速，不能推断 GPU 训练加速。结果见 [cpu-optimization.json](cpu-optimization.json)。
评估 watcher 首个待评估 checkpoint 出现前不初始化仿真，首次评估后仍保留环境；
它不是空闲 GPU 调度器。

## 复现

先从 canonical `surface_v2` 实际求解器导出单世界 `solver-collision.mjb`
及配套 `.mjb.json`（`NativeContacts.export_collision_model`）。
脚本验证权威 URDF/USD 指纹，不下载或替换机器人模型。

```bash
python packages/supermimic/tools/probe_contact_backends.py \
  --collision_model /path/to/solver-collision.mjb \
  --mode mjwarp --dt .002 --steps 600 --sample_every 1 \
  --kh 1e11 --voxel .002 --output tmp/contact-comparison/mjwarp
```

另两组只改变 `--mode hard` / `--mode hydro` 和输出目录。
探针不训练策略，但使用 GPU；仅在获分配的空闲 GPU 上运行。
原始轨迹、大型 checkpoint 和本地故障快照未包含在本次代码发布中。
