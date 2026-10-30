# 不训练启动检查

先执行 `python3 scripts/setup_workflow_environment.py`，再确认 `configs/workflow_resources.json` 中的解释器、IsaacLab 和外部 runtime 路径。本机的独立 venv 继承现有 SuperMimic 的 torch/CUDA/Newton 库，额外依赖由 `workflow-extras.lock.txt` 固定；不改基础环境。网页 TestClient 测试需要执行器允许本地 socket 通信。

所有命令都通过 `scripts/run_workflow.py`，`--` 后是原生参数。先用各阶段的 `--help` 验证解析与依赖导入；`--dry-run` 只打印命令，不能证明物理仿真可启动。

## Tracking

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_workflow.py tracking -- \
  --startup_check --num_envs 40 --headless --amp none \
  --task_manifest /perception/workspace/zhouxia/supermimic_vibed/data/lowclimb_boxes/task.json \
  --contact_model force6 --collision_profile surface_v2 --contact_response manifold_v2 \
  --log_dir /external/new-check/tracking
```

`--startup_check` 完成环境创建、策略前向和一步物理仿真后退出，检查 s/s1/action 的有限性。报告写入 `startup-check.json`，优化更新数为零，不保存模型。GPU 编号和新输出目录应按本机空闲资源调整；首次 CUDA/物理内核编译可能较慢。

## 采集、DAgger 与 contact

采集使用原生 `--steps 2` 做短物理/求解器接触读取检查，只产生诊断 rollout。它不证明演示成功，不能直接替代训练示范。旧 teacher 若迁移到 surface_v2，必须显式 `--allow_collision_transfer`，输出保留未验证迁移标记，不绕过 checkpoint 合约。

DAgger 使用匹配任务、物理设置和成功 recording，传入 `--startup_check --bc_steps 0`。它加载 teacher、创建学生和真实仿真，检查 teacher 对实际学生状态的动作输出，然后退出。`num_envs` 必须满足每个接触片段的原生环境分配，报告也不会保存初始化权重。

contact 使用匹配的 tracking teacher、recording、resume 和 prior，传入 `--profile_collection --freeze_actor --freeze_wm`。它建立正常 ContactTask、原生接触读取和碰撞表面目标，完成短采集后退出；程序断言 actor/WM 全部参数和缓冲区不变，不保存训练 checkpoint。`collection-profile.json` 与 `status.json` 记录实际执行情况。

这些检查覆盖进程启动、资源加载、GPU 物理、策略前向和原生接触读取。它们不证明训练收敛或动作成功率，也不重新认证历史权重。PPO 仍仅保留原 tracking，未添加 PPO→DAgger 适配器。

## 仓库迁移与实际验证范围

checkpoint 的机器人身份按完整资产指纹、碰撞配置及接触响应合约校验。仅 `urdf_path` 和 `usd_path` 可随仓库迁移而改变，资产内容或物理合约变化仍拒绝加载，不用旧路径继续导入旧仓库源码。

2026-10-30 的 GPU 检查使用 2 号卡。tracking 用当前 surface_v2 + manifold_v2，40 个环境执行一步物理；其余阶段用现存历史 teacher/recording/prior 严格匹配的 legacy 物理，分别执行两步原生采集、800 环境/19 片段 DAgger 初始化，以及 64 环境 contact 冻结采集。contact 的 policy_updates 和 wm_updates 均为 0，actor_frozen_unchanged 与 wm_frozen_unchanged 均为 true。没有生成新 checkpoint，四个输入文件的哈希保持不变。历史 prior 仅作启动样例，不因本检查成为新主流程的接受基线。
