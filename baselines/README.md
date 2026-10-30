# Predictor baseline

当前唯一基线为[predictor.v1_latest.20261010](predictor_v1_latest_20261010/README.md)。
机器可读入口为[current.json](current.json)，配置及依赖也登记于项目资产和训练清单。
它同时指定scratch训练配方和固定step1000分析checkpoint；新训练仍从零初始化。

| 历史记录 | 封存方式 | 用途 |
| --- | --- | --- |
| [scratch_recipe.20261002](predictor_scratch_recipe_20261002/ARCHIVED.md) | 原地封存，状态archived | 原始配方、源代码和完整历史结果 |
| native_interval_scratch1100_20261007_r4 | 原运行目录保留，登记于archives.json | 只到550轮的同轮次历史对照 |
| [control500.20261009](predictor_control500_20261009/ARCHIVED.md) | 原地封存，状态archived | source550之后500次低学习率更新及架构消融证据 |

旧数据、权重、快照及实验结论均保留，未移动或删除。历史配置不会自动升级为新规则。
当前文档、训练默认配置及项目清单均指向新基线；历史记录仅供显式对照。
v2整体改版未被采用；其失败证据保留在control500归档，代码可显式选择做实验。
