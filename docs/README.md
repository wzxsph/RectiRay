# 文档索引

先读仓库 [README](../README.md) 的当前状态。此目录保留各阶段的原始研究记录；旧文档里的“当前”“下一步”只代表写作当时，不自动覆盖后续检查点和验收结论。计划或讨论稿也不是运行结果。

| 类型 | 文档 | 用途 |
| --- | --- | --- |
| 当前入口 | [复现指南](reproducibility.md)、[代码地图](code-map.md)、[关键权重与源码对应表](checkpoint-source-map.md) | 环境、资产绑定、父 2000 → 新分支、完整开发验证与模块职责；权重在分阶段私有备份中 |
| 研究权重备份 | [空间头](../models/release-notes/backup-ptrn-spatial-v2-v1.0.0.md)、[完整流水线](../models/release-notes/backup-ptrn-pipeline-v1-v1.0.0.md)、[2000 步分支](../models/release-notes/backup-ptrn-step2000-branch-v1.0.0.md) | 三个私有 Pre-release 的来源、角色、限制与资产核验要求 |
| 当前实现协议 | [可见平面 V3](partial-v3.md)、[并行流水线](pipeline-v1.md) | 模型与几何接口、训练/评估调度；以实际源码与检查点绑定为准 |
| 运行经验 | [AutoDL 与 GPU 训练实践](autodl-training-practice.md) | 性能、并发问题、恢复审计和关机/无卡下载流程；其中的时间点状态为历史快照 |
| 本轮设计记录 | [从旧 2000 步分支的训练计划](plans/2026-09-25-ptrn-step2000-autodl-training-plan.md) | 新配方、3000 新增步、检查点与验收设计；最终状态须从运行产物核对 |
| 外观审样记录 | [拍摄干扰 v1](capture-appearance-v1.md) | 独立 100 张预览阶段的配方与技术核验；文内“尚未进入训练”是当时状态 |
| 讨论稿 | [过裁与下一轮数据分布](2026-09-25-overcrop-discussion.md) | 2026-09-25 对实拍内缩的假设与候选方案，不作已证因果 |
| 历史实验 | [空间 v2](spatial-v2.md)、[统一几何头](unified-v1.md)、[比例实验](ratio-v1.md)、[比例结果](ratio-v1-results.md) | 各自冻结协议、限制与结果；不能混用选权重门槛 |
| 早期记录 | [B1 首轮](first-run.md)、[B1.1](second-run.md)、[资源估算](resources.md) | 早期模型和硬件基线，仅供追溯 |

本仓库保存可公开审阅的协议、代码和**非逐例**汇总结论。固定开发集的 5736 个视图、47 张已暴露实拍、v1.1.0 混合来源标注图各有不同用途；它们都不是独立患者测试。病例图、逐例网页、原始清单、权重和运行目录留在受控位置，不在文档里写个人路径或源文件名。
