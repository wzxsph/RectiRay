---
title: "PTRN 关键权重与训练源码对应表"
status: current
owner_role: "PTRN research maintainer"
created: 2026-09-26
updated: 2026-09-26
review_cycle: event-driven
source_of_truth: true
supersedes: null
---

# 关键权重与训练源码

关键权重按研究阶段保存在本仓库的三个私有备份 Pre-release，普通 Git 不包含权重：

| 阶段 | 私有备份 |
|---|---|
| 空间头 1250／5000 | [spatial-v2](https://github.com/wzxsph/ptrn-hip/releases/tag/backup-ptrn-spatial-v2-v1.0.0) |
| 完整流水线 1250／2000／5000 | [pipeline-v1](https://github.com/wzxsph/ptrn-hip/releases/tag/backup-ptrn-pipeline-v1-v1.0.0) |
| 从父 2000 步出发的 +2000／+2250 | [step2000-branch](https://github.com/wzxsph/ptrn-hip/releases/tag/backup-ptrn-step2000-branch-v1.0.0) |

这些备份只供私有研究字节保全；原始副本也在主项目的
[私有备份 Pre-release](https://github.com/wzxsph/hip-xray-ai/releases/tag/backup-ptrn-workflows-20260926-v1.0.0)。
各 Release 的聚合 JSON 列出 ZIP 和内部权重的 SHA-256、训练源码绑定及角色，
`SHA256SUMS.txt` 校验上传资产。主项目 Release 的资产大小及摘要曾与其
`release-catalog.json` 核对。备份不表示模型通过独立实拍验收、获准公开分发或可用于应用部署。

| 研究阶段 | Release 内部权重 SHA-256 | 训练源码绑定 | 当前代码关系 |
|---|---|---|---|
| spatial head 1250 | `04a557a2a4b5dac3efa8d19d19d9f6298282af395c95bcf234d8b502b2e0b775` | `config.source_binding.sha256 = 4753aa89522ecd28e500ce07c6b8813984ae10132afa457038a78507616078c4`；18 个文件 | 当前 18 个文件的字节均匹配 |
| spatial head 5000 | `48f2d04f90a00e90114540cf83ed599e5505337e9b01420d04d36f91b22183bf` | 同上 | 过裁对照，不是当前候选 |
| pipeline 1250 | `3db6be2a46d7e89ccb5064b6757eebf4c198545dd4bbc353ec7f61b54d2ae153` | 48 个文件的代码映射摘要 `18873dba17e23078a6dc180da644e37d786b28e741a435e184d01418a69c6099` | 3 个训练时文件在[冻结覆盖层](../training-source-overlays/README.md) |
| pipeline 2000 | `016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535` | 同上 | 后续分支的起点 |
| pipeline 5000 | `b82b0dacd188149564c2ced08acf14743b5eb3547432489d3a125d9905f37d5d` | 同上 | 旧全程训练对照 |
| branch +2000 | `50904ca61f0b6994733b59a637e6600f4595f84e78eaac29be8ab58f3f93519b` | 76 个文件的代码映射摘要 `27ac0c6ad9d24a84e8a8e52821b89b67495946d8dcfad3ffc257199e5779859e` | 当前仅渲染器与训练时不同 |
| branch +2250 | `35f0101260b19ee1484022de3a4ed40fd3ce2f4bdb0ef73f379c4fd458a53b3b` | 同上 | 已暴露资料上的研究候选，不是自动部署选择 |

“代码映射摘要”按检查点中的 `binding.code` 以排序键、紧凑 JSON 编码后计算
SHA-256；具体每个文件的 SHA 保存在检查点本身。空间组使用其 `config.source_binding`
记录。原 pipeline 的 `binding.code` 以包内文件名为键；分支组改为仓库相对路径，
两者不能直接拼成同一目录解释。

原 pipeline 的 `partial_data.py`、`partial_synthesis.py` 后来为分支训练更新；
`partial_geometry.py` 后来为减少展平摩尔纹更新为面积重采样。原 pipeline 与分支组
训练时的渲染器 SHA 都是 `188b9e6bc8b3d423fee6432a5906c4df8402c334914c1c4b6559e87356d4cc9d`；
当前渲染器 SHA 为 `a23bb919dd5cea1437477b0cb6478247e6bf9735599962787d7b48e42ef92549`。
实拍网页工具默认要求精确源码绑定。使用当前取图渲染器时，必须显式添加
`--allow-renderer-update`；同时比较旧 pipeline 节点时还需
`--allow-parent-source-updates`，工具只允许上表对应的已核实字节偏差，逐权重报告原 SHA 和当前 SHA。
两套训练输出与取图渲染版本应分别记录，不能把更新后的网页称为当时训练环境逐像素重放。

旧 B1/B1.1 和官方演示初始化仅在上述主项目的旧私有备份 Release 中，
不包含在本仓库的三个分阶段 Release；角色与 SHA 见主项目备份的 JSON 清单。
官方演示权重的再分发依据尚未确认，因此不随未来公开的代码仓库传播。
