---
title: "PTRN 并行训练与验证流水线"
status: current
owner_role: "PTRN research maintainer"
created: 2026-09-25
updated: 2026-09-25
review_cycle: "event-driven"
source_of_truth: false
supersedes: null
---

# PTRN pipeline v1

AutoDL 环境准备、显卡调优经验与“完成后关机，再用无卡模式下载”的操作流程见 [AutoDL 与 GPU 训练实践](autodl-training-practice.md)。

目标是缩短训练加完整验证的总耗时，提前供数，减少 GPU 等待 CPU。实现位于独立的 `ptrn_pipeline` 包，原 `ptrn_hip` 模型、数据配方、损失和验收规则不变。此文描述运行机制；实测结果保存在本地研究目录，不代表模型精度通过。

## 调度

`runner` 在 CPU 工作池预合成 32 个 batch 后启动训练，后续按绝对样本游标持续准备。按训练轮次生成确定性排列，恢复时只读取检查点中已消费的游标，丢弃的预取不改变样本次序。训练实 batch、有效 batch 都为 64；主进程最多 4 线程，准备数据使用 12 个单线程进程；验证活动期间为准备 8 个、指标 4 个，工作池在检查点边界调整。

`DoubleBuffer` 在独立传输流上提前把固定页内存中的下一批送到 GPU。CUDA 事件保护主存生命周期，消费流等待对应事件并登记 GPU 张量使用。队列有界，两个固定页缓冲的合计上限为 1 GiB。

验证使用冻结检查点的单独模型及提前装入 RAM 的开发集输入。GPU 在训练更新之间执行少量验证预测，CPU 指标进程独立读取预测分片。模型仍保持 batch 1；更大的 batch 没有通过既定数值检查。CUDA Graph 只捕获可兼容的编码器和空间解码过程，比例计算及原有有限值检查继续执行。单个 GPU 所有者调度训练和验证，独占命令使用 GPU 锁，避免不受控地启动多个计算进程。

CPU 指标异常会生成 `metrics-failed.json`，不会终止健康训练。预测进程的 PID 与启动标识写入绑定，进程被杀且没有完整结束标记时，指标进程会报告失败，避免永久等待。只有样本编号恰好覆盖 `0..5735`、没有重复、所有指标完成，才生成完整开发集结果。小集合诊断会显式标为非完整验证。

## 缓存和版本

固定开发集拆成轻量网络输入和完整几何/掩码信息，合计不超过 12 GiB；训练所有缓存版本合计不超过 8 GiB；总缓存上限 20 GiB，磁盘至少保留 15 GiB。校验和、分片原子替换和配额索引保护缓存写入；损坏项可重新生成。清理范围仅限缓存分片，不能删除原片、检查点或指标。RAM 输入库上限 8 GiB，进程组内存超过 48 GiB 时，生产队列暂停补充。

旧版 CPU 渲染保留原有 uint8 像素、浮点标签和分数掩码，不增加 JPEG 压缩，不量化标签。缓存键包含数据指纹、样本编号、轮次、输入尺寸、渲染版本和代码 SHA。

`torch-photo-v1` 是单独的 GPU 渲染版本：CPU 解码、确定场景参数、求解缺损几何和执行 JPEG；GPU 做透视采样、背景混合、光照、噪声与模糊；事件保护的异步回传交给后台写盘线程，下一批几何规划也可提前执行。原片占幅、缺损分层和训练/开发隔离仍须审计。它不填补缺失的医学影像内容，也不能中途替换固定开发集。

GPU 渲染不是默认启用项。必须在包含规划、传输、压缩、读取及训练的完整流程中达到至少 10% 收益，并通过数据审计，才可用于后续训练；仅 GPU 核函数更快不算通过。

## 兼容性与运行示例

输入仍为等比例补边 RGB；输出仍为 V3 有序四角、比例与辅助掩码，检查点保留原契约和原绑定。运行时配置、精度模式、流水线源文件 SHA 另存于运行目录。严格 FP32 与旧 TF32 的参考结果分开比较，不能混用。

以下变量均由本地部署环境设置，不把数据地址、密钥或映射写入仓库：

```bash
python -m ptrn_pipeline.evaluation prepare \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" \
  --cache "$CACHE" --output "$RUN/dev-cache" --workers 12

python -m ptrn_pipeline.runner \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --resume "$CHECKPOINT" --output "$RUN/training" --steps 5000 \
  --gc-interval 4 --eval-batch 1 --eval-current

python -m ptrn_pipeline.evaluation predict \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --checkpoint "$CHECKPOINT" --output "$RUN/evaluation" \
  --batch 1 --workers 4 --graph
```

`gc-interval` 使用相同起点与数据的 500 步实测选择，候选为 1、4、8。最后 100 步显存分配波动超过 128 MiB、峰值超过 22 GiB 或出现 OOM 都不能通过；不合格时保留 1。性能诊断的更新只保留在 scratch 目录，不替换正式权重。

## 独立监控

服务器 `telemetry` 每 10 秒采集进程身份、训练与各阶段进度、CPU 配额利用率、GPU、内存和磁盘。告警事件和去重状态落盘；本地 Codex 断线不影响采样，重连后可补读事件。GPT‑6 Luna 使用一个固定任务每分钟读取，只报告新故障、恢复、关键完成或需要人工操作的事项；不会修改训练或自动恢复，也不会为每次检查创建新任务。

训练结束和评估结束分别记录。全部实施与验收完成前，不能只因训练到 5000 步就标记整个流水线任务完成。

参考：[PyTorch 固定页内存与异步传输](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)、[CUDA Graph 约束](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

## 本轮已选配置

[服务器配置](../configs/pipeline-v1.json)记录本轮 4090 D 的参数。GC 1、4、8 各跑 500 步：每步回收约 0.411 秒/步，每 4 步回收约 0.300 秒/步；每 8 步略快，但最后 100 步的显存分配波动为 133.22 MiB，超过本轮采用的 128 MiB 上限，因此选择 4。

固定缓存供数的 80 步实测约 0.314 秒/步，取数等待占比约 0.0021%。这是热缓存结果，不能据此断言无限长度的冷缓存训练也没有供数等待。另有独立 `prefill` 命令可在 GPU 执行其他阶段时，提前准备下一个训练区间：

```bash
python -m ptrn_pipeline.prefill \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --output "$RUN/ahead-cache" --cursor 80000 --count 5120 --workers 12
```

原模型的 CUDA 反向计算不是逐位确定性的；原代码自身的 A/A 对照也有差异。CPU 确定性对照仍严格检查损失、参数、Adam 状态与恢复结果逐位一致；GPU 恢复时也严格核对加载瞬间的模型、优化器、RNG 和样本游标，不能把后续 GPU 轨迹的差异当作“状态没有恢复”。原有成图检查和数值容差未降低。

GPU 渲染已完成包含 916 张不同原片、1024 个视图的三轮配对实验。使用常驻 CPU 工作池后，CPU 在线供数加训练约为 10 秒，GPU 预生成、读取加训练约为 32～36 秒。因此本轮保持 CPU 提前合成和缓存，GPU 渲染仅作为独立版本的实验接口，不默认接入后续训练。
