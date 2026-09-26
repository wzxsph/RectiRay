# 训练与验证复现指南

本指南对应 `ptrn-step2000-unified-v1`：以旧 `step-2000.pt` 为父，从新数据游标训练最多 3000 个**新增**更新步。所需医疗影像和初始化权重不在仓库中。下列变量只在本机或获准服务器设置；不要把其实际值、清单或日志写入 Git。

## 1. 环境和资产

Python 3.11/3.12；`pyproject.toml` 是项目依赖声明，`requirements.lock` 是一次 Linux 环境快照。AutoDL 旧验证环境为 RTX 4090 D 24 GiB、16 核 CPU 配额、约 60 GiB RAM、Python 3.12、PyTorch CUDA 版与 Keras 3 的 torch 后端。重新开机后先核对实际驱动、PyTorch、BF16、CPU cgroup 和剩余空间，不把这组历史数值当作新设备保证。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
KERAS_BACKEND=torch .venv/bin/python -c 'import torch, keras; print(torch.__version__, keras.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)'
CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=torch .venv/bin/python -m pytest -q
```

准备获准使用的本地 `BUNDLE`：`data/manifest.json`、`data/split.json`、`data/backgrounds-manifest.json`、清单引用的只读源片，以及 `weights/b11.weights.h5` 和 `weights/g0.pt`。`BACKGROUNDS` 是背景清单相对路径指向的素材根目录。另需父 `step-2000.pt`、与其一起保存的 `frozen-protocol.json` 和本地缓存目录。母片划分必须为训练 4114、开发 478、隔离 28；开发集每片 12 个固定场景，共 5736。所有资产均先校验 SHA 和使用权限；图像、路径映射与背景不进入普通 Git。

## 2. 新分支训练与恢复

运行前令 `BUNDLE`、`BACKGROUNDS`、`CACHE`、`PARENT`、`PARENT_PROTOCOL`、`RUN` 指向**本地**路径。`RUN` 必须是未使用的新目录。父权重预期 SHA256 为 `016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535`；不匹配会被拒绝。`--skip-training` 只建立并核对零步新分支，可作为 CPU 预检；应使用独立输出目录，不能再拿它当正式 run 的空目录。

```bash
KERAS_BACKEND=torch .venv/bin/python -m ptrn_pipeline.step2000_runner \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --parent "$PARENT" --parent-protocol "$PARENT_PROTOCOL" \
  --output "$RUN" --steps 3000 --gc-interval 4
```

首次运行创建 `branch-step-0.pt`、`branch-binding.json`、`branch-provenance.json` 和确定性 `sample-sequence`；每 250 步保存一个 `branch-step-N.pt`。读取的是**已经消费**的游标，预取数据不算训练步。中断后只从同一运行目录内、SHA 已核对的已保存检查点恢复；保持所有参数、代码、资产及样本序列相同，并增加 `--resume "$RUN/branch-step-N.pt"`。源码或资产变更时绑定校验会阻止错误续训；有审计迁移记录的旧实验例外不能当作普通恢复方法。

实际 batch 与有效 batch 均为 64、224 输入、冻结 BN、`stage=head`。CPU 合成提前供数；GPU 训练与检查点诊断预测交错，CPU 同步计算指标。训练中采用经过并发验证的 eager 诊断预测；要求既有数值口径的 batch 1 CUDA Graph 完整评估在输入线程退出后运行。较大预测 batch 曾超出既有一致性容差，不自行增大或放宽容差。`--eval-limit` 只用于短程诊断，正式完整运行应省略。

2026-09-26 的一次 +250 中断涉及 CUDA Graph 捕获与固定页内存线程并发，恢复审计见 [AutoDL 实践记录](autodl-training-practice.md)。需要复现该历史检查点时应使用其绑定的源码及经核验迁移版本；不要把一个故障 run 与另一个 run 的同名步数混用。

## 3. 固定开发集验证

训练 `completion.json` 将训练终态与开发评估终态分开；只有 `full_development_evaluated=true` 且 `evaluation_errors` 为空、相应 `evaluation.json` 覆盖全部 5736 个唯一编号，才算该检查点完整开发验证。代码明确把 `effectiveness_verified` 留为 `false`：完整计算不表示真实取图合格。当前开发集已参与研究和选模，不能称独立验证。

单独重算一个检查点时，使用相同 bundle、背景、缓存、输入尺寸、场景种子及数值口径，先核对检查点 SHA 与 `branch-binding.json` 的代码/资产映射；验证目录必须全新：

```bash
KERAS_BACKEND=torch .venv/bin/python -m ptrn_pipeline.evaluation prepare \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --output "$RUN/dev-prepare-separate" --workers 4
KERAS_BACKEND=torch .venv/bin/python -m ptrn_pipeline.evaluation predict \
  --bundle "$BUNDLE" --backgrounds "$BACKGROUNDS" --cache "$CACHE" \
  --checkpoint "$CHECKPOINT" --output "$RUN/eval-separate" \
  --batch 1 --workers 4 --graph
```

`predict` 让 CPU 指标子进程继续工作；命令返回不代表评估完成。待 `metrics-complete.json` 和 `evaluation.json` 均落盘，再确认 `full_development=true`、`count=5736`、编号 `0..5735` 无重漏、`binding.json` 的检查点 SHA、开发指纹与后端/TF32 设置正确。失败标记须报告，不通过漏样本或只保留成功样本来提高结果。`--strict-fp32` 是另一数值审计模式，不能与旧 TF32 参考混合称等价。

需要比较若干完整评估时，用聚合工具逐项校验样本覆盖、预测分片、场景对应和检查点 SHA；结果放在忽略目录，不输出逐例内容到 Git：

```bash
.venv/bin/python tools/summarize_development.py \
  --run parent="$PARENT_EVAL" --run plus2250="$BRANCH_2250_EVAL" \
  --output .local/reviews/development-summary.json
```

## 4. 实拍与交付判断

使用同一批获准的手机照片，对父 2000、+1250、+2000、+2250、+3000 等实际存在且已核 SHA 的检查点，固定读图、补边、取图与显示方式；逐图同时看原照片、四角叠加和最终输出。记录有效影像是否被切掉、背景、方向、比例、无法确认的边界。多留少量背景可接受，漏掉可见医学内容或不必要形变需单独计数。47 张实拍已参与开发，只能作为候选审阅；可靠的独立真实泛化和真实比例准确性需要新的、未参与开发且有适当参考的照片。

在本地有 CUDA 的机器上可用独立工具生成相同读图和取图口径的并排网页。默认会严格验证检查点中的全部源码 SHA。当前本地 +2250 对照使用训练后经审阅的 V3 面积重采样取图器，其 `src/ptrn_hip/partial_geometry.py` 与训练时 SHA 不同；此时需显式传入 `--allow-renderer-update`。该开关**只**允许这一文件与工具内固定的已审阅 SHA 匹配，其余训练源码仍须与检查点完全一致，并在本地 `report.local.json` 中逐权重记录训练时/当前 SHA。它只用于实拍渲染对照，不改变训练恢复或完整开发评估的严格绑定。**输出包含原照片副本**，所以 `--output` 必须在受控的忽略目录：

```bash
KERAS_BACKEND=torch .venv/bin/python tools/compare_real_photos.py \
  --bundle "$BUNDLE" --photos "$AUTHORIZED_PHOTOS" \
  --checkpoint plus2000="$BRANCH_2000" \
  --checkpoint plus2250="$BRANCH_2250" \
  --output .local/reviews/real-compare --allow-renderer-update
```

若把旧正式 pipeline 1250/2000/5000 权重也加入同页，还需显式加
`--allow-parent-source-updates`；它只接受旧训练的 `partial_data.py`、
`partial_synthesis.py` 与当前文件之间已核 SHA 的两项差异。精确关系见
[权重与源码对应表](checkpoint-source-map.md)。工具把原照片解码并按推理方向无损重编码为 PNG，
不把原文件的 EXIF 元数据复制到网页；逐例影像仍必须只留本地。

权重、逐例网页与图像只留受控位置。交付时记录代码版本、配置、父/分支权重 SHA、资产/场景指纹、完整评估结果、真实图审阅限制与 CPU 导出检查。若训练/验证都结束，AutoDL 关机及无卡模式下载按 [实践记录](autodl-training-practice.md) 操作，不能以空闲显存单独判断完成。
