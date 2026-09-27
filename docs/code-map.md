# 代码地图

阅读顺序：[README](../README.md) → [复现指南](reproducibility.md) → 对应实验协议。模型输出与训练绑定以源码为准；本地图像和检查点不随仓库提供。

| 位置 | 职责与入口 |
| --- | --- |
| `src/ptrn_hip/partial_model.py`、`partial_geometry.py` | 当前 V3 的共享空间解码、无边界裁钳的四角、完整参考平面比例、辅助可见掩码与原图一次采样。 `RawPartial` 为明确的导出契约 |
| `src/ptrn_hip/partial_data.py`、`partial_synthesis.py`、`capture_synthesis.py` | 4114/478/28 母片与背景隔离、确定性场景、旧/新配方、拍摄外观和可见几何；固定开发集仍使用 `partial-v3` 场景 |
| `src/ptrn_hip/partial_runner.py` | 旧 V3 正式训练入口和 `step-2000.pt` 的父协议；保留供来源核对，不能把新配方当成旧样本流续训 |
| `src/ptrn_pipeline/step2000_runner.py`、`branch_checkpoint.py`、`step2000_data.py` | 从父 2000 步创建零步新分支，绑定新数据、源码、资产与游标；保存每 250 步检查点、交错诊断及训练后的严格参考验证 |
| `src/ptrn_pipeline/evaluation.py`、`async_eval.py`、`graph.py` | 固定开发场景缓存、GPU 预测和 CPU 指标；完整结果须覆盖 5736 张，批量模式与 FP32/TF32 口径不能混称 |
| `src/ptrn_pipeline/cache.py`、`training.py`、`telemetry.py`、`resources.py` | 有界缓存、固定页内存/传输、运行进度与资源门槛；GPU 合成在 `gpu_render.py` 中保留为非默认实验路径 |
| `src/ptrn_hip/cli.py`、`src/ptrn_hip/{model,data,synthesis}.py` | 早期 H8/B1 系列接口与历史复现；不是当前 V3 分支的训练入口 |
| `configs/`、`tests/` | 历史与流水线配置，以及合成夹具上的单元/集成验证；测试通过不等于真实取图通过 |
| `tools/prepare_data.py`、`tools/data_assets.py` | 将用户已划分的源片与照片背景按字节复制为本地数据包；检查解码、SHA、重复及训练/开发泄漏，不自动切分或修改源文件 |
| `tools/check_assets.py` | 分别预检数据、历史起训、恢复与评估资产；核对权重、配套 JSON、协议、源码与样本序列，只读检查且不放宽原训练契约。新数据包不能直接用于固定历史训练入口 |
| `tools/compare_real_photos.py`、`tools/summarize_development.py` | 分别生成受控本地实拍取图对照网页、比较完整开发集的非逐例聚合；工具源码可提交，生成物不可提交 |

`src/ptrn_hip/*.py`、`src/ptrn_pipeline/*.py`、`pyproject.toml` 与 `configs/pipeline-v1.json` 的文件 SHA 会进入新分支绑定。给某个训练过的模块改名、重排代码或格式化，会改变绑定，即使算法看似相同。复现旧权重时使用其绑定对应的代码版本；实验改动应建立新版本与新运行目录。
