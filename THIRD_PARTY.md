# 外部来源与许可记录

本项目自有几何、合成、数据与训练编排代码独立编写；未将上游仓库源码复制进来。

| 项目 | 来源 | 本次用途与边界 |
|---|---|---|
| PTRN论文 | [arXiv:2210.05954](https://arxiv.org/abs/2210.05954) | 方法与架构参考 |
| 官方代码 | [maxium0526/ptrn](https://github.com/maxium0526/ptrn) | 核对训练、坐标及预处理约定；2026-09-21核对根目录无LICENSE，GitHub许可字段为空 |
| 演示权重 | [作者README](https://github.com/maxium0526/ptrn#readme)、[Drive下载](https://drive.google.com/uc?id=1WMT6nNrO0WAvOHjARQfG7o6LDiVr3UaD) | 仅本地加载与研究微调；不是论文报告的同一训练实例，未确认再分发许可 |
| DenseNet201 | [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201) | 架构重建，严格加载上述已有参数；没有重新下载ImageNet权重 |
| Keras 3 / PyTorch | [Keras多后端迁移说明](https://keras.io/guides/migrating_to_keras_3/) | 用同一Keras架构和H5权重，在已可用的PyTorch GPU后端训练；3张FP32预测与旧TensorFlow CPU最大差异7.2×10⁻⁷ |
| 自然背景 | [COCO128说明](https://docs.ultralytics.com/datasets/detect/coco128/)、[下载归档](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip) | 128张JPEG，仅本地合成；不附带分发图片，来源图片许可另行核对 |
| 数字髋部片 | 使用者本地MTDDH与公开数据目录 | 原片只读；不因为下载可用便推断可以再分发、训练商用或上传云端 |

当前官方H5 SHA-256：

```text
fef1704284ad34056b3f83af49cfee8d7b7234d366016bb1d9cd9de8e54f4c4f
```

旧模型的序列化配置不能直接被现代Keras完整反序列化时，按相同架构重建并严格 `load_weights()`。禁止忽略形状不匹配或静默跳过层。

未来公开代码、公开权重及公开数据是三个不同的决定。本轮没有发布任何一项。
