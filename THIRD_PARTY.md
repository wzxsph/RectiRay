# 外部来源与许可记录

本项目有权授权的几何、合成、数据与训练编排代码独立编写，按仓库 [Apache-2.0](LICENSE) 许可；未将上游仓库源码复制进来。许可证仅覆盖这些自写源码和文档，**不授予**第三方代码、数据、背景图、演示权重、微调权重、病例资料或生成结果的使用或再分发权。

| 项目 | 来源 | 本次用途与边界 |
|---|---|---|
| PTRN论文 | [arXiv:2210.05954](https://arxiv.org/abs/2210.05954) | 方法与架构参考 |
| 官方代码 | [maxium0526/ptrn](https://github.com/maxium0526/ptrn) | 核对训练、坐标及预处理约定；截至2026-09-26页面文件列表未见许可证。本仓库不得将其源码默认为 Apache-2.0；公开前重新核对 |
| 演示权重 | [作者README](https://github.com/maxium0526/ptrn#readme)、[Drive下载](https://drive.google.com/uc?id=1WMT6nNrO0WAvOHjARQfG7o6LDiVr3UaD) | 仅本地加载与研究微调；不是论文报告的同一训练实例，未确认再分发许可 |
| DenseNet201 | [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201) | 架构重建，严格加载上述已有参数；没有重新下载ImageNet权重 |
| Keras 3 / PyTorch | [Keras多后端迁移说明](https://keras.io/guides/migrating_to_keras_3/) | 用同一Keras架构和H5权重，在已可用的PyTorch GPU后端训练；3张FP32预测与旧TensorFlow CPU最大差异7.2×10⁻⁷ |
| 自然背景 | [COCO128说明](https://docs.ultralytics.com/datasets/detect/coco128/)、[下载归档](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip) | 128张JPEG，仅本地合成；不附带分发图片，来源图片许可另行核对 |
| 数字髋部片 | 使用者获准使用的本地资料与公开数据目录 | 原片只读；不因为下载可用便推断可以再分发、训练商用或上传云端 |
| 研究权重 | 本仓库训练产物与主项目受控备份 | 仅在授权范围内保存与核对 SHA；不受本仓库源码许可证覆盖，不进入普通 Git |

当前官方H5 SHA-256：

```text
fef1704284ad34056b3f83af49cfee8d7b7234d366016bb1d9cd9de8e54f4c4f
```

旧模型的序列化配置不能直接被现代Keras完整反序列化时，按相同架构重建并严格 `load_weights()`。禁止忽略形状不匹配或静默跳过层。

私有代码备份、公开代码、共享权重和共享数据是不同决定。私有仓库访问权限不改变上述资产许可；未来若公开本仓库，仍须复核 Git 历史与第三方来源。权重、病例和背景不随代码公开。
