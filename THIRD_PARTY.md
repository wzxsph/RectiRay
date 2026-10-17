# Third-party sources and licenses / 外部来源与许可记录

This repository licenses only the code and documentation it can authorize under [Apache-2.0](LICENSE). That license does **not** extend to source radiographs, annotations, COCO photographs, upstream code, initialization or fine-tuned weights, case images, or generated outputs. A private backup preserves bytes; it does not establish a right to redistribute them.

本仓库仅将有权授权的自写代码和文档按 [Apache-2.0](LICENSE) 提供。此许可**不覆盖**源 X 光片、标注、COCO 照片、上游源码、初始化及微调权重、病例图或生成结果。私有备份只是字节保全，不等于取得再分发权。

| Asset / 项目 | Source / 来源 | Use and boundary / 用途与边界 |
| --- | --- | --- |
| PTRN paper / 论文 | [Chong et al., arXiv:2210.05954](https://arxiv.org/abs/2210.05954) | Method and architecture reference; the original research addressed chest radiographs. / 方法与架构参考；原研究对象为胸片。 |
| Official implementation / 官方实现 | [maxium0526/ptrn](https://github.com/maxium0526/ptrn) | Used to check training, coordinates, and preprocessing; upstream source was not copied into this repository. The upstream repository's redistributable license has not been established. / 用于核对约定，未复制上游源码；尚未确定其再分发许可。 |
| Demonstration initialization / 演示初始化 | [Author README](https://github.com/maxium0526/ptrn#readme) and [Drive link](https://drive.google.com/uc?id=1WMT6nNrO0WAvOHjARQfG7o6LDiVr3UaD) | Local research initialization only. It is not established as the same training instance as the published result; redistribution permission remains unverified. / 仅作本地研究初始化；不等同论文结果的同一训练实例，尚未核实再分发许可。 |
| DenseNet201 and runtime / 架构与运行时 | [Keras DenseNet201](https://keras.io/api/applications/densenet/), [Keras 3 migration guide](https://keras.io/guides/migrating_to_keras_3/) | Reconstructed the architecture and strictly loaded the existing local parameters on the PyTorch backend; no separate ImageNet weight download is part of this repository. / 重建架构并严格加载已有本地参数；仓库不附带另行下载的 ImageNet 权重。 |
| [MTDDH](https://www.nature.com/articles/s41597-025-05146-x) source radiographs / 源片 | Local inventory: 1,849 files / 本地清单：1,849 个文件 | Original-version redistribution terms have **not** been directly verified. No MTDDH image is used in repository effect examples. / 尚未直接核实原始版本再分发条款；仓库效果图不用 MTDDH。 |
| AV-DDH v2 source radiographs / 源片 | [Mendeley Data v2](https://data.mendeley.com/datasets/4gvcb6gmh2/2), DOI: 10.17632/4gvcb6gmh2.2; 2,417 raw X-ray files stated on page | Dataset page states **CC BY 4.0**. Any illustrated source must also pass individual pixel-text and metadata review, carry attribution, and identify synthetic changes. / 页面标注 CC BY 4.0；入图素材仍须逐张检查像素文字与元数据、署名并说明合成改动。 |
| Fraiwan DDH X-ray v2 source radiographs / 源片 | [Mendeley Data v2](https://data.mendeley.com/datasets/jf3pv98m9g/2), DOI: 10.17632/jf3pv98m9g.2; 354 subjects stated on page | Dataset page states **CC BY 4.0**. The local 354-file count is an inventory count, not proof of independent patients. Same per-image review and attribution rule applies. / 页面标注 CC BY 4.0；本地 354 个文件的数量不证明患者独立，逐图审查及署名要求相同。 |
| Natural background pool / 自然背景池 | 4,400 local COCO 2017 photographs / 本地 4,400 张 COCO 2017 照片; [COCO Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) | Used for local synthesis only and **not included** in repository effect examples. COCO's CC BY 4.0 applies to annotations and website; photographs follow their respective Flickr terms. This is the current pool, replacing the earlier COCO128 description. / 仅用于本地合成，不进入仓库效果图；COCO 标注与网站的 CC BY 4.0 不覆盖照片，照片遵循各自 Flickr 条款。此为当前背景池，替代旧 COCO128 描述。 |
| Procedural interface pool / 程序界面池 | 1,100 locally generated viewer/interface images / 本地生成 1,100 张查看器／界面图 | Synthetic backgrounds for local training. Repository effect examples use programmatically generated backgrounds. / 供本地训练合成；仓库效果图背景由程序生成。 |
| Research weights / 研究权重 | Local training outputs and controlled private backups held outside this public source repository / 本地训练产物及本公开源码仓库之外的受控私有备份 | Verify SHA and source binding within authorized access; see the [checkpoint/source map](docs/checkpoint-source-map.md). Never part of ordinary Git, public Releases or the repository source license; private Pre-release is not a deployment registration. / 在授权范围内核对 SHA 与源码绑定，保存入口见[权重与源码对应表](docs/checkpoint-source-map.md)；不进入普通 Git、公开 Release 或源码许可，私有 Pre-release 不登记为部署模型。 |

The historical official H5 file used for initialization has SHA-256:

```text
fef1704284ad34056b3f83af49cfee8d7b7234d366016bb1d9cd9de8e54f4c4f
```

When an older serialized model cannot be deserialized by current Keras, the equivalent architecture is rebuilt and weights are loaded strictly. Shape mismatches are not ignored. The source inventory and research checkpoint relationships are summarized in the [README](README.md), its [Chinese version](i18n/README.zh-CN.md), and the [checkpoint/source map](docs/checkpoint-source-map.md). Public source-code availability does not grant redistribution rights for source images, backgrounds or weights; each requires its own rights review.

旧模型的序列化配置若无法由当前 Keras 完整反序列化，需按相同架构重建并严格加载权重，不能忽略形状不匹配。来源清单与研究权重对应关系见[英文 README](README.md)、[中文 README](i18n/README.zh-CN.md)及[权重与源码对应表](docs/checkpoint-source-map.md)。源码公开不授予源片、背景或权重的再分发权，它们仍需各自审查使用权。
