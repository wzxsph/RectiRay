# 六例合成效果图：来源与审计

[English credits and audit](../CREDITS.md) · [中文总览图](overview.webp)

这六例是固定素材和种子的**研究示意图**，不是随机抽样的准确率评估。中文图直接从仓库已审阅的六张英文合成图复制三个影像区域，再重绘中文标题、列名、图例和状态；因此**照片、四角线和两张展平图与英文版逐像素相同**。本地化不重新推理，也不重新采样。原英文图使用相同的六张公开原片、合成几何和拍摄干扰；所有背景均由程序生成，不包含 COCO 照片或 47 张手机实拍。

原片只来自以下两个已标注许可的版本：

- Bassem Haddad 等，[*Acetabular-vision hip developmental dysplasia’s (AV-DDH), version 2*](https://data.mendeley.com/datasets/4gvcb6gmh2/2)，DOI [10.17632/4gvcb6gmh2.2](https://doi.org/10.17632/4gvcb6gmh2.2)。数据集页面标为 **CC BY 4.0**。
- Mohammad Fraiwan、Noran Al-Kofahi、Omar Hanatleh、Ali Ibnian，[*A dataset of DDH x-ray images, version 2*](https://data.mendeley.com/datasets/jf3pv98m9g/2)，DOI [10.17632/jf3pv98m9g.2](https://doi.org/10.17632/jf3pv98m9g.2)。数据集页面标为 **CC BY 4.0**。

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 要求注明来源和修改。这些图片在原片上叠加模拟拍摄场景、干扰、角点线和模型取图；AV-DDH 三例还在合成前遮盖了原图两处通用阅片器页脚文字。没有发布原始文件名、文件路径映射、原片、标注、患者信息或实拍资料。

| 案例 | 数据集；该版本图像目录按相对路径排序后的零基序号 | 固定种子；视图序号 | 场景与实际占幅 |
| --- | --- | --- | --- |
| [01 · 近满幅照片](01-near-full.png) | AV-DDH v2；1228 | 20260926；0 | 程序化阅片背景、实体胶片拍摄；占幅 92.39% |
| [02 · 轻度透视](02-mild-perspective.png) | Fraiwan v2；135 | 20260927；1 | 轻度投影、程序化混色背景；占幅 81.72% |
| [03 · 胶片缺角](03-missing-corner.png) | AV-DDH v2；2022 | 20260928；2 | 拍摄后的局部缺损，源平面约 6.05% 未入镜；占幅 65.54% |
| [04 · 近色融边](04-near-color.png) | Fraiwan v2；244 | 20260929；3 | 程序化近色背景及柔化边缘；占幅 55.78% |
| [05 · 屏幕干扰](05-screen.png) | AV-DDH v2；2346 | 20260930；4 | 程序化阅片背景、较强屏幕翻拍干扰；占幅 75.73% |
| [06 · 原片直通](06-original.png) | Fraiwan v2；276 | 20261001；5 | 外部确认来源可信的数字原片，RGB 像素原样保留；占幅 100% |

[README 双栏示意图](hero-missing-corner.png)（1100 × 670、RGB PNG；SHA256 `9f5c811893c64d97fe9e2f8e990a243a8888ee8fea8f8850c6a17b511b1f1477`）及[窄屏上下排列图](hero-missing-corner-mobile.png)（540 × 1240、RGB PNG；SHA256 `28bbe7ea747f17073a24856cc5cff53fa4c4e51850ce4d8bd9f25bd694189e79`）从[案例 03](03-missing-corner.png)逐像素复制输入及分支 +2250 的影像区域，只重绘标题、说明与版式。可用 [`tools/build_readme_hero.py`](../../../../tools/build_readme_hero.py)在不读取私有数据、不执行模型推理的情况下重建；脚本校验已审阅源图的 SHA 和导出元数据。棋盘格表示原照片未覆盖的像素，不表示补造解剖影像。英文版沿用相同的影像像素。上述 PNG 哈希标识仓库内的成品；字体或 Pillow 版本变化可能改变文字光栅化和文件字节，但不会改变复制的影像区域。

输入列的**白线**为已知参考四角，**青线**为父模型预测，**橙线**为分支模型预测。01–05 的两个结果列由同一合成照片展平得到；透明棋盘格表示照片没有覆盖的区域，**未拍到的影像内容不会被补造**。06 的预测四角只用于诊断显示：原片直通来自事先可信的来源标记，程序已经核验输出与输入逐像素相同；模型本身并没有判断出“这是原片”。

比较使用父权重 `step-2000.pt`（SHA256 `016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535`）和分支权重 `branch-step-2250.pt`（SHA256 `35f0101260b19ee1484022de3a4ed40fd3ce2f4bdb0ef73f379c4fd458a53b3b`）。两者都经过源码与资产绑定检查，并使用同一经审阅的 V3 面积重采样渲染器；它与训练时渲染器有差别，见[权重与源码对应表](../../../checkpoint-source-map.md)。+2250 是从父 2000 步重新计数的分支步数。

六张中文 PNG（1700 × 720）和[中文总览图](overview.webp)（850 × 2198、无损 WebP）由[同一生成器](../../../../tools/build_public_examples.py)的本地化路径生成。它逐个核对英文 PNG 的固定 SHA，复制已发布的影像像素，并使用英文图上已显示的三位小数比例。生成器默认仍执行原英文生成流程；中文使用 `--locale zh-CN`，只需 Pillow、仓库内英文图和 Noto Sans CJK SC 字体，不需要原始数据、权重或 GPU。必要时用 `--zh-font` 和 `--zh-font-index` 指定字体：

    python tools/build_public_examples.py --locale zh-CN \
      --reference-dir docs/figures/examples \
      --output "$EMPTY_CHINESE_OUTPUT_DIRECTORY"

| 文件 | SHA256 |
| --- | --- |
| `01-near-full.png` | `d2e65e5b400631a6d844416e498fa5395f7771be7df54af5455c599a043a44c4` |
| `02-mild-perspective.png` | `31198efb2334005017f3a6c5b6537c343b590c46b0048dcb8f9d30eb6f562b6a` |
| `03-missing-corner.png` | `1bb0d6faa354bf4c81a42784817e9486e35c7782569f3f1282b0c77cb6a2c152` |
| `04-near-color.png` | `157695db5de34ecbb8d66979d4bac7c0fa4de44832714def5b25c913b7270605` |
| `05-screen.png` | `01a4c822c05bf5eff14780c1aacba566bc098877f5367e8e31e26cd2d0048757` |
| `06-original.png` | `c9a035584f5133f3651f9dafd95338d35958fb998eaf3ee4662648370e1e58c2` |
| `overview.webp` | `a5ef10ee0acef1f357ceb3d91707171d6f4b9e06cd6438c4fcaee9f73d39984c` |

上述 WebP **文件字节**哈希由 Pillow 10.2.0 / libwebp 1.3.2 生成。Pillow 12.2.0 / libwebp 1.6.0 会将完全相同的总览图 RGB 像素编码成不同的 WebP 字节；两版解码后的 RGB 像素 SHA256 均为 `a9f0a207c99b813a1682c59f5b6b998a35ec8bb6f0bea68f6ca4a98698407812`。六张 PNG 在上述两个本地环境中逐字节一致。

英文图制作时已逐张检查源片与输出图的可见文字；中文本地化后再次检查没有姓名、编号、日期、检查号或其他可识别个人信息。所选源文件没有 EXIF，六张中文 RGB PNG 的元数据字典为空；WebP 只有一段 VP8L 图像数据，没有 EXIF、XMP、ICC、动画或文本块。中文与已发布英文图的三个影像展示区域已逐像素核对一致。可见的 “L” 是左右侧标记。原始路径及本地审阅表均不进入 Git。

这些例子可能与研究训练或已暴露的开发素材重合，不能据此推断独立实拍表现，也不构成临床适用性验证。

[100 场景对照总览](contact-sheet-100.png)复用同六张已审计公开源片；[固定配方、GPU 推理审计与哈希](../../README.md#100-scene-recipe-and-audit--100-场景配方与审计)单独记录。没有引入新的源片或私人实拍。
