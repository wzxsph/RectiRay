# 六例合成效果图：来源与审计

[English credits and audit](../CREDITS.md) · [中文总览图](overview.webp)

这六例是固定素材和种子的**研究示意图**，不是随机抽样的准确率评估。中文图直接从仓库已审阅的六张英文合成图复制三个影像区域，再重绘中文标题、列名、图例和状态；因此**照片、四角线和两张展平图与英文版逐像素相同**。本地化不重新推理，也不重新采样。原英文图使用相同的六张公开原片、合成几何和拍摄干扰；所有背景均由程序生成，不包含 COCO 照片或 47 张手机实拍。

原片只来自以下两个已标注许可的版本：

- Bassem Haddad 等，[*Acetabular-vision hip developmental dysplasia’s (AV-DDH), version 2*](https://data.mendeley.com/datasets/4gvcb6gmh2/2)，DOI [10.17632/4gvcb6gmh2.2](https://doi.org/10.17632/4gvcb6gmh2.2)。数据集页面标为 **CC BY 4.0**。
- Mohammad Fraiwan、Noran Al-Kofahi、Omar Hanatleh、Ali Ibnian，[*A dataset of DDH x-ray images, version 2*](https://data.mendeley.com/datasets/jf3pv98m9g/2)，DOI [10.17632/jf3pv98m9g.2](https://doi.org/10.17632/jf3pv98m9g.2)。数据集页面标为 **CC BY 4.0**。

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 要求注明来源和修改。这些图片在原片上叠加模拟拍摄场景、干扰、角点线和模型取图；AV-DDH 三例还在合成前遮盖了原图两处通用阅片器页脚文字。

维护者于 2026-10-17 明确授权保留这六例已审查公开夹具及其溯源。下表的数据集版本和排序序号可以定位公开数据集中的源文件名，**不代表匿名化**。仓库文件不写入原始文件名、私有病例映射、原始数据集标注或人口学字段；47 张私人手机实拍、其源文件映射及患者／分组信息仍不公开。该授权只覆盖这六例已审查公开夹具，不扩展到其他素材。

| 案例 | 数据集；该版本图像目录按相对路径排序后的零基序号 | 固定种子；视图序号 | 场景与实际占幅 |
| --- | --- | --- | --- |
| [01 · 近满幅照片](01-near-full.png) | AV-DDH v2；1228 | 20260926；0 | 程序化阅片背景、实体胶片拍摄；占幅 92.39% |
| [02 · 轻度透视](02-mild-perspective.png) | Fraiwan v2；135 | 20260927；1 | 轻度投影、程序化混色背景；占幅 81.72% |
| [03 · 胶片缺角](03-missing-corner.png) | AV-DDH v2；2022 | 20260928；2 | 拍摄后的局部缺损，源平面约 6.05% 未入镜；占幅 65.54% |
| [04 · 近色融边](04-near-color.png) | Fraiwan v2；244 | 20260929；3 | 程序化近色背景及柔化边缘；占幅 55.78% |
| [05 · 屏幕干扰](05-screen.png) | AV-DDH v2；2346 | 20260930；4 | 程序化阅片背景、较强屏幕翻拍干扰；占幅 75.73% |
| [06 · 原片直通](06-original.png) | Fraiwan v2；276 | 20261001；5 | 外部确认来源可信的数字原片，RGB 像素原样保留；占幅 100% |

[README 双栏示意图](hero-missing-corner.png)及[窄屏上下排列图](hero-missing-corner-mobile.png)统一展示“输入 / RectiRay 输出”。输入用案例 03 的同一公开源片、种子、几何、裁剪与拍摄干扰重新合成，**不绘制角点叠加线**，并沿用既有 AV 页脚遮盖。输出直接逐像素复制案例 03 的既有结果，未重新推理、选择或编辑；历史三栏对照图保持原样。

[共享干净输入](../hero-input.png)为 492 × 465 RGB PNG，无元数据，SHA256 为 `1aaf54bdbf9b1a46f3e11957d6d7285a2fbd75a5177d4932ac06a6b8bab82476`，解码像素 SHA256 为 `b64c6c6b078f574c1e568b59d69f933ae69afe4035be925c8d1e040f6e97bbe9`；来源署名与合成修改沿用案例 03。新输入及四份主图导出均已检查，没有个人识别文字或嵌入元数据；“L”为左右侧标记。

[`tools/build_readme_hero.py`](../../../../tools/build_readme_hero.py)默认只读仓库已审计公开图片重建版式。加 `--av-root "$AV_V2_RAW"` 并设置 `KERAS_BACKEND=torch PYTHONPATH=src` 可重新合成输入，不加载模型或使用 GPU。脚本校验输入像素 SHA、既有输出源图 SHA 和复制区域。棋盘格也用于排版留白；输出图内的无效区域表示原照片没有覆盖的像素，不表示补造影像。英文与中文使用相同影像像素。字体或 Pillow 版本变化可能影响文字栅格化和文件字节。

| 主图 | 尺寸 | SHA256 |
| --- | --- | --- |
| 中文双栏 | 1100 × 670 | `0b8cd3907542b4ccd47f3ca586514ca121a210e6410782d6be3837dafcbd1bf7` |
| 中文窄屏 | 540 × 1240 | `783ca73338a2f2aba96b5c582253f09eee3b00568279f26fa7894f42d7b588f3` |

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
