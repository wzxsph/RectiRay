# PTRN Hip

用现有数字髋部X光自动合成翻拍图片和几何标签，微调已有PTRN，再在真实手机照片上比较展平效果。本仓库独立于标注工具，首轮只处理几何展平，不训练去摩尔纹网络。

## 当前：B1.1训练与原片保护

首轮B1和B1.1试验均已完成。B1.1使用30%正常原片、60%轻微翻拍、10%较大变形，保持全部底片及原划分；本机batch=16完成5000步，用时约28.2分钟，峰值预留显存3806MiB，Luna全程监控。**按用户明确的展平与视角优先目标，B1.1作为后续手机实验的优先基线，B1保留对照；背景杂乱不扣分。** 原联合数值门槛未全部通过这一事实保留，但撤回此前以软件边缘增多否定整体改善的评价。详情见[B1.1结果](docs/second-run.md)。

已实现数字原片在读图转换和推理前按字节保留；来源未知默认保留，明确手机照片才生成展平候选。三组固定开发验证分别报告，数字直通与回归器的恒等能力分开验收。478张真实数字原片的直通SHA-256全部一致；这些真实开发片均为8位，高位深保留仅有合成测试覆盖。

完整实施方案统一维护在`hip-xray-ai`仓库的`docs/plans/2026-09-21-ptrn-identity-mild-training-plan.md`。B1.1配置见[hip-v2.yaml](configs/hip-v2.yaml)，运行方式见下节；B1结果见[首轮记录](docs/first-run.md)。

## B1.1运行与来源选择

以下命令使用已准备的数据和B1权重。复现实验时复制配置并修改`run`为新目录，已有训练记录与图片不会被覆盖。训练会在1000步自动检查数字组至少改善一半且轻微组不退步；失败则停止并记录原因。只有同时满足数字、轻微、较大变形三组目标才保存`best.weights.h5`，否则保留明确标为未合格的`diagnostic.weights.h5`用于分析。

```bash
.venv/bin/ptrn-hip --config configs/hip-v2.yaml preview
.venv/bin/ptrn-hip --config configs/hip-v2.yaml audit-digital
.venv/bin/ptrn-hip --config configs/hip-v2.yaml benchmark --steps 100 --batch-size 8
.venv/bin/ptrn-hip --config configs/hip-v2.yaml benchmark --steps 100 --batch-size 16
.venv/bin/ptrn-hip --config configs/hip-v2.yaml train
.venv/bin/ptrn-hip --config configs/hip-v2.yaml evaluate --checkpoint .local/runs/b1-1/diagnostic.weights.h5
.venv/bin/ptrn-hip --config configs/hip-v2.yaml report-digital
.venv/bin/ptrn-hip --config configs/hip-v2.yaml compare \
  --inputs /path/to/anonymous-phone-input \
  --checkpoint .local/runs/b1-1/diagnostic.weights.h5 \
  --output .local/runs/b1-1/eval47
```

本轮没有通过原定全部联合门槛的权重，上述命令因此明确使用第4000步诊断候选；当前推荐其用于手机实验，不改名冒充原协议下的最优合格模型。以上命令仍复现原协议，本次未修改训练门槛；后续按几何目标的新选模方案需另行固化。三组各478张均用于开发选权重，不是独立患者测试；47张实拍分为IMG 20张和weixin 27张，仅用于现有匿名批次的评审，分组不参与来源判断。

```bash
# 数字原片：不加载模型，不解码或转8位；按字节复制到新文件。
.venv/bin/ptrn-hip restore --source-kind digital \
  --input /path/to/digital.png --output /path/to/new-copy.png
# 来源未知：默认保留，也可不提供output而返回原文件摘要。
.venv/bin/ptrn-hip restore --input /path/to/image.png
# 明确手机照片：从原分辨率生成PNG候选，不覆盖输入。
.venv/bin/ptrn-hip restore --source-kind phone \
  --input /path/to/phone.jpg --output /path/to/new-candidate.png \
  --checkpoint /path/to/candidate.weights.h5
```

直通保留原文件的位深和元数据，不代表新增DICOM解析或显示能力。图像模型与真实对照仍只处理几何，不应用ESDNet。数字原片直通文件和用于浏览的RGB预览必须区分。

## 首轮复现步骤

1. 只读导入全部数字片，约90%训练、10%开发验证；不做复杂去重或人工挑选。
2. 查看20张“原片 → 合成照片 → 自动四角 → 真值还原”示例。
3. 测100步速度和显存，从官方权重微调500步输出层、2500步最后一个DenseNet块。
4. 在同一批真实照片上生成“原照片 / B0官方权重 / B1髋部权重”对照。

现有照片已经用于开发，不能用它们声称独立测试成功率。代码不会写回原图，也不会上传数据。

## 环境和运行

Linux、Python 3.11、可用NVIDIA驱动。使用Keras 3的PyTorch后端，严格加载原PTRN的H5权重。第一次安装GPU运行库可能下载数GB；本机已从现有完整安装复用到独立环境，未修改原环境，也没有跨环境路径注入。以下命令在本仓库根目录运行。

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
mkdir -p .local/weights .local/backgrounds
```

将作者公开的演示权重放在 `.local/weights/ptrn_best_iou.h5`；下载入口、校验值与许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。将自然背景JPEG放在 `.local/backgrounds/`。本次使用COCO128的128张背景，仅本地使用。

```bash
.venv/bin/ptrn-hip prepare \
  --source mtddh-part1=/path/to/mtddh_part1 \
  --source mtddh-part2=/path/to/mtddh_part2 \
  --source av-ddh-v2=/path/to/av-ddh-v2 \
  --source ddh-xray-354-v2=/path/to/ddh-xray-354-v2
.venv/bin/ptrn-hip preview
.venv/bin/ptrn-hip benchmark --steps 100
.venv/bin/ptrn-hip train
.venv/bin/ptrn-hip evaluate --checkpoint .local/runs/b1/best.weights.h5
.venv/bin/ptrn-hip compare \
  --inputs /path/to/anonymous-phone-input \
  --checkpoint .local/runs/b1/best.weights.h5
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 --directory .local
```

照片入口约定为 `p001.jpg`、`p002.jpg` 等匿名本地副本。打开 `http://127.0.0.1:8768/preview/` 检查合成图，打开 `/eval47/` 查看真实照片。配置在 [configs/hip.yaml](configs/hip.yaml)。新实验修改 `run`，已存在的训练记录不会被覆盖。

```bash
.venv/bin/python -m pytest -q
```

本次实际依赖版本保存在 [requirements.lock](requirements.lock)；它记录本机Linux/Python 3.11的完整环境。云端若使用预装PyTorch，先检查CUDA驱动与PyTorch是否可用，再按 `pyproject.toml` 安装本项目；不要为了版本号机械替换服务器上已正常工作的驱动。

## 模型及自动标签

```text
RGB 224×224 → DenseNet201 → 全局平均池化1920维 → 线性Dense(8)
    → 补上h33=1 → 3×3矩阵H → 原分辨率照片上的四角 → OpenCV展平
```

H使用 `[-1,1]` 归一化坐标，方向是“数字原片 → 合成照片”。训练时先采样H，再按H渲染照片，所以标签直接来自程序，不需要手工点角。四角顺序为左上、右上、右下、左下。像素归一化遵循上游的 `2*x/width-1`，不能混用width−1。

合成器保留PTRN思路，由本项目独立实现：±45°旋转、缩放、错切、平移、透视、自然背景，30%概率屏幕黑框，再加亮度、颜色、噪声、模糊和JPEG扰动。标签对应原片整体，包括原片已有黑边；不对应额外屏幕框。黑边很大的源片也按用户要求保留。

这不是逐行复刻官方增强分布，也没有真实屏幕工具栏、反光和摩尔纹物理仿真。原片缩放到224×224是对官方输入约定的沿用；输出长宽由预测四边形估计，不保证原始物理比例。模型不生成骨骼像素，不负责去摩尔纹。

## 训练与评估约定

- 全部原片进入数据池；完全相同的224像素内容只用于同分区分配，副本仍保留。没有跨库感知去重。
- 所有训练原片每轮均参与，在线生成不同扰动。患者映射目前不可用；分区只能视为原片级开发划分。
- 首轮batch=8、混合精度、冻结BatchNorm，先训练回归头，再解冻最后一个DenseNet块；切阶段重新创建Adam优化器。
- 固定256张开发原片及合成种子，按有序角点误差选择最佳权重。误差除以图片对角线；IoU辅助观察，不能单独识别180°倒置。
- 训练后用 `evaluate` 在全部开发原片上做FP32比较；其中包含选权重的256张，仍是开发评估，不是独立测试。
- B0/B1真实照片使用同样的方向规则和几何检查，无逐张人工纠正。数学上有效的四边形不代表裁得正确。
- `best.weights.h5`是微调候选中最优；若仍不如B0，应保留B0，不自动替换现有流程。

## 本地文件边界

所有图片、缓存、源文件名映射、权重、日志和逐例结果都在Git忽略的 `.local/`；环境在 `.venv/`。原片路径仅写本地清单。不要将 `.local/` 全量上传到云端或GitHub。租机时可只传经确认允许使用的匿名224像素缓存及必要权重/背景，患者内容和授权范围仍需独立核对。

本仓库尚未公开远端；未来开源前单独核实上游权重和各数据来源许可，参见 [THIRD_PARTY.md](THIRD_PARTY.md)。租机选择见 [docs/resources.md](docs/resources.md)。

## 已完成的首轮

4620张全部入池，3000步训练和47张真实照片对照已完成，本机训练及阶段验证约13.5分钟。B1主体定位与方向改善，裁切边界仍需调整；完整结果与本地产物入口见 [首轮记录](docs/first-run.md)。
