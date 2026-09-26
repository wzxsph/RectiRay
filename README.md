# PTRN Hip

**From a smartphone photograph of a hip X-ray to one straightened image of the film that was actually captured.**

[English](README.md) · [简体中文](i18n/README.zh-CN.md)

> **Research status:** Branch **+2250** is a candidate selected using previously viewed synthetic and real photographs. It has not passed independent real-photo or clinical validation, and it has not replaced the application model.

<picture>
  <source media="(max-width: 600px)" srcset="docs/figures/examples/hero-missing-corner-mobile.png">
  <img src="docs/figures/examples/hero-missing-corner.png" alt="Synthetic hip radiograph photograph beside the portion rectified by branch +2250; a stacked version appears on narrow screens.">
</picture>

*A fixed synthetic example made from an audited public-source radiograph. Part of the film is outside the photograph; the output preserves what was captured and does not invent the missing anatomy. [Source and image audit](docs/figures/examples/CREDITS.md).*

The pipeline has three jobs:

1. **Find the film.** Predict its four ordered corners and the width-to-height ratio of its full reference plane, even when a corner lies outside the photo.
2. **Straighten the captured part.** Map the visible portion into one canvas by sampling the original photograph once. Pixels that were never photographed remain invalid; they are not reconstructed.
3. **Leave trusted originals alone.** A digital original with verified provenance can be kept pixel for pixel. The network cannot infer that provenance merely because an image looks flat.

Choose a starting point:

- **See the result:** [six fixed synthetic examples](#six-synthetic-examples), including a difficult missing corner, a near-color edge, and an unchanged original.
- **Inspect the implementation:** [code map](docs/code-map.md) and the [code-only checks](#reproduce).
- **Reproduce with authorized assets:** [training and evaluation guide](docs/reproducibility.md). Source images and checkpoints are not bundled with Git.

## How it works

![PTRN Hip architecture: 224-pixel input, four DenseNet201 scales, spatial fusion, ordered corners, ratio correction, auxiliary mask, and a single sampling pass.](docs/figures/architecture.svg)

The network sees a photograph resized without stretching and padded to **224 × 224** pixels. DenseNet201 supplies four spatial feature scales; a 64-channel decoder fuses them before predicting corners in **top-left, top-right, bottom-right, bottom-left** order, in input pixel-center coordinates. A ratio residual adjusts the edge-derived log aspect ratio of the full reference plane. The visible-film mask helps during training but does not decide a second crop or act as an approval gate. Inputs use ImageNet channel normalization.

The renderer intersects the predicted plane with the original photo, samples that photo **once**, and returns a validity mask for observed pixels. A predicted corner can sit outside the photo when the film is partially captured. The output can retain background near an uncertain edge; the model does not fill in unphotographed X-ray content. See the [visible-plane geometry contract](docs/partial-v3.md) for coordinates, clipping, and renderer details.

## What we know

On the same **4,589 photographed views from an exposed synthetic development set**, branch +2250 traded a little more background for less visible-film loss than branch +2000:

| Measure, mean across the same 4,589 views | Branch +2000 | Branch +2250 |
| --- | ---: | ---: |
| **Additional crop** of visible film | 3.43% | **2.25%** |
| **Extra background** inside the predicted region | **2.54%** | 3.54% |

“Additional crop” means visible film missed by the predicted region; “extra background” means predicted area outside the true visible-film mask. Both use **synthetic ground truth**. The full fixed development set has 5,736 views, but its keep-original views are excluded here so they do not dilute photographed-image behavior. The set and previously reviewed real photos influenced model selection; these numbers are **not independent real-photo accuracy** or a clinical result.

![The exposed synthetic development trade-off between visible-film crop and residual background.](docs/figures/tradeoff.svg)

### Six synthetic examples

The fixed cases cover near-full-frame capture, mild perspective, a missing corner, a near-color boundary, screen interference, and a trusted original. They remain in their predetermined order, including imperfect outcomes. The input panel shows the known plane and predicted corners; the other panels show parent step 2000 and branch +2250 under the same reviewed renderer. In the original case, verified source provenance triggers a pixel-exact keep path; the network's corners are diagnostic only.

<details>
<summary>Open the six-case input and output comparison</summary>

![Six fixed synthetic cases showing input, parent output, and branch +2250 output.](docs/figures/examples/overview.webp)

</details>

The radiographs in these illustrations come only from the two versioned datasets marked CC BY 4.0 below; the backgrounds are procedural. Both checkpoints use the same reviewed V3 area-resampling renderer for this comparison. That renderer differs from the training-time source, so the illustrations are not pixel-identical replays of training. [Example credits, seeds, modifications, and pixel/metadata audit](docs/figures/examples/CREDITS.md).

## Data and training

The inventory counts **source files, not patients**. Its split preserves historical development files and quarantines 28 duplicate or similarity-review candidates. It is **not known to be patient-independent**.

| Source collection | Files in local inventory | Use in repository illustrations |
| --- | ---: | --- |
| [MTDDH](https://www.nature.com/articles/s41597-025-05146-x) | 1,849 | Original-version redistribution terms not directly verified; no MTDDH image used. |
| [AV-DDH v2](https://data.mendeley.com/datasets/4gvcb6gmh2/2) | 2,417 | Source page states CC BY 4.0; illustrated files reviewed individually. |
| [Fraiwan DDH X-ray v2](https://data.mendeley.com/datasets/jf3pv98m9g/2) | 354 | Source page states CC BY 4.0; illustrated files reviewed individually. |
| **Total** | **4,620** | **4,114 training · 478 exposed development · 28 quarantine** |

Training generates photographs with variable film size, perspective, partial capture, near-color edges, backgrounds, and camera/screen artifacts. Its local background pool uses **4,400 COCO 2017 photos** and **1,100 procedural interface images**. COCO photos retain their individual Flickr terms and are not redistributed here. The fixed development set keeps the earlier `partial-v3` scene recipe: **12 scenes per development file = 5,736 views**. The newer recipe changes **training only**, so the checkpoint comparison uses the same frozen scenes.

![Synthetic-scene construction from a source radiograph, geometry, appearance, missing-area variation, and background.](docs/figures/synthesis.svg)

The branch begins with the verified pipeline **step-2000** model and Adam state, then uses a **new deterministic sample cursor** for up to **3,000 additional updates**. Thus “+2250” means 2,250 updates *after* that parent checkpoint, not absolute step 2250 of the old run. Checkpoints are saved every 250 updates. Training uses 224-pixel input, actual and effective batch **64**, Adam head learning rate **1 × 10⁻⁴**, and a frozen DenseNet201 encoder and BatchNorm at `stage=head`. Losses cover corners, full-plane ratio, geometry validity, auxiliary visible mask, and missed visible film with a greater initial cost than extra background. There is no blanket penalty for a reference corner falling outside the photo. Checkpoints bind the recipe, source, assets, optimizer and RNG state, and consumed sample cursor for exact resume auditing.

<details>
<summary>Open the exact synthetic training recipe targets</summary>

These are **targets**, not measured frequencies of every realized training sample. Geometry feasibility and retries can shift the distribution. About 20% of all slots use the original-image keep path; the percentages below other than that keep rate are conditional on a photographed slot.

| Component | Training recipe target |
| --- | --- |
| Original-image keep path | About 20% of all views; photographed views about 80%. |
| Film area within photograph | 90–98%: 20%; 75–90%: 30%; 50–75%: 35%; 30–50%: 10%; 10–30%: 5%. |
| Missing area of photographed reference plane | None: 80%; 1–3%: 12%; >3–8%: 6%; >8–15%: 2%. |
| Background family | Procedural viewer/interface: 40%; COCO context: 25%; near-color: 25%; mixed-color: 10%. |
| Source pre-crop | 12.5% of photographed views; target loss of 5–10% of source area. |
| Capture appearance | Screen or physical film, with independently sampled interference and edge blending. |

Source pre-crop defines a new complete reference plane. A crop *after* photographing instead makes part of that plane unobserved; the renderer cannot recover its missing content. The train-only probabilities and frozen development path are implemented in [partial_data.py](src/ptrn_hip/partial_data.py).

</details>

## Availability and limitations

This is a research implementation, not a published hip-radiograph paper or a clinical tool. The **478 development files**, all **5,736 fixed scenes**, and **47 real photographs** have been viewed during research. Those photos lack reliable paired reference boundaries, so their true crop loss cannot be measured; their images and per-case review pages remain local. Ambiguous or partially visible edges can still leave background or introduce geometric distortion. The current checkpoint has not passed an independent real-photo study, and the application model is unchanged.

The trained weights are controlled **private research backups**, not deployable model releases. Access to a backup does not establish redistribution rights for upstream initialization, images, or fine-tuned weights. Dataset-page licensing alone does not settle pixel text, privacy, or every downstream use. See [third-party terms](THIRD_PARTY.md) and the [checkpoint/source map](docs/checkpoint-source-map.md). The repository's [Apache-2.0 license](LICENSE) applies only to code and documentation we can license.

## Reproduce

Code-only checks need Python 3.11 or 3.12, PyTorch, and Keras 3 with the PyTorch backend:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=torch .venv/bin/python -m pytest -q
```

Training and full fixed-development evaluation additionally require separately **authorized local source images, backgrounds, initialization and parent checkpoints**, and SHA-checked manifests. None are in Git. The [reproduction guide](docs/reproducibility.md) gives the actual commands, checkpoint/source binding and **5,736-view** completion checks. `--eval-limit` is diagnostic only; private Pre-releases, where accessible, are not deployment registrations.

## Attribution

- Chong et al., [*Image Projective Transformation Rectification with Synthetic Data for Smartphone-captured Chest X-ray Photos Classification*](https://arxiv.org/abs/2210.05954), arXiv:2210.05954. The original PTRN study concerns chest photographs; this is a separate hip-radiograph implementation.
- Haddad et al., [AV-DDH v2](https://data.mendeley.com/datasets/4gvcb6gmh2/2), DOI: 10.17632/4gvcb6gmh2.2, CC BY 4.0 as stated on the source page.
- Fraiwan et al., [DDH X-ray images v2](https://data.mendeley.com/datasets/jf3pv98m9g/2), DOI: 10.17632/jf3pv98m9g.2, CC BY 4.0 as stated on the source page.
- [MTDDH dataset publication](https://www.nature.com/articles/s41597-025-05146-x); original-version redistribution terms have not been directly verified for repository illustrations.
- COCO Consortium, [dataset terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm); COCO photographs follow their individual Flickr terms.
