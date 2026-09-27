# Why we changed the training photographs

[English](research-journey.md) · [简体中文](../i18n/docs/research-journey.zh-CN.md) · [Project overview](../README.md)

This record explains the September 2026 overcrop investigation and resulting data changes. It is not independent validation.

## The problem: better synthetic scores, smaller real-photo crops

The task is to retain the X-ray content actually photographed, remove surroundings and straighten it. A tight box around the bones is not the target.

In the earlier run, mean corner error on **4,589 synthetic development photographs** improved from **2.514% to 1.768% of the input diagonal** between steps 1250 and 5000. Yet the later model often returned a smaller region on previously reviewed phone photographs. Across those photos, the median ratio of the later predicted area to the earlier predicted area was **78.24%**.

The earlier prediction is **not ground truth**: the area ratio cannot quantify lost anatomy. Lower coordinate error was insufficient for choosing a checkpoint. The [original investigation](2026-09-25-overcrop-discussion.md) records the measurements and limits.

## Inspection: separate the prediction from the renderer

Both comparisons used the same [visible-plane renderer](../src/ptrn_hip/partial_geometry.py), without the acceptance gate or another crop from the auxiliary mask. The predicted region had contracted in the original photo's coordinates before rendering. Relaxing a gate would not recover excluded pixels.

The generator also revealed a mismatch worth addressing. Ordinary complete photographed films were constructed in smaller occupancy bands, while full-frame examples were largely clean digital originals. Cropping a photograph could increase its final occupancy, so high-occupancy examples were not entirely absent. The specific gap was **near-full photographs with capture artifacts**, rather than near-full images in general.

## The change: a single revised data recipe

We retained the improved background families and changed the training photographs together:

| Aspect | Earlier recipe | Revised recipe |
| --- | --- | --- |
| Partial capture | 40% of photographed slots requested missing content; the moderate stratum removed 12–28% of the reference plane. | 80% request a complete plane; partial captures favor mild cuts, with routine missing area capped at 15%. |
| Framing | Complete photographs were constructed at 30–75% film occupancy before photo cropping. | Explicit near-full photographed views, common close views and a small distant-view component. |
| Appearance | Earlier light capture effects. | Screen or physical film sampled separately from background, with reflections, blur, appropriate screen interference and narrow edge blending. |
| Source pre-crop | More frequent, potentially larger source crops. | Less frequent, milder crops; the cropped source becomes a new complete reference plane. |

These are **targets**; feasibility checks and retries can alter realized frequencies. See [sampling code](../src/ptrn_hip/partial_data.py), [exact recipe](figures/README.md#exact-training-targets--精确训练目标) and [data sources](../THIRD_PARTY.md). Unlike source pre-crop, later photo cropping leaves part of the reference plane unobserved. It does not authorize further loss of visible content.

This phase **did not change the architecture**. It started from the earlier step-2000 model **and Adam state**, with a new deterministic sample cursor. The encoder and BatchNorm remained frozen. The existing loss already penalized missed visible film more than extra background; its coefficients were **inherited unchanged**, not increased for this experiment. The [training entry point](../src/ptrn_pipeline/step2000_runner.py) preserves those distinctions.

## Evaluation: retain content, then examine the trade-off

The frozen synthetic development scenes remained unchanged. Evaluation tracks additional crop, residual background, ratio and valid geometry alongside corner error. The same phone photographs informed visual review, but have no reliable paired boundary truth and are already exposed development material.

Several data factors changed together: this cannot isolate one cause or prove independent real-world improvement. A larger crop helps only if retained content, background and geometry are acceptable. See the [reproduction guide](reproducibility.md) for practical steps and the [execution plan](plans/2026-09-25-ptrn-step2000-autodl-training-plan.md) for historical decisions.
