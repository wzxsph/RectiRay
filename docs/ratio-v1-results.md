# Ratio-v1.1 experiment results — 2026-09-22

**Decision: retain B1.1. Neither joint-ratio model met the frozen ratio gate.** G1 and G2 passed every clean/noisy per-geometry corner gate but failed all three mild-ratio requirements. The exports are failed-research diagnostics only; no application model, deployment catalog or runtime was changed.

Training source snapshot: `858945b4b8b1b1286f4598db115ddae5394efd81`. Restoration baseline `8b67e07` is not asserted as B1.1's historical training commit. Training ran on the local RTX 3000 Ada 8GB from 2026-09-21 21:49:20 to 2026-09-22 01:04:02 (Asia/Shanghai), including frozen B1.1 evaluation and all scheduled development evaluations: 3.245 hours.

Each group completed 1250 optimizer updates and 80000 exposures; total 240000 exposures. The 150-head/1100-last-block schedule, Adam learning rates, effective batch 64, frozen BatchNorm, loss, clipping and shared random streams were retained. Every group selected step 1250 under its predetermined checkpoint rule. All 15 checkpoints, optimizer/RNG states, configs, SHA sidecars, histories and clean/noisy evaluation records remain local. All checkpoint file SHA256 values were verified against their sidecars.

## Data and validity

All 4620 original file hashes and EXIF dimensions matched the read-only manifest. The v1.1 split retained 478 historical development sources and quarantined 28 historical training sources (3 strict duplicate-linked plus 25 additional similarity-candidate-linked sources), leaving 4114 training sources. Mean training log(W/H) was 0.2280023265. Each group had approximately 19.446 exposures per retained source. Fourteen exact original copies were found; eight additional strict near-duplicate pairs included three cross-partition clusters. Similarity candidates are not verified patient identities. The original manifest was not changed.

**This is exposed synthetic source-image development, not independent patient validation.** Patient grouping is unavailable. The new quarantine prevents this round from reusing those training sources, but cannot remove their possible historical exposure through B1.1 initialization. Forty-seven unpaired real photographs provide visual checks only; they cannot establish ratio accuracy. Canvas W/H includes original black borders and does not establish physical anatomical proportions.

## Frozen gates

All percentages below are multiplicative ratio error `exp(abs(log(pred/GT)))-1`. The edge baseline uses the same predicted corners and `max(top,bottom)/max(left,right)`.

| Model | Mild median | Mild p90 | Same-corner edge median | Median improvement vs edge | Corner gate | Ratio gate |
|---|---:|---:|---:|---:|---|---|
| G1 | 5.268% | 13.836% | 3.238% | -62.73% | pass | fail |
| G2 | 4.733% | 16.398% | 3.163% | -49.63% | pass | fail |

Required: median ≤3%, p90 ≤10%, and ≥20% median improvement over the same-corner edge baseline. Negative improvement means worse performance. G2 mild ratio p90 was 18.52% worse than G1, so it also failed the additional ≥10% p90 improvement required for G2 priority. CPU speed cannot override these failures.

| Model | Maximum corner mean factor vs G0 | Maximum corner p90 factor vs G0 | Invalid quads (clean/noisy) |
|---|---:|---:|---:|
| G1 | 1.049385 | 1.037200 | 0 / 0 |
| G2 | 1.020771 | 1.080556 | 0 / 0 |

Factors are the worst of all identity/mild/hard clean/noisy strata. Limits were 1.05 for mean and 1.10 for p90; invalid counts could not increase.

## Clean and fixed-noisy development results

Each condition contains 1434 views (478 sources × 3 geometry strata). Corner NME is expressed as percent of input diagonal. G0/B1.1 ratio entries use the maximum-edge baseline; G1/G2 entries use the learned head.

| Model | Condition | All-view corner mean | All-view corner p90 | Invalid | Mild ratio median | Mild ratio p90 |
|---|---|---:|---:|---:|---:|---:|
| B1.1 | clean | 3.468% | 5.095% | 6 | 5.836% | 17.955% |
| B1.1 | noisy | 3.470% | 5.128% | 6 | 6.003% | 18.413% |
| G0 | clean | 1.236% | 1.927% | 0 | 3.281% | 9.369% |
| G0 | noisy | 1.266% | 1.949% | 0 | 3.427% | 9.167% |
| G1 | clean | 1.280% | 1.985% | 0 | 5.268% | 13.836% |
| G1 | noisy | 1.312% | 2.029% | 0 | 5.344% | 13.919% |
| G2 | clean | 1.183% | 1.918% | 0 | 4.733% | 16.398% |
| G2 | noisy | 1.226% | 2.024% | 0 | 4.966% | 16.387% |

G0 substantially improves synthetic corner error over frozen B1.1, but it also changes input preprocessing, synthesis and loss. This comparison does not isolate model size or augmentation effects. All three trained groups received noise augmentation, so none is a clean-training ablation. In hard geometry the learned-ratio median improves over the same-corner edge baseline, but mild/identity performance fails the accepted decision rule; hard-group gains do not substitute for the required gate.

## Realized training augmentation

All three groups produced exactly equal counts: identity 23985 (29.98125%), mild 47978 (59.97250%), hard 8037 (10.04625%). Requested clear/light/medium/strong counts were 36258 / 27944 / 11822 / 3976. The hard-geometry rule downgraded 422 strong appearances to light, yielding actual counts 36258 / 28366 / 11822 / 3554 (45.3225% / 35.4575% / 14.7775% / 4.4425%). Moire appeared in 16594/80000 views (20.7425%). Texture covered the whole simulated screen including viewer; appearance left labels unchanged.

Peak CUDA reserved memory reported per complete run was G0 6030 MiB, G1 5546 MiB, G2 5358 MiB. No nonfinite value, skipped bad batch, OOM, or hyperparameter change was recorded. BatchNorm remained unchanged.

## Unpaired real-photo visual review

All 47 source photos were rendered for G0/G1/G2 corner overlays against B1.1 and as six-panel actual rectifications: B1.1 max-edge, G0 max-edge, G1 max-edge, G1 learned ratio, G2 max-edge, G2 learned ratio. The learned/edge comparisons reuse each candidate's exact same predicted quad. Original photo hashes were unchanged. Representative/high-disagreement contacts 1, 4, 16, 25, 40 and 47 were visually inspected; the other contact images remain available for human review.

The contacts show visible horizontal-scale differences, residual screen tilt and differing crop extents. For example, anonymous ordinal 16 has G1 same-corner edge ratio 1.4335 versus learned ratio 1.1084; this is a disagreement, not a measured error without paired truth. Strong moire remains visible; PTRN predicts geometry and does not denoise these review images. No real-photo noise-robustness accuracy or true ratio recovery has been established.

Geometric rejection flagged one real image for each of B1.1/G0/G1 (`outside_photo`, anonymous ordinal 40), and none for G2. The rejected panels are explicitly marked INVALID and were not silently rectified. Passing these geometric checks is not evidence of correct film localization or clinical accuracy.

Final original-integrity audit rehashed all 4620 training/development source files, the old manifest and B1.1 initialization weights after training; all remained unchanged. The complete repository test suite passed 33 tests, including three added postprocessing geometry tests.

## Diagnostic exports

G1/G2 exports use distinct RGB224/RGB384, H8+log(W/H) nine-output contracts. They use source-normalized to input-normalized H, pixel-center normalization 0..size-1, explicit letterbox preprocessing and model_estimated ratio evidence. They are not compatible replacements for the eight-output B1.1 interface.

Both exported models were compared with the eager CPU training implementation on all 47 real photographs plus two synthetic inputs (49 per model). Maximum absolute output difference was 0.0. Sanitized versus raw state tensors were identical; contract metadata and missing-resource failure checks passed. Windows acceptance and redistribution eligibility remain separate, unfulfilled conditions.

| Model | Diagnostic TorchScript SHA256 |
|---|---|
| G1 | `8abd7f2a7a8c2df94e71bd535c723be65d3fb8a0f94543c3586933371ea24cc2` |
| G2 | `48814883a26f61a964fada4d143a1e7303cfb6705585d965a39ef566e6b084ee` |

Export-task CPU timings are single-model warm inference only (3 warmups, 10 measurements per thread count), not main-application or native Windows acceptance:

| Model | Threads | Warm p50 | Warm p95 |
|---|---:|---:|---:|
| G1 | 2 | 128.6 ms | 132.5 ms |
| G1 | 4 | 106.4 ms | 123.0 ms |
| G2 | 2 | 364.6 ms | 391.2 ms |
| G2 | 4 | 223.4 ms | 241.2 ms |

## Local evidence index

Artifacts remain under the ignored `.local/runs/ratio-v1/`; none are uploaded or committed:

- `selection.json`, `gate-details.json`: checkpoint choices and exact acceptance comparisons.
- `checkpoint-inventory.json`: all 15 checkpoint paths, sizes and verified SHA256 values.
- `G0|G1|G2/step-*.pt`, `evaluation-*.json`, `history.jsonl`, `config.json`, `realized-distributions.json`: complete training evidence.
- `data-audit.json`, `split.local.json`, `interpretation.json`, `final-original-integrity.json`: original integrity, source-only quarantine and exposure limitations.
- `G1|G2/diagnostic-export/`: explicit nine-output TorchScript assets and export equivalence reports.
- `G0|G1|G2/real47-overlay/`: baseline-versus-model corner overlays, anonymous ordinals and predicted ratios.
- `real47-rectified/`, `real47-visual-notes.json`: 47 six-panel rectification contacts, source-axis transforms, individual local outputs and the explicit scope of visual inspection.
- `preview/`: four-level augmentation full frames and 1:1 crops, plus geometry overlays.

All training, export and real-photo postprocessing processes have exited. The main application remains on B1.1; idle 2/4-thread whole-pipeline acceptance is handed to the main integration task. There is no approved joint-ratio candidate to integrate under this experiment.
