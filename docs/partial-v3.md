# PTRN visible-plane v3 research code

This is an isolated, runnable research candidate, not a promoted model. The prior
B diagnostic failed; the C runs were user-authorized exploration. V3 has not yet
shown better real-photo quality or passed the automatic-output acceptance goal.
Old source modules, weights and thresholds are preserved.

## Geometry and data

`partial_synthesis` distinguishes source pre-crop (new rectangular reference and
width/height ratio) from post-photo crop (same full-plane ratio, translated
homography, possibly out-of-photo virtual corners). Both retain the mother
source's split. It records visible polygon, captured corner/edge visibility,
film mask, separate screen mask, crop chain and deterministic scene seed.
Missing fractions are measured in the reference plane, not projected photo area.
The width/height label convention remains unchanged from v2.

The portable manifest contains 4114 train, 478 exposed-development and 28
quarantined sources. Every train source has 16 views per round (65824); the fixed
development has 12 per source (5736). Quarantine is never sampled. No independent
patient or new-real-test claim is made. SHA, source membership and background
split/content overlap are checked before updates. COCO attribution remains in
the staged background manifest.

Keep views occupy approximately 20%; photos approximately 80%. Within photos,
complete/light/moderate missing-area quotas are 60/30/10%, and background family
probabilities are viewer/COCO/near-color/mixed-color = 40/25/25/10%. Source pre-crop
probability is 25%, retaining at least 70% of the source area. All photo views
have visible occupancy >=30%; a balanced schedule requests >50% for 15 of every
16 photo slots (93.75%, above the approved minimum 80%). Other proportions are
sampling probabilities, not exact per-minibatch counts. The scene audit records
realized frequencies. Geometry retries preserve the requested strata and are
counted; no failed source is silently dropped.

Viewer scenes retain the prior nested screen/window/toolbar generator. COCO
comes from the matching existing background split; near-color and mixed-color
backgrounds use source border colors. This is a first implementation of the data
families, not a claim to cover every real phone reflection/interface. Occlusion
inside the film and missing regions other than rectangular camera crops are not
simulated in this revision.

## Candidate architecture and objectives

The first candidate reuses DenseNet201 and the existing four-scale 64-channel
spatial decoder, selecting its **unbounded direct quad readout** instead of the
bounded heatmap expectation. A lightweight visible-mask readout supplies
auxiliary boundary/region supervision. It does not hard-trim the image. The
ratio residual is separately weighted and logged against its edge-ratio baseline.
No pretrained v2 heatmap head is represented as a trained v3 head: initialization
uses the verified B1.1/G0 encoder, with newly initialized v3 readouts.

The contract is `ptrn-visible-plane-v3`: normalized NHWC input -> ordered full
plane TL/TR/BR/BL corners in input pixel centers, full-plane log ratio, and visible
film logits. `RawPartial` provides raw RGB 0..255 -> quad/log-ratio/probability for
CPU export. Old H8 or v2 checkpoints cannot be silently loaded as v3.

Losses are coordinate Smooth L1, visible-area missed-film/background penalties
(initial relative costs 3:1), convexity, visible-mask BCE and ratio Smooth L1.
There is **no blanket photo-boundary penalty**. Mask, coverage and ratio weights
are CLI ablations. Output-grid error is evaluated on the captured portion of a
17x17 reference lattice; both P90 and maximum are reported. It is not yet an added
training loss: adding it remains a later matched experiment, after localization
is verified. Temporary collapsed predicted quads receive finite training losses
rather than being skipped. Invalid output geometry remains in the evaluation
denominator.

`rectify_visible` maps captured content to a bounding canvas and returns a valid
pixel mask. Uncaptured areas stay empty. When the photo-to-output map uniformly
shrinks the visible region, the renderer samples at 3x output resolution and
reduces with area interpolation to suppress resampling aliases. It normalizes
partial pixel coverage so clipped photo edges do not acquire a dark halo. For
magnification or mixed local scales it retains bilinear sampling; the output
geometry and valid mask do not depend on this choice. It never invents pixels or
clamps virtual corners to the photo border. Trusted originals are copied
pixel-exactly. The renderer is a candidate visualization path; it is not a new
automatic approval policy and does not replace old gates.

## Entry point

Use the staged Python environment and set `KERAS_BACKEND=torch`. Select CPU
visibility before importing Keras. Example code smoke (scratch optimizer updates,
not an effectiveness experiment):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m ptrn_hip.partial_runner \
  --bundle /path/to/portable-bundle --backgrounds /path/to/backgrounds \
  --output /path/to/new-cpu-smoke --device cpu --steps 1 \
  --microbatch 1 --effective-batch 2 --stage last_block --eval-limit 2
```

GPU is opt-in with `--device cuda`; it checks CUDA/BF16 availability. Initial
performance candidates are 224 with physical batch 16/32/64 at effective batch
64. CUDA memory, sustained throughput and worker counts require GPU measurement.
Do not use CPU-smoke timing as a GPU estimate. Initial head/conv5 learning rates
are 1e-4/1e-5, BN is frozen, clipping is 1. Mixed precision covers model forward;
geometry/losses remain FP32. Gradient accumulation uses actual sample counts.

The runner freezes data/code/weight hashes and objective settings before updating,
saves every 250 steps plus 500/1250/5000 and final, and evaluates at 500/1250/5000
and final. Omit `--eval-limit` for the full exposed development. Resume with the
same output and `--resume /path/to/step-N.pt --steps TARGET`; mismatched bindings
are refused. The sample cursor resumes exactly despite DataLoader prefetch.
Scratch weights, output images and results belong in ignored local run storage.

CPU code verification, diagnostic fit, sustained GPU validation and real-photo
acceptance are separate milestones. The 47 exposed real photos support local
input/export checks now; new v3 weights must be trained before a useful quality
comparison can be made. The old 90% success goal is not redefined by these raw
geometric diagnostics.
