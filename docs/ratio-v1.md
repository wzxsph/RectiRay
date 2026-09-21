# Ratio-v1 frozen experiment

This independent research run uses local B1.1 initialization and does not change the
application model. The source restoration baseline `8b67e07` is not claimed as the
historical B1.1 training commit. Input data and weights are read-only; every output
(including source-image previews) must remain in the ignored `.local/runs/ratio-v1`.
No patient names, source paths, grouping hashes, images or model files belong in Git.

## Contract

G0 is DenseNet201 + the original H8 head, 224 center-span letterbox. G1 adds a
1921-parameter log(W/H) head at 224; G2 uses the same architecture at 384. Each starts
independently from B1.1 SHA
`89d48a8c88200aa3c6916f2839556615138d6c5b1c7b4f803cd8f78ce8cc0520`.
The ratio head is initialized with zero weights and the retained training set's mean
log ratio. B1.1 is evaluated frozen with its original direct-square preprocessing.

Baseline GT uses the actual direct-resize pixel-center map `(x+.5)*size/W-.5`;
this corrects evaluation geometry without changing B1.1 weights or its legacy H decoder.

H maps normalized original canvas corners to normalized input-photo corners; TL,
TR, BR, BL order never changes. New H normalization maps pixel centers 0..size-1 to
[-1,1]; B1.1's legacy decoder uses size. The ratio is EXIF-correct original W/H,
including preexisting black borders and excluding synthetic viewer framing. It is
not (W-1)/(H-1), anatomical calibration, or proof of physical film proportions.

The film is placed in a physical plane with equal x/y pixel scale. A pinhole camera
with isotropic focal length, rotations and translation projects the viewer and film.
The common photograph has long side 768. Final letterbox uses an explicit isotropic
warpAffine and its exact center-span transform, including the identity branch.
Geometry uses 30/60/10 identity/mild/hard; yaw/pitch limits 0/15/40 degrees and roll
0/12/45. The screen stays entirely visible. Procedural backgrounds are independent
of all medical sources, so background source leakage is absent.

Appearance is sampled independently at 45/35/15/5 clear/light/medium/strong. Strong
appearance is downgraded to light for hard geometry; realized counts are saved.
The frozen moire probabilities are 0/.30/.50/.60; normalized amplitudes
0/.02/.05/.08; noise sigma maxima 0/1/2/3; gamma 1/.9–1.1/.8–1.2/.75–1.3;
blur maxima 0/.5/.8/1.2; JPEG minima 100/85/70/60; light amplitudes
0/.025/.05/.08. Moire sums rotated, phased sine grids at periods 2–8, 8–40 and
40–160 common-photo pixels. Its mask covers the entire screen, including viewer.
Camera degradation samples none/blur/JPEG/noise with .4/.2/.2/.2 probabilities.
These are appearance approximations, not a calibrated physical screen model.

Seed 20260921, source ordinal, global view ordinal and domain select independent
geometry/background/appearance streams. No group, resolution, microbatch or worker
identity enters these keys. All three groups receive exactly the same common photos
in the same order. Labels are identical before and after appearance changes.

## Audit and split v1.1

The audit verifies all original file hashes and EXIF dimensions against the old
manifest, compares original decoded pixels and the historical 224 pixel fingerprints,
and checks perceptual near duplicates. Hashes and connected-group keys exist only
in memory. The output retains source ordinals, never patient grouping hashes.

Strict near duplicates require pHash Hamming <=4, gray64 correlation >=.995,
MAE <=.02 and log aspect difference <.01. Conservative candidates require pHash
Hamming <=4 and correlation >=.98. Candidate status is not proof of patient identity.
The main task authorized on 2026-09-21 retaining all 478 historical dev sources and
quarantining training sources linked to dev by the connected closure of exact,
strict-near and candidate links. The old manifest is never rewritten. The audit
records unique excluded counts, thresholds, new train count and decision provenance.
Patient grouping remains unknown; dev is exposed and not independent validation.

## Optimization and evaluation

Each group receives 1250 optimizer updates, effective batch 64, 80000 sample
exposures. The first 150 updates train only heads at Adam 5e-5; 1100 updates train
heads and the final dense block at 1e-5. All BatchNorm parameters and state are frozen.
Global gradient norm clipping is 1.0. bfloat16 autocast runs forward; projective
geometry and loss run in float32. There is no loss-scaler batch skipping. A nonfinite
loss, gradient, weight, singular denominator or mixed-sign projective horizon stops the run with a failure record.

G0 loss is mean SmoothL1(pixel corner difference / (.01 * input diagonal), beta=1).
G1/G2 add .25 * mean SmoothL1(log ratio difference / .05, beta=1). Initial microbatch
is 16 with 4 accumulation steps for 224, and 4 with 16 for 384. Only micro/accum may
change together to retain effective 64, explicitly in a fresh config.

Every 250 updates, save all model/optimizer/RNG/config/source SHA states and evaluate
478 x 3 fixed geometry views, clean and noisy. Fixed noisy levels use the training
non-clear distribution conditional on non-clear (7/11 light, 3/11 medium, 1/11
strong), with the same hard-geometry downgrade. B1.1 sees these exact common photos
through its legacy input transform. Metrics include per-geometry corner mean/p90,
illegal quads, multiplicative ratio errors, maximum-opposite-edge ratio max(top,bottom)/max(left,right) using the same predicted
corners, and GT-corner/predicted-ratio versus predicted-corner/GT-ratio decompositions.
No invalid predictions are omitted. Evaluation is float32.

Checkpoint selection is fixed before training: G0 minimizes clean all-view mean
corner NME; G1/G2 minimize clean mild ratio median; ties favor earlier steps. Candidate
corner gates apply to every clean and noisy geometry stratum (mean <=1.05x G0,
p90 <=1.10x G0, illegal count no larger). Clean mild learned-ratio median <=3%,
p90 <=10%, median <=80% of same-corner edge baseline. G2 additionally needs >=10%
p90 improvement over G1 and external CPU acceptance. Since the training protocol
provides no numeric CPU budget, G2 CPU acceptance remains explicitly pending review.
No failing model replaces B1.1. Real unpaired photographs are visual checks only.

## Execution

Use the existing environment's Python with `PYTHONPATH` pointing to this checkout's
`src`. Set the following shell variables locally; their values are never committed:

```bash
export PYTHONPATH="$PWD/src"
PTRN_PYTHON=/path/to/existing/venv/bin/python
PTRN_MANIFEST=/path/to/readonly/manifest.local.json
PTRN_WEIGHTS=/path/to/readonly/diagnostic.weights.h5
PTRN_RUN=.local/runs/ratio-v1
```

Run `python -m ptrn_hip.ratio_runner --manifest ... --weights ... --run ... audit`,
then `preview`, `smoke`, and the synthetic test suite. Inspect local full-frame and
1:1 preview sheets. After the main task commits and explicitly confirms the source
snapshot, use the same command with `--source-sha CONFIRMED_SHA run-all`. The sequential
runner evaluates frozen B1.1, then G0/G1/G2. Resume with exactly the same command.
Source SHA/config mismatches fail. A checkpoint is saved before dev evaluation;
interrupted dev evaluation is rerun before continuing. Updates after the last saved
checkpoint are replayed deterministically, not credited twice toward the budget.

Status files expose PID, phase, update/microstep, sample count, throughput, ETA,
loss and GPU memory; JSONL history retains every completed update. Keep all five
checkpoints per group. Read-only monitoring may report failure/stall, but must not
change hyperparameters. Check disk space before training and each checkpoint.

Export is a separate explicit 9-output contract for G1/G2, with model_estimated
ratio evidence. It cannot be installed as an old eight-output B1.1 asset. CPU
forward equivalence, artifact SHA, metadata, missing-resource errors, Windows
runtime acceptance and redistribution licensing remain distinct checks.
