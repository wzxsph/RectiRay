# [PRIVATE RESEARCH BACKUP] PTRN pipeline v1

> **Storage update · 2026-10-17.** The original assets for this historical stage are retained in the maintainer-controlled [private archive](https://github.com/wzxsph/hip-xray-ai/releases/tag/backup-rectiray-prepublic-20261017-v1.0.0); access requires authorization. This public source repository provides no model downloads. The original research notes below are preserved for provenance.

**Channel:** private Pre-release; `latest=false`

**Role:** byte preservation for pipeline 1,250-, 2,000-, and 5,000-step checkpoints. This is not a deployable model release.

The ZIP contains only these three checkpoints and a stage metadata file. The same-tag JSON records each checkpoint's SHA-256, source-code binding, research role, and evaluation limitations; `SHA256SUMS.txt` covers the uploaded ZIP and JSON. The tag refers to the documented repository `main` after merge, **not** to the exact training source snapshot. Three files changed after pipeline training; the frozen source overlay and exact binding are explained in the [checkpoint source map](../../docs/checkpoint-source-map.md).

The 2,000-step checkpoint is the parent of the later branch, which starts a new sample cursor and optimizer sequence. The 1,250- and 5,000-step checkpoints are comparison points, not model-selection outcomes. Existing synthetic development and phone-photo material was exposed during research. No independent real-world or clinical acceptance is claimed. No patient images, annotations, per-case predictions, or upstream demo initializer are included.

This private backup does not establish permission for public distribution of weights, upstream initialization, or training data. Do not register these checkpoints as application models or use them for clinical decisions.
