# [PRIVATE RESEARCH BACKUP] PTRN spatial v2

> **Storage update · 2026-10-17.** The original assets for this historical stage are retained in the maintainer-controlled [private archive](https://github.com/wzxsph/hip-xray-ai/releases/tag/backup-rectiray-prepublic-20261017-v1.0.0); access requires authorization. This public source repository provides no model downloads. The original research notes below are preserved for provenance.

**Channel:** private Pre-release; `latest=false`

**Role:** byte preservation for the spatial-head 1,250- and 5,000-step research checkpoints. This is not a deployable model release.

The ZIP contains only the two checkpoints and a stage metadata file. The same-tag JSON records each checkpoint's SHA-256, source-code binding, research role, and evaluation limitations; `SHA256SUMS.txt` covers the uploaded ZIP and JSON. The tag refers to the documented repository `main` after merge, **not** to the exact training source snapshot. The per-checkpoint binding and differences from current code are documented in the [checkpoint source map](../../docs/checkpoint-source-map.md).

These checkpoints are earlier spatial-v2 comparators. The 5,000-step checkpoint is retained for over-cropping analysis; neither checkpoint is the current research choice. Existing synthetic development and phone-photo material was exposed during research. No independent real-world or clinical acceptance is claimed. No patient images, annotations, per-case predictions, or upstream demo initializer are included.

This private backup does not establish permission for public distribution of weights, upstream initialization, or training data. Do not register either checkpoint as an application model or use it for clinical decisions.
