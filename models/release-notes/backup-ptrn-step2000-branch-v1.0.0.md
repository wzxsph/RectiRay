# [PRIVATE RESEARCH BACKUP] PTRN step-2000 branch

**Channel:** private Pre-release; `latest=false`

**Role:** byte preservation for branch +2,000 and +2,250 checkpoints trained from the separate pipeline 2,000-step parent. This is not a deployable model release.

The ZIP contains only these two checkpoints and a stage metadata file. The same-tag JSON records each checkpoint's SHA-256, source-code binding, research role, and evaluation limitations; `SHA256SUMS.txt` covers the uploaded ZIP and JSON. The tag refers to the documented repository `main` after merge, **not** to the exact training source snapshot. The current renderer differs from the training-time renderer; the [checkpoint source map](../../docs/checkpoint-source-map.md) gives both hashes and the reviewed comparison path.

On the **exposed synthetic development** photo subset (4,589 views), the +2,000 and +2,250 checkpoints had average added crop of 3.43% and 2.25%, respectively, with average residual background of 2.54% and 3.54%. The latter is a research choice favoring completeness over background removal. These figures are not independent phone-photo accuracy or a guarantee for any individual image. Existing 47 phone photos were used for local review and are **not** in this Release. No independent real-world or clinical acceptance is claimed. No patient images, annotations, per-case predictions, or upstream demo initializer are included.

This private backup does not establish permission for public distribution of weights, upstream initialization, or training data. Do not register either checkpoint as an application model or use it for clinical decisions.
