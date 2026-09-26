# Credits and audit for the six synthetic examples

[简体中文图与来源说明](zh-CN/CREDITS.md) · [中文总览图](zh-CN/overview.webp)

These six panels are fixed illustrations of the research pipeline, not an accuracy sample. Source radiographs come only from these versioned datasets:

- Bassem Haddad et al., [*Acetabular-vision hip developmental dysplasia’s (AV-DDH), version 2*](https://data.mendeley.com/datasets/4gvcb6gmh2/2), DOI [10.17632/4gvcb6gmh2.2](https://doi.org/10.17632/4gvcb6gmh2.2), **CC BY 4.0** as stated on the dataset page.
- Mohammad Fraiwan, Noran Al-Kofahi, Omar Hanatleh, and Ali Ibnian, [*A dataset of DDH x-ray images, version 2*](https://data.mendeley.com/datasets/jf3pv98m9g/2), DOI [10.17632/jf3pv98m9g.2](https://doi.org/10.17632/jf3pv98m9g.2), **CC BY 4.0** as stated on the dataset page.

The [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require attribution and notice of changes. Each panel is a modified work: a radiograph was projected into a simulated photograph or passed through as an original; procedural surroundings, capture effects, corner overlays and model rectifications were added. The AV examples also have the two small generic viewer-footer text strips covered with black before synthesis. No source filename, source-file mapping, annotation, demographic field or patient identifier is distributed here.

| Panel | Source dataset; zero-based ordinal in relative-path lexicographic order within the versioned image tree | Fixed scene seed; view index | Scene and realized visible area |
|---|---|---|---|
| [01 · Near-full frame](01-near-full.png) | AV-DDH v2; 1228 | 20260926; 0 | Procedural viewer surroundings, film capture, 92.39% occupancy |
| [02 · Mild perspective](02-mild-perspective.png) | Fraiwan v2; 135 | 20260927; 1 | Mild homography, procedural mixed-color background, 81.72% occupancy |
| [03 · Missing film corner](03-missing-corner.png) | AV-DDH v2; 2022 | 20260928; 2 | Partial-film crop after projection, 6.05% source-plane content absent, 65.54% occupancy |
| [04 · Near-color boundary](04-near-color.png) | Fraiwan v2; 244 | 20260929; 3 | Procedural near-color background and softened seam, 55.78% occupancy |
| [05 · Screen interference](05-screen.png) | AV-DDH v2; 2346 | 20260930; 4 | Procedural viewer surround and strong screen capture artifacts, 75.73% occupancy |
| [06 · Original retained](06-original.png) | Fraiwan v2; 276 | 20261001; 5 | Trusted-provenance original pass-through; exact RGB pixels retained, 100% occupancy |

The [vertical six-case overview](overview.webp) presents the same inputs and outputs in six rows with labels set for an 850-pixel reading width. It is an 850 × 2198 lossless WebP (SHA256 5656651d9b390a65dca2c0dfee4df18cb2a7421c3232b0b8f188d594f1a9520c). The file was rebuilt from the six panels by the same generator and contains one VP8L image chunk with no EXIF, XMP, ICC, animation or text chunk.

The [README two-panel teaser](hero-missing-corner.png) (1100 × 670 RGB PNG; SHA256 `231b26a6fbca41560c1a1b96100a04a268339e0aca0d8187378b710f59806272`) and [narrow-screen stack](hero-missing-corner-mobile.png) (540 × 1240 RGB PNG; SHA256 `65c671ff90ef8149dd3b061f91ab9c7dce3efde53a840ea0a7d0cf99cce30d96`) copy the input and branch +2250 image rectangles pixel-for-pixel from [case 03](03-missing-corner.png). Only headings, captions, and surrounding layout are new. It can be rebuilt without private assets or inference using [`tools/build_readme_hero.py`](../../../tools/build_readme_hero.py); the script checks the reviewed source SHA and exported image metadata. The checkerboard denotes unobserved pixels, not reconstructed anatomy. [Chinese teaser and credits](zh-CN/hero-missing-corner.png) use the same picture pixels. The listed PNG hashes identify the checked-in renderings; font or Pillow versions can change caption rasterization and thus file bytes without changing the copied picture rectangles.

The source ordinals and seeds were locked before either checkpoint was run. All six panels remain in the set even where a result loses film or retains background. In the input column, **white** is the known full-plane quadrilateral, **cyan** is the parent prediction, and **orange** is the branch prediction. For cases 01–05, the latter two columns show rectification using the same photograph. In case 06, both outputs use the existing trusted-original keep path: the model corners are displayed for diagnosis but do not determine the action or crop. The keep action copies all RGB pixels exactly because the source is explicitly marked trusted; the network did not identify an original image. Checkerboard shows pixels outside observed photo support; missing anatomy is never synthesized.

The parent is step-2000.pt (SHA256 016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535). The branch is branch-step-2250.pt (SHA256 35f0101260b19ee1484022de3a4ed40fd3ce2f4bdb0ef73f379c4fd458a53b3b). Both were checked against their source and bundle bindings. Panels use the reviewed V3 area-resampling renderer (SHA256 a23bb919dd5cea1437477b0cb6478247e6bf9735599962787d7b48e42ef92549); the training-time renderer differs, as recorded in [the checkpoint source map](../../checkpoint-source-map.md). The same updated renderer was used for both checkpoints.

To rebuild, install the project dependencies with CUDA support and run [the generator](../../../tools/build_public_examples.py) with local paths:

    KERAS_BACKEND=torch PYTHONPATH=src python tools/build_public_examples.py \
      --av-root "$AV_V2_RAW" --fraiwan-root "$FRAIWAN_V2_IMAGES" \
      --bundle "$LOCAL_BUNDLE" --parent "$PARENT_STEP2000" \
      --branch "$BRANCH_PLUS2250" --output "$EMPTY_OUTPUT_DIRECTORY"

The local audit checked the six selected source images and every rendered panel at full size. No patient name, ID, date, accession number or other identifying text was visible; the remaining “L” is a laterality marker. The selected source files had no EXIF, and all six exported RGB PNGs have empty metadata dictionaries. Only procedural backgrounds were used; no COCO photograph or real phone photograph appears in these panels. Original source paths and any local review sheets remain outside Git.

These examples can overlap the research training or exposed development sources. They do not measure performance on independent real photographs and do not establish clinical suitability.

The [100-scene contact sheet](contact-sheet-100.png) reuses these same six audited public sources. Its [fixed recipe, GPU inference audit and hashes](../README.md#100-scene-recipe-and-audit--100-场景配方与审计) are documented separately. It contains no additional source images or private phone photographs.
