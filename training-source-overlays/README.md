# Frozen pipeline source overlays

The original pipeline 1250/2000/5000 checkpoints record 48 source-file SHA-256
values by package basename. Three files subsequently changed. The exact
training-time bytes are preserved in [`pipeline-parent/`](./pipeline-parent/):
`partial_data.py`, `partial_geometry.py`, and `partial_synthesis.py`.

The step-2000 branch +2000/+2250 checkpoints record 76 repository-relative
source SHA-256 values. Their only current source difference is
`partial_geometry.py`; the same training-time bytes are in
[`pipeline-parent/partial_geometry.py`](./pipeline-parent/partial_geometry.py).
The current renderer uses reviewed area resampling for reduced moiré and has a
different SHA. This is an output rendering change, not a retrained checkpoint.

These files are byte-preservation material, not a second importable package.
Never copy them over the active `src/ptrn_hip/` tree in place. Reconstruct a
historical source tree in a separate checkout and verify every digest from the
checkpoint binding before claiming exact replay or resume. See the
[checkpoint/source map](../docs/checkpoint-source-map.md).
