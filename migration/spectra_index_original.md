2026-09-25 Muon extension: the user replaced the unlaunched16-layer AdamW
cell with Muon depths8/12/20 on4/4/8 GPUs. See the
[Muon case](adamw_spectra/MUON_CASE.md), [independent review](adamw_spectra/MUON_REVIEW.md)
and current [research state](../RESEARCH_STATE.md) for execution handles.
The distributed runner records separate momentum and actual post-NS update
spectra using the existing fixed-width, batch and token horizon. Active handles:
8-layer **2618555.9** on g20; 12-layer **2620620** on four A6000 GPUs;
20-layer **2620623** queued for four96 GB GPUs (replaces canceled2620619). See
[qualification](adamw_spectra/MUON_QUALIFICATION.md).

2026-09-24 depth-only extension: submitted eight-GPU jobs **2620279**
(12 layers) and **2620278** (20 layers), width 512, heads 8, fixed global batch
and the same 1.540B-token horizon and LR 0.0012 as the eight-layer reference.
See the [corrected depth case](adamw_spectra/SCALE_UP_CASE.md) and its job links
in [research state](../RESEARCH_STATE.md). Each job qualifies the distributed
execution before training. The 12-layer run is now complete: final validation
NLL 3.83965, 1,469 updates and 60 spectral snapshots. The 20-layer job remains
pending for resources on the 96 GB GPU pool (checked 2026-09-25 01:23 UTC).

2026-09-24 user-requested implementation: [77M AdamW adaptive-update
spectra](adamw_spectra/README.md), with a fixed small LR sweep, exact singular
values across 24 body matrices and static analysis. CPU and full-size CUDA
qualification passed; the pilots, selected-rate run and plotting completed
successfully on g20 allocation 2618555, step .3. Final validation NLL 3.87925,
1,469 updates, 60 snapshots of 24 matrices. Spectral interpretation is pending.
[Completed run](../logs/adamw_spectra/g20_20260924_212246_r2/README.md).
See its [protocol](adamw_spectra/PROTOCOL.md) for reconstruction limits.

