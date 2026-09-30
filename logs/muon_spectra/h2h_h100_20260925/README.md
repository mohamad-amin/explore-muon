# Head-to-head on H100: PD vs Newton-Muon vs PMuon vs tuned Muon (2026-09-25)

Protocol, decision argument and predeclared reading: `research/adamw_spectra/MUON_CASE.md`,
"Head-to-head with Newton-Muon and PMuon on H100". Runs on Google Cloud Spot `a3-highgpu-4g`
(4×H100) via `cloud/run_arms.sh`; one VM per lane, deleted when its lane ends.

- `frozen/` and `frozen_manifest.json`: the live `research/adamw_spectra` after the port (37 files).
  `unit_tests.log`: 67 tests pass on the frozen copy.
- `make_arms.py`: arm configs from the frontier-norm reference config, every key explicit.
- Width 512, depth 8, 1× horizon (1469 steps). Arms are paired by seed on H100.
- Seed 260924 tuned Muon and PD on H100 are in `../gcp_validation_20260925` (wave-9 code).

## What runs

The planned 14-run design (LR brackets plus seeds 260925 and 260926) was launched at 21:08 UTC on
five VMs and stopped within minutes. VMs h2h-2 to h2h-5 were deleted before any arm started, two of
them after Spot preemption at boot. At 21:20 UTC the user directed: no LR brackets and no extra seeds
for now. The unrun arms are in `cancelled_before_start/`.

One run per method on seed 260924, paired with `../gcp_validation_20260925` (tuned Muon@0.007
3.70386, PD@0.01 3.68530), both on VM h2h-1, one after the other:
- `NM_lr0.005_s260924_h100`: Newton-Muon at LR 0.005, 0.71× Muon's. That is within the ratio of the
  Newton-Muon record's own tuned LRs to its Muon baseline (0.020–0.030 vs 0.035).
- `PM_lr0.007_s260924_h100`: PMuon at Muon's LR 0.007, as the PMuon record did.

Single-seed screens: report the differences, make no claim.

## Results (2026-09-25 21:40 UTC; H100, seed 260924, single-seed screen)

| method | LR | final val NLL | vs tuned Muon | PD − method |
|---|---|---|---|---|
| tuned Muon | 0.007 | 3.70386 | — | −0.0186 |
| PD α = ¼ | 0.01 | 3.68530 | −0.0186 | — |
| Newton-Muon | 0.005 | 3.70296 | −0.0009 | −0.0177 |
| PMuon | 0.007 | 3.70167 | −0.0022 | −0.0164 |

Both ported arms passed their tiny and full-size replica audits and completed (exit 0). The traceback
at the end of `lane_h2h-1.log` is expected: the unstarted `M_lr0.007_s260925_h100` arm was removed from
the VM, so the lane stopped after Newton-Muon (see `cancelled_before_start/`). PMuon ran in lane `h2h-1-pm`.
