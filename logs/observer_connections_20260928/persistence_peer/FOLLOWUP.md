# Arithmetic review and dense-lag provenance

2026-09-28. Independent follow-up, CPU only, two threads. The machine-readable
checks are `followup_checks.json`; all writes stay in this peer directory.

## New observer analysis

Reviewed `../persistence_groups.py` and independently recomputed its sixteen
rank cells by direct rectangular tensor slices, rather than the script's
flattened bin accumulation. Checked every matrix-kind aggregate and whole-body
aggregate, both available lags, for Muon and PD at steps 200, 500 and 900.
All raw moments, coordinate counts, update energies, and decay-subtracted
energies agree; the largest relative discrepancy is below 3.8e-13.

The derived signed moments, ratio validity checks, and sensitivity factors
have the intended algebra. The pair-average and pair-difference energies are
the debiased energies of half-sum and half-difference, respectively. They
should not be named stationary and oscillatory components without extra
temporal assumptions.

One labeling detail matters: per-kind `update_energy_share`,
`coordinate_share`, and `update_energy_enrichment` use whole-body denominators.
They describe total-body shares and enrichment against total-body mean energy
per parameter. Within-kind plots must divide group energy/count by that
kind's own `all` entry. The script currently retains kind totals, not per-layer
heterogeneity; a broad kind can be dominated by one layer.

## Dense-lag sources and state

The relevant training runs are
`logs/muon_spectra/soaudit_denselag_20260927/{M,PD}`. Their launchers run the
original trajectory cohort's frozen package, with the full 1,539,870,720-token
budget (1469 updates), and merely stop execution after update 532. These are
fresh four-A4000 runs, not resumed versions of the L40S/Ada trajectories.

All source hashes recorded in both training metadata files match both the
per-run `source/` snapshots and the executable
`soaudit_traj_20260926/frozen/adamw_spectra/` files. Every retained full
checkpoint at 501/503/507/515/531 and next-step weights file has the expected
step and token count; full-checkpoint configs match the launch configs.

The actual probe starts at **501**, and targets 502, 503, 504, 507, 508, 515,
516, 531, and 532. Thus the lags are **1, 2, 3, 6, 7, 14, 15, 30, 31**. The
training README's step-500 wording is a prospective description and should
not supply the plot labels. The job launchers, job logs, saved probe metadata,
and `plot_lags.py` agree on the actual measurement.

Checked possible endpoint/probe artifacts:

* Every logged LR from 501 through 532 is 0.007 (Muon) or 0.01 (PD). The
  frozen training loop uses the full token budget in its LR function; the
  stopping limit cannot introduce a cooldown. Neither run clips in this window.
* The full checkpoint serializer saves model/optimizer/RNG states without
  resetting them. No restart occurs. The next-weight serializer saves the
  same model's state dictionary after the next update.
* Full checkpoints and weights-only checkpoints both pass through the same
  FP32 eager `build_model` in eval mode. Measurement disables input-statistic
  tracking. The gradient accumulator is newly constructed for every state,
  so later probes do not inherit earlier per-sequence moments.
* Although full versus next-weight files alternate with even versus odd lag,
  source inspection finds no loading-path behavior that would itself cause
  alternating signs.
* Muon's logged training gradient norm rises to 0.4149 and 0.4724 at updates
  531 and 532. These are gradients evaluated before their respective updates,
  on training batches; they are corroborating nearby dynamics, not identical
  observations to the held-out gradient at the saved post-update state.

This audit found no schedule, reset, or checkpoint-label explanation for the
late recurrence. It does not prove the proposed dynamics explanation. In
particular, one anchored correlation series cannot show that a period-two
oscillation retained phase for thirty steps: data forcing, transient mode
amplification, and a changing mixture of modes are alternatives. The common
training batch at update 529 has conspicuously lower loss in both methods
(Muon 3.9934, PD 3.9610), before the Muon burst; this is a reason to retain
data forcing as a competing explanation, not evidence that it caused the
burst.

The saved probe metadata does **not** include measurement-source hashes.
The executable probe path and present code agree with artifact structure and
logs, but this is weaker provenance than the hash-verified frozen training
sources. Later-state measurements also reuse sequence set B, as discussed in
the first review. These limitations do not prevent a clearly labeled fixed
rank-group exploration of the dense-lag artifact.
