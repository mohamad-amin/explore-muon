# β 0.8 at 4M for the whitening methods (second-order audit, 2026-09-28)

Is the whitening methods' 16M momentum gain (PD −0.124, S∘PD −0.10) a large-batch effect? Decision argument:
`research/adamw_spectra/MUON_CASE.md`, "Decision argument: is the whitening methods' momentum gain a large-batch
effect?" (14:39 CDT).

- **Arms.** 4M, seed 260925, 368 steps, α ½ @0.02 at β 0.8:
  - S∘PD on priv-g14 (L40S), reference 3.8260;
  - PD on g20 (Ada), reference 3.8589.
  Each runs after its node's current queue.
