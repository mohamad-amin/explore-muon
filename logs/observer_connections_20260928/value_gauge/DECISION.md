# Identification check: value route after the output projection

Pre-analysis decision, 2026-09-28. The saved V-only mean/centered energies are
not invariant under an exact block-diagonal per-head V/O change of basis:
Wv' = S Wv, Wo' = Wo S^-1. Therefore a difference in that statistic could reflect
an internal coordinate choice. Compute the constant attention residual output
c = Wo Wv μ and measured output mean m = Wo μ_z from the same 32 retained states.
Both survive that internal gauge. Retain their norms, angle, relative mismatch,
and layerwise/timewise results. Residual-stream global scale remains a separate
ambiguity; absolute norms do not identify causal route preservation. Full O-input
covariance is absent, so no centered residual output energy is invented.

At the primary step500 states also use actual saved next weights and compute
Wo Dv μ, Do Wv μ, Do Dv μ, and their exact sum Δ(WoWv)μ at fixed μ. Keep signed
cross terms: individual route energies do not add. This tests whether the
reported small PD value perturbation survives its output projection and whether
simultaneous O movement materially changes it. No loss usefulness or training
intervention follows from this coordinate check. CPU-only sparse mmap tensor
reads, two threads, under three minutes, no models, forwards, GPUs or jobs.
