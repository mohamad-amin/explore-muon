# A larger learned shared-value route moves less on the next step

**Later V/O qualification:** the pre-O quantities below depend on internal
value coordinates. `../value_gauge/REPORT.md` shows that the learned magnitude
ordering does not generally survive O, while the smaller current joint V/O
constant-route perturbations do. `../value_split/REPORT.md` supplies the
completed V-only local functional scores; no extra mean-route gain is established.

2026-09-28. Read-only contraction of the original 1M-token trajectories, seed
260925, all eight V matrices at the seven retained preterminal checkpoints
10/50/100/200/500/900/1300. No models, forwards, GPUs or scheduler jobs. Two CPU
threads; measured analysis runtime 13.20 seconds.

The learned-route observation has an important qualification: **PD preserves a
larger shared-value component while moving that component much less on the next
actual update.** Learned amplitude and current perturbation tell opposite stories.

Let W be the saved V weight, D = W_next − W the actual parameter displacement,
μ the independently measured normalized attention-input mean, and
C = E[xxᵀ]. Compute mean energy ||Dμ||² and centered energy
tr(D(C−μμᵀ)Dᵀ). Their sum is E||Dx||² before attention. Use the identical
calculation on W. This is a functional activation-space scale, not a parameter
Frobenius fraction or a gradient/momentum fraction.

## Primary checkpoint: step 500 to 501

Entries except explicitly summed energies are medians over the eight layers.

| Quantity | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| Learned W mean fraction | .0262 | .1582 | .0984 | .2995 |
| Actual D mean fraction | .1848 | .0411 | .2730 | .0544 |
| Sum of ||Dμ||² over layers | .03243 | .00336 | .04779 | .00325 |
| Sum of centered D energy | .14927 | .09059 | .13599 | .06841 |
| ||Dμ|| / ||Wμ|| | .0250 | .0035 | .0189 | .0027 |
| Centered perturbation / centered W energy, square root | .0082 | .0066 | .0092 | .0073 |
| D mean gain / D centered gain, corrected for input energies | 1.043 | .111 | .978 | .096 |
| Wμ·Dμ positive, layer count | 2/8 | 8/8 | 3/8 | 6/8 |
| Wμ·D_adaptive μ positive, layer count | 2/8 | 8/8 | 3/8 | 8/8 |

PD's constant-route perturbation energy is 9.7× smaller than Muon's; SOAP-PD's is
14.7× smaller than SOAP-Muon's. The centered reductions are much smaller, 1.65×
and 1.99×. These are their actual recipes, including LR differences, at their own
states; they do not isolate the input map's instantaneous action at fixed weights.

The median signed radial fraction (Wμ·Dμ)/||Wμ||² is
−.000923/+.000408/−.000241/+.000343. The directions mostly turn rather than simply
scale Wμ: median cosine(Wμ,Dμ) is −.063/+.131/−.012/+.148. A small positive radial
term under PD coexists with much smaller total movement. A fixed-μ increase is
2 Wμ·Dμ + ||Dμ||²; its polarization identity is checked in the script. The next
state also changes μ, so this is not the complete change in learned route energy.

## Full retained trajectory

Median fraction of pre-attention D activation energy in its constant route:

| Step | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| 10 | .7615 | .2532 | .3929 | .0735 |
| 50 | .0927 | .0517 | .4655 | .0518 |
| 100 | .1055 | .0363 | .1946 | .0413 |
| 200 | .1244 | .0269 | .1523 | .0377 |
| 500 | .1848 | .0411 | .2730 | .0544 |
| 900 | .2290 | .0571 | .3825 | .0676 |
| 1300 | .2575 | .0703 | .4731 | .0793 |

The absolute constant-route perturbation energy and its fraction are smaller
under PD than Muon in all eight layers at steps 200, 500, 900, 1300. SOAP-PD is
smaller in both measures in all layers at 500, 900, 1300; at 200 its absolute
energy is smaller in seven of eight layers and fraction in eight. These paired
layer/time comparisons are correlated observations in one training seed, not
independent replicates. Every layer and checkpoint is in `layers.csv`; both
median and pooled summaries are in `summary.csv` and `result.json`.

## Decay, uncertainty, and implication for the functional probe

The frozen training source uses decoupled parameter decay followed by the
optimizer-direction write. All four run configs confirm `data_norm_decay` is
`decoupled`, with WD=.01. The script reproduces the next update's LR schedule
(including warmup at step 11), forms ideal D_decay=−lr*wd*W, and labels the
remainder **adaptive_plus_rounding**. It is not the pre-NS momentum, nor the
unscaled optimizer direction. The maximum scalar-decay FP32 write discrepancy
relative to ||D|| is 3.77e−5. Rounding of the subsequent addition remains in the
adaptive remainder. This bounded accounting is sufficient to distinguish the
signed radial decay from the observed much larger full perturbation; it is not
an exact reconstruction of the optimizer's pre-write tensor.

At 500 decay's mean-route norm is .28%/2.88%/.37%/3.71% of the full D mean-route
norm. Subtracting decay makes two SOAP-PD layers' small negative radial terms
positive, but does not explain the large perturbation-size distinction.

Because each causal attention row sums to one, Dμ is passed through attention
exactly. The centered term becomes A D(x−μ), however; its expectation can be
nonzero, and its interaction with Dμ need not vanish after attention or in the
loss/GN metric. These saved marginals cannot score usefulness, GN coupling,
data-dependent selection, or cross-layer interactions. Nor do they include the
simultaneous changes in Q/K, output weights, or upstream activations. The parent
probe must keep the constant/centered cross term and actual signed descent.

A plausible reading is that PD permits a learned bias-like route to persist
while damping its fluctuations. An equally live alternative is co-adaptation
with no productive loss contribution from this route. The functional probe is
needed to distinguish those readings. No architecture or training proposal is
licensed here. Mean-only whitening, removal/capping of the mean step and whole-
body centering already failed to recover full PD's gain; this evidence does not
resurrect those interventions or establish that stronger mean suppression helps.

## Reproduction and provenance

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python logs/observer_connections_20260928/value_step_geometry/analyze.py

`result.json` preserves every checkpoint/nextweights path, size and mtime, each
accessed original V tensor's SHA-256, every used C/μ tensor hash, all configs,
step/token counts, and analysis-source hash. W and W_next hashes are rechecked
after contractions; no full-checkpoint scan touches embeddings or optimizer
storage. All sources are mmap-loaded on CPU. The common marginal probe used
2048 held-out sequences per state. These are fixed empirical activation moments
without saved independent-split error bars. No finite-sample confidence or
causal attribution is claimed.
