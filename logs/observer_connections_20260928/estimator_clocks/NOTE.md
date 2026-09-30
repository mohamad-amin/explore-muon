# Estimator clocks: distinguish update age from sampling precision

**Subsequent archive qualification:** ordinary model/optimizer checkpoints
omit covariance state, but a separate rank0 input-statistics sidecar was found
for the16M checkpoints. It is not all owners' state or the cached R from the
previous refresh. The conditional repeatability test in
`../ts_repeatability/REPORT.md` uses it as a fixed common proxy. That completed
test finds large Euclidean map variation but much closer predictive responses;
the proposed sampling bottleneck remains unestablished. The historical audit
and sampling-budget distinction below remain valid within that qualification.

This is a read-only scientific/source audit, extending the input-EMA observation in `logs/observer_20260928/mean_geometry/NOTE.md`. New computation uses only Python's standard library. No model forward, checkpoint loading, GPU call, training, or job submission. Live Slurm state was checked: allocations 2567578 (`priv-g14`) and 2618555 (`g20`) remained running; neither was touched. All new files are in this directory.

**Strongest connection:** increasing training batch gives SOAP a more precise global gradient observation, but does **not** give the online TS output factor a larger sample. This leaves a viable estimator explanation for some of SOAP's growing advantage over TS. It does not show that TS is noisy, nor refute the geometric advantage. Conversely, the input covariance's dramatic apparent acceleration in optimizer steps is largely an expected consequence of its token clock. It should not be presented as an overlooked 16-fold arbitrary EMA change.

## What the sources establish

`extract_clocks.py` reads actual scientific metadata for nine completed PD/SOAP-PD/TS arms, extracts constants from their frozen source AST, and verifies **27 frozen-source SHA256 values against the archived run metadata**. Re-run with:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python logs/observer_connections_20260928/estimator_clocks/extract_clocks.py
```

The generated `TABLE.md` and `evidence.json` retain run paths, hashes, exact source spans, parameters, and computed quantities. The nine arms all use four ranks, 16 sequences per microbatch, and context 512. Representative 1M arms use alpha 1/4 and momentum .95; the tuned 4M and 16M arms use alpha 1/2 and momentum .9. This is a descriptive clock audit, not a controlled causal batch comparison.

| Estimator | Observation | Update clock | Rank handling |
|---|---|---|---|
| Input C | 16 positions per sequence, from positions 1,33,...,481 | decay .998 per local training forward | Owner rank's statistic, not pooled |
| Input root R | inverse fractional power of normalized C | cached 10 steps at 1M/4M, 2 at 16M | Owner computes; resulting parameter direction shared |
| Body momentum | clipped full gradient | per optimizer step | DDP global gradient |
| SOAP Gram | `(G R)(G R)^T` and its input counterpart | decay .9 per optimizer step | DDP global gradient; owner-local state |
| SOAP denominator | squares of projected momentum, or gradient in the explicit alternative | decay .9 per optimizer step after readiness | Owner-local state |
| SOAP basis | one eigendecomposition to initialize, then one sorted QR iteration | after every step's preconditioned direction is computed | Next update uses that new basis |
| TS output B | sampled-label output-error outer products from **8 local sequences** | decay .8 each refresh: 10 / 10 / 2 steps | **Not pooled** across ranks |
| Auxiliary Adam | first and second moment of auxiliary gradients | per optimizer step, beta1 .9, beta2 .95 | DDP global gradient |

Source anchors in the 16M frozen tree (`logs/muon_spectra/soaudit_batch16m_20260927/frozen/adamw_spectra/`): `model.py:60–100`; `muon.py:162–211,243–260,460–499,619–718,832–870`; `distributed.py:262–335`. The corresponding earlier sources were individually hash verified. DDP has `broadcast_buffers=False`; no all-reduce of C or B is present. Each matrix is assigned one update owner, which uses its own rank's statistics before sharing the final direction. SOAP sees `parameter.grad` after the gradient reduction and clipping.

These are different statistical objects: a Gram of the batch-mean parameter gradient is not the per-token output-gradient second moment. A sample-budget check can distinguish estimation quality from a useful difference in target, but cannot equate those targets.

## The quantitative clock comparison

Numbers below are stationary mean observation ages, relative to the current pre-update model state. Cache age is averaged uniformly over refresh phase. They describe the linear estimators; matrix inversion, polar factors and changing eigenbases need not inherit the same effective lag.

| Batch | C forwards / step | Effective C step decay | C age + root cache, steps | Momentum age, steps | SOAP Gram age when used, steps | TS B age + cache, steps |
|---|---:|---:|---:|---:|---:|---:|
| 1M | 32 | .937945 | 19.615 | 19 | 10 | 44.5 |
| 4M | 128 | .773944 | 7.924 | 9 | 10 | 44.5 |
| 16M | 512 | .358787 | 1.060 | 9 | 10 | 8.5 |

The C age at refresh is `.998^n / (1 - .998^n)`, where n is forwards per optimizer step. Cached-root mean age adds `(refresh - 1)/2`. Momentum age is `beta/(1-beta)`. SOAP's Gram is used one step after its observation was included, so its .9 EMA contributes 9 steps plus 1; its denominator includes the current observation and has a nominal age of 9. Those ages should **not** be added mechanically to the momentum age: the squared momentum and changing basis are nonlinear history dependence.

In millions of elapsed global training tokens the C-plus-cache ages are **20.57 / 33.23 / 17.78M**, not a monotonic 16-fold reduction. Ignoring model-step grouping and cache, C's microbatch-age clock is exactly the same **16.35M global tokens** at all three batch sizes. The 16M refresh change was explicitly predeclared in `MUON_CASE.md:2898–2910`, to keep refreshes near a common token interval. It was not an accidental hidden change. A fixed .9 step EMA, by contrast, has age **9.44 / 37.75 / 150.99M tokens** as batch grows.

This says neither tokens nor steps are the uniquely correct clock. Sampling noise is reduced by observations; nonstationarity is driven by parameter/function changes. At fixed parameters, grouping the same stream into different optimizer batches does not improve the stationary precision of C. With learning, larger groups mean more C samples at the same state, so C is fresher in the model trajectory. LR, update shape, clipping and early training speed decide whether that matters.

## The output factor has a fixed sample budget and a distinctive startup

For independent stationary observations, an EMA with decay b has weight ESS `(1+b)/(1-b)`. These are nominal weight ESS values, **not measured effective independent tokens**:

- C: 999 local microbatch observations × 256 sampled token positions = **255,744 weighted token rows**, unchanged with batch. Sequence and token dependence can lower the information content.
- B: 9 refresh observations × 8 owner-local sequences = **72 weighted sequences**, or **36,864 token rows**, unchanged with batch. The four ranks do not make one factor a 32-sequence observation.
- SOAP: 19 gradient observations per .9 Gram EMA; each gradient averages **2048 / 8192 / 32768 global sequences** at 1M / 4M / 16M. These are nonlinear gradient Grams, so multiplying those two counts would not make an ESS for the same target as B. The gradient's sampling noise nevertheless decreases with training batch; B's fixed local subsample does not automatically improve.

The whole TS run has **147 / 37 / 46 B refreshes** at 1M / 4M / 16M. Their stationary B ages including cache are **46.66 / 186.65 / 142.61M global tokens**. In steps the 16M B age is 8.5, comparable to SOAP's 9–10, and substantially *fresher* than B at 4M. Therefore, a simple claim that "TS loses at 16M because B refreshes too slowly relative to SOAP" is not supported by the schedule alone. A fixed sample budget and factor nonstationarity remain separate possibilities.

The first B estimate is copied directly; later calls use `.8 B + .2 B_new`. At 16M step 9, there have been five observations and the first **8 sequences still have coefficient .8^4 = 40.96%**. At the approximately equal-token 4M step 37, there have been four observations and their coefficient is **51.2%**. By 16M step 46 the first coefficient is about **0.74%**. These are algebraic weights; their matrix-norm contribution also depends on B's changing scale. The code averages raw B, then normalizes its eigenvalues at root construction, so an unusually large early B could influence longer than its scalar coefficient suggests. No archived online B magnitude series was found here, and this possibility is unmeasured.

SOAP also has startup and coordinate-history issues. The first direction is unpreconditioned, its Gram is initialized after that direction, and denominator accumulation begins on the second step. QR refresh reorders denominator entries but does not rotate a full second-moment tensor into the new basis. In S∘PD, the stored Grams also combine gradients transformed by historical R matrices. These facts justify measuring coordinate drift; they do not establish a consequential approximation error.

## What this changes, and what it does not

The legitimate existing result is that **these implemented SOAP-PD recipes beat these implemented TS recipes at 16M**. The stronger interpretation "output gradient statistics are intrinsically preferable to output curvature at large batch" is not isolated by those runs: the targets, sample budgets, startup, and history differ. A carefully estimated curvature factor could still lose because SOAP's sign-like normalization is the right geometry; this audit is not evidence that extra curvature data will fix TS.

The input C clock cannot by itself explain SOAP-PD versus PD at a fixed batch: their C clock and cache agree. It could contribute to the batch dependence of both versus Muon. The persistence of gains in the longer 16M runs and the successful lower-momentum arms also limits an explanation based only on an initial transient. No single-clock explanation accounts for all of those observations.

World-size and microbatch matter operationally. At fixed global batch, increasing world size reduces local forwards per step, making C older in steps while leaving per-factor stationary C ESS unchanged. Doubling microbatch sequences with the same per-forward decay likewise reduces forwards, but doubles sampled rows per forward and hence the nominal C ESS. B still uses the configured 8 sequences on the owner. SOAP's globally averaged gradient is largely invariant to such partitioning, apart from numerical effects and the changed R. All audited runs use the same world size and microbatch, so this is a reproducibility constraint, not an explanation for their actual difference.

## Smallest discriminator

**First separate sample instability from lag before proposing training.** At an existing authorized measurement opportunity, hold one 16M state, momentum and input R fixed. Form B from disjoint 8-sequence groups; compare the resulting normalized TS directions with directions formed from their pooled B. Repeat sampled labels on fixed sequences to separate label Monte Carlo noise from sequence noise. Report step cosine and held-out slope/curvature or true-loss profiles with fixed normalization, alongside PD. The relevant stability is the direction after inverse power and polar mapping, not just top-eigenspace overlap. No full training arm is needed for this discriminator.

- If independent small B estimates already give near-identical directions and pooling barely changes them, the fixed sample-budget explanation weakens.
- If pooling makes the direction reproducible and improves its held-out profile, the implemented TS estimator is a plausible bottleneck. This still does not prove a sustained training benefit.
- To test lag, retain the trainer's actual cached B/R and compare them with a sufficiently sampled current-state factor. The ordinary checkpoints omit C, SOAP state and B, so offline recomputation alone cannot reconstruct the historical online estimator or adjudicate its lag.

The archived exact-GN preflight compares two full GN samples and finds instability, but it does not compare two B estimates. That is not substitute evidence for instability of the Kronecker output factor. This note recommends the bounded discriminator as a scientific next measurement, and launches nothing.
