# Primary precedents for the coordinate/normalization qualification

Checked2026-09-28. Bounded prior-art check, not an exhaustive novelty search.
The algebra in`REPORT.md`is the observer's derivation/application; the
following are specific claims supported by the primary sources.

**K-FAC.** Martens and Grosse's Theorem1/Corollary2 establishes equivalence
under fixed affine network reparameterizations for a basic formulation.
The path result assumes matched initialization, absent/negligible damping,
no momentum, and a parameterization-independent learning-rate choice. It
does not certify invariance of arbitrary normalized polar updates or their
practical state handling. [Paper, section4](https://www.cs.toronto.edu/~rgrosse/publications/icml2015-kfac.pdf#page=6)

**GO-MUON.** Section3.2 distinguishes the raw matched spectral oracle from
Frobenius rescaling. The positive rescaling retains its direction but changes
the weighted constraint radius. The paper states orthogonal/scalar properties
and does not infer general affine invariance for the quarter-power recipe.
This directly limits an invariance-based reading of the project's practical
PD/TSnormalization. [Paper, section3.2](https://arxiv.org/html/2608.09763v1#S3.SS2)

**Circuit-Muon.** The original implementation record already couples attention
V/O legs using partner per-head norms and a trace-balancing correction along
their gauge. It retains the baseline's broader optimizer machinery. This is
prior art for circuit coupling and balancing, not evidence that the project's
observed mean-route differences identify a balance defect or that the same
intervention would help this recipe. [Original record](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/results/20260523_circuit_muon/README.md)

Local source context: `research/adamw_spectra/data_norm_muon.py` specifies
input-root normalization, damping and parameter-Frobenius matching;
`muon.py` implements the geometry-shaped decay. Historical controls are in
`MUON_CASE.md` around wave10 and the Track3geometry-decay/hyperball decisions.
The early curvature-exponent derivation of alpha¼ was later withdrawn and
is not used here.
