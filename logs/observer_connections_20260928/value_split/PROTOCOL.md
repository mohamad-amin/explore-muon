# Actual value-step functional split

2026-09-28. Previous goal turn is progress: completed independent-frame and
dense persistence analyses, an architecture contraction, and a qualified CPU
body/auxiliary counterfactual changed the next question. No previous observer
computation remains live. Main Slurm state was checked at13:53; all observer
work remains CPU only and in this directory tree.

## Question and alternatives

PD/SOAP-PD learned value maps preserve the shared input mean much more strongly
than Muon, even after accounting for input mean size. Does their *actual next
V displacement* make productive use of that shared route? Alternatively the
larger route is mostly co-adaptation and its current contribution is small or
unproductive. The saved-step geometry check independently found that PD's
middle-checkpoint constant perturbation energy is much smaller than Muon's
despite the larger learned route. Do not equate activation size with step size.

## Fixed design, before evaluating losses

Original1M Muon and PD own states at500, actual saved V displacements to501.
Only the8 V maps move; every other learned parameter stays at its base state.
For each layer fix mu from the archived2048-sequence validation marginal,
independent of the two new never-trained training-stream score banks. Inject
at the value output

    a D mu + b D(x−mu)

using the live input x, including changes caused by earlier perturbed layers.
At a=b=c this is exactly the model family W_v→W_v+cD_v (up to FP32 roundoff).
The component paths separately include a temporary value bias and are
functional counterfactuals, not valid bias-free parameter steps in isolation.

At(0,0), measure logit JVPs for each component, loss slopes and predictive GN
2×2 matrix including the cross term. Score actual losses at(1,0),(0,1),(1,1)
and(.5,.5), plus base. Report pure and conditional component effects and
finite interaction. No line search, independent re-normalization, parameter
update, training or new method is selected from these scores.

Exactly two banks of4 sequences per state, offsets2,500,098,304 and
2,600,098,304; sequence length512. Small samples qualify a local premise only.
Archive per-sequence values and input tokens; do not grow banks based on effect.

## Instrument and resource gates

CPU FP32 eager, two numerical threads, no visible CUDA devices; frozen model
helpers. Use explicit math attention for all paths so forward AD is supported.
Before science, qualify diagonal-family equivalence to directly changed V
weights (relative logit RMS error<=1e-5), JVP linear additivity(<=1e-5), and
central finite differences at fixed coefficient h=.02 (relative tangent RMS
error<=.02). Reverse-mode coefficient gradients must match JVP-derived slopes
within max(1e-6, .002*abs(slope)). No repair may relax these criteria silently.

One-sequence timed qualification precedes full scoring. If projected total
CPU cost exceeds15 minutes, retain qualification and stop before the banks.
All means, accessed V tensors, source and exact score tokens have provenance.
No GPU, scheduler job, optimizer or original artifact mutation.

## Interpretation and next decision

Consistent negative slope and actual improvement by a component supports its
local usefulness under this actual step. A small or uphill shared effect does
not prove that the learned route is useless. Large cross terms mean independent
component prescriptions are unsupported. Own-state comparisons do not isolate
intrinsic geometry from co-adaptation or establish Muon-specific mis-scaling.
Retain relative logit/GN magnitudes alongside raw effects, but do not choose
new step scales. Any subsequent architecture/optimizer experiment needs a
separate argument and independent review.

Fresh independent review: `../value_split_peer/REVIEW.md`; agrees this is a
worthwhile bounded premise check. It emphasizes conditional effects, mean
dependence and the distinction between GN cross-curvature and finite loss
interaction. The user requirement to isolate observer outputs takes precedence
over placing this new note in the shared main protocol.
