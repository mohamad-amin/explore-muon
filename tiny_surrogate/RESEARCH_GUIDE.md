# Small-surrogate research guide

Find a small standalone setup reproducing the reference optimizer ordering,
growing geometry benefit with batch, and the fixed-token batch–momentum
interaction. The original goal is still active; partial reproduction is not
completion. The user additionally requires evaluation-overfit controls and all
study work to remain in this directory.

Use only GPUs strictly below 48 GB. Do not invoke the parent full-size training
pipeline. This study owns its small trainer and a pinned copy of the pure GPT
and optimizer kernels. The shared Python environment is read-only.

Read RESEARCH_STATE.md and the relevant PROTOCOL.md entries before planning.
Preserve configurations, sources, hashes, failed attempts and predeclared
criteria. Keep gradients, momentum, normalized directions and actual parameter
changes distinct. Fixed tokens are not fixed compute or fixed update counts.

All inspected validation scores are development evidence. Choose recipes there,
then lock the complete comparison family before independent confirmation. Keep
test results sealed until every declared run is ready. Report ties, reversals,
censored crossings, seed uncertainty and document heterogeneity. Do not turn a
failed criterion into a favorable claim by changing windows or thresholds.

Compare ordinary next-token NLL and common-loss progress. Geometry is supporting
evidence, not a replacement for learning. Momentum controls classify phase and
horizon dependence; they do not license a universal batch-only causal claim.

Before a substantial new branch, state the missing premise, mechanism, simplest
competitor, discriminating comparison, decision-changing outcomes and bounded
cost in PROTOCOL.md. Obtain a fresh independent peer discussion before a major
scientific commitment or repair cascade. Current user authorization governs;
idle GPUs and historical launchers do not authorize unrelated work.

Use `./run` for Python, and keep all scratch/caches inside `scratch/`. Historical
paths are preserved by compatibility links. Never rewrite frozen metadata merely
to update those paths. The migration record is in `migration/relocation.json`.
