## 2026-09-25: completed eight-layer AdamW/Muon comparison

At the user's request, compared the completed eight-layer runs and generated
[figures, CSV exports and provenance](logs/muon_spectra/comparison_depth8_20260925/README.md),
including [validation curves](logs/muon_spectra/comparison_depth8_20260925/validation.png),
[median spectral trajectories](logs/muon_spectra/comparison_depth8_20260925/median_all24.png)
and [the full PDF](logs/muon_spectra/comparison_depth8_20260925/comparison.pdf).
Both reached 1,469 updates / 1,539,870,720 tokens and have all 60 snapshots.
Initial-weight hashes, frozen model code, data manifests, global batch,
validation bank and token exposure match. All 24 matrices in each spectral
object were checked against the full archived singular values and energy.

Final validation NLL is **3.720990943 for Muon**, **3.879250005 for AdamW**,
a difference of **−0.158259062 nats/token**; perplexities 41.305 and 48.388
(14.64% lower for this Muon recipe). At the endpoint, the top singular direction
contains 5.96–17.00% of AdamW update energy across the 24 matrices, versus
0.209–0.389% of Muon's post-NS update energy. Muon's pre-NS momentum remains
concentrated (15.26–86.01%). Flattening of post-NS spectra is expected from
orthogonalization and does not establish the cause of the NLL improvement.

This is one seed and two optimizer recipes: AdamW peak LR 0.0012; Muon body
LR 0.01 and auxiliary AdamW LR 0.002, with different effective decay and GPU
counts (one versus four RTX 6000 Ada GPUs). There is no isolated causal NS,
statistical-significance, exact paper-initialization, wall-time speedup or
formal stabilization claim. The predeclared 1100–1300 pre-cooldown summaries
are retained. This establishes an observed loss and geometry difference in
the local eight-layer regime. The next decision is to compare the authorized
deeper paired runs when complete; no extra seed, tuning or job was launched.

## 2026-09-25: Muon depth comparison launched

The user canceled the **16-layer AdamW** cell before launch and authorized
Muon at depths **8/12/20 on4/4/8 GPUs**. Status verified01:50 UTC:
- **8 layers:** scientific training on g20 under **2618555.9**, four48 GiB
  RTX6000 Ada GPUs; reached at least126 updates. Momentum and post-NS archives
  at steps1/25/50/75 each contain24 matrices with512 singular values each.
- **12 layers:** job **2620620**, four48 GB RTX A6000 GPUs on g14; tiny CUDA
  qualification passed and full-size peak-rate qualification is running.
- **20 layers:** replaced pending eight-GPU Muon job2620619 with **2620623**,
  requesting **four96 GB GPUs**, at 2026-09-25 01:55 UTC. The user requested
  moving one depth20 job to four GPUs. Original job2620619 was canceled before
  launch; sources, config, global batch and token horizon are unchanged.
  [Scheduler change](logs/muon_spectra/depth20_w512_20260925_r2/gpu_count_change.json).
The completed AdamW8/12 baselines and queued AdamW20 job2620278 are preserved.
The sealed head-initialization study remains untouched.

Width512, heads8, context512, global batch1,048,576 tokens, seed260924,
initialization recipe, data order, warmup50 and the1,539,870,720-token horizon
remain fixed (1,469 updates). Muon uses body LR.01, plain momentum.95 and
five paper NS polynomials; auxiliary AdamW uses LR.002. Separate archives
record pre-NS FP32 momentum and captured post-NS direction spectra, excluding
LR/decay, every25 steps plus first/final. This is a paper-inspired optimizer
recipe comparison, not isolated NS causality or exact paper reproduction.
[Decision argument](research/adamw_spectra/MUON_CASE.md),
[independent review](research/adamw_spectra/MUON_REVIEW.md).

[Qualification](research/adamw_spectra/MUON_QUALIFICATION.md) records16 passing
unit checks, uneven-batch distributed CPU checks and the initial CUDA failure.
The final implementation computes each NS matrix on one owner and shares its
FP32 direction; strict replica and communicated-owner checks pass. NS is eager
and batched after compiled NS disagreed with its BF16 reference beyond the
fixed check; model compilation remains enabled. Originals are preserved.
Eight-layer full-size peak-rate qualification passed:1.02s/update, .74s for
both spectral panels, 3.44 GiB peak per rank, rough31.5-minute main forecast.
Public jobs independently qualify before science and all pipelines cap at7h45m.
No Muon spectral or comparative-loss conclusion is yet recorded.

[8-layer run](logs/muon_spectra/depth8_w512_20260925_r2/README.md),
[12-layer run](logs/muon_spectra/depth12_w512_20260925_r2/README.md),
[20-layer run](logs/muon_spectra/depth20_w512_20260925_r2/README.md).
Next decision: inspect completed paired trajectories and NLL; do not infer
an optimizer speedup from hardware-dependent elapsed times.

Initialization audit (user question, 2026-09-25): frozen AdamW and Muon
model code, seeds and model configs match at each depth; actual initial-weight
hashes match for the launched8/12 pairs. The initializer is GPT-style normal
std.02, with attention-output/MLP-down std.02/sqrt(2L), zero biases and default
LayerNorm. It does not explicitly apply Appendix A.1's stated initialization
factors sqrt(d_out/d_in) and1/sqrt(d_in) for the head. The paper does not specify
a complete base initialization or experiment revision, so equivalence cannot
be asserted or a unique paper initializer inferred from those factors alone.
The current cohort is a matched local recipe, not verified paper initialization.
See [paper A.1](https://arxiv.org/html/2606.04058v2#A1) and
[frozen initializer](logs/muon_spectra/depth8_w512_20260925_r2/frozen/adamw_spectra/model.py).
No configuration or job was changed for this read-only audit.


## 2026-09-24: submitted fixed-width depth comparison

The user requested two eight-GPU jobs, then clarified that width and batch
must remain fixed to study depth. No jobs from the initial width-scaling
interpretation were submitted. Corrected jobs are **2620279** (12 layers,
48 GB or RTX Pro) and **2620278** (20 layers, 96 GB RTX Pro), both in `gpu`,
eight-hour limits. Status verified 2026-09-25 01:23 UTC: the **12-layer job
completed successfully** on g22 (eight RTX A6000 GPUs), exit 0:0, allocation
elapsed 00:27:45. Its scientific run reached all 1,469 updates and
1,539,870,720 tokens, with final validation NLL **3.8396469429135323**,
all 60 spectra and generated plots. Main invocation time was 1474.68 seconds
(about 24.6 minutes). The **20-layer job remains pending for Resources**, with
no node assigned or training output yet; it still requests eight 96 GB GPUs.
Both fix width 512, 8 heads, context 512, global batch 1,048,576, total tokens
1,539,870,720, seed 260924 and the eight-layer selected LR 0.0012. Thus each
has 1,469 updates and 60 spectral samples; only depth changes under the same
recipe (including its residual initialization scaling with depth). Counts
are 89,603,072 and 114,822,144 parameters. No per-depth LR retuning is added.

[Depth case](research/adamw_spectra/SCALE_UP_CASE.md),
[12-layer job](logs/adamw_spectra/depth12_w512_20260924/README.md),
[20-layer job](logs/adamw_spectra/depth20_w512_20260924/README.md).
The new DDP runner keeps the global mean gradient with actual-token weighting,
handles uneven final microbatches, clips after reduction and distributes the
24 independent SVDs. Three-rank CPU qualification passed against the single
device and on resume; all replicas' parameters/moments were byte-identical
within a run. Distributed resume differed by at most 1.86e-9, so numerical
agreement, not bitwise identity, is claimed. The 12-layer job passed tiny and
full-size eight-GPU qualification and exact replica weight/moment audits
before its full run. The 20-layer job will do the same after allocation.
[12-layer summary](logs/adamw_spectra/depth12_w512_20260924/scientific/summary.json),
[12-layer median spectra](logs/adamw_spectra/depth12_w512_20260924/scientific/plots/median.png).
No depth-effect or spectral-stabilization interpretation is yet recorded.
Frozen sources isolate these
jobs from the now-completed eight-layer g20 run. The current original 32 data
shards suffice; the separate 7.2B-token staging done before clarification is
unused. No sealed initialization outcomes were read.

## 2026-09-24: separate AdamW update-spectrum implementation

The user requested a simple approximately 77M AdamW setup comparable to
*Spectral Scaling Laws of Muon*, measuring the adaptive update rather than
only the first moment. The isolated [implementation](research/adamw_spectra/README.md)
and [protocol](research/adamw_spectra/PROTOCOL.md) are complete with CPU
[qualification](research/adamw_spectra/QUALIFICATION.md). This is a new
descriptive question, not a head-remedy result or a Track-3 modification.
The 76,993,536-parameter reconstruction uses 20N = 1,539,870,720 tokens,
1,048,576 tokens/update and 1,469 updates. Batch size and untied vocabulary
matrices are supported by the plotted horizon and parameter count; exact
architecture details remain documented reconstruction choices.
Subsequent user steering sets spectral sampling to every 25 optimizer updates
plus first/final (60 samples), while training loss remains logged every update.

The subsequent [efficiency update](research/adamw_spectra/EFFICIENCY.md) enables
compiled CUDA execution, fused AdamW, bulk token transfer and cached validation.
Eleven CPU tests, a 53-step comparison against the original implementation,
and tiny compiled/BF16/fused CUDA checks passed. The user then explicitly
requested execution on the other idle allocation, g20. Allocation 2618555
was verified to contain four idle 48 GiB RTX 6000 Ada GPUs. A detached
[frozen pipeline](logs/adamw_spectra/g20_20260924_212246_r2/README.md) completed
under Slurm step **2618555.3**: five-update full-size qualification, three
paired 300-update LR pilots in parallel, then one selected-rate scientific
run and analysis. Full-size qualification now passed; worker **2698379**
exited 0. All three pilots completed successfully; the fixed selection
chose **LR 0.0012**, the upper grid boundary. The scientific eight-layer run
and plotting completed at **2026-09-24 23:45:21 UTC** (18:45 Central).
Slurm accounting confirms **COMPLETED, exit 0:0**, elapsed 02:13:16 for the
pipeline. All workers exited 0; the final checkpoint is at update **1469**
and **1,539,870,720 tokens**. Final held-out validation NLL is
**3.879250004887581**. All **60** spectral snapshots covering the selected
**24** matrices and all plot families are present. The main training
invocation took 6569.51 seconds (about 1h49m); measurement time was 69.48 seconds.
[Summary](logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924/summary.json),
[median trajectories](logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924/plots/median.png).
Completion was verified 2026-09-25 01:20 UTC. This is a completed baseline
dataset, not yet an interpretation of stabilization or a depth effect. The
next research decision is to inspect trajectories and compare qualified
depth runs when available. Do not restart the completed baseline. Source hashes, configs, logs and
worker handles are preserved in that run directory. The [selection record](logs/adamw_spectra/g20_20260924_212246_r2/selection.json)
keeps all candidate scores. No extra candidate or second eight-layer seed
is added. No spectral-mechanism conclusion has yet been drawn.
The original step .2 completed its five-update training worker (about 4 s
per steady update, 3.24 GiB peak allocated) but stopped before pilots: FP32
Frobenius accumulation produced 4.77e-4 normalized-energy error, above the
unchanged 2e-4 tolerance. The scalar norm now accumulates in FP64, with no
training-update change; remeasurement of the real checkpoint gives error
below 1e-6. Original outputs and failure are retained in the first attempt.
The corrected [qualification](logs/adamw_spectra/g20_20260924_212246_r2/QUALIFICATION_PASSED.json)
measures 3.94 s/update after startup, 1.17 s per spectral panel collection,
and 3.24 GiB peak allocated GPU memory. Pilot initial-model hashes,
source hashes and training/validation data manifests match. All 12 CPU
tests passed after the precision fix. The rough forecast is 20 minutes for
parallel pilots, then 98 minutes for the main run, plus startup/I/O/analysis.
This is runtime qualification, not a scientific conclusion about the spectra.
The separate sealed
head-initialization study remains untouched. The older research snapshot
below retains its original date and must not be read as current job state.

