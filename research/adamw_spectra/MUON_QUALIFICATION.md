# Muon execution qualification, 2026-09-25 UTC

Sixteen final unit checks pass: the existing AdamW suite and four Muon
contracts covering square/tall/wide/zero matrices against an independent
scalar-polynomial/SVD reference, auxiliary AdamW equivalence, exhaustive
parameter routing, actual parameter writes, read-only measurement and
mixed-state resume with scheduled group rates.

Four-rank CPU checks use a 288-token toy run with a 32-token final batch;
two ranks therefore contribute only a zero-weight synchronization graph.
Within-run weights and all optimizer state agree exactly. The mixed-state
resume check is bit-identical in this instance. Relative to one rank, the
maximum parameter difference is 2.0342e-6 and moment differences are below
6.52e-9. The single-owner NS execution also passes that comparison. These
are numerical qualification results, not a general bitwise-resume claim.

The initial tiny CUDA run stopped before science when one MLP-down matrix
differed across replicas. Gradients and optimizer states matched, narrowing
the disagreement to direction computation/application. Compute NS once per
matrix owner and sum disjoint FP32 direction slices; qualification verifies
that each communicated owner slice equals its original and that all replicas
agree. The literal eager BF16 polynomial is retained after compiled NS
disagreed with it by 3.407% on an archived input, beyond the chosen 2% check.
Original failure evidence is preserved; no threshold or LR was adjusted.

Final four-GPU CUDA toy qualification and five full-size 8-layer updates at
peak rates .01/.002 passed, including exact first/final replica hashes and
both spectral energy checks. Initial-model and train/validation manifests
match the completed 8-layer AdamW baseline. Qualification measured 1.0196
seconds per steady update, .7394 seconds per double spectral collection,
and 3.44 GiB peak allocated per rank. Forecast main invocation is about
31.5 minutes including allowances; actual elapsed time remains the outcome.

The scientific run started fresh with warmup50 under Slurm step2618555.9.
Both spectrum archives at steps1/25/50 contain all24 expected matrices and
512 singular values each. Twelve- and twenty-layer jobs perform their own
tiny and peak-rate full-size qualification before entering science.

Evidence:
- [CPU and numerical checks](../../logs/muon_spectra/qualification_20260925/CPU_QUALIFIED.json)
- [Distributed-owner check](../../logs/muon_spectra/qualification_20260925/OWNER_CPU_QUALIFIED.json)
- [Final unit log](../../logs/muon_spectra/qualification_20260925/final_unit_tests.log)
- [Full-size GPU check](../../logs/muon_spectra/depth8_w512_20260925_r2/QUALIFICATION_PASSED.json)
- [Final recipe](MUON_CASE.md), [independent discussion](MUON_REVIEW.md)
