# Tiny surrogate observation: fixed-bank reversal and transfer

Dated snapshot: 2026-09-28 14:25 CDT. Read-only with respect to the main branch;
stdlib archive reading only, no model evaluation, GPU, or job submission.
All 26 discovery runs (arrays2626585,2626631,2626637) were confirmed
COMPLETED/0:0 in sacct. Dynamics array2626669 had22 completed arms, two
running and remaining pending at the snapshot. Do not interpret its unfinished
sweep. The main allocations2567578/2618555 were RUNNING and untouched.

## What transferred

The discovery report preserves a substantial Muon→PD/SOAP-PD gain and a
growing endpoint SOAP-PD−Muon difference across batches2048/8192/32768:
−.02343/−.03464/−.08553 on full validation. The predeclared loss2.0 step
speedups are1.151/1.217/1.127. Therefore endpoint gain transferred but
monotone matched-loss rate gain is not established. Neither observation
establishes the entire surrogate phenomenon or a mechanism. The rates at
small/large batch were carried from8192 and are not yet tuned/replicated.

The matched training-window hashes and fixed4,194,304-token exposure
mean the batch comparison groups the same training sample stream differently.
It is not simply more distinct data in the large-batch arms. However the
small corpus is repeatedly exposed (4.178 corpus equivalents), and there are
2048/512/128 parameter updates. EMA and root-refresh clocks are in optimizer
steps in this harness, so their token scales change16× across the endpoints.
This differs from the large-study input-statistic clock audit and limits a
literal transfer of its clock explanation. Shared exposure does not match
optimizer age, clipping history, or schedule-in-token details automatically.
The missing controls are already acknowledged by the main report; this note
neither launches nor prescribes a new main branch.

## The TS/PD reversal can be localized using saved losses

At selected LR.016, beta.9, batch8192, seed20260928, TS−PD is−.001611 on
the fixed bank but+.003137 on full validation. Frozen evaluation code shows
the fixed bank is256 evenly spaced windows selected from the same871
non-overlapping128-character windows used for full validation. It is not a
separately collected validation corpus. `final_validation.pt` retains all871
per-window losses; `windows.pt` retains the exact fixed-bank starts.

`analyze_windows.py` decodes only the explicitly whitelisted one-dimensional
CPU tensor archive records with stdlib; no torch/model imports. It verifies
paired starts and identical training-window streams. Source hashes and the
complete numerical output are in`window_result.json`.

| subset | windows | mean TS−PD | SD across paired window differences | TS better windows |
|---|---:|---:|---:|---:|
| entire saved validation |871|+.00313644|.05820|437|
| fixed-bank windows within full evaluation |256|−.00161009|.05902|132|
| complement |615|+.00511224|.05778|305|

The archived full-pass bank subset reconstructs stored bank means within
3.53e−6(PD) and4.55e−6(TS), and the **contrast** within1.03e−6.
These small execution differences are recorded, not silently forced to zero;
they cannot explain a.00475 change in the contrast. Evaluating all871
windows changes the sign because of which retained windows are averaged.

Contiguous deciles also vary substantially: paired means range−.01052 to
+.02664. Deciles8/9 favor PD by+.02664/+.02150, whereas several earlier
blocks favor TS. Across all windows the median paired difference is−.000227
and almost exactly half favor each optimizer. The full mean is therefore a
weak aggregate direction amid large content-dependent variation at this
single endpoint. Windows are not independent randomized examples, so this
is descriptive heterogeneity, not a t-test or evidence for distinct latent
corpora. It does not establish a temporal endpoint fluctuation, a repeated-seed
ordering, or a semantic explanation for the blocks.

## Connection to earlier observer anomalies

The justified commonality with the body/aux bank-sign result is narrow:
small aggregate differences can change sign when the scoring sample changes,
and paired item-level data locate that change. The tiny example is an
endpoint-loss difference between two trained models; the earlier example
was a gradient-alignment counterfactual at fixed weights. They are not the
same statistic and no shared physical mechanism follows. The SNR issue was
an estimator bias under pure noise; there is no analogous bias established
here. The fixed-bank/full-split reversal alone does not support SNR, curvature
sampling, or a SOAP mechanism.

## Original execution failures retained

1. First archive read used the wrong parent depth and raised FileNotFoundError
   before reading any tensor. Corrected logs-root lookup.
2. An initial1e−6 bank-reconstruction assertion failed. Replaced that exact
   reproduction premise with explicit error reporting above. Full means
   independently match summary values within2.28e−7. No model or source
   results were modified to make a check pass.

Sources: `logs/tiny_spectra/shakespeare_discovery_20260928/report/README.md`,
`DISCOVERY_SPEEDUP.json`, selected PD/TS screen summaries and final-validation
archives, and the narrowly inspected frozen train.py evaluation/save logic.
No report on the unfinished momentum arms is attempted.
