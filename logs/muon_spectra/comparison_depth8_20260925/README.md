# Eight-layer Muon / AdamW comparison

Both runs completed all 1,469 updates and 1,539,870,720 tokens, with
60 snapshots of all 24 matrices. Initial weights, frozen model
code, data manifests, token exposure, batch and validation bank agree.

Final validation NLL: **Muon 3.720991**, **AdamW 3.879250**;
Muon minus AdamW is **-0.158259 nats/token**. Perplexities are
41.305 and 48.388
(14.64% lower for the Muon recipe in this one run).

At the endpoint, the leading singular direction contains
5.96–17.00%
of AdamW update energy across the 24 matrices, versus
0.209–0.389%
for Muon's post-NS update. Muon's momentum remains concentrated before NS:
15.26–86.01%.
The post-NS flattening is expected from orthogonalization; it does not by itself
establish why validation loss improved. The shared normalized reference is
1/sqrt(512)=0.04419 for a matrix with 512 equal singular values.

This is a comparison of optimizer recipes at one seed. AdamW peak LR is 0.0012;
Muon body peak LR is 0.01 with auxiliary AdamW peak LR 0.002. The auxiliary rate and
effective decay therefore also change. AdamW used one RTX6000 Ada GPU and
Muon used four; elapsed-time differences do not isolate optimizer efficiency.
Both use our same local GPT initialization, not a verified paper initializer.

Figures (PNG/PDF): [validation](validation.png), [median trajectories](median_focus.png),
[all 24 medians](median_all24.png), [final spectra](final_spectra_focus.png),
[all 24 final spectra](final_spectra_all24.png), [leading energy](top_energy_all24.png),
[adaptive step magnitude](step_norm_all24.png). [Complete PDF](comparison.pdf).

Raw exports: [validation](validation.csv), [all sampled spectral metrics](spectral_metrics.csv),
[all final singular values](final_spectra.csv), [summary and provenance](summary.json).
The summary preserves the predeclared 1100–1300 pre-cooldown window. No formal
stabilization criterion or uncertainty band is introduced after seeing the curves.

Sources: `/share/data/dl-theory/amin/projects/last_layer/logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924` and `/share/data/dl-theory/amin/projects/last_layer/logs/muon_spectra/depth8_w512_20260925_r2/scientific`. Figures reuse saved exact singular
values; no training or new SVD computation is performed. `compare.py` is copied
here with its source hash and can be rerun through the research module.
