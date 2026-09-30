# Fixed-recipe batch comparison

Development batch-component gate passed: **False**.
D's five-method ordering remains failed. No test panel was scored. This result does not complete the surrogate goal.

Common post-warmup threshold range: [5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6]. Bounds reflect crossing resolution, not statistical confidence.

| Seed | Muon LR | Growth in mean log step ratio | Resolution bounds | All gates pass |
|---|---:|---:|---|---|
| 20261001 | 0.01 | 0.03373922723075544 | [-0.07858521791569169, 0.1366376766174532] | False |
| 20261001 | 0.02 | 0.05375236879983698 | [-0.05183159040348245, 0.16284191189772554] | False |
| 20261002 | 0.01 | 0.046464572258729755 | [-0.06346589146709578, 0.15084189346605895] | False |
| 20261002 | 0.02 | 0.0496897675242991 | [-0.04863669492348435, 0.1645019780863105] | False |

Both fixed Muon controls, both seeds, all thresholds, censored targets and all endpoint document groups are retained in results.json. No envelope or favorable checkpoint/threshold selection is used. No automatic follow-up sweep is authorized.


All 12 new runs completed successfully, using **7.7983 GPU-hours**. The endpoint interaction is reproducible across both seeds and both fixed Muon controls:

| Muon LR | Small-batch mean gap | Midpoint mean gap | Large-batch mean gap |
|---|---:|---:|---:|
| 0.01 | +0.01990 | -0.07231 | -0.21341 |
| 0.02 | +0.01034 | -0.08470 | -0.22492 |

Gaps are SoapMuon+PD minus Muon NLL; negative favors SoapMuon+PD. The growing endpoint advantage does not establish the prespecified common-loss growth criterion. All four conservative resolution intervals for that growth include zero. Against Muon 0.01, the interpolated midpoint advantage also slightly exceeds the large-batch advantage in both seeds. This result is therefore recorded as **unqualified**, with endpoint interaction supported and common-loss growth unresolved.

See [the plot](batch_comparison.png) or [the PDF](batch_comparison.pdf). All original thresholds, censoring and paired trajectories remain in results.json and all_crossings.csv.
