# Candidate D optimizer ordering

30 initial native-code runs: five methods, three fixed LRs, two paired seeds. Each array task uses one 16 GB A4000; there is no array throttle or arbitrary aggregate compute cap. Slurm schedules available qualified GPUs. Full-development NLL chooses one LR per method jointly across seeds. Up to ten predeclared edge arms are conditional on a complete initial readout. Every test panel remains sealed.

Expected cost: 12.41 GPU-hours initially, 16.55 with all edges. Each job has an ordinary one-hour walltime for an expected 17–30-minute run. No automatic retries. Frozen numerical sources are copied unchanged from the successful 21-update qualification. The earlier eight-hour planning-limit failure and graph fidelity failure remain preserved.

## Initial screen readout

All30initial arms completed with exit0 and passed the full artifact readout. Actual allocation time was18.263GPU-hours. At the jointly selected rates, the desired five-method ordering holds in both development seeds. Mean NLL is AdamW5.130812, Muon4.564518, PD4.516786, TS4.495898, SOAP-PD4.492976. SOAP-PD’s mean advantage over TS is0.002922, below the predeclared0.005material-gap gate. AdamW, TS and SOAP-PD still have boundary minima. The six originally allowed checks are submitted asarray2627283: AdamW0.0003, TS0.02 andSOAP-PD0.02 on both seeds. No ordering qualification or test-set claim is made yet.

Naming: **SOAP-PD** means **SoapMuon + PD**, configured as `spd`: PD input geometry, SOAP-style preconditioning in those coordinates, and Muon orthogonalization.

## Final36-run result

All36runs completed successfully and passed the full artifact checks, consuming22.1128GPU-hours. The best joint mean NLLs remain ordered AdamW5.130812, Muon4.564518, PD4.516786, TS4.495898, SoapMuon+PD4.492208. Joint selected LRs are0.0006/0.01/0.01/0.01/0.02.

The robust-ordering gate fails. SoapMuon+PD−TS is+0.00311697 in seed20261001 and−0.01049699 in20261002; its mean advantage0.00369001 is below0.005. Its selected0.02LR remains an upper-bound winner. AdamW, Muon, PD andTS brackets are closed. The original shared0.01comparison remains recorded, but it does not replace the predeclared joint-rate selection.

The candidate is closed without qualification. No further LR, architecture, corpus, power or horizon repair follows automatically. All three test panels remain unscored, and the full batch/momentum/independent-confirmation goal remains unfulfilled. See `CANDIDATE_D_VERDICT.json` and `report_final/`.
