# Candidate D: separate momentum component and horizon controls

Base array **2627389** contains 20 new runs. Together with 12 identical completed runs, it forms a balanced comparison of Muon and SoapMuon+PD, batches 65,536 and 1,048,576, body momentum 0.8 and 0.9, body learning rates 0.01 and 0.02, and two paired development seeds. Qualified native numerical sources are unchanged.

After all base runs finish, choose one common control rate per method/batch by mean endpoint loss across both momentum values and both seeds. Then run all 24 declared controls: 16 at 128 updates and eight at twice the large-batch token horizon. Controls run regardless of the base scientific sign. Their rates cannot be retuned.

The primary momentum rule is fixed-rate and methodwise; joint per-momentum LR choices are secondary only. All two-seed/rate contrasts, failed criteria, crossing intervals and censored targets remain visible. This separate question is motivated by the replicated endpoint batch interaction; it does not change the failed original batch-rate-growth or five-method-ordering gates. No independent test panel is scored.

The full family has 44 new runs, forecast roughly 20–40 GPU-hours. Each task uses one 16 GB A4000 with no array throttle. All footprint remains in this isolated study. PLAN.json, PEER_DISCUSSION.json and LAUNCH_CHECK.json preserve the decision, source/configuration identities and prelaunch checks.

CPU continuation job **2627400** depends on successful completion of array2627389. It automatically performs the fixed base readout, launches the24 prescribed controls, and schedules their final CPU readout. Do not race it with manual stage launches. Subsequent handles will be in submission_edges.json and submission_final_analysis.json.
