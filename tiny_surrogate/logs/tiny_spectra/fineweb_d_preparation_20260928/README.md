# Candidate D data preparation

CPU-only Slurm job **2627148**, 8 CPUs and 32 GiB. The external preparation
limit is 7,200 seconds; Slurm allows one additional minute for termination and
exit recording. No GPU, model training, model inference, or test score is used.

Sources and configuration are frozen and verified by driver.py. The job must
finish the decoder, prior-exposure, role integrity, training-only tokenizer,
all-target evaluation and fresh-capacity gates before any GPU qualification.

- PLAN.json: fixed data and later conditional experiment rules.
- SOURCE_ROLE_AUDIT.json: scoped source history and public hash verification.
- CODE_DATA_REVIEW.json: independent code/data-contract review and retained failure.
- DECODER_SAMPLE_QUALIFICATION.json: bounded real-source round trips.
- source_manifest.json and frozen/: executed Python sources.
- submission.json, EXECUTION_*.json, PROCESS_EXIT.json and console_2627148.log:
  execution evidence. Check the live job before continuing or restarting.

Prepared data and stage receipts live in data/fineweb_d_20260928. A missing
PREPARATION_COMPLETE.json means the data are not yet qualified for training.
