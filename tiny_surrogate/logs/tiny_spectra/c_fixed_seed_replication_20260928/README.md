# Fixed-recipe development seed replication

Exactly four arms: TS and SOAP-PD at seeds 20260929 and 20260930. All recipe
fields are copied unchanged from the scale diagnostic's LR 0.016 arms. Only the
seed and run identifier change. See PLAN.json and PROTOCOL.md. Complete all
four arms before interpreting outcomes.

The primary criterion is a SOAP-PD minus TS full-development endpoint difference
of at most -0.005 in each fresh pair. The original selection seed is excluded
from the primary estimate. Retain every trajectory and document group.

Failure closes this fixed-recipe lead without repairs. Success only replicates
this development contrast. The failed scale-grid gate and all full qualification
requirements remain unchanged. No test scoring is authorized by this stage.

Maximum two concurrent 11 GB RTX 2080 Ti workers; 15 minutes per arm;
one GPU-hour aggregate limit. Submitted as Slurm array **2626974**.
Verify live state; do not resubmit.
