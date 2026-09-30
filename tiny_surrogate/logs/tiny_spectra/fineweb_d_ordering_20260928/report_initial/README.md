# Candidate D development ordering

Run only the declared boundary checks before finalizing ordering.

| Method | Joint selected LR | Mean full NLL | Bracket closed |
|---|---:|---:|---|
| adamw | 0.0006 | 5.130811885681701 | False |
| muon | 0.01 | 4.564517547830024 | True |
| pd | 0.01 | 4.516786080322758 | True |
| ts | 0.01 | 4.495898233584496 | False |
| spd | 0.01 | 4.49297639299296 | False |

All model scores are development evidence. Both fixed seeds select one common learning rate per method; no favorable window subset or per-seed LR envelope is used. Every declared adjacent sign, material-gap requirement and bracket is reported in results.json. The complete surrogate goal is not yet qualified.

Naming: **SOAP-PD** means **SoapMuon + PD**, configured as `spd`: PD input geometry, SOAP-style preconditioning in those coordinates, and Muon orthogonalization.
