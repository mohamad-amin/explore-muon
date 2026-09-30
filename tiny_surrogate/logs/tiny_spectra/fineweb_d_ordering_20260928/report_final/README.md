# Candidate D development ordering

The predeclared ordering/bracketing requirement fails; preserve and close this candidate.

| Method | Joint selected LR | Mean full NLL | Bracket closed |
|---|---:|---:|---|
| adamw | 0.0006 | 5.130811885681701 | True |
| muon | 0.01 | 4.564517547830024 | True |
| pd | 0.01 | 4.516786080322758 | True |
| ts | 0.01 | 4.495898233584496 | True |
| spd | 0.02 | 4.492208224917604 | False |

All model scores are development evidence. Both fixed seeds select one common learning rate per method; no favorable window subset or per-seed LR envelope is used. Every declared adjacent sign, material-gap requirement and bracket is reported in results.json. The complete surrogate goal is not yet qualified.
