# How much averaging does Muon need at batch 4M? (second-order audit, 2026-09-27)

Sequential (Gauss-Seidel) curvature corrections, like Newton scaling, help only on fresh gradients: in one step they add 71% for Muon on a fresh 4M gradient and lose 29% on the momentum. A staged optimizer would therefore have to run with little or no momentum. This cohort measures what low momentum costs plain Muon at 4M: Muon @0.014 with β 0.5 and β 0, plus a β 0.9 reference on the same GPU type. Earlier L40S runs: β 0.95 3.9524, 0.9 3.9218, 0.81 3.9437. A6000, 4 GPUs each, seed 260925.
