# Independent transmission-source review

2026-09-29. Reviewed `TRANSMISSION_PROTOCOL.md` and
`analyze_transmission.py` before execution. No tensors were loaded, no
contractions or synthetic qualifications executed, and no models called.

**Verdict:** the proposed architectural tangent and pooled factorization
are implemented correctly. Two small defensive checks are recommended
before execution: handle zero-GN contrast denominators explicitly, and
assert exact state/panel/schema coverage rather than relying on total counts.
These do not alter the scientific panel or require new measurements.

## Algebra and units

The analytic derivative matches the specified tanh-approximate GELU:

    u=sqrt(2/pi)*(z+.044715*z³),
    phi'(z)=.5*(1+tanh(u))
             +.5*z*(1−tanh(u)²)*sqrt(2/pi)*(1+3*.044715*z²).

The fixed synthetic autograd comparison is against that same approximation,
in FP64, including both saturated tails. The source keeps the original
preactivation Z fixed, as required for this derivative at the checkpoint.

`F.linear(x,D)` produces the individual preactivation kick; multiplying by
phi'(Z) and applying `F.linear(...,W_down)` produces the immediate residual-
branch tangent. No transpose, activation placement, or residual-identity
term is missing. Up-input x is unchanged by an isolated up-weight intervention.
This does not reconstruct finite joint changes, and the script never claims
to do so.

Pre/post squared norms sum the last feature axis, leaving [4,512] token
energies. Their means and GN use exactly the same context/position selection:
bank0 rows0–1, bank1 rows2–3, all4 rows0–3. Thus the factorization

    GN/Epre = (Epost/Epre)*(GN/Epost)

uses ratios of pooled quantities rather than means of tokenwise ratios.
This is the correct whole-context descriptive comparison at early layers,
where later attention couples locations. The existing slope and true H
are retained without folding H−GN into this GN explanation.

## Provenance and coverage

The down matrix comes from the producer's base checkpoint record, not its
next-weight record. Its bytes are hashed against the exact model-tensor
hash saved by the producer; checkpoint size/mtime are checked before and
after reading. That qualifies the current down matrix used for the tangent.
The recomputed preactivation energy is checked against the saved kick radius
for every context, token and direction, connecting the loaded X/D to the
producer's measurements.

Add explicit equality of checkpoint step, state metadata step, and panel
summary step, along with exact method/block/direction/scales and shape checks;
alternatively make successful prior atlas validation an explicit prerequisite.
The intended sequence already runs this after both existing analyses, but
the current script itself only checks producer completion and final row
counts. Counts alone do not prove that every unique panel is present once.

The intended output is270 metric rows and216 control-contrast rows, covering
18 panels, five directions and three aggregates. The token-energy archive
retains all four contexts, so separate context values remain recoverable;
the CSVs themselves contain banks and all4, not individual-context rows.
Original X/Z/D remain in the producer archive. Source hashes and input file
identity should remain alongside this derived result, as stipulated by the
protocol.

## Zero denominators and interpretation

The current `gn>=0` check allows exact zero, while contrast construction
divides by the control's GN-normalized quantities. Preserve zero raw GN and
energies. For a zero denominator, emit an undefined ratio with its reason,
without inserting an arbitrary positive floor or dropping the panel. The
same principle applies if a structural metric annihilates a direction.
This is a robustness issue, not evidence that any archived direction is zero.

For defined ratios, the before-transmission contrast equals the transmission
contrast times the remaining downstream contrast. Displaying all three
factors is informative, but a smaller remaining contrast is descriptive
support for this fixed architectural decomposition, not proof of causal
optimizer mediation or finite-radius isotropy. The plot title and protocol
keep those limits clear. One-thread CPU tensor operations and plotting add
no network evaluation, training, or optimizer reconstruction.
