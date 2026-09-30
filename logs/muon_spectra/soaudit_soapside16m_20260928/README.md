# Which SOAP side carries the large-batch gain? (second-order audit, 2026-09-28)

Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: which SOAP side carries the
large-batch gain?". S∘PD α ½ @0.028 at 16M (92 steps) with SOAP's basis on one side only: `soap_basis` left (output
side) and right (input side). References: S∘PD 4.5742 (Ada), PD 4.6711 (L40S). gpu partition, 4× RTX A6000 each.
Frozen from the live source (84 unit tests pass).

**Revision (04:0x CDT).** The gpu-partition jobs (2626259/2626260) were still pending when g20 and priv-g14 freed up. They were cancelled before starting. The arms were renamed from _a6000 to the hardware they actually run on (S_left on g20, Ada; S_right on priv-g14, L40S) and run through node_queue.
