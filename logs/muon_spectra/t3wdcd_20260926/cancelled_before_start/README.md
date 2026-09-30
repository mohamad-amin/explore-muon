# Cancelled before start (2026-09-26 16:20 CDT)

These two A6000 arms were still queued (jobs 2624286 and 2624287, never started, no outputs) when the user asked to run
the queued arms on the g20 and priv-g14 allocations instead. Their configs were copied unchanged to
`PDgeo2_a0.125_wd0.1_cd0.9_t3recipe_s260925_ada` (g20, RTX 6000 Ada) and
`PDgeo2_a0.125_wd0.3_cd0.5_t3recipe_s260925_l40s` (priv-g14, L40S). The comparison with the A6000 center and the
A6000 Muon arms is therefore across GPU types for these two cells.
