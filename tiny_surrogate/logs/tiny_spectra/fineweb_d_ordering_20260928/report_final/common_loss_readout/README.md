# Fixed-recipe common-loss readout

Exploratory accounting from the saved full-development curves, with the final joint endpoint-selected LR fixed for each method. All thresholds from 4.0 to 9.5 in increments of 0.1 are retained in `results.json` and `all_crossings.csv`. No model inference or test scoring was performed.

The modest SoapMuon+PD advantage over TS is not consistently resolved by common-loss progress either. At loss 4.6, TS/SoapMuon+PD interpolated step ratios are 0.9833 and 1.0045 in the two seeds. At higher losses the ratios are modestly above one, but their observed crossing intervals mostly overlap or touch one. The complete grid, including censored targets, is authoritative; these examples are not a new decision threshold.

Ratios above one mean the later method reaches the same loss in fewer steps. Bounds reflect evaluation spacing only, not statistical confidence. No per-seed or per-threshold LR selection is used. D's failed primary ordering gates remain unchanged.
