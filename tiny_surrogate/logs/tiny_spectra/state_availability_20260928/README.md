# Saved-state operator comparison: availability failure

A fresh independent review proposed one falsifiable question: does SOAP make
substantially less directional correction beyond PD in the tiny models than in
the successful reference regime? The comparison requires the same stored
momentum, cached input roots, SOAP bases, and second moments at each state.

All three long-C SOAP-PD endpoints retain 24 momentum matrices, 24 SOAP states,
24 cached input roots, and 24 input-statistic modules. The two reference 4M
SOAP-PD checkpoints retain 48 momentum matrices but no cached input roots,
SOAP bases, or SOAP second moments. Their kept input-statistic sidecars retain
input means/covariances/EMA weights only. The checked reference kept checkpoints
use the ordinary optimizer state-dictionary schema as well.

The proposed comparison is therefore unidentifiable from these artifacts. No
directional correction, covariance participation ratio, denominator spread,
model loss, or gradient-noise statistic was calculated. Missing historical
state was not reconstructed. This result does not support selecting a larger
model or different corpus to repair the failed ranking.

See availability.json for paths, checkpoint/sidecar hashes, state fields,
serialization compatibility failures, and interpretive limits. All reference
artifacts were read only; all new files remain in tiny_surrogate. No GPU was
allocated and neither confirmation panel was accessed.
