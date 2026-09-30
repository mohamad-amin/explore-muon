# Prior art and the limited contribution of this analysis

Primary sources checked 2026-09-28. This is not an exhaustive novelty search.

- [Eschenhagen et al., K-FAC for modern neural architectures](https://arxiv.org/html/2311.00636v2#A2.SS3.SSS1)
  treats weight-sharing and simplified self-attention. Its analysis shows
  why standard expand/reduce factorization assumptions can fail even in
  simplified attention. The broader premise that routing changes curvature
  and the grouping of shared linear operations matters is established prior
  art. Its simplified example omits softmax; our conditional value-map
  identity allows fixed softmax attention but does not remove its statistical
  dependence on inputs when expectations are taken.
- [Seung, Lee and Ko, mean activation curvature](https://link.springer.com/article/10.1007/s10115-026-02781-7)
  uses an attention-weighted mean activation for the value projection in a
  rank-one curvature approximation. Attention-aware value statistics and
  mean-based preconditioners are therefore not a new optimizer concept here.
  That method is distinct from the full pooled-input second moment and the
  temporal error covariance in the identity examined in this folder.

The project-specific contribution is narrower: derive the predicted
coefficients in the **archive's empirical sequence-centering convention**,
separate b from b/a, retain the general downstream-covariance term, and check
the resulting necessary relation against every saved value fit. The analytic
qualification and archive inconsistency do not establish a better optimizer,
novel prior art, population rejection, or a causal mechanism of the momentum
training gains. No external implementation was installed or executed.
