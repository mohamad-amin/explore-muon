# Why the output projection matters

This qualification was identified after the completed functional value probe,
before the new saved-weight output-projection contractions. It changes what
the earlier activation-amplitude observations identify, not their numbers.

The model has no normalization on V. An invertible block-diagonal S that mixes
features within each head gives an exact symmetry:

    Wv' = S Wv,   Wo' = Wo S^-1.

Attention mixing commutes with S within each head, so the whole attention
branch output is unchanged for every input. Yet ||Wv mu||², its fraction of
value energy, and its mean-versus-centered gain can change greatly under this
symmetry. These are real coordinates learned by the optimizer, but they do
not by themselves identify an invariant functional route.

`gauge_identity.py` gives a2D example: mean-versus-centered V gain changes
from1 to.0198 and value mean fraction from.9802 to.495 while attention output
is identical to floating-point precision. This is a possibility proof, not
a claim that all real optimizer differences are gauge transformations.

The constant residual component Wo Wv mu and the measured mean Wo mu_z are
invariant under this V/O symmetry. Compute those from saved weights and
moments before interpreting the learned route. Their absolute sizes can
still depend on other residual-stream scaling symmetries across models.
The archive lacks full O-input covariance orientation, so do not invent an
invariant centered-output denominator from its eigenvalues alone.

The completed value functional probe used full-model logits and predictive
GN, so those local responses are already function-space quantities. A fixed
gauge transform applied consistently to Wv,Wo and Dv leaves them unchanged.
It still evaluated V changes with O fixed, not the complete simultaneous V/O
step. The actual combined constant-route displacement at fixed mu is

    (Wo+Do)(Wv+Dv)mu − Wo Wv mu
      = Wo Dv mu + Do Wv mu + Do Dv mu.

Keep all three terms when using it as the next robustness check. This does
not score the moving input mean or the full layer's actual loss change.
