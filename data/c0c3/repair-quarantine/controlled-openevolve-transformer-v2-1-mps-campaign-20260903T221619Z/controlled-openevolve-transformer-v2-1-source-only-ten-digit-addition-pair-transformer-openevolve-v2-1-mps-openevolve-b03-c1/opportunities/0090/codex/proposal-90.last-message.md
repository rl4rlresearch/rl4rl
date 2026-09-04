MECHANISM: Five-dimensional tied lexical bottleneck

HYPOTHESIS: Reducing the tied token representation and logit projection from rank six to rank five will lower the model from 1,107 to 1,095 learned parameters while retaining at least 99% accuracy, because digit identity and output classification do not intrinsically require six independent lexical coordinates.

INTENDED_EDIT: Use a rank-five learned token code with a five-dimensional output bottleneck, leaving the proven attention-routing and MLP mechanisms unchanged.

EVIDENCE: The 1,107-parameter rank-six model achieved 99.91%, while the direct bilinear attention refactor collapsed to 30.33%, showing that attention-score structure is load-bearing. This patch instead challenges the shared assumption that six lexical dimensions are necessary and cleanly tests a different representational bottleneck without perturbing routing.

<<<<<<< SEARCH
        # The final state reaches the logits only through a rank-six lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining two null directions in the LayerNorm bias.
        token_rank = 6
=======
        # The final state reaches the logits only through a rank-five lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining three null directions in the LayerNorm bias.
        token_rank = 5
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A rank-six product is invariant under an invertible change of latent
        # basis. Fix that 36-dimensional gauge by choosing the best-conditioned
        # six feature columns as an identity pivot and learning only the two
        # remaining columns.
=======
        # A rank-five product is invariant under an invertible change of latent
        # basis. Fix that 25-dimensional gauge by choosing the best-conditioned
        # five feature columns as an identity pivot and learning only the three
        # remaining columns.
>>>>>>> REPLACE