MECHANISM: Four-dimensional tied lexical bottleneck

HYPOTHESIS: Reducing the tied token representation and output projection from rank five to rank four will remove `VOCAB_SIZE` learned parameters while retaining at least 99% accuracy, because the rank-five model achieved 99.92% and this change leaves its proven attention-routing and MLP mechanisms intact.

INTENDED_EDIT: Set `token_rank` to four and retain the existing adaptive pivot-gauge factorization for the resulting four-dimensional lexical space.

EVIDENCE: The rank-five lexical bottleneck reached 99.92% with 995 parameters, whereas altering attention scoring collapsed accuracy to 30.33%; the next informative compression is therefore another lexical-rank reduction without perturbing attention.

<<<<<<< SEARCH
        # The final state reaches the logits only through a rank-five lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining three null directions in the LayerNorm bias.
        token_rank = 5
=======
        # The final state reaches the logits only through a rank-four lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining four null directions in the LayerNorm bias.
        token_rank = 4
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A rank-five product is invariant under an invertible change of latent
        # basis. Fix that 25-dimensional gauge by choosing the best-conditioned
        # five feature columns as an identity pivot and learning only the three
        # remaining columns.
=======
        # A rank-four product is invariant under an invertible change of latent
        # basis. Fix that 16-dimensional gauge by choosing the best-conditioned
        # four feature columns as an identity pivot and learning only the four
        # remaining columns.
>>>>>>> REPLACE