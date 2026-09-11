MECHANISM: Three-dimensional tied lexical bottleneck

HYPOTHESIS: Reducing the tied token representation and logit projection from rank four to rank three will remove `VOCAB_SIZE + 2` learned parameters while retaining at least 99% accuracy, because the prior rank reduction improved accuracy to 99.98% without changing attention or MLP computation.

INTENDED_EDIT: Set `token_rank` to three and retain the adaptive pivot-gauge factorization for the resulting three-dimensional lexical space.

EVIDENCE: The rank-five model achieved 99.92% with 995 parameters, and the rank-four model achieved 99.98% with 881 parameters; this supports testing the next lexical-rank reduction while preserving the proven attention-routing structure.

<<<<<<< SEARCH
        # The final state reaches the logits only through a rank-four lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining four null directions in the LayerNorm bias.
        token_rank = 4
=======
        # The final state reaches the logits only through a rank-three lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining five null directions in the LayerNorm bias.
        token_rank = 3
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A rank-four product is invariant under an invertible change of latent
        # basis. Fix that 16-dimensional gauge by choosing the best-conditioned
        # four feature columns as an identity pivot and learning only the four
        # remaining columns.
=======
        # A rank-three product is invariant under an invertible change of latent
        # basis. Fix that nine-dimensional gauge by choosing the best-conditioned
        # three feature columns as an identity pivot and learning only the five
        # remaining columns.
>>>>>>> REPLACE