MECHANISM: Rank-five tied lexical bottleneck

HYPOTHESIS: Reducing the gauge-fixed tied lexical representation from rank six to rank five will reduce learned parameters from 1,209 to 1,097 while retaining at least 99% accuracy, because the successful rank-six model reached 99.96% and the full-width attention and residual pathways remain unchanged.

INTENDED_EDIT: Use five-dimensional token codes, a fixed five-column identity chart, and a learned 5-by-3 lexical tail for both embeddings and logits.

EVIDENCE: The gauge-fixed rank-six lexical bottleneck achieved 99.96% accuracy at 1,223 parameters, and subsequent independent MLP compression retained 99.89%; lowering lexical rank is the most direct test of whether six latent token features contain excess capacity.

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