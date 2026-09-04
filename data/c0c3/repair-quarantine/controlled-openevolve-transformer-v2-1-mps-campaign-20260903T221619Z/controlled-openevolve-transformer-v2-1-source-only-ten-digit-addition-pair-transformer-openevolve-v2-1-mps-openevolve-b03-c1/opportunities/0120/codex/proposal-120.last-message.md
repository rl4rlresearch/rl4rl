MECHANISM: Fixed spherical lexical readout

HYPOTHESIS: Replacing the final learned eight-coordinate LayerNorm gain with non-affine normalization will reduce the model from 606 to 598 parameters while retaining at least 99% accuracy, because it preserves the initialized function exactly and the learned rank-four token projection can define the decoder geometry.

INTENDED_EDIT: Make the final LayerNorm non-affine while retaining its four-dimensional learned lexical bias.

EVIDENCE: The 606-parameter model achieved 99.89% accuracy with both internal LayerNorms already non-affine. The failed 595-parameter experiment shows the MLP decoder directions are load-bearing, so this cleanly tests a different assumption: whether a separate learned coordinate-wise metric is needed after the transformer when the rank-four lexical projection already supplies learned readout directions.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # The final state reaches the logits only through a rank-four lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining four null directions in the LayerNorm bias.
        token_rank = 4
        self.ln_f_token_bias = nn.Parameter(
            self.ln_f.bias.new_zeros(token_rank)
        )
        self.ln_f.bias = None
=======
        # Decode a spherically normalized residual state directly through the
        # learned rank-four lexical projection. The removed coordinate-wise
        # gain initializes to the identity, so this preserves the initialized
        # model function while testing whether a second learned decoder metric
        # is necessary.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )

        # The final state reaches the logits only through a rank-four lexical
        # projection. Learn its observable bias directly in that space.
        token_rank = 4
        self.ln_f_token_bias = nn.Parameter(
            self.token_emb.weight.new_zeros(token_rank)
        )
>>>>>>> REPLACE