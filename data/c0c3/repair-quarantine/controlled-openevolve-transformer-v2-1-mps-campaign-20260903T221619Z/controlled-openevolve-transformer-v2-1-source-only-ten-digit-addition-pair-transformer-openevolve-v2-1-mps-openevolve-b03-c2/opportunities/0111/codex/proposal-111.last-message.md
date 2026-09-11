MECHANISM: Terminal LayerNorm gain tied to the fixed scale anchor

HYPOTHESIS: Fixing the seventh gain equal to the already-fixed eighth gain will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store six learned final-LayerNorm gains and reconstruct the final two adjacent gains as the fixed unit scale.

EVIDENCE: Sharing the preceding pair of adjacent learned gains reached 98.91%, the closest verified 1484 result, while a nonadjacent pair reached only 95.17%; this tests the remaining adjacent terminal pair without coupling two learned coordinates.

<<<<<<< SEARCH
        # A common positive scale of this affine transform changes only the
        # global logit temperature, not autoregressive argmax decoding.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))
=======
        # A common positive scale of this affine transform changes only the
        # global logit temperature, not autoregressive argmax decoding. Tie
        # the adjacent terminal gain to that fixed unit-scale anchor.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 2))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
=======
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(2),
            )
        )
>>>>>>> REPLACE