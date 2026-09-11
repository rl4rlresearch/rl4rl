MECHANISM: Leading-pair terminal LayerNorm gain sharing

HYPOTHESIS: Sharing the first two final-LayerNorm gains will reduce the verified 1489-parameter model to 1488 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store six learned final-LayerNorm gains and reuse the first scalar for coordinates zero and one, while retaining the fixed final gain.

EVIDENCE: The current design achieved 99.99% accuracy at 1489 parameters, and the identical leading-pair gain sharing previously achieved 99.8% in Reference Design 3; this tests that qualified reduction on the higher-margin current attention geometry.

<<<<<<< SEARCH
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
=======
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 2))
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
                self.ln_f_weight[:1].expand(2),
                self.ln_f_weight[1:],
                self.ln_f_weight.new_ones(1),
            )
        )
>>>>>>> REPLACE