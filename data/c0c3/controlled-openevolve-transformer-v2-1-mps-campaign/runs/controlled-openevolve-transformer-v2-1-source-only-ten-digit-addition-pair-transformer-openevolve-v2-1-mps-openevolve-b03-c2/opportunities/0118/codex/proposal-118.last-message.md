MECHANISM: Disjoint adjacent-pair final-LayerNorm gain sharing

HYPOTHESIS: Sharing coordinates two and three while retaining the verified leading-pair sharing will reduce the model from 1484 to 1483 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Store five learned final-LayerNorm gains and independently reuse the first two scalars for coordinate pairs zero/one and two/three.

EVIDENCE: Leading-pair gain sharing achieved 99.8% at 1484 parameters, while extending that same group to three coordinates collapsed accuracy; a separate adjacent pair tests a less restrictive, previously untested reduction.

<<<<<<< SEARCH
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 2))
=======
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight[:1].expand(2),
                self.ln_f_weight[1:],
                self.ln_f_weight.new_ones(1),
            )
        )
=======
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight[:1].expand(2),
                self.ln_f_weight[1:2].expand(2),
                self.ln_f_weight[2:],
                self.ln_f_weight.new_ones(1),
            )
        )
>>>>>>> REPLACE