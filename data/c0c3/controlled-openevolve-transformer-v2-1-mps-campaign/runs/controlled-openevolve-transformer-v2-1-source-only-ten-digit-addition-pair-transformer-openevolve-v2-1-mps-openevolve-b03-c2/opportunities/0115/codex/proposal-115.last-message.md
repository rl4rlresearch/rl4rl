MECHANISM: Leading-pair final-LayerNorm bias sharing

HYPOTHESIS: Sharing the first two final-LayerNorm biases will reduce the qualified 1484-parameter model to 1483 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store seven final-LayerNorm biases and reuse the first bias for coordinates zero and one, leaving the successful gain sharing unchanged.

EVIDENCE: Leading-pair gain sharing achieved 99.8% at 1484 parameters, while extending that gain group to three coordinates collapsed accuracy; sharing the corresponding two zero-initialized biases tests a distinct local constraint without further restricting the gains.

<<<<<<< SEARCH
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))
=======
        self.ln_f_bias = nn.Parameter(
            torch.zeros(cfg.d_model - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x * ln_f_weight + self.ln_f_bias
=======
        ln_f_bias = torch.cat(
            (
                self.ln_f_bias[:1].expand(2),
                self.ln_f_bias[1:],
            )
        )
        x = x * ln_f_weight + ln_f_bias
>>>>>>> REPLACE