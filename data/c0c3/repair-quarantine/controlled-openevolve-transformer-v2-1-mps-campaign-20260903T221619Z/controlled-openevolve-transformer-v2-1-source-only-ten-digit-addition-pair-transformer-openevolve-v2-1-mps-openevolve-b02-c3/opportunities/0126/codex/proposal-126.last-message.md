MECHANISM: Attention-output quotient-bias fixing

HYPOTHESIS: A 670-parameter transformer will retain at least 99% accuracy because fixing one attention-output bias coordinate preserves the qualified 671-parameter model’s learned routing, lexical representation, MLP bias, and successful second-head query gauge.

INTENDED_EDIT: Fix the final quotient-space bias coefficient of the attention output projection at zero, removing one learned parameter.

EVIDENCE: The 671-parameter design achieved 99.22%; further query-bias fixing, lexical tying, final-normalization fixing, and removal of the last MLP bias failed, motivating an isolated reduction in the previously untested attention-output bias.

<<<<<<< SEARCH
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.proj = QuotientOutputLinear(
            d_model, d_model, fixed_bias_coeffs=1
        )
>>>>>>> REPLACE