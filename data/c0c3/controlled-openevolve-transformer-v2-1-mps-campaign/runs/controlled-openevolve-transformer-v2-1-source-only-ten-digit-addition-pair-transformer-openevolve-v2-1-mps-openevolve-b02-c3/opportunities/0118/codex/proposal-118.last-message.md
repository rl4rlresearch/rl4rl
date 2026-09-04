MECHANISM: Fourth-coordinate MLP output-bias fixing

HYPOTHESIS: A 672-parameter transformer will retain at least 99% accuracy because it preserves the qualified 673-parameter architecture while fixing only one additional zero-initialized MLP output-bias coordinate.

INTENDED_EDIT: Increase the MLP output projection’s fixed bias coefficients from one to four, reconstructing the qualified 673-parameter design and removing its fourth learned bias coordinate.

EVIDENCE: Successive MLP output-bias reductions achieved 99.97% at 675 parameters, 99.89% at 674, and 99.71% at 673 without altering the qualified attention-routing layout, directly motivating the next one-coordinate reduction.

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=1
        )
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE