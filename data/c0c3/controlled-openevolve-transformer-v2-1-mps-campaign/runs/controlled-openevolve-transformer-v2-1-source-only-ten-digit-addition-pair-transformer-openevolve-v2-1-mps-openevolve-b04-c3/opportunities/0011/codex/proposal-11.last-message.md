MECHANISM: Parameter-free RMS output normalization

HYPOTHESIS: Replacing the affine final LayerNorm with parameter-free RMS normalization will retain at least 99% accuracy after 10,000 training steps while reducing learned parameters from 1,612 to 1,596.

INTENDED_EDIT: Use the existing parameter-free RMSNorm for the model’s final normalization, removing its eight learned scales and eight learned biases.

EVIDENCE: The current design replaced both affine pre-LayerNorms with RMSNorm, removed 32 parameters, and still achieved 99.98% accuracy at 9,999 steps; this directly motivates testing the same normalization simplification at the output.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = RMSNorm(cfg.d_model)
>>>>>>> REPLACE