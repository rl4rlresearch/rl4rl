MECHANISM: Bias-free final normalization

HYPOTHESIS: Removing the final LayerNorm bias will reduce parameters from 1,596 to 1,588 while retaining at least 99% accuracy.

INTENDED_EDIT: Keep the successful architecture and training schedule unchanged, but disable only the eight-element bias in `ln_f`.

EVIDENCE: The 1,596-parameter model reached 99.99% after removing affine parameters from both block LayerNorms, while the larger 36-parameter nullspace reparameterization fell to 97.76%; this motivates a conservative normalization-affine reduction.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE