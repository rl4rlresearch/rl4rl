MECHANISM: Absorb pre-MLP normalization scale into the first linear projection

HYPOTHESIS: Removing the eight learned `ln2` scale parameters will reduce the model from 1,628 to 1,620 parameters while retaining at least 99% accuracy, because with no `ln2` bias its elementwise scale can be represented exactly by rescaling the columns of `fc1.weight`.

INTENDED_EDIT: Disable all affine parameters in `ln2` while preserving the full `d_ff=12` MLP, attention architecture, initialization RNG stream, and training procedure.

EVIDENCE: The current 1,628-parameter model reached 99.77% after successfully removing `ln2` bias, while reducing feed-forward width failed badly; removing the remaining mathematically absorbable `ln2` scale tests a redundant parameterization without reducing MLP width.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE