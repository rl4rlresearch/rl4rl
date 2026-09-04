MECHANISM: Affine-free pre-MLP normalization

HYPOTHESIS: Removing all three pre-MLP LayerNorm bias coefficients will produce a 1,039-parameter transformer with at least 99% accuracy because their constant contribution is exactly representable by `fc1.bias`.

INTENDED_EDIT: Replace the three-direction pre-MLP LayerNorm with an affine-free LayerNorm while preserving the qualified architecture and training procedure.

EVIDENCE: The one-direction 1,040-parameter design achieved 99.81% accuracy after every preceding incremental bias reduction qualified; eliminating its final redundant direction is the narrowest remaining parameter reduction, although earlier complete removal showed optimization risk.

<<<<<<< SEARCH
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE