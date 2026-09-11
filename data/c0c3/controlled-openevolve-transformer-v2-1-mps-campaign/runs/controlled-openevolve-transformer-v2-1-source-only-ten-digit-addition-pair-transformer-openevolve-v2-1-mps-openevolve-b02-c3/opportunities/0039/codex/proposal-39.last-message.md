MECHANISM: Fixed positional-amplitude gauge

HYPOTHESIS: Fixing the sinusoidal position scale at its initialization value on the qualified affine-free pre-MLP design will produce a 1,038-parameter transformer with at least 99% accuracy, because token and residual-branch weights can learn the required relative scaling.

INTENDED_EDIT: Remove all four pre-MLP LayerNorm bias coefficients and replace the learned positional-scale scalar with a fixed nonpersistent buffer.

EVIDENCE: The affine-free pre-MLP reference achieved 99.94% accuracy with 1,039 parameters; fixing the remaining positional-scale scalar at its existing initialization is the narrowest next reduction and preserves the successful initial signal balance.

<<<<<<< SEARCH
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_scale = nn.Parameter(torch.tensor(0.02))
=======
        self.register_buffer("pos_scale", torch.tensor(0.02), persistent=False)
>>>>>>> REPLACE