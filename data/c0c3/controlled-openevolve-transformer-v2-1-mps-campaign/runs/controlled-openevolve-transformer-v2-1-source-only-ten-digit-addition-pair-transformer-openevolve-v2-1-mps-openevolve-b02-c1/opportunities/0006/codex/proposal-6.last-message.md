MECHANISM: MLP pre-normalization affine absorption

HYPOTHESIS: Removing the second LayerNorm’s 16 affine parameters will reduce the model from 1,628 to 1,612 parameters while retaining at least 99% accuracy, because its learned scale and shift can be absorbed exactly into the following MLP input weights and bias.

INTENDED_EDIT: Make the LayerNorm immediately before the MLP non-affine; leave all attention, embeddings, residual width, and MLP capacity unchanged.

EVIDENCE: The 1,628-parameter shared-bias design reached 99.99% accuracy, whereas reducing positional or token representation capacity failed. This motivates eliminating an exact affine reparameterization rather than removing representational dimensions.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE