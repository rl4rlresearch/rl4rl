MECHANISM: Additional post-attention LayerNorm-bias absorption

HYPOTHESIS: Restricting `ln2.bias` to six orthonormal directions will reduce the model to 1,602 parameters while retaining at least 99% accuracy, because the omitted bias direction is exactly absorbable by the unrestricted `fc1.bias`.

INTENDED_EDIT: Remove one additional coordinate from `ln2.bias` and its reconstruction basis; the existing full-coordinate quotient optimizer already supports the reduced basis.

EVIDENCE: The 1,603-parameter design reached 99.91%, while adding either the next late positional quotient or row-2 positional quotient scored 94.63% and 98.71%. This tests a distinct redundancy whose effect can be absorbed directly before GELU by `fc1.bias`.

<<<<<<< SEARCH
        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
=======
        # Any fixed component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove the uniform direction and one
        # additional zero-sum direction while retaining six trainable modes.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
>>>>>>> REPLACE