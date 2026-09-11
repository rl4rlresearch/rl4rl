MECHANISM: Incremental two-direction pre-MLP bias quotient

HYPOTHESIS: Removing one additional direction from `ln2.bias` will reduce the model to 1,619 parameters while retaining at least 99% accuracy, because the one-direction quotient achieved 99.95% and is the smallest incremental step toward the eight-direction removal that failed.

INTENDED_EDIT: Represent `ln2.bias` in a six-dimensional orthonormal subspace and retain its reconstructed full-coordinate AdamW updates.

EVIDENCE: The verified 1,620-parameter model removed one `ln2.bias` direction at 99.95% accuracy, while removing all eight bias coordinates fell to 90.52%; testing exactly one additional direction most directly locates the compression boundary.

<<<<<<< SEARCH
        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
=======
        # Every post-normalization bias direction is absorbable by the
        # unrestricted fc1 bias. Incrementally remove a second direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
>>>>>>> REPLACE