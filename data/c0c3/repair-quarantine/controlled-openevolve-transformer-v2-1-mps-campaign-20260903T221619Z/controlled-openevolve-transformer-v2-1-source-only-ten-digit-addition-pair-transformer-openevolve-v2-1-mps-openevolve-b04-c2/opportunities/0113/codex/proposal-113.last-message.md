MECHANISM: Downstream-bias LayerNorm gauge quotient

HYPOTHESIS: Removing one additional `ln2` bias direction will reduce the verified 1,568-parameter model to 1,567 parameters while maintaining at least 99% accuracy, because any constant shift it induces before `fc1` is exactly absorbable by the unrestricted `fc1` bias.

INTENDED_EDIT: Restrict `ln2.bias` from seven to six orthonormal directions while leaving all other verified token, positional, and value-basis reductions unchanged.

EVIDENCE: The current 1,568-parameter design achieved 100% accuracy; unlike the failed deeper value rotations and query-key constraint, this reduction targets a representationally redundant affine direction immediately followed by a biased linear layer.

<<<<<<< SEARCH
        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        # Every constant post-normalization shift is absorbable by the
        # unrestricted fc1 bias. Remove one additional direction while
        # retaining six coordinates for nearly unchanged optimization.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE