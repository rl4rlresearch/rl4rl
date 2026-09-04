MECHANISM: Orthonormal gauge fixing of attention output bias

HYPOTHESIS: Representing the attention output bias in a seven-dimensional orthonormal zero-mean basis will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because it removes the same exact LayerNorm-invariant degree of freedom as the failed fixed-coordinate parameterization without its poorly conditioned optimization geometry.

INTENDED_EDIT: Replace the eight-parameter attention output bias with seven learned coefficients mapped through a fixed orthonormal Helmert basis spanning the zero-mean subspace.

EVIDENCE: Fixing one output-bias coordinate at zero reached only 73.43%, despite removing an exact gauge degree of freedom; an orthonormal basis preserves the full bias equivalence classes while avoiding the asymmetric, ill-conditioned coordinates of that experiment.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        bias_basis = torch.zeros(d_model, d_model - 1)
        for j in range(d_model - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            bias_basis[: j + 1, j] = 1.0 / scale
            bias_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("proj_bias_basis", bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = self.proj(y) + self.proj_bias_basis @ self.proj_bias
        y = self.resid_drop(y)
>>>>>>> REPLACE