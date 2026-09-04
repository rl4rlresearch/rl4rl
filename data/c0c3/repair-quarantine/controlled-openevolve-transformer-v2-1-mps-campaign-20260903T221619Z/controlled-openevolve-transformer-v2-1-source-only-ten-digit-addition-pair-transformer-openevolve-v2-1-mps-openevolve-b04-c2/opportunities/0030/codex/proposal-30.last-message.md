MECHANISM: Columnwise MLP residual common-mode quotient

HYPOTHESIS: Constraining every `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the verified 1,620-parameter model to 1,608 parameters while retaining at least 99% accuracy, because the removed component contributes only a token-local uniform residual shift erased by the final LayerNorm.

INTENDED_EDIT: Preserve full baseline initialization, project `fc2.weight` into compact zero-sum coordinates, reconstruct its eight output rows in the forward pass, and extend virtual full-coordinate AdamW updates to the compact matrix.

EVIDENCE: The current 1,620-parameter design achieved 99.95%, and its successful `fc2.bias` quotient removes the same function-null output direction. The previous columnwise attempt could not be verified rather than failing accuracy, motivating an isolated implementation on the robust current design.

<<<<<<< SEARCH
        # The following final LayerNorm removes a uniform residual-channel
        # shift, so parameterize this bias in an orthonormal zero-sum basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
=======
        # The following final LayerNorm removes every token-local uniform
        # residual-channel shift. Store the bias and each output column of
        # fc2 in an orthonormal zero-sum basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        weight = self.bias_basis @ self.fc2.weight
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
=======
        self.apply(self._init_weights)

        # Preserve the baseline full-matrix initialization and RNG
        # consumption, then discard only each fc2 column's function-null
        # uniform output component.
        with torch.no_grad():
            for blk in self.blocks:
                initialized_fc2 = blk.mlp.fc2.weight.detach()
                compact_fc2 = (
                    blk.mlp.bias_basis.transpose(0, 1) @ initialized_fc2
                )
                blk.mlp.fc2.weight = nn.Parameter(compact_fc2)

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
        )
=======
        for pair in (
            (blk.mlp.fc2.weight, blk.mlp.bias_basis),
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in gauge_params
    ]
=======
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
        }
        for param, basis in gauge_params
    ]
>>>>>>> REPLACE