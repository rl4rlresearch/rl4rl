MECHANISM: Columnwise MLP residual common-mode quotient

HYPOTHESIS: Constraining every `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the model from 1,618 to 1,606 parameters while retaining at least 99% accuracy, because each removed component produces only a token-local uniform residual shift erased by subsequent LayerNorms.

INTENDED_EDIT: Store `fc2.weight` as a 7-by-12 compact matrix, reconstruct its eight output coordinates during inference, preserve baseline initialization, and train it with virtual full-coordinate AdamW moments.

EVIDENCE: The 1,618-parameter design achieved 99.94% while already quotienting the uniform component of `fc2.bias`; applying the same verified function-null output direction independently to every `fc2.weight` column is the closest structural extension.

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
        # The following final LayerNorm removes every uniform residual-channel
        # shift, so parameterize both the bias and each weight column in an
        # orthonormal zero-sum output basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def compact_output_weight(self) -> None:
        with torch.no_grad():
            self.fc2.weight = nn.Parameter(
                self.bias_basis.transpose(0, 1) @ self.fc2.weight.detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        weight = self.bias_basis @ self.fc2.weight
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
=======
        self.apply(self._init_weights)

        # Preserve full-matrix initialization and RNG consumption before
        # removing the function-null common output mode of every fc2 column.
        for block in self.blocks:
            block.mlp.compact_output_weight()

        # Preserve baseline initialization and RNG consumption, then remove
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
=======
        for pair in (
            (blk.mlp.fc2.weight, blk.mlp.bias_basis),
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
>>>>>>> REPLACE

<<<<<<< SEARCH
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
=======
            "exp_avg": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
>>>>>>> REPLACE