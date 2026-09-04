MECHANISM: Single-column MLP residual common-mode quotient

HYPOTHESIS: Constraining only the first `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the model to 1,618 parameters while retaining at least 99% accuracy, because the removed activation-dependent uniform residual shift is erased by the final LayerNorm.

INTENDED_EDIT: Compact one `fc2.weight` column after baseline initialization, reconstruct the full matrix during inference, and train it with projected full-coordinate AdamW moments.

EVIDENCE: The 1,619-parameter current design achieved 99.85%, while quotienting all twelve `fc2.weight` columns at once fell to 38.54%; applying the same function-null quotient to exactly one column tests whether the failure came from the simultaneous twelve-direction optimization change.

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
=======
        # The following final LayerNorm removes a uniform residual-channel
        # shift, so parameterize this bias in an orthonormal zero-sum basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        # A uniform component of any fc2 column contributes only a token-local
        # common-mode residual shift. Quotient just the first column while
        # retaining unrestricted coordinates for every other column.
        self.d_model = d_model
        self.d_ff = d_ff
        weight_basis = torch.zeros(d_model * d_ff, d_model * d_ff - 1)
        first_col_indices = torch.arange(d_model) * d_ff
        weight_basis[first_col_indices, : d_model - 1] = basis
        compact_index = d_model - 1
        for row in range(d_model):
            for col in range(1, d_ff):
                weight_basis[row * d_ff + col, compact_index] = 1.0
                compact_index += 1
        self.register_buffer("weight_basis", weight_basis, persistent=False)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        weight = (self.weight_basis @ self.fc2.weight).view(
            self.d_model, self.d_ff
        )
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
=======
        self.apply(self._init_weights)

        # Preserve the full baseline initialization and RNG consumption before
        # projecting the first fc2 column into its compact coordinates.
        with torch.no_grad():
            for blk in self.blocks:
                initialized_weight = blk.mlp.fc2.weight.detach().reshape(-1)
                compact_weight = (
                    blk.mlp.weight_basis.transpose(0, 1) @ initialized_weight
                )
                blk.mlp.fc2.weight = nn.Parameter(compact_weight)

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
=======
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.mlp.fc2.weight, blk.mlp.weight_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
>>>>>>> REPLACE