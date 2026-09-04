MECHANISM: Two-column MLP residual-output common-mode quotient

HYPOTHESIS: Compacting the first two `fc2` output columns on the qualified 1,602-parameter design will yield 1,601 parameters while retaining at least 99% accuracy, because each omitted uniform output component is independently erased by the final LayerNorm.

INTENDED_EDIT: Reproduce the qualified first-two/final-nine positional compaction, then parameterize the first two MLP output columns in the seven-dimensional zero-sum basis and train them with full-coordinate AdamW moments.

EVIDENCE: Compacting the first `fc2` column produced 99.60% accuracy at 1,602 parameters; the second column has the same exact LayerNorm-null uniform direction, making it the closest incremental reduction.

<<<<<<< SEARCH
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
=======
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def compact_output_columns(self) -> None:
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first two columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :2]
            )
            remaining_weight = weight[:, 2:].clone()
        self.fc2_compact_columns = nn.Parameter(compact_columns)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        compact_columns = self.bias_basis @ self.fc2_compact_columns
        weight = torch.cat((compact_columns, self.fc2.weight), dim=1)
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()

        # Remove LayerNorm-invariant common modes from the first two and final
        # five positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 7
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Remove LayerNorm-invariant common modes from the first two and final
        # nine positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 11
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-5:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-5].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
=======
        for blk in model.blocks
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.mlp.fc2_compact_columns, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE