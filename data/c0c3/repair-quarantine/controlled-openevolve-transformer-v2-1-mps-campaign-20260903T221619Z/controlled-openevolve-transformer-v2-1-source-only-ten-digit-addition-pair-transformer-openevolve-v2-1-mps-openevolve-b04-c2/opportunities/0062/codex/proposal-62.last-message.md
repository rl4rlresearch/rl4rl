MECHANISM: MLP residual-output common-mode quotient

HYPOTHESIS: Compacting one `fc2` output column on top of the qualified 1,603-parameter design will yield 1,602 parameters while retaining at least 99% accuracy, because its uniform residual-channel component is exactly erased by the final LayerNorm.

INTENDED_EDIT: Reproduce the qualified first-two/final-nine positional compaction, then parameterize the first MLP output column in the existing seven-dimensional zero-sum basis and optimize it with full-coordinate AdamW moments.

EVIDENCE: Reference Design 3 achieved 99.91% at 1,603 parameters, while additional positional and `ln2.bias` restrictions failed; the successful attention-output common-mode quotients motivate testing the analogous, previously untested exact quotient in the MLP output path.

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

    def compact_output_column(self) -> None:
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first column in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_column = self.bias_basis.transpose(0, 1) @ weight[:, 0]
            remaining_weight = weight[:, 1:].clone()
        self.fc2_first_column = nn.Parameter(compact_column)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        first_column = (
            self.bias_basis @ self.fc2_first_column
        ).unsqueeze(1)
        weight = torch.cat((first_column, self.fc2.weight), dim=1)
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final eight positions.
        self.compact_pos_count = 10
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_column()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final nine positions.
        self.compact_pos_count = 11
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-8:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-8].reshape(-1),
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
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
=======
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.mlp.fc2_first_column, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-8:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-8].reshape(-1),
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