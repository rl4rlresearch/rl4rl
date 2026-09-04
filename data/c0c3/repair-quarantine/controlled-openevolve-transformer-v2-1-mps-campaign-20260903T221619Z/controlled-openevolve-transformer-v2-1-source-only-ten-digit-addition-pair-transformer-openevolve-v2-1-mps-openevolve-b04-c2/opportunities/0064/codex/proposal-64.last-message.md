MECHANISM: Three-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the verified 1,601-parameter design by quotienting the third `fc2` output column will produce a 1,600-parameter model with at least 99% accuracy, because its uniform residual-channel component is independently erased by the final LayerNorm.

INTENDED_EDIT: Reproduce the qualified two-column attention quotient and first-two/final-nine positional compaction, then parameterize the first three MLP output columns in the seven-dimensional zero-sum basis with full-coordinate AdamW moments.

EVIDENCE: Compacting the first two `fc2` columns achieved 99.80% accuracy at 1,601 parameters; extending the same exact LayerNorm-null quotient to the adjacent third MLP column is the closest incremental reduction.

<<<<<<< SEARCH
            # Any uniform output-row component of a projection column only
            # adds a per-token common-mode residual shift. Subsequent
            # LayerNorms erase it, so quotient one column by that direction.
            compact_proj_column = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, 0]
            )
            remaining_proj_weight = proj_weight[:, 1:].clone()
        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_first_column = nn.Parameter(compact_proj_column)
        self.proj.weight = nn.Parameter(remaining_proj_weight)
=======
            # Uniform output-row components add only per-token common-mode
            # residual shifts, which subsequent LayerNorms erase. Quotient
            # the first two projection columns by that direction.
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()
        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_compact_columns = nn.Parameter(compact_proj_columns)
        self.proj.weight = nn.Parameter(remaining_proj_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = self.proj_bias_basis @ self.proj.bias
        proj_first_column = (
            self.proj_bias_basis @ self.proj_first_column
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (proj_first_column, self.proj.weight), dim=1
        )
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
>>>>>>> REPLACE

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
        # erases. Store the first three columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :3]
            )
            remaining_weight = weight[:, 3:].clone()
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

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final nine positions.
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
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_first_column, blk.attn.proj_bias_basis),
        )
=======
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.mlp.fc2_compact_columns, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_compact_columns, blk.attn.proj_bias_basis),
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
                (basis.size(0), *param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0), *param.shape[1:])
            ),
        }
        for param, basis in gauge_params
    ]
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