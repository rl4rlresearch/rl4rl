MECHANISM: Dual common-mode quotient for attention output bias and one late positional row

HYPOTHESIS: Adding the exact two quotients from qualified Reference Design 2 will reduce the current model from 1,619 to 1,617 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the attention projection bias in a seven-dimensional zero-sum basis and extend positional compaction from the first two plus final four rows to the first two plus final five rows, preserving full-coordinate AdamW moments.

EVIDENCE: Reference Design 2 used this exact 1,617-parameter structure and achieved 100% accuracy; the subsequent 1,616-parameter failure indicates this qualified boundary is safer than compacting another positional row.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)

        # A uniform projection-bias shift is erased by both subsequent
        # LayerNorms, so store only its zero-sum component.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-5:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-5].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
=======
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
>>>>>>> REPLACE