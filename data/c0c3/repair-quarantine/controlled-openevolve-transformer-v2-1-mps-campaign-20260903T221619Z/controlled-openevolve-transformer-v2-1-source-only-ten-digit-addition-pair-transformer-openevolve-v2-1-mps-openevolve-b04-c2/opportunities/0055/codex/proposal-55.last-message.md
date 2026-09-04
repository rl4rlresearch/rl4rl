MECHANISM: Two-column residual-output quotient plus eighth positional common-mode quotient

HYPOTHESIS: Extending the verified 1,607-parameter design by removing the LayerNorm-invariant common mode of one additional late positional row will produce a 1,606-parameter model with at least 99% accuracy.

INTENDED_EDIT: Quotient the first two attention output-projection columns as in Reference Design 1, then compact the final six rather than final five positional rows, with full-coordinate AdamW moments for both matrix and positional quotient parameters.

EVIDENCE: Reference Design 1 achieved 99.96% at 1,607 parameters by quotienting two projection columns, while the current design achieved 99.87% with seven positional common modes removed. Extending the already successful positional quotient is a distinct alternative to the third-column and cross-head projection allocations that failed at 1,606 parameters.

<<<<<<< SEARCH
            self.proj.weight.copy_(proj_weight)
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)
        self.qkv.weight = nn.Parameter(compact_weight)
=======
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)

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
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final six positions.
        self.compact_pos_count = 8
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
                (initialized_pos[:2], initialized_pos[-6:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-6].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
=======
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
=======
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_compact_columns, blk.attn.proj_bias_basis),
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
                    (full_update[:2], full_update[-6:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-6].reshape(-1),
                    )
                )
>>>>>>> REPLACE