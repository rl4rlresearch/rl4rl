MECHANISM: Alternate early-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,603-parameter design by quotienting positional row 2 instead of the failed tenth late row will produce a 1,602-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified two-column attention-output quotient and first-two/final-nine positional compaction, then additionally compact positional row 2 with consistent reconstruction and full-coordinate AdamW moments.

EVIDENCE: The first two and final nine positional quotients achieved 99.91% at 1,603 parameters, while adding the next late row collapsed to 94.63%; reallocating the same exact LayerNorm-invariant quotient to the adjacent early row tests whether the failure was specific to that late-row optimization path.

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
            # Uniform output-row components add only common-mode residual
            # shifts erased by subsequent LayerNorms. Quotient the first two
            # projection columns independently by that exact null direction.
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
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
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
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first three and final
        # nine positional rows.
        self.compact_pos_count = 12
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:3], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[3:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:2],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:3],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[3:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_first_column, blk.attn.proj_bias_basis),
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
                full_grad = torch.cat(
                    (
                        selected_full_grad[:2].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[2:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:3].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[3:].reshape(-1),
                    )
                )
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
                    (full_update[:3], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[3:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE