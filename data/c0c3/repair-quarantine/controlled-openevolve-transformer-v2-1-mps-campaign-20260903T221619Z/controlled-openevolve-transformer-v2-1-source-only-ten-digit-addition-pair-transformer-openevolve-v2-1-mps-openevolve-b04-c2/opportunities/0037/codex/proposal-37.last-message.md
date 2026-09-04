MECHANISM: Split-coordinate positional common-mode quotient

HYPOTHESIS: Compacting the sixth-from-last positional row with native compact-coordinate AdamW while preserving the verified seven rows’ full-coordinate updates will produce 1,616 parameters and at least 99% accuracy.

INTENDED_EDIT: Store the additional late positional row as a separate seven-parameter zero-sum vector, optimize it normally, and retain the qualified optimizer behavior for all existing compact rows.

EVIDENCE: The seven-row quotient achieved 100% at 1,617 parameters, while adding this same eighth row with reconstructed full-coordinate AdamW fell to 88.94%; isolating the new row under compact-coordinate AdamW directly tests whether optimizer geometry caused that failure.

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
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Preserve baseline initialization and RNG consumption. Keep the
        # verified first two and final five compact rows together, while
        # parameterizing the sixth-from-last row separately so it can use
        # native compact-coordinate AdamW updates.
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
                    initialized_pos[2:-6].reshape(-1),
                )
            )
            extra_compact_pos = initialized_pos[-6] @ pos_basis
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.extra_compact_pos = nn.Parameter(extra_compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
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
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        extra_compact_row = (
            self.extra_compact_pos.unsqueeze(0)
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows[:2],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count - 1,
                    self.cfg.d_model,
                ),
                extra_compact_row,
                compact_rows[2:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }
=======
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(
            (model_cfg.max_seq_len - 1) * model_cfg.d_model
        ),
        "exp_avg_sq": pos_basis.new_zeros(
            (model_cfg.max_seq_len - 1) * model_cfg.d_model
        ),
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len, model_cfg.d_model)
=======
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len - 1, model_cfg.d_model)
>>>>>>> REPLACE