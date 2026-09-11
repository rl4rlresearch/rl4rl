MECHANISM: Index-mapped positional common-mode quotient

HYPOTHESIS: Starting from the qualified 1,564-parameter LayerNorm design, quotienting positional row 14 through a shared explicit index partition will produce 1,563 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply the qualified six-coordinate `ln1.bias` and four-coordinate `ln2.bias`, compact positional row 14, and use the same index buffers for initialization, inference reconstruction, gradients, and full-coordinate AdamW updates.

EVIDENCE: The 1,564-parameter LayerNorm design achieved 99.86% accuracy, and earlier positional common-mode compaction retained at least 99%; repeated row-14 attempts yielded no contrary accuracy measurement, while their verification failures motivate replacing fragile slice bookkeeping with one shared bijective index map.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # Retain the qualified six-coordinate pre-attention bias quotient.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain five coordinates to test the next
        # progressive quotient beyond the qualified six-coordinate design.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 3))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 3)
        for col in range(cfg.d_model - 3):
=======
        # Retain the qualified four-coordinate post-normalization bias
        # quotient; omitted shifts are absorbable by the unrestricted fc1 bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_token = self.token_emb.weight.detach()
            token_mean = initialized_token.mean(dim=0)
            compact_token = token_basis.transpose(0, 1) @ initialized_token

            initialized_pos = self.pos_emb.weight.detach() + token_mean
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )

        self.lm_head = None
        self.token_emb.weight = nn.Parameter(compact_token)
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows. A single explicit partition keeps
        # initialization, inference, and optimizer reconstruction bijective.
        compact_pos_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
        compact_pos_mask[:7] = True
        compact_pos_mask[8:13] = True
        compact_pos_mask[14] = True
        compact_pos_mask[-9:] = True
        compact_pos_indices = torch.nonzero(
            compact_pos_mask, as_tuple=False
        ).squeeze(1)
        unrestricted_pos_indices = torch.nonzero(
            ~compact_pos_mask, as_tuple=False
        ).squeeze(1)
        self.compact_pos_count = int(compact_pos_indices.numel())

        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_token = self.token_emb.weight.detach()
            token_mean = initialized_token.mean(dim=0)
            compact_token = token_basis.transpose(0, 1) @ initialized_token

            initialized_pos = self.pos_emb.weight.detach() + token_mean
            selected_pos = initialized_pos.index_select(
                0, compact_pos_indices
            )
            unrestricted_pos = initialized_pos.index_select(
                0, unrestricted_pos_indices
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )

        self.lm_head = None
        self.token_emb.weight = nn.Parameter(compact_token)
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
        self.register_buffer(
            "compact_pos_indices", compact_pos_indices, persistent=False
        )
        self.register_buffer(
            "unrestricted_pos_indices",
            unrestricted_pos_indices,
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:],
                compact_rows[12:],
            ),
            dim=0,
        )
=======
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
        pos_weight = compact_rows.new_zeros(
            self.cfg.max_seq_len, self.cfg.d_model
        )
        pos_weight = pos_weight.index_copy(
            0, self.compact_pos_indices, compact_rows
        )
        pos_weight = pos_weight.index_copy(
            0, self.unrestricted_pos_indices, unrestricted_rows
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
=======
    compact_pos_count = model.compact_pos_count
    compact_pos_indices = model.compact_pos_indices
    unrestricted_pos_indices = model.unrestricted_pos_indices
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
                    )
                )
=======
                full_grad_rows = pos_basis.new_zeros(
                    model_cfg.max_seq_len, model_cfg.d_model
                )
                full_grad_rows.index_copy_(
                    0, compact_pos_indices, selected_full_grad
                )
                full_grad_rows.index_copy_(
                    0, unrestricted_pos_indices, unrestricted_full_grad
                )
                full_grad = full_grad_rows.reshape(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:13],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[13:-9]), dim=0
                )
=======
                selected_update = full_update.index_select(
                    0, compact_pos_indices
                )
                unrestricted_update = full_update.index_select(
                    0, unrestricted_pos_indices
                )
>>>>>>> REPLACE