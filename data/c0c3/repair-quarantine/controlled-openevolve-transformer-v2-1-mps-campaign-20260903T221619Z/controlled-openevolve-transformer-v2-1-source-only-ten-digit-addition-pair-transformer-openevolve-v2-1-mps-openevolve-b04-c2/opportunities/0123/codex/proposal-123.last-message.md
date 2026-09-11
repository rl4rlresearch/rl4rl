MECHANISM: Index-safe positional common-mode quotient

HYPOTHESIS: Starting from the qualified 1,564-parameter LayerNorm design, removing positional row 14’s LayerNorm-invariant common mode with a single bijective index map will yield 1,563 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply the qualified six-coordinate `ln1.bias` and four-coordinate `ln2.bias`, then compact positional row 14 using shared indices for initialization, inference reconstruction, gradients, and full-coordinate AdamW updates.

EVIDENCE: The 1,564-parameter LayerNorm design achieved 99.86% accuracy, and prior positional common-mode quotients remained above 99%; repeated row-14 attempts produced no contrary accuracy result but motivate eliminating fragile slice bookkeeping.

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the query bias and
        # output-projection bias; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
=======
        # Constant pre-attention shifts are absorbable by the query and
        # output-projection biases; key shifts cancel from attention softmax.
        # Retain the six coordinates qualified at 1,564 parameters.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Every constant post-normalization shift is absorbable by the
        # unrestricted fc1 bias. Remove one additional direction while
        # retaining six coordinates for nearly unchanged optimization.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
=======
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain the four coordinates qualified at
        # 1,564 parameters.
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
        # row 14, and the final nine rows. One explicit partition controls
        # initialization, reconstruction, and optimizer-coordinate mapping.
        compact_pos_indices = torch.tensor(
            [
                *range(7),
                *range(8, 13),
                14,
                *range(cfg.max_seq_len - 9, cfg.max_seq_len),
            ],
            dtype=torch.long,
        )
        unrestricted_mask = torch.ones(cfg.max_seq_len, dtype=torch.bool)
        unrestricted_mask[compact_pos_indices] = False
        unrestricted_pos_indices = torch.arange(cfg.max_seq_len)[
            unrestricted_mask
        ]
        stored_pos_indices = torch.cat(
            (compact_pos_indices, unrestricted_pos_indices)
        )
        pos_inverse_indices = torch.argsort(stored_pos_indices)
        self.compact_pos_count = compact_pos_indices.numel()

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
            selected_pos = initialized_pos[compact_pos_indices]
            unrestricted_pos = initialized_pos[unrestricted_pos_indices]
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
        self.register_buffer(
            "pos_inverse_indices", pos_inverse_indices, persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        stored_rows = torch.cat((compact_rows, unrestricted_rows), dim=0)
        pos_weight = stored_rows[self.pos_inverse_indices]
>>>>>>> REPLACE

<<<<<<< SEARCH
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
=======
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    compact_pos_indices = model.compact_pos_indices
    unrestricted_pos_indices = model.unrestricted_pos_indices
    pos_inverse_indices = model.pos_inverse_indices
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
                stored_full_grad = torch.cat(
                    (selected_full_grad, unrestricted_full_grad), dim=0
                )
                full_grad = stored_full_grad[pos_inverse_indices].reshape(-1)
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
                selected_update = full_update[compact_pos_indices]
                unrestricted_update = full_update[unrestricted_pos_indices]
>>>>>>> REPLACE