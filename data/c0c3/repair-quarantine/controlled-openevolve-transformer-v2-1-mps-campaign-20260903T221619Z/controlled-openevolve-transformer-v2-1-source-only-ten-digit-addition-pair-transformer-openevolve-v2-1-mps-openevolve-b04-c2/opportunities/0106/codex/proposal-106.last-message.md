MECHANISM: Indexed row-14 positional common-mode quotient atop the qualified value gauge

HYPOTHESIS: Adding the verified head-1 value rotation and quotienting positional row 14 with index-based reconstruction will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the qualified ninth value-basis rotation, compact positional row 14, and use explicit row-index maps for inference and full-coordinate AdamW updates.

EVIDENCE: The head-1 rotation achieved 99.99% accuracy at 1,576 parameters. Row-12 positional compaction achieved 99.78%; row 13 was sensitive, while row-14 attempts produced no accuracy result because they could not be verified.

<<<<<<< SEARCH
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
=======
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
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
            initialized_pos = self.pos_emb.weight.detach()
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
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows. Explicit index maps preserve the
        # original ordering while keeping sensitive rows 7 and 13 unrestricted.
        compact_row_set = (
            set(range(7))
            | set(range(8, 13))
            | {14}
            | set(range(cfg.max_seq_len - 9, cfg.max_seq_len))
        )
        compact_pos_indices = torch.tensor(
            sorted(compact_row_set), dtype=torch.long
        )
        unrestricted_pos_indices = torch.tensor(
            [
                row
                for row in range(cfg.max_seq_len)
                if row not in compact_row_set
            ],
            dtype=torch.long,
        )
        self.compact_pos_count = compact_pos_indices.numel()
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
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
        self.pos_emb.weight = nn.Parameter(compact_pos)
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
                full_grad = pos_basis.new_zeros(
                    model_cfg.max_seq_len, model_cfg.d_model
                )
                full_grad.index_copy_(
                    0, model.compact_pos_indices, selected_full_grad
                )
                full_grad.index_copy_(
                    0,
                    model.unrestricted_pos_indices,
                    unrestricted_full_grad,
                )
                full_grad = full_grad.reshape(-1)
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
                    0, model.compact_pos_indices
                )
                unrestricted_update = full_update.index_select(
                    0, model.unrestricted_pos_indices
                )
>>>>>>> REPLACE