MECHANISM: Head-local value gauge plus skip-sensitive-row positional common-mode quotient

HYPOTHESIS: Adding the qualified head-1 value rotation and compacting positional rows 12 and 14 while preserving rows 7 and 13 will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the verified ninth value-basis rotation, then compact positional rows 0–6, 8–12, 14, and the final nine rows with balanced reconstruction and full-coordinate AdamW updates.

EVIDENCE: The row-12 design with the added head-1 rotation achieved 99.99% accuracy at 1,576 parameters. Row 13 was optimization-sensitive, while row-14 attempts produced no accuracy measurement and therefore provide no contrary evidence.

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
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-11,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 20
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
                    initialized_pos[8:12],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[12:-9]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows. Keep sensitive rows 7 and 13
        # unrestricted, along with the remaining middle positions.
        self.compact_pos_count = 22
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
                    initialized_pos[14:15],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (
                    initialized_pos[7:8],
                    initialized_pos[13:14],
                    initialized_pos[15:-9],
                ),
                dim=0,
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:11],
                unrestricted_rows[1:],
                compact_rows[11:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:2],
                compact_rows[12:13],
                unrestricted_rows[2:],
                compact_rows[13:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:11].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[11:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:2].reshape(-1),
                        selected_full_grad[12:13].reshape(-1),
                        unrestricted_full_grad[2:].reshape(-1),
                        selected_full_grad[13:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:12],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[12:-9]), dim=0
                )
=======
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:13],
                        full_update[14:15],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (
                        full_update[7:8],
                        full_update[13:14],
                        full_update[15:-9],
                    ),
                    dim=0,
                )
>>>>>>> REPLACE