MECHANISM: Alternate-row positional common-mode quotient through row 12

HYPOTHESIS: Extending the verified 1,578-parameter design by quotienting positional row 12 while retaining optimization-sensitive row 7 unrestricted will produce a 1,577-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–6, 8–12, and the final nine rows; reconstruct their original ordering; and retain full-coordinate AdamW moments.

EVIDENCE: The 1,578-parameter reference compacted rows 0–6, 8–11, and the final nine rows while achieving 99.52% accuracy; row 12 is the closest incremental application of the same exact LayerNorm-null quotient.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-9,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 18
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
                    initialized_pos[8:10],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[10:-9]), dim=0
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:9],
                unrestricted_rows[1:],
                compact_rows[9:],
            ),
            dim=0,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:9].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[9:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:10],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[10:-9]), dim=0
                )
=======
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
>>>>>>> REPLACE