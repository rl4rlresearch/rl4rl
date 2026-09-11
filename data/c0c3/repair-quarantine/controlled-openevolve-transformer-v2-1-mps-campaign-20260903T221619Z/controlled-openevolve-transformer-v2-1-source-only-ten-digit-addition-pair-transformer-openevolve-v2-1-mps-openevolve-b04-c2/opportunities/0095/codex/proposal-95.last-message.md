MECHANISM: Skip-sensitive-row positional common-mode quotient

HYPOTHESIS: Compacting positional row 14 while retaining sensitive rows 7 and 13 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with explicit reconstruction of both skipped rows during inference and full-coordinate AdamW optimization.

EVIDENCE: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction failed at 97.72%, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. Prior row-14 submissions were unverifiable and provide no contrary accuracy result.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, row 8, and
        # the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 17
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
                    initialized_pos[8:9],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[9:-9]), dim=0
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
        # unrestricted.
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
                compact_rows[7:8],
                unrestricted_rows[1:],
                compact_rows[8:],
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
                        selected_full_grad[7:8].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[8:].reshape(-1),
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
                        full_update[8:9],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[9:-9]), dim=0
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