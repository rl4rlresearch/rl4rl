MECHANISM: Skip-sensitive-row positional common-mode quotient

HYPOTHESIS: Quotienting positional row 14 while leaving sensitive rows 7 and 13 unrestricted will reduce the model to 1,576 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct both skipped rows exactly; and preserve full-coordinate AdamW moments.

EVIDENCE: The 1,577-parameter design achieved 99.78% accuracy through row 12, while row 13 compaction failed at 97.72%; earlier, bypassing sensitive row 7 and compacting row 8 restored accuracy to 99.80%, motivating the analogous correctly reconstructed bypass to row 14.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows while leaving sensitive rows 7 and
        # 13 unrestricted.
        self.compact_pos_count = 22
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
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
                        full_update[8:13],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[13:-9]), dim=0
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