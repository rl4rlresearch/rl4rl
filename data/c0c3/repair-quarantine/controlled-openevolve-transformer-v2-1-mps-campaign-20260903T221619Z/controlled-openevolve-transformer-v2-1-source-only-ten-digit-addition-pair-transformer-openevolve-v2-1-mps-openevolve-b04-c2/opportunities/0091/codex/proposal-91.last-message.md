MECHANISM: Alternate-row positional common-mode quotient through row 13

HYPOTHESIS: Extending the verified 1,577-parameter design by quotienting positional row 13 while retaining optimization-sensitive row 7 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–6, 8–13, and the final nine rows; reconstruct their original ordering; and retain full-coordinate AdamW moments.

EVIDENCE: The 1,577-parameter reference achieved 99.78% accuracy after compacting through row 12 while skipping sensitive row 7; row 13 is the closest incremental application of the same repeatedly successful LayerNorm-null quotient.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, row 8, and
        # the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 17
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-13,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 22
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:14],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[14:-9]), dim=0
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
                compact_rows[7:13],
                unrestricted_rows[1:],
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
                        selected_full_grad[7:13].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
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
                        full_update[8:14],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[14:-9]), dim=0
                )
>>>>>>> REPLACE