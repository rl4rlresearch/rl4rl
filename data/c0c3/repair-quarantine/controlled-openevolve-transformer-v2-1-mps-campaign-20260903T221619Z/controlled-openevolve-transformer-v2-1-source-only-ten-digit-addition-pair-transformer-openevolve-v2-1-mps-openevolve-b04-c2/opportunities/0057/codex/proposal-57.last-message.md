MECHANISM: Tenth positional common-mode quotient

HYPOTHESIS: Extending the verified 1,605-parameter design by quotienting the next late positional row will produce a 1,604-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact the first two and final eight positional rows, updating initialization and full-coordinate AdamW projection consistently.

EVIDENCE: Successive late-row extensions reached 99.94% at 1,606 parameters and 99.62% at 1,605 parameters; this tests the closest remaining incremental reduction using the same LayerNorm-invariant direction.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final seven positions.
        self.compact_pos_count = 9
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final eight positions.
        self.compact_pos_count = 10
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-7:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-7].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-8:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-8].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-7:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-7].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-8:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-8].reshape(-1),
                    )
                )
>>>>>>> REPLACE