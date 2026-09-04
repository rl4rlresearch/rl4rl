MECHANISM: Tail-localized positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Quotienting positional rows zero, one, and the final two rows will produce a 1,631-parameter model with at least 99% accuracy because the verified final-row quotient reached 99.99%, while the penultimate row has similarly limited causal influence compared with the failed row-two quotient.

INTENDED_EDIT: Add the penultimate positional row to the orthonormal zero-sum quotient and extend its reconstructed full-coordinate AdamW update accordingly.

EVIDENCE: The current 1,632-parameter design achieved 99.99% after quotienting the final row, whereas quotienting row two achieved only 98.97%; this motivates testing the adjacent penultimate row as the smallest tail-localized reduction.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final position. The final row has the shortest causal influence.
        self.compact_pos_count = 3
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final two positions, which have the shortest causal influence.
        self.compact_pos_count = 4
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-1:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-1].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-2:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-2].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-1:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-1].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-2:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-2].reshape(-1),
                    )
                )
>>>>>>> REPLACE