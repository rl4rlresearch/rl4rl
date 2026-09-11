MECHANISM: Thirteenth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,586-parameter design by quotienting positional row 3 will produce a 1,585-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by LayerNorm.

INTENDED_EDIT: Compact positional rows 0–3 and the final nine rows, reconstruct their original ordering, and retain full-coordinate AdamW moments for positional optimization.

EVIDENCE: The current design achieved 99.91% accuracy at 1,586 parameters after extending the positional quotient through row 2; applying the same exact common-mode quotient to adjacent row 3 is the closest incremental reduction.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first three and final
        # nine positional rows.
        self.compact_pos_count = 12
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first four and final
        # nine positional rows.
        self.compact_pos_count = 13
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:3], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[3:-9].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:4], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[4:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:3],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[3:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:4],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[4:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:3].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[3:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:4].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[4:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:3], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[3:-9].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:4], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[4:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE