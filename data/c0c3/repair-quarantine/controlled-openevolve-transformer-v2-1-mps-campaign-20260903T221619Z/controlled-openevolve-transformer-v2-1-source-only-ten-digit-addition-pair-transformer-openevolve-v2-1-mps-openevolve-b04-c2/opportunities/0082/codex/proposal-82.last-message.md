MECHANISM: Fourteenth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,585-parameter design by quotienting positional row 4 will produce a 1,584-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by LayerNorm.

INTENDED_EDIT: Compact positional rows 0–4 and the final nine rows, reconstruct their original ordering, and update the full-coordinate AdamW projection for the expanded 14-row quotient.

EVIDENCE: The immediately preceding design compacted rows 0–3 and the final nine rows, achieving 99.44% accuracy with 1,585 parameters; row 4 is the closest incremental application of the same LayerNorm-null quotient.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final nine positions.
        self.compact_pos_count = 11
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first five and final
        # nine positional rows.
        self.compact_pos_count = 14
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-9].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:5], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[5:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:2],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:5],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[5:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:2].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[2:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:5].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[5:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-9].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:5], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[5:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE