MECHANISM: Seventeenth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,582-parameter design by quotienting positional row 7 will produce a 1,581-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the transformer’s LayerNorms.

INTENDED_EDIT: Compact positional rows 0–7 and the final nine rows, reconstruct their original ordering, and retain full-coordinate AdamW moments for positional optimization.

EVIDENCE: The current design achieved 99.53% accuracy after compacting rows 0–6 and the final nine; quotienting adjacent row 7 is the closest incremental application of the same repeatedly verified LayerNorm-null reduction.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first seven and final
        # nine positional rows.
        self.compact_pos_count = 16
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first eight and final
        # nine positional rows.
        self.compact_pos_count = 17
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:7], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[7:-9].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:8], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[8:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[7:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:8],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[8:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[7:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:8].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[8:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:7], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[7:-9].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:8], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[8:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE