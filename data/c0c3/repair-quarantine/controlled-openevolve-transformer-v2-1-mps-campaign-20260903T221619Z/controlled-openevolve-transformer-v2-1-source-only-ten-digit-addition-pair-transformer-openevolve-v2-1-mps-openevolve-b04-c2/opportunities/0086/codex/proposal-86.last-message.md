MECHANISM: Alternate-row positional common-mode quotient

HYPOTHESIS: Compacting positional rows 0–6 and 8 while leaving optimization-sensitive row 7 unrestricted will produce a 1,581-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the positional quotient to rows 6 and 8, reconstruct the skipped row ordering exactly, and retain full-coordinate AdamW moments.

EVIDENCE: Row 6 compaction achieved 99.53% at 1,582 parameters, while adding adjacent row 7 fell to 98.83%; bypassing that sensitive coordinate mirrors the successful alternate-column attention strategy.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from the first six and final
        # nine positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 15
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, row 8, and
        # the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 17
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:6], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[6:-9].reshape(-1),
                )
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:6],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[6:],
            ),
            dim=0,
        )
=======
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:6].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[6:].reshape(-1),
                    )
                )
=======
                unrestricted_full_grad = pos_param.grad[
                    pos_compact_size:
                ].view(
                    model_cfg.max_seq_len - compact_pos_count,
                    model_cfg.d_model,
                )
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:8].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[8:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:6], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[6:-9].reshape(-1),
                    )
                )
=======
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
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
>>>>>>> REPLACE