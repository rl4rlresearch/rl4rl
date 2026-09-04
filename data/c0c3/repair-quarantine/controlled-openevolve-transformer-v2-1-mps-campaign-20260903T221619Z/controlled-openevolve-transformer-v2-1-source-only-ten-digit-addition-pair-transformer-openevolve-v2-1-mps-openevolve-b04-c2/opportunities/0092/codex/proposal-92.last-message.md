MECHANISM: Alternate-row positional common-mode quotient beyond sensitive row 13

HYPOTHESIS: Compacting positional row 14 while leaving optimization-sensitive rows 7 and 13 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct skipped rows exactly; and retain full-coordinate AdamW moments.

EVIDENCE: The 1,577-parameter design compacting through row 12 achieved 99.78%, while compacting row 13 fell to 97.72%; previously, bypassing sensitive row 7 and compacting row 8 restored accuracy to 99.80%, motivating the analogous bypass to row 14.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first seven and final
        # nine positional rows.
        self.compact_pos_count = 16
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
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
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows while leaving sensitive rows 7 and
        # 13 unrestricted.
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
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[7:],
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
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[7:].reshape(-1),
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
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
>>>>>>> REPLACE