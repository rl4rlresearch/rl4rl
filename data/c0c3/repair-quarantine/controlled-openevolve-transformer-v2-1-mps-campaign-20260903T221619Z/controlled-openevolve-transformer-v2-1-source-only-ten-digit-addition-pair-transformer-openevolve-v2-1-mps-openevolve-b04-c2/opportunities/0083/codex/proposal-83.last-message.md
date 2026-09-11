MECHANISM: Fifteenth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,584-parameter design by quotienting positional row 5 will produce a 1,583-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first LayerNorm.

INTENDED_EDIT: Reproduce the qualified all-but-column-3 attention quotient, then compact positional rows 0–5 and the final nine rows while retaining full-coordinate AdamW moments.

EVIDENCE: The 1,584-parameter reference compacted positional rows 0–4 and the final nine rows with 99.83% accuracy; extending the identical exact quotient to adjacent row 5 is the closest incremental reduction.

<<<<<<< SEARCH
        # each coefficient made zero. Also quotient projection columns zero
        # through two and four through six, leaving column three unrestricted.
=======
        # each coefficient made zero. Also quotient every projection column
        # except optimization-sensitive column three.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6)]
            )
            remaining_proj_weight = torch.cat(
                (proj_weight[:, 3:4], proj_weight[:, 7:]), dim=1
            ).clone()
=======
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6, 7)]
            )
            remaining_proj_weight = proj_weight[:, 3:4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.proj.weight[:, :1],
                compact_columns[:, 3:],
                self.proj.weight[:, 1:],
            ),
            dim=1,
        )
=======
        proj_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.proj.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from the first two and final
        # nine positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 11
=======
        # Remove LayerNorm-invariant common modes from the first six and final
        # nine positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 15
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
                (initialized_pos[:6], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[6:-9].reshape(-1),
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
                compact_rows[:6],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[6:],
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
                        selected_full_grad[:6].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[6:].reshape(-1),
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
                    (full_update[:6], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[6:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE