MECHANISM: Sixteenth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,583-parameter design by quotienting positional row 6 will produce a 1,582-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first and final LayerNorms.

INTENDED_EDIT: Reproduce the qualified all-but-column-3 attention quotient, compact positional rows 0–6 and the final nine rows, and retain full-coordinate AdamW moments.

EVIDENCE: The 1,583-parameter reference achieved 99.90% accuracy after compacting attention columns 0, 1, 2, and 4–7 plus positional rows 0–5 and the final nine; row 6 is the closest incremental application of the same exact LayerNorm-null quotient.

<<<<<<< SEARCH
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient projection columns zero through two and columns four
        # and five, leaving optimization-sensitive column three unrestricted.
=======
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient every projection column except optimization-sensitive
        # column three.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5)]
            )
            remaining_proj_weight = torch.cat(
                (proj_weight[:, 3:4], proj_weight[:, 6:]), dim=1
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
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final nine positions.
        self.compact_pos_count = 11
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
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
                compact_rows[:7],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[7:],
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
                        selected_full_grad[:7].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[7:].reshape(-1),
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
                    (full_update[:7], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[7:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE