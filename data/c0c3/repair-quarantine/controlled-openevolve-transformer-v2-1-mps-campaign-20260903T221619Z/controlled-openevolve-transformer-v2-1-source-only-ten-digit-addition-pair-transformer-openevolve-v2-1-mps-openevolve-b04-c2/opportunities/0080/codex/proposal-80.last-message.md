MECHANISM: Twelfth-row positional common-mode quotient

HYPOTHESIS: Extending the verified 1,587-parameter design by quotienting positional row 2 will produce a 1,586-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first LayerNorm.

INTENDED_EDIT: Reproduce the qualified all-but-column-3 MLP and attention projection quotients, then compact positional rows 0–2 and the final nine rows while retaining full-coordinate AdamW moments.

EVIDENCE: Reference Design 2 achieved 99.86% accuracy at 1,587 parameters with all attention projection columns except column 3 compacted; its existing eleven positional-row quotients also succeeded, motivating the closest incremental extension to adjacent row 2.

<<<<<<< SEARCH
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
=======
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient every projection column except optimization-sensitive
        # column three.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()
=======
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6, 7)]
            )
            remaining_proj_weight = proj_weight[:, 3:4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
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
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through eight,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 9:]), dim=1
            ).clone()
=======
        # Uniform output components are erased by the final LayerNorm.
        # Quotient every column except column three, which remains an
        # unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11)]
            )
            remaining_weight = weight[:, 3:4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight[:, :1],
                compact_columns[:, 3:],
                self.fc2.weight[:, 1:],
            ),
            dim=1,
        )
=======
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight,
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
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first three and final
        # nine positional rows.
        self.compact_pos_count = 12
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
                (initialized_pos[:3], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[3:-9].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (
                compact_rows[:2],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
            ),
=======
            (
                compact_rows[:3],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[3:],
            ),
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
                        selected_full_grad[:3].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[3:].reshape(-1),
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
                    (full_update[:3], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[3:-9].reshape(-1),
                    )
                )
>>>>>>> REPLACE