MECHANISM: Incremental tail-localized positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Quotienting positional rows zero, one, and the final three rows will reduce the model from 1,633 to 1,630 parameters while retaining at least 99% accuracy, because the verified final-two-row design achieved 99.96% and tail rows have substantially shorter causal influence than the failed row-two quotient.

INTENDED_EDIT: Store positional rows zero, one, and the final three rows in the seven-dimensional orthonormal zero-sum basis, reconstruct them during inference, and preserve virtual eight-coordinate AdamW moments during training.

EVIDENCE: The 1,631-parameter design quotienting rows zero, one, and the final two rows achieved 99.96%, whereas adding early row two produced only 98.97%; this supports testing the adjacent third-from-last row as the smallest tail-localized reduction.

<<<<<<< SEARCH
        # Preserve the baseline initialization and its RNG consumption, then
        # remove rows zero and one's LayerNorm-invariant common-mode coordinates.
        self.compact_pos_rows = 2
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            compact_pos = torch.cat(
                (
                    (
                        initialized_pos[: self.compact_pos_rows] @ pos_basis
                    ).reshape(-1),
                    initialized_pos[self.compact_pos_rows :].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final three positions, which have the shortest causal influence.
        self.compact_pos_count = 5
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-3:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-3].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        compact_size = self.compact_pos_rows * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_rows, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows,
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_rows,
                    self.cfg.d_model,
                ),
            ),
            dim=0,
        )
=======
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    compact_pos_rows = model.compact_pos_rows
    pos_compact_size = compact_pos_rows * (model_cfg.d_model - 1)
=======
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        (
                            pos_param.grad[:pos_compact_size].view(
                                compact_pos_rows, model_cfg.d_model - 1
                            )
                            @ pos_basis.transpose(0, 1)
                        ).reshape(-1),
                        pos_param.grad[pos_compact_size:],
                    )
                )
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                full_update = state["exp_avg"] / denom
                full_compact_size = compact_pos_rows * model_cfg.d_model
                compact_update = torch.cat(
                    (
                        (
                            full_update[:full_compact_size].view(
                                compact_pos_rows, model_cfg.d_model
                            )
                            @ pos_basis
                        ).reshape(-1),
                        full_update[full_compact_size:],
                    )
                )
=======
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                full_grad = torch.cat(
                    (
                        selected_full_grad[:2].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[2:].reshape(-1),
                    )
                )
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len, model_cfg.d_model)
                selected_update = torch.cat(
                    (full_update[:2], full_update[-3:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-3].reshape(-1),
                    )
                )
>>>>>>> REPLACE