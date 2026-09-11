MECHANISM: Index-safe position-specific LayerNorm common-mode quotient

HYPOTHESIS: Compacting positional row 14 with an explicit bijective row map will reduce the qualified model to 1,563 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add row 14 to the compact positional set and use shared explicit indices for initialization, inference reconstruction, gradients, and full-coordinate AdamW updates.

EVIDENCE: The current 1,564-parameter design achieved 99.86%, and an earlier positional quotient retained 99.53%; prior row-14 attempts produced no accuracy result, so a corrected index-safe implementation remains the most informative unmeasured reduction.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows. Explicit complementary index maps
        # keep compact storage, inference, and optimizer reconstruction aligned.
        compact_pos_indices = torch.tensor(
            [
                *range(7),
                *range(8, 13),
                14,
                *range(cfg.max_seq_len - 9, cfg.max_seq_len),
            ],
            dtype=torch.long,
        )
        unrestricted_pos_indices = torch.tensor(
            [7, 13, *range(15, cfg.max_seq_len - 9)],
            dtype=torch.long,
        )
        self.compact_pos_count = compact_pos_indices.numel()
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
            )
=======
            selected_pos = initialized_pos.index_select(
                0, compact_pos_indices
            )
            unrestricted_pos = initialized_pos.index_select(
                0, unrestricted_pos_indices
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
        self.register_buffer(
            "compact_pos_indices", compact_pos_indices, persistent=False
        )
        self.register_buffer(
            "unrestricted_pos_indices",
            unrestricted_pos_indices,
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:],
                compact_rows[12:],
            ),
            dim=0,
        )
=======
        pos_weight = compact_rows.new_zeros(
            self.cfg.max_seq_len, self.cfg.d_model
        )
        pos_weight = pos_weight.index_copy(
            0, self.compact_pos_indices, compact_rows
        )
        pos_weight = pos_weight.index_copy(
            0, self.unrestricted_pos_indices, unrestricted_rows
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
=======
    compact_pos_count = model.compact_pos_count
    compact_pos_indices = model.compact_pos_indices
    unrestricted_pos_indices = model.unrestricted_pos_indices
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
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
                    (
                        full_update[:7],
                        full_update[8:13],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[13:-9]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
=======
                full_grad = selected_full_grad.new_empty(
                    model_cfg.max_seq_len, model_cfg.d_model
                )
                full_grad.index_copy_(
                    0, compact_pos_indices, selected_full_grad
                )
                full_grad.index_copy_(
                    0, unrestricted_pos_indices, unrestricted_full_grad
                )
                full_grad = full_grad.reshape(-1)
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
                selected_update = full_update.index_select(
                    0, compact_pos_indices
                )
                unrestricted_update = full_update.index_select(
                    0, unrestricted_pos_indices
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
>>>>>>> REPLACE