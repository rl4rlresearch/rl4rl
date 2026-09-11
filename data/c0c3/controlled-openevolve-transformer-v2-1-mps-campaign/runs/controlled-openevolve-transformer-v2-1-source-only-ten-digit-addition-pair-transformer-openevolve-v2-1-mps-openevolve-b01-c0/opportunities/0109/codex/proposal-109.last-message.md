MECHANISM: Gamma-aware optimizer-preserving fourth key-row gauge quotient

HYPOTHESIS: Compressing the fourth key-projection row to seven coordinates while emulating full eight-coordinate AdamW dynamics will reduce the model to 1589 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Keep the third key row full, anchor the fourth row’s final coordinate, reconstruct its redundant gradient from the current `ln1` gains, and optimize it with virtual full-coordinate moments.

EVIDENCE: The 1590-parameter design retains 99.92% accuracy with two anchored key rows, while the third-row attempt failed at 54.56%; testing the fourth row independently is the clearest test of row-specific key-gauge compressibility.

<<<<<<< SEARCH
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 2, d_model))
=======
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_third_row = nn.Parameter(torch.empty(d_model))
        self.key_fourth_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 4, d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 2 :])
=======
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        fourth_key_row = weight[self.d_model + 3]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.key_third_row.copy_(weight[self.d_model + 2])
            self.key_fourth_row.copy_(
                fourth_key_row[:-1] - fourth_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 4 :])
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )
=======
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        fourth_key_row = torch.cat(
            (self.key_fourth_row, self.key_fourth_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                self.key_third_row.unsqueeze(0),
                fourth_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize the compressed attention columns and fourth, sixth, ninth, and
    # tenth MLP output columns with virtual moments for their full representations.
    # This preserves AdamW's original coordinate geometry without adding
    # learned parameters.
    virtual_columns = [
        column
        for block in model.blocks
        for column in (
            block.attn.proj.fourth_column,
            block.attn.proj.fifth_column,
            block.attn.proj.sixth_column,
            block.mlp.fc2_fourth_column,
            block.mlp.fc2_sixth_column,
            block.mlp.fc2_ninth_column,
            block.mlp.fc2_tenth_column,
        )
    ]
    virtual_column_ids = {id(column) for column in virtual_columns}
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in virtual_column_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    virtual_first_moments = [
        column.new_zeros(column.numel() + 1) for column in virtual_columns
    ]
    virtual_second_moments = [
        column.new_zeros(column.numel() + 1) for column in virtual_columns
    ]
=======
    # Optimize the compressed attention-output and MLP-output columns with
    # virtual moments for their full representations.
    virtual_columns = [
        column
        for block in model.blocks
        for column in (
            block.attn.proj.fourth_column,
            block.attn.proj.fifth_column,
            block.attn.proj.sixth_column,
            block.mlp.fc2_fourth_column,
            block.mlp.fc2_sixth_column,
            block.mlp.fc2_ninth_column,
            block.mlp.fc2_tenth_column,
        )
    ]
    # A key-row shift proportional to the inverse LayerNorm gains changes that
    # key coordinate by the same constant at every position and is therefore
    # invisible to attention softmax.
    virtual_key_rows = [
        (block.attn.qkv.key_fourth_row, block.ln1.weight)
        for block in model.blocks
    ]
    virtual_parameter_ids = {
        id(parameter)
        for parameter in virtual_columns
    } | {
        id(key_row)
        for key_row, _ in virtual_key_rows
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in virtual_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    virtual_first_moments = [
        column.new_zeros(column.numel() + 1) for column in virtual_columns
    ]
    virtual_second_moments = [
        column.new_zeros(column.numel() + 1) for column in virtual_columns
    ]
    virtual_key_first_moments = [
        key_row.new_zeros(key_row.numel() + 1)
        for key_row, _ in virtual_key_rows
    ]
    virtual_key_second_moments = [
        key_row.new_zeros(key_row.numel() + 1)
        for key_row, _ in virtual_key_rows
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            for column in virtual_columns:
                if column.grad is not None:
                    virtual_last_grad = -column.grad.detach().float().sum()
                    total_grad_sq = total_grad_sq + virtual_last_grad.square()

            clip_scale = min(
=======
            for column in virtual_columns:
                if column.grad is not None:
                    virtual_last_grad = -column.grad.detach().float().sum()
                    total_grad_sq = total_grad_sq + virtual_last_grad.square()
            for key_row, gamma in virtual_key_rows:
                if key_row.grad is not None:
                    gamma_snapshot = gamma.detach().float()
                    virtual_last_grad = -gamma_snapshot[-1] * (
                        key_row.grad.detach().float()
                        / gamma_snapshot[:-1]
                    ).sum()
                    total_grad_sq = total_grad_sq + virtual_last_grad.square()

            clip_scale = min(
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        beta1, beta2 = optimizer.defaults["betas"]
=======
        virtual_key_full_gradients = []
        for key_row, gamma in virtual_key_rows:
            gamma_snapshot = gamma.detach().float()
            stored_gradient = key_row.grad.detach().float()
            virtual_last_gradient = -gamma_snapshot[-1] * (
                stored_gradient / gamma_snapshot[:-1]
            ).sum()
            virtual_key_full_gradients.append(
                torch.cat(
                    (stored_gradient, virtual_last_gradient.reshape(1))
                )
            )

        optimizer.step()

        beta1, beta2 = optimizer.defaults["betas"]
>>>>>>> REPLACE

<<<<<<< SEARCH
                column.mul_(1.0 - lr_now * train_cfg.weight_decay)
                column.add_(normalized_moment[:-1], alpha=-step_size)
                column.add_(normalized_moment[-1], alpha=step_size)

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
                column.mul_(1.0 - lr_now * train_cfg.weight_decay)
                column.add_(normalized_moment[:-1], alpha=-step_size)
                column.add_(normalized_moment[-1], alpha=step_size)

            for (
                (key_row, gamma),
                first_moment,
                second_moment,
                full_gradient,
            ) in zip(
                virtual_key_rows,
                virtual_key_first_moments,
                virtual_key_second_moments,
                virtual_key_full_gradients,
            ):
                first_moment.mul_(beta1).add_(
                    full_gradient, alpha=1.0 - beta1
                )
                second_moment.mul_(beta2).addcmul_(
                    full_gradient,
                    full_gradient,
                    value=1.0 - beta2,
                )
                denominator = (
                    second_moment.sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(adam_eps)
                )
                normalized_moment = first_moment / denominator
                gauge_ratio = (gamma[-1] / gamma[:-1]).detach()

                key_row.mul_(1.0 - lr_now * train_cfg.weight_decay)
                key_row.add_(normalized_moment[:-1], alpha=-step_size)
                key_row.add_(
                    normalized_moment[-1] * gauge_ratio,
                    alpha=step_size,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE