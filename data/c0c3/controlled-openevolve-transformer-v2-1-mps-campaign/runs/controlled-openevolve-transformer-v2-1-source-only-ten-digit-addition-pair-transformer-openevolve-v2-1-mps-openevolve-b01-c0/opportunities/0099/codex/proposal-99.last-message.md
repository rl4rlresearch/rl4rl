MECHANISM: Gamma-aware optimizer-preserving key-row gauge quotient

HYPOTHESIS: Compressing the third key-projection row to seven coordinates while emulating full eight-coordinate AdamW updates along the LayerNorm-scaled key-shift gauge will retain at least 99% accuracy with 1593 parameters.

INTENDED_EDIT: Anchor the third key row’s final coordinate, reconstruct its redundant gradient using the current `ln1` gains, and optimize it with virtual full-coordinate moments alongside the existing compressed attention-output columns.

EVIDENCE: The verified 1594-parameter model already tolerates two anchored key rows, while virtual full-coordinate optimization rescued sensitive attention-output reductions through the sixth column and retained 99.48% accuracy.

<<<<<<< SEARCH
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 2, d_model))
=======
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_third_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 3, d_model))
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
        third_key_row = weight[self.d_model + 2]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.key_third_row.copy_(
                third_key_row[:-1] - third_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 3 :])
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
        third_key_row = torch.cat(
            (self.key_third_row, self.key_third_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                third_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize the compressed fourth through sixth projection columns with
    # virtual moments for their full eight-coordinate representations. This preserves
    # AdamW's original coordinate geometry without adding learned parameters.
    virtual_columns = [
        column
        for block in model.blocks
        for column in (
            block.attn.proj.fourth_column,
            block.attn.proj.fifth_column,
            block.attn.proj.sixth_column,
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
    # Optimize compressed projection columns and the third key row with
    # virtual moments for their full eight-coordinate representations.
    virtual_columns = [
        column
        for block in model.blocks
        for column in (
            block.attn.proj.fourth_column,
            block.attn.proj.fifth_column,
            block.attn.proj.sixth_column,
        )
    ]
    virtual_key_rows = [
        (block.attn.qkv.key_third_row, block.ln1.weight)
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
            for key_row, ln_weight in virtual_key_rows:
                if key_row.grad is not None:
                    virtual_last_grad = -ln_weight[-1].detach().float() * (
                        key_row.grad.detach().float()
                        / ln_weight[:-1].detach().float()
                    ).sum()
                    total_grad_sq = total_grad_sq + virtual_last_grad.square()

            clip_scale = min(
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        beta1, beta2 = optimizer.defaults["betas"]
=======
        virtual_column_full_gradients = [
            torch.cat(
                (
                    column.grad.detach().float(),
                    -column.grad.detach().float().sum().reshape(1),
                )
            )
            for column in virtual_columns
        ]
        virtual_key_full_gradients = []
        for key_row, ln_weight in virtual_key_rows:
            key_gradient = key_row.grad.detach().float()
            ln_weight_float = ln_weight.detach().float()
            last_gradient = -ln_weight_float[-1] * (
                key_gradient / ln_weight_float[:-1]
            ).sum()
            virtual_key_full_gradients.append(
                torch.cat((key_gradient, last_gradient.reshape(1)))
            )

        optimizer.step()

        beta1, beta2 = optimizer.defaults["betas"]
>>>>>>> REPLACE

<<<<<<< SEARCH
            for column, first_moment, second_moment in zip(
                virtual_columns,
                virtual_first_moments,
                virtual_second_moments,
            ):
                full_gradient = torch.cat(
                    (column.grad, -column.grad.sum().reshape(1))
                )
                first_moment.mul_(beta1).add_(
=======
            for column, first_moment, second_moment, full_gradient in zip(
                virtual_columns,
                virtual_first_moments,
                virtual_second_moments,
                virtual_column_full_gradients,
            ):
                first_moment.mul_(beta1).add_(
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
                (key_row, ln_weight),
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
                gauge_ratio = ln_weight[-1] / ln_weight[:-1]

                key_row.mul_(1.0 - lr_now * train_cfg.weight_decay)
                key_row.add_(normalized_moment[:-1], alpha=-step_size)
                key_row.add_(
                    normalized_moment[-1] * gauge_ratio,
                    alpha=step_size,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE