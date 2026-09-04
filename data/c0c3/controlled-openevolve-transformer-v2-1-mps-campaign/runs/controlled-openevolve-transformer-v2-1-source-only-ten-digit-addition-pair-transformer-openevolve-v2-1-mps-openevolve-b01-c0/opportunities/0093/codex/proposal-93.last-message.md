MECHANISM: Optimizer-preserving fourth-column gauge quotient

HYPOTHESIS: Anchoring the fourth attention projection column will achieve at least 99% accuracy with 1596 parameters when its updates emulate full eight-coordinate AdamW dynamics, because the prior anchored and orthonormal failures indicate optimizer-geometry sensitivity rather than loss of model expressivity.

INTENDED_EDIT: Store seven learned coordinates for the fourth projection column and maintain virtual eight-coordinate Adam moments during training so its quotient-space updates match the successful full-column parameterization.

EVIDENCE: The current three-column anchor reached 99.98% at 1597 parameters, while naïve anchoring of the fourth column reached 84.5% and an orthonormal parameterization reached 40.36%; preserving the original AdamW coordinate dynamics directly targets the distinction between those failures and the successful design.

<<<<<<< SEARCH
        self.third_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 3))
=======
        self.third_column = nn.Parameter(torch.empty(d_model - 1))
        self.fourth_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_column = weight[:, 2]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.second_column.copy_(
                second_column[:-1] - second_column[-1]
            )
            self.third_column.copy_(
                third_column[:-1] - third_column[-1]
            )
            self.rest.copy_(weight[:, 3:])
=======
        third_column = weight[:, 2]
        fourth_column = weight[:, 3]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.second_column.copy_(
                second_column[:-1] - second_column[-1]
            )
            self.third_column.copy_(
                third_column[:-1] - third_column[-1]
            )
            self.fourth_column.copy_(
                fourth_column[:-1] - fourth_column[-1]
            )
            self.rest.copy_(weight[:, 4:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_column = torch.cat(
            (self.third_column, self.third_column.new_zeros(1))
        )
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                third_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )
=======
        third_column = torch.cat(
            (self.third_column, self.third_column.new_zeros(1))
        )
        fourth_column = torch.cat(
            (self.fourth_column, self.fourth_column.new_zeros(1))
        )
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                third_column.unsqueeze(1),
                fourth_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    # Optimize each compressed fourth projection column with virtual moments
    # for the full eight-coordinate representation. This preserves AdamW's
    # original coordinate geometry without adding a learned model parameter.
    virtual_columns = [
        block.attn.proj.fourth_column for block in model.blocks
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            total_grad_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    total_grad_sq = (
                        total_grad_sq
                        + parameter.grad.detach().float().square().sum()
                    )
            for column in virtual_columns:
                if column.grad is not None:
                    virtual_last_grad = -column.grad.detach().float().sum()
                    total_grad_sq = total_grad_sq + virtual_last_grad.square()

            clip_scale = min(
                1.0,
                train_cfg.grad_clip
                / (float(total_grad_sq.sqrt().item()) + 1e-6),
            )
            if clip_scale < 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(clip_scale)

        optimizer.step()

        beta1, beta2 = optimizer.defaults["betas"]
        adam_eps = optimizer.defaults["eps"]
        bias_correction1 = 1.0 - beta1 ** (step + 1)
        bias_correction2 = 1.0 - beta2 ** (step + 1)
        step_size = lr_now / bias_correction1
        with torch.no_grad():
            for column, first_moment, second_moment in zip(
                virtual_columns,
                virtual_first_moments,
                virtual_second_moments,
            ):
                full_gradient = torch.cat(
                    (column.grad, -column.grad.sum().reshape(1))
                )
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

                column.mul_(1.0 - lr_now * train_cfg.weight_decay)
                column.add_(normalized_moment[:-1], alpha=-step_size)
                column.add_(normalized_moment[-1], alpha=step_size)
>>>>>>> REPLACE