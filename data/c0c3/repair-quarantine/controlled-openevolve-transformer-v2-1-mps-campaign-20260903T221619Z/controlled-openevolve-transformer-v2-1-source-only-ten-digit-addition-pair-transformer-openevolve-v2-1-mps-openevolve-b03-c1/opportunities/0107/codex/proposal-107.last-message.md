MECHANISM: Radius-scale gauge fixing with ambient AdamW moments

HYPOTHESIS: Removing one globally redundant token-radius coordinate will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy, because virtual full-coordinate AdamW updates preserve the successful 606-parameter training trajectory up to an exact scale gauge.

INTENDED_EDIT: Fix the first token radius as a buffer, learn the remaining radii, and train the gauge-fixed lexical parameters through virtual full-radius optimizer states mapped back into the 605-parameter model after every step.

EVIDENCE: The current 606-parameter model achieved 99.89% accuracy. The earlier phase gauge collapsed to 4.8%, showing optimizer-coordinate changes can be destructive; this patch therefore retains the omitted coordinate only as transient optimizer state and reproduces full-coordinate gradients, clipping, AdamW moments, and weight decay before exact gauge normalization.

<<<<<<< SEARCH
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
=======
        self.token_phase = nn.Parameter(token_phase.clone())
        self.register_buffer(
            "token_radius_reference", token_radius[:1].clone()
        )
        self.token_radius = nn.Parameter(token_radius[1:].clone())
        self.token_proj = nn.Parameter(token_proj.clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
=======
        token_radius = torch.cat(
            (self.token_radius_reference, self.token_radius)
        )
        token_code = token_radius.unsqueeze(1) * torch.stack(
>>>>>>> REPLACE

<<<<<<< SEARCH
    fc2_bias_ids = {id(param) for param in fc2_bias_params}
    token_phase_id = id(model.token_phase)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) not in fc2_bias_ids
                    and id(param) != token_phase_id
                ],
            },
            {
                "params": [model.token_phase],
                "weight_decay": 0.0,
            },
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    fc2_bias_states = [
=======
    fc2_bias_ids = {id(param) for param in fc2_bias_params}
    token_phase_id = id(model.token_phase)
    gauge_token_params = [
        model.token_radius,
        model.token_proj,
        model.ln_f_token_bias,
    ]
    gauge_token_ids = {id(param) for param in gauge_token_params}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) not in fc2_bias_ids
                    and id(param) not in gauge_token_ids
                    and id(param) != token_phase_id
                ],
            },
            {
                "params": [model.token_phase],
                "weight_decay": 0.0,
            },
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # Optimize the gauge-fixed lexical factors through virtual coordinates
    # matching the original full-radius parameterization. This preserves
    # AdamW's coordinate-wise moments and the successful training trajectory;
    # only the gauge-normalized 605-parameter model enters checkpoints.
    virtual_token_radius = torch.cat(
        (
            model.token_radius_reference,
            model.token_radius.detach(),
        )
    ).clone()
    virtual_token_proj = model.token_proj.detach().clone()
    virtual_token_bias = model.ln_f_token_bias.detach().clone()
    virtual_token_params = [
        virtual_token_radius,
        virtual_token_proj,
        virtual_token_bias,
    ]
    virtual_token_states = [
        {
            "exp_avg": param.new_zeros(param.shape),
            "exp_avg_sq": param.new_zeros(param.shape),
        }
        for param in virtual_token_params
    ]

    fc2_bias_states = [
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for bias in fc2_bias_params:
            bias.grad = None
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        with torch.no_grad():
            adam_step = step + 1
=======
        optimizer.zero_grad(set_to_none=True)
        for bias in fc2_bias_params:
            bias.grad = None
        for param in gauge_token_params:
            param.grad = None
        loss.backward()

        with torch.no_grad():
            # Map reduced-coordinate gradients back to the virtual full
            # parameterization. The missing reference-radius derivative follows
            # from differentiating the exact common-radius scale symmetry.
            gauge_scale = (
                model.token_radius_reference[0]
                / virtual_token_radius[0]
            )
            virtual_radius_grad = torch.empty_like(
                virtual_token_radius
            )
            virtual_radius_grad[1:] = (
                gauge_scale * model.token_radius.grad
            )
            virtual_proj_grad = (
                model.token_proj.grad / gauge_scale
            )
            virtual_bias_grad = (
                model.ln_f_token_bias.grad / gauge_scale
            )
            virtual_radius_grad[0] = (
                -(
                    model.token_radius.grad
                    * model.token_radius
                ).sum()
                + (
                    model.token_proj.grad
                    * model.token_proj
                ).sum()
                + (
                    model.ln_f_token_bias.grad
                    * model.ln_f_token_bias
                ).sum()
            ) / virtual_token_radius[0]
            virtual_token_gradients = [
                virtual_radius_grad,
                virtual_proj_grad,
                virtual_bias_grad,
            ]

            if train_cfg.grad_clip > 0:
                grad_norm_sq = virtual_radius_grad.new_zeros(())
                for param in model.parameters():
                    if (
                        id(param) not in gauge_token_ids
                        and param.grad is not None
                    ):
                        grad_norm_sq.add_(param.grad.square().sum())
                for grad in virtual_token_gradients:
                    grad_norm_sq.add_(grad.square().sum())
                clip_coefficient = (
                    train_cfg.grad_clip
                    / (grad_norm_sq.sqrt() + 1e-6)
                ).clamp(max=1.0)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)
                for grad in virtual_token_gradients:
                    grad.mul_(clip_coefficient)

            adam_step = step + 1
>>>>>>> REPLACE

<<<<<<< SEARCH
                bias.mul_(1.0 - lr_now * train_cfg.weight_decay)
                bias.add_(reduced_update, alpha=-lr_now)
        optimizer.step()
=======
                bias.mul_(1.0 - lr_now * train_cfg.weight_decay)
                bias.add_(reduced_update, alpha=-lr_now)

            for param, grad, state in zip(
                virtual_token_params,
                virtual_token_gradients,
                virtual_token_states,
            ):
                state["exp_avg"].mul_(beta1).add_(
                    grad, alpha=1.0 - beta1
                )
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2
                )
                update = state["exp_avg"] / (
                    1.0 - beta1 ** adam_step
                )
                denom = (
                    state["exp_avg_sq"] / (1.0 - beta2 ** adam_step)
                ).sqrt().add_(1e-8)
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.addcdiv_(update, denom, value=-lr_now)

            gauge_scale = (
                model.token_radius_reference[0]
                / virtual_token_radius[0]
            )
            model.token_radius.copy_(
                gauge_scale * virtual_token_radius[1:]
            )
            model.token_proj.copy_(
                virtual_token_proj / gauge_scale
            )
            model.ln_f_token_bias.copy_(
                virtual_token_bias / gauge_scale
            )
        optimizer.step()
>>>>>>> REPLACE