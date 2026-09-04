MECHANISM: Coupled value–projection ambient AdamW gauge

HYPOTHESIS: Removing one value-bias coordinate while transferring its full eight-coordinate AdamW update through the learned output projection will produce a 1,625-parameter model with at least 99% accuracy.

INTENDED_EDIT: Represent the value bias with seven learned coordinates, reconstruct an anchored eighth coordinate, and manually optimize the value and projection biases with full ambient AdamW moments while transferring the omitted value update into the projection bias.

EVIDENCE: The naïve value-bias gauge failed at 55.22%, closely matching the naïve terminal-bias gauge’s 54.75% failure; preserving ambient optimizer geometry raised the latter to at least 99.95%, motivating the analogous coupled treatment here.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.full_v_bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        full_v_bias = torch.cat(
            (self.v_bias, self.v_bias.new_zeros(1))
        )
        if torch.is_grad_enabled():
            full_v_bias.retain_grad()
            self.full_v_bias = full_v_bias
        v = v + full_v_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_ids = {id(p) for p in gauge_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in gauge_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0
=======
    value_bias_params = [
        blk.attn.v_bias for blk in model.blocks
    ]
    projection_bias_params = [
        blk.attn.proj.bias for blk in model.blocks
    ]
    value_attentions = [
        blk.attn for blk in model.blocks
    ]
    special_ids = {
        id(p)
        for p in (
            gauge_params
            + value_bias_params
            + projection_bias_params
        )
    }
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in special_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    value_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in value_bias_params
    ]
    value_v = [torch.zeros_like(moment) for moment in value_m]
    projection_m = [
        torch.zeros_like(p) for p in projection_bias_params
    ]
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()
=======
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
=======
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_value_grads = [
            attention.full_v_bias.grad.detach()
            for attention in value_attentions
        ]
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            gauge_params, full_gauge_grads, gauge_m, gauge_v
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(ambient_grad, alpha=0.1)
            variance.mul_(0.999).addcmul_(
                ambient_grad, ambient_grad, value=0.001
            )
            m_hat = moment / (1.0 - 0.9 ** gauge_step)
            v_hat = variance / (1.0 - 0.999 ** gauge_step)
            direction = m_hat / (v_hat.sqrt() + 1e-8)
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )
=======
        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            gauge_params, full_gauge_grads, gauge_m, gauge_v
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(ambient_grad, alpha=0.1)
            variance.mul_(0.999).addcmul_(
                ambient_grad, ambient_grad, value=0.001
            )
            m_hat = moment / (1.0 - 0.9 ** gauge_step)
            v_hat = variance / (1.0 - 0.999 ** gauge_step)
            direction = m_hat / (v_hat.sqrt() + 1e-8)
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )

        for (
            value_param,
            projection_param,
            attention,
            full_grad,
            projection_grad,
            value_moment,
            value_variance,
            projection_moment,
            projection_variance,
        ) in zip(
            value_bias_params,
            projection_bias_params,
            value_attentions,
            full_value_grads,
            projection_grads,
            value_m,
            value_v,
            projection_m,
            projection_v,
        ):
            value_grad = full_grad * clip_scale
            projection_grad = projection_grad * clip_scale
            value_moment.mul_(0.9).add_(value_grad, alpha=0.1)
            value_variance.mul_(0.999).addcmul_(
                value_grad, value_grad, value=0.001
            )
            projection_moment.mul_(0.9).add_(
                projection_grad, alpha=0.1
            )
            projection_variance.mul_(0.999).addcmul_(
                projection_grad, projection_grad, value=0.001
            )
            value_direction = (
                value_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    value_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            projection_direction = (
                projection_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    projection_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                value_param.mul_(decay)
                value_param.add_(
                    value_direction[:-1], alpha=-lr_now
                )
                projection_param.mul_(decay)
                projection_param.add_(
                    projection_direction, alpha=-lr_now
                )
                projection_param.add_(
                    attention.proj.weight[:, -1]
                    * value_direction[-1],
                    alpha=-lr_now,
                )
>>>>>>> REPLACE