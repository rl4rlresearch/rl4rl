MECHANISM: Ambient-Adam attention residual-shift gauge

HYPOTHESIS: Gauge-fixing the attention projection bias’s exact all-ones residual-shift invariance on top of the verified 1,470-parameter design will retain at least 99% accuracy with 1,469 parameters when its full eight-coordinate AdamW and coupled value-bias dynamics are preserved.

INTENDED_EDIT: Port the verified bias-free `ln1` and folded affine-free `ln2` design, then store seven attention projection-bias differences while optimizing its full ambient bias and transferring the omitted value-bias update in quotient coordinates.

EVIDENCE: Reference Design 2 achieved 99.94% accuracy with 1,470 parameters. The terminal-bias and relative-bias ambient gauges also retained at least 99%, while complete attention-output-bias removal failed; this patch retains all seven functionally relevant projection-bias differences and removes only the common shift erased by downstream LayerNorms.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = GaugeFixedBiasLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize compact gauges through their corresponding ambient-coordinate
    # AdamW moments, including coordinates omitted from learned storage.
    gauge_params = [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
    position_bias_param = model.pos_bias.bias
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
            + [position_bias_param]
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
    position_m = torch.zeros(
        model.pos_bias.n_head,
        model.pos_bias.max_seq_len,
        device=device,
        dtype=position_bias_param.dtype,
    )
    position_v = torch.zeros_like(position_m)
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
=======
    # Optimize compact gauges through their corresponding ambient-coordinate
    # AdamW moments, including coordinates omitted from learned storage.
    gauge_params = [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
    position_bias_param = model.pos_bias.bias
    value_bias_params = [
        blk.attn.v_bias for blk in model.blocks
    ]
    projection_bias_params = [
        blk.attn.proj.bias for blk in model.blocks
    ]
    value_attentions = [
        blk.attn for blk in model.blocks
    ]
    mlp_weight_params = [
        blk.mlp.fc1.weight for blk in model.blocks
    ]
    mlp_weight_ids = {id(p) for p in mlp_weight_params}
    special_ids = {
        id(p)
        for p in (
            gauge_params
            + [position_bias_param]
            + value_bias_params
            + projection_bias_params
            + mlp_weight_params
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
    position_m = torch.zeros(
        model.pos_bias.n_head,
        model.pos_bias.max_seq_len,
        device=device,
        dtype=position_bias_param.dtype,
    )
    position_v = torch.zeros_like(position_m)
    value_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in value_bias_params
    ]
    value_v = [torch.zeros_like(moment) for moment in value_m]
    projection_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in projection_bias_params
    ]
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]

    # The model stores the exact product of the eliminated ln2 scales and
    # their ambient fc1 weights.
    mlp_ambient_weights = [
        p.detach().clone() for p in mlp_weight_params
    ]
    mlp_ambient_scales = [
        torch.ones(p.shape[1], device=device, dtype=p.dtype)
        for p in mlp_weight_params
    ]
    mlp_weight_m = [
        torch.zeros_like(p) for p in mlp_ambient_weights
    ]
    mlp_weight_v = [
        torch.zeros_like(p) for p in mlp_ambient_weights
    ]
    mlp_scale_m = [
        torch.zeros_like(p) for p in mlp_ambient_scales
    ]
    mlp_scale_v = [
        torch.zeros_like(p) for p in mlp_ambient_scales
    ]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        position_bias_param.grad = None
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()

        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_position_grad = model.pos_bias.full_bias.grad.detach()
        full_value_grads = [
            attention.full_v_bias.grad.detach()
            for attention in value_attentions
        ]
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
        ]
        clip_scale = 1.0
=======
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        position_bias_param.grad = None
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        for mlp_weight_param in mlp_weight_params:
            mlp_weight_param.grad = None
        loss.backward()

        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_position_grad = model.pos_bias.full_bias.grad.detach()
        full_value_grads = [
            attention.full_v_bias.grad.detach()
            for attention in value_attentions
        ]
        projection_grads = [
            attention.proj.full_bias.grad.detach()
            for attention in value_attentions
        ]
        effective_mlp_grads = [
            p.grad.detach().clone() for p in mlp_weight_params
        ]
        ambient_mlp_weight_grads = [
            grad * scale.unsqueeze(0)
            for grad, scale in zip(
                effective_mlp_grads, mlp_ambient_scales
            )
        ]
        ambient_mlp_scale_grads = [
            (grad * weight).sum(dim=0)
            for grad, weight in zip(
                effective_mlp_grads, mlp_ambient_weights
            )
        ]
        clip_scale = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            grad_sq = (
                grad_sq
                + full_position_grad[:, -1]
                .float()
                .square()
                .sum()
            )
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
=======
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None and id(p) not in mlp_weight_ids
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            grad_sq = (
                grad_sq
                + full_position_grad[:, -1]
                .float()
                .square()
                .sum()
            )
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for full_grad in projection_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for weight_grad, scale_grad in zip(
                ambient_mlp_weight_grads,
                ambient_mlp_scale_grads,
            ):
                grad_sq = (
                    grad_sq
                    + weight_grad.float().square().sum()
                    + scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
                projection_param.mul_(decay)
                projection_param.add_(
                    projection_direction, alpha=-lr_now
                )
                projection_param.add_(
                    attention.proj.weight[:, -1]
                    * value_direction[-1],
                    alpha=-lr_now,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
                projection_param.mul_(decay)
                projection_param.add_(
                    projection_direction[-1]
                    - projection_direction[:-1],
                    alpha=lr_now,
                )
                projection_param.add_(
                    (
                        attention.proj.weight[:-1, -1]
                        - attention.proj.weight[-1:, -1]
                    )
                    * value_direction[-1],
                    alpha=-lr_now,
                )

        for (
            stored_weight,
            ambient_weight,
            ambient_scale,
            weight_grad,
            scale_grad,
            weight_moment,
            weight_variance,
            scale_moment,
            scale_variance,
        ) in zip(
            mlp_weight_params,
            mlp_ambient_weights,
            mlp_ambient_scales,
            ambient_mlp_weight_grads,
            ambient_mlp_scale_grads,
            mlp_weight_m,
            mlp_weight_v,
            mlp_scale_m,
            mlp_scale_v,
        ):
            weight_grad = weight_grad * clip_scale
            scale_grad = scale_grad * clip_scale
            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    weight_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            scale_direction = (
                scale_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    scale_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                ambient_weight.mul_(decay)
                ambient_weight.add_(
                    weight_direction, alpha=-lr_now
                )
                ambient_scale.mul_(decay)
                ambient_scale.add_(
                    scale_direction, alpha=-lr_now
                )
                stored_weight.copy_(
                    ambient_weight * ambient_scale.unsqueeze(0)
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE