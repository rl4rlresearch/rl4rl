MECHANISM: Single-column residual-output weight gauge

HYPOTHESIS: Extending the verified 1,469-parameter design by removing one all-ones row-shift coordinate from `fc2` will retain at least 99% accuracy with 1,468 parameters, because the induced token-wise scalar residual shift is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Port the qualified relative-bias, affine-free LayerNorm, folded-MLP, and projection-bias gauges; then compact the first `fc2` weight column to seven row differences and optimize it through full ambient moments.

EVIDENCE: The 1,469-parameter projection-bias gauge reached 99.89%, confirming that common residual-output shifts are removable. Earlier multi-row `fc1` gauges timed out, so this tests one exact `fc2` weight gauge with only one additional vectorized ambient update.

<<<<<<< SEARCH
class LearnedRelativePositionBias(nn.Module):
    """Shift-equivariant positional routing learned in attention-logit space."""

    def __init__(self, n_head: int, max_seq_len: int, rng_width: int):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.rng_width = rng_width
        self.bias = nn.Parameter(torch.empty(n_head, max_seq_len))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Draw the former full positional tensor to preserve the initialization
        # stream of every unchanged transformer parameter.
        raw = self.bias.new_empty(self.max_seq_len, self.rng_width)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.bias.copy_(
            raw.flatten()[: self.bias.numel()].view_as(self.bias)
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(seqlen, device=self.bias.device)
        distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        return self.bias[:, distance]
=======
class GaugeFixedRelativePositionBias(nn.Module):
    """Per-head relative-lag bias with softmax-invariant shifts removed."""

    def __init__(self, n_head: int, max_seq_len: int, rng_width: int):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.rng_width = rng_width
        self.bias = nn.Parameter(
            torch.empty(n_head, max_seq_len - 1)
        )
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.bias.new_empty(
            self.max_seq_len, self.rng_width
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        ambient = raw.flatten()[: self.n_head * self.max_seq_len]
        ambient = ambient.view(self.n_head, self.max_seq_len)
        self.bias.copy_(
            ambient[:, :-1] - ambient[:, -1:]
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        positions = torch.arange(seqlen, device=self.bias.device)
        distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        return full_bias[:, distance]
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, self.weight, full_bias)


class SingleColumnGaugeFixedBiasLinear(nn.Module):
    """Output linear with bias and one weight-column row shift removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        compact_size = (
            out_features - 1
            + out_features * (in_features - 1)
        )
        self.weight = nn.Parameter(torch.empty(compact_size))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std=None) -> None:
        ambient_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        if std is None:
            nn.init.kaiming_uniform_(
                ambient_weight, a=math.sqrt(5)
            )
        else:
            nn.init.normal_(
                ambient_weight, mean=0.0, std=std
            )

        split = self.out_features - 1
        first_column = ambient_weight[:, 0]
        self.weight[:split].copy_(
            first_column[:-1] - first_column[-1]
        )
        self.weight[split:].copy_(
            ambient_weight[:, 1:].reshape(-1)
        )

        if std is None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                ambient_weight
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            raw_bias = self.bias.new_empty(self.out_features)
            nn.init.uniform_(raw_bias, -bound, bound)
            self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
        else:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.out_features - 1
        first_column = torch.cat(
            (self.weight[:split], self.weight.new_zeros(1))
        )
        remaining = self.weight[split:].view(
            self.out_features, self.in_features - 1
        )
        full_weight = torch.cat(
            (first_column.unsqueeze(1), remaining), dim=1
        )
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(1))
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight = full_weight
            self.full_bias = full_bias
        return F.linear(x, full_weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = GaugeFixedBiasLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
=======
        self.fc2 = SingleColumnGaugeFixedBiasLinear(d_ff, d_model)
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
        self.pos_bias = LearnedRelativePositionBias(
            cfg.n_head, cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_bias = GaugeFixedRelativePositionBias(
            cfg.n_head, cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, LearnedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
=======
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, SingleColumnGaugeFixedBiasLinear):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize each seven-coordinate MLP-bias gauge through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
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
    compact_weight_layers = [
        blk.mlp.fc2 for blk in model.blocks
    ]
    compact_weight_params = [
        layer.weight for layer in compact_weight_layers
    ]
    special_ids = {
        id(p)
        for p in (
            gauge_params
            + [position_bias_param]
            + value_bias_params
            + projection_bias_params
            + mlp_weight_params
            + compact_weight_params
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
    compact_weight_m = [
        torch.zeros(
            layer.out_features,
            layer.in_features,
            device=device,
            dtype=layer.weight.dtype,
        )
        for layer in compact_weight_layers
    ]
    compact_weight_v = [
        torch.zeros_like(moment) for moment in compact_weight_m
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
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()
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
        for compact_weight_param in compact_weight_params:
            compact_weight_param.grad = None
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
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
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()
=======
        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach()
            for blk in model.blocks
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
        compact_weight_grads = [
            layer.full_weight.grad.detach()
            for layer in compact_weight_layers
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
                + full_position_grad[:, -1].float().square().sum()
            )
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for full_grad in projection_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for full_grad in compact_weight_grads:
                grad_sq = (
                    grad_sq
                    + full_grad[-1, 0].float().square()
                )
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
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()
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

        position_grad = full_position_grad * clip_scale
        position_m.mul_(0.9).add_(position_grad, alpha=0.1)
        position_v.mul_(0.999).addcmul_(
            position_grad, position_grad, value=0.001
        )
        position_direction = (
            position_m / (1.0 - 0.9 ** gauge_step)
        ) / (
            (
                position_v / (1.0 - 0.999 ** gauge_step)
            ).sqrt()
            + 1e-8
        )
        with torch.no_grad():
            position_bias_param.mul_(
                1.0 - lr_now * train_cfg.weight_decay
            )
            position_bias_param.add_(
                position_direction[:, -1:]
                - position_direction[:, :-1],
                alpha=lr_now,
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
            weight_param,
            layer,
            full_grad,
            moment,
            variance,
        ) in zip(
            compact_weight_params,
            compact_weight_layers,
            compact_weight_grads,
            compact_weight_m,
            compact_weight_v,
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(ambient_grad, alpha=0.1)
            variance.mul_(0.999).addcmul_(
                ambient_grad, ambient_grad, value=0.001
            )
            direction = (
                moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    variance / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            split = layer.out_features - 1
            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                weight_param[:split].mul_(decay)
                weight_param[:split].add_(
                    direction[-1, 0] - direction[:-1, 0],
                    alpha=lr_now,
                )
                weight_param[split:].mul_(decay)
                weight_param[split:].add_(
                    direction[:, 1:].reshape(-1),
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
>>>>>>> REPLACE