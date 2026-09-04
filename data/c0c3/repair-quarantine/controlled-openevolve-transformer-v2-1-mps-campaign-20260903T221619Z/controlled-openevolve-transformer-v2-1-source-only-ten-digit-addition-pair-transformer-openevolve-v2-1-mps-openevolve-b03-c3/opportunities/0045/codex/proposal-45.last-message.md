MECHANISM: Ambient AdamW for two multiplicatively absorbed LayerNorm scales

HYPOTHESIS: A 1,538-parameter model will retain at least 99% accuracy because the two omitted `ln2` scales are exactly absorbed into their `fc1` columns, while optimizer-side factorization reproduces the full model’s separate AdamW updates instead of the optimization geometry that caused the naive two-scale design to reach only 80.28%.

INTENDED_EDIT: Reproduce the qualified attention projection bias/first-column gauges, replace `ln2` with a six-scale LayerNorm, and train the two absorbed scale/weight factorizations in ambient coordinates while continuously materializing only their products in the learned model.

EVIDENCE: The full-scale 1,540-parameter attention-gauged design achieved 99.99%, whereas directly fixing a second `ln2` scale fell to 80.28%; since each scale is functionally absorbable into the following `fc1` column, this isolates and restores the lost optimizer dynamics while removing both parameters.

<<<<<<< SEARCH
        return F.linear(x, weight, full_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and one weight-column output gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.weight_prefix.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = torch.cat(
            (self.weight_prefix, self.weight_prefix.new_zeros(1))
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_weight_prefix.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (full_weight_prefix.unsqueeze(1), self.weight_rest), dim=1
        )
        return F.linear(x, weight, full_bias)


class TwoFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with two scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoFixedScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedTerminalLinear):
=======
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                module.weight_prefix.copy_(
                    raw_weight[:-1, 0] - raw_weight[-1, 0]
                )
                module.weight_rest.copy_(raw_weight[:, 1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedTerminalLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and four terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
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
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.attn.proj.weight_prefix)
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
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

    # The last two ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(2, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -2:])
        for blk in model.blocks
    ]
    absorbed_weight_v = [
        torch.zeros_like(moment) for moment in absorbed_weight_m
    ]
    absorbed_scale_m = [
        torch.zeros_like(scale) for scale in absorbed_scales
    ]
    absorbed_scale_v = [
        torch.zeros_like(moment) for moment in absorbed_scale_m
    ]
    absorbed_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )
        clip_scale = 1.0
=======
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.attn.proj.full_weight_prefix.grad.detach()
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -2:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -2:].detach()
                / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )

        clip_scale = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
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
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        # These effective columns are updated below through their ambient
        # weight/scale factorization rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -2:].zero_()

        optimizer.step()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        gauge_step += 1
=======
        optimizer.step()

        absorbed_step += 1
        absorbed_bc1 = 1.0 - 0.9 ** absorbed_step
        absorbed_bc2 = 1.0 - 0.999 ** absorbed_step
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = absorbed_weight_m[i]
            weight_variance = absorbed_weight_v[i]
            scale_moment = absorbed_scale_m[i]
            scale_variance = absorbed_scale_v[i]
            virtual_scale = absorbed_scales[i]

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / absorbed_bc1
            ) / (
                (weight_variance / absorbed_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / absorbed_bc1
            ) / (
                (scale_variance / absorbed_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                virtual_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                virtual_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )
                blk.mlp.fc1.weight[:, -2:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
>>>>>>> REPLACE