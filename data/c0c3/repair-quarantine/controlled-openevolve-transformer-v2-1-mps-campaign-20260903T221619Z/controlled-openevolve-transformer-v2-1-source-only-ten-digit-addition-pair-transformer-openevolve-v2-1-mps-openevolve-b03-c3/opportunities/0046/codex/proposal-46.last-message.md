MECHANISM: Third multiplicatively absorbed LayerNorm scale with ambient AdamW

HYPOTHESIS: Extending the qualified two-scale optimizer-side factorization to a third `ln2` scale will produce a 1,537-parameter model with at least 99% accuracy, because the third scale is exactly absorbable into its corresponding `fc1` column and its separate AdamW dynamics are preserved outside the deduplicated learned model.

INTENDED_EDIT: Reproduce the qualified 1,538-parameter shared-key/value, bias-free-value, bias-free-`ln1`, attention-gauged, four-terminal-gauge design, then absorb three rather than two `ln2` scales into `fc1` with ambient-coordinate moments and clipping.

EVIDENCE: The two-scale ambient-factorization design achieved 99.94% accuracy at 1,538 parameters, whereas directly fixing two scales achieved only 80.28%; this isolates optimizer geometry as the failure cause and motivates extending the verified factorization by one coordinate.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and three weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

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


class ThreeFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with three scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(3)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_bias
=======
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = ThreeFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and three terminal-weight gauge vectors.
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
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

    # The last three ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(3, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -3:])
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
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
                blk.mlp.fc1.weight.grad[:, -3:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -3:].detach()
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
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
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

        gauge_step += 1
=======
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
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
            blk.mlp.fc1.weight.grad[:, -3:].zero_()

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
                blk.mlp.fc1.weight[:, -3:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
>>>>>>> REPLACE