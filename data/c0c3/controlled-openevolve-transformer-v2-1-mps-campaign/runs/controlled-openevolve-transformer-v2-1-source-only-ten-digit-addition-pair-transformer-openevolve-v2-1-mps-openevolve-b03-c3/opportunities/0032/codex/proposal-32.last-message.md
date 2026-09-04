MECHANISM: Ambient-Adam LayerNorm-to-MLP scale folding

HYPOTHESIS: Folding the bias-free second LayerNorm’s eight learned scales into the first MLP weight matrix will reduce the qualified 1,486-parameter model to 1,478 parameters while retaining at least 99% accuracy, because the fold is functionally exact and full-space AdamW dynamics are reproduced during training.

INTENDED_EDIT: Port the qualified gauge-fixed relative-position design, remove `ln2`’s affine parameters, and manually optimize virtual LayerNorm scales and MLP weights before folding their product into the stored learned weight after every step.

EVIDENCE: The 1,486-parameter relative-lag gauge design achieved 99.90% accuracy. Its `ln2` is already bias-free, so each remaining scale is exactly multiplicatively redundant with the corresponding `fc1` weight column; ambient-coordinate optimization follows the successful optimizer-preserving gauge strategy used by that design.

<<<<<<< SEARCH
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
    vocab_size: int


class GaugeFixedRelativePositionBias(nn.Module):
    """Per-head relative-lag bias with softmax-invariant shifts removed."""

    def __init__(self, n_head: int, max_seq_len: int, rng_width: int):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.rng_width = rng_width
        self.bias = nn.Parameter(torch.empty(n_head, max_seq_len - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.bias.new_empty(self.max_seq_len, self.rng_width)
        nn.init.normal_(raw, mean=0.0, std=std)
        ambient = raw.flatten()[: self.n_head * self.max_seq_len]
        ambient = ambient.view(self.n_head, self.max_seq_len)
        self.bias.copy_(ambient[:, :-1] - ambient[:, -1:])

    def forward(self, seqlen: int) -> torch.Tensor:
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.n_head, 1)),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        positions = torch.arange(seqlen, device=self.bias.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        return full_bias[:, distance]


class GaugeFixedBiasLinear(nn.Module):
    """Linear layer whose output bias omits its all-ones gauge scalar."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.full_v_bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
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
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + position_bias.unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), position_bias)
        x = x + self.mlp(self.ln2(x))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_bias = GaugeFixedRelativePositionBias(
            cfg.n_head, cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
=======
        x = self.drop(self.token_emb(idx))
        position_bias = self.pos_bias(seqlen)

        for blk in self.blocks:
            x = blk(x, position_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    gauge_params = [blk.mlp.fc2.bias for blk in model.blocks]
    position_bias_param = model.pos_bias.bias
    value_bias_params = [blk.attn.v_bias for blk in model.blocks]
    projection_bias_params = [blk.attn.proj.bias for blk in model.blocks]
    value_attentions = [blk.attn for blk in model.blocks]
    mlp_weight_params = [blk.mlp.fc1.weight for blk in model.blocks]
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
        torch.zeros_like(p) for p in projection_bias_params
    ]
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]

    # These tensors are optimizer state for the eliminated ln2 scales and
    # their ambient fc1 weights. The model always stores their exact product.
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
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
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
            blk.mlp.fc2.full_bias.grad.detach()
            for blk in model.blocks
        ]
        full_position_grad = model.pos_bias.full_bias.grad.detach()
        full_value_grads = [
            attention.full_v_bias.grad.detach()
            for attention in value_attentions
        ]
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
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
                    projection_direction, alpha=-lr_now
                )
                projection_param.add_(
                    attention.proj.weight[:, -1]
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
>>>>>>> REPLACE