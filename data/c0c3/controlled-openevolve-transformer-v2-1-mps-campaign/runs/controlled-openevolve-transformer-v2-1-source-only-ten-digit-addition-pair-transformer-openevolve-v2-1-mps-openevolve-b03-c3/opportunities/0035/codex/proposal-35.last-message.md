MECHANISM: Orthonormal quotient coordinates for mean-zero MLP inputs

HYPOTHESIS: Directly optimizing the 84-dimensional orthonormal quotient of `fc1` will complete within the time limit and retain at least 99% accuracy with 1,466 parameters, because it preserves every linear function on bias-free LayerNorm outputs while avoiding the expensive ambient-coordinate machinery that caused the prior 1,466- and 1,477-parameter attempts to time out.

INTENDED_EDIT: Port the qualified relative-bias gauge, remove redundant `ln2` affine scales, and represent every eight-coordinate `fc1` row by seven learned coefficients in a fixed orthonormal basis of the mean-zero subspace.

EVIDENCE: The 1,478-parameter LayerNorm-to-MLP fold achieved 99.88%, while the exact 1,466-parameter row-gauge extension timed out rather than failing accuracy; using compact quotient coordinates directly targets that implementation overhead while preserving the same function class.

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
        self.bias.copy_(ambient[:, :-1] - ambient[:, -1:])

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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map stored in an orthonormal quotient of mean-zero inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        basis = torch.zeros(in_features, in_features - 1)
        for column in range(in_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        self.weight.copy_(raw_weight @ self.basis)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
            raw_weight
        )
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def reset_normal_parameters(self, std: float) -> None:
        raw_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(raw_weight, mean=0.0, std=std)
        self.weight.copy_(raw_weight @ self.basis)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.weight @ self.basis.transpose(0, 1)
        return F.linear(x, full_weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
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
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LearnedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, MeanZeroInputLinear):
            module.reset_normal_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
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
    # Preserve ambient AdamW dynamics for the previously qualified compact
    # biases; the MLP input weights use their direct orthonormal coordinates.
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

        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_value_grads = [
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
        loss.backward()

        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_position_grad = model.pos_bias.full_bias.grad.detach()
        full_value_grads = [
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for full_grad in full_value_grads:
=======
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            grad_sq = (
                grad_sq
                + full_position_grad[:, -1].float().square().sum()
            )
            for full_grad in full_value_grads:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            value_param,
            projection_param,
            attention,
            full_grad,
=======
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
>>>>>>> REPLACE