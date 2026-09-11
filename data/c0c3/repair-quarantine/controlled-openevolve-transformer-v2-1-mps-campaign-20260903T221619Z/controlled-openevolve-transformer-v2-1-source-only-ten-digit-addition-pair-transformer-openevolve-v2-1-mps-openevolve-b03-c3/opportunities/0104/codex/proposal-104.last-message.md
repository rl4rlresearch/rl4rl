MECHANISM: Fourth orthogonal value/output basis gauge

HYPOTHESIS: A 1,374-parameter model will retain at least 99% accuracy because the qualified 1,375-parameter model reached 99.97%, and a fourth Givens rotation removes one remaining exact value-basis redundancy while preserving the initialized function and full four-dimensional value stream.

INTENDED_EDIT: Reproduce the qualified four-column attention gauge and three-scale absorption, then gauge-fix one coordinate of the second value-projection column and compensate both attention-output head blocks at initialization.

EVIDENCE: Successive value/output basis gauges achieved 99.96% at 1,377 parameters, 99.88% at 1,376, and 99.97% at 1,375; extending that consistently successful symmetry is more strongly supported than the failed LayerNorm, attention-output, MLP, and query-key reductions.

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and three weight-column output gauges removed."""

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
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and four output-shift gauges removed."""

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
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)

    @torch.no_grad()
    def rotate_value_basis_(self, rotation: torch.Tensor) -> None:
        """Compensate an orthogonal change of basis in every value head."""
        prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        weight = torch.cat(
            (
                torch.stack(prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        head_dim = rotation.size(0)
        rotated = torch.cat(
            [
                weight[:, start : start + head_dim]
                @ rotation.transpose(0, 1)
                for start in range(0, self.in_features, head_dim)
            ],
            dim=1,
        )
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                rotated[:-1, column] - rotated[-1, column]
            )
        self.weight_rest.copy_(
            rotated[:, len(self.weight_prefix) :]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class TwoAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final two scales absorbed downstream."""

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
=======
class GaugeFixedValueProjection(nn.Module):
    """Value projection with four orthogonal basis gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 4:
            raise ValueError("out_features must be at least four")
        self.in_features = in_features
        self.out_features = out_features
        self.first_column = nn.Parameter(
            torch.empty(out_features - 3)
        )
        self.second_column = nn.Parameter(
            torch.empty(out_features - 1)
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 2)
        )
        self.full_weight = None
        self.initial_rotation = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = None) -> None:
        raw = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        if std is None:
            nn.init.kaiming_uniform_(raw, a=math.sqrt(5))
        else:
            nn.init.normal_(raw, mean=0.0, std=std)

        pair = raw[:2, 0]
        radius = pair.square().sum().sqrt()
        rotation = torch.eye(
            self.out_features, device=raw.device, dtype=raw.dtype
        )
        if float(radius.item()) > 0.0:
            cosine = pair[0] / radius
            sine = pair[1] / radius
            rotation[0, 0] = cosine
            rotation[0, 1] = sine
            rotation[1, 0] = -sine
            rotation[1, 1] = cosine

        triple_radius = (
            radius.square() + raw[2, 0].square()
        ).sqrt()
        if float(triple_radius.item()) > 0.0:
            second = torch.eye(
                self.out_features,
                device=raw.device,
                dtype=raw.dtype,
            )
            cosine = radius / triple_radius
            sine = raw[2, 0] / triple_radius
            second[0, 0] = cosine
            second[0, 2] = sine
            second[2, 0] = -sine
            second[2, 2] = cosine
            rotation = second @ rotation

        fourth_radius = (
            triple_radius.square() + raw[3, 0].square()
        ).sqrt()
        if float(fourth_radius.item()) > 0.0:
            third = torch.eye(
                self.out_features,
                device=raw.device,
                dtype=raw.dtype,
            )
            cosine = triple_radius / fourth_radius
            sine = raw[3, 0] / fourth_radius
            third[0, 0] = cosine
            third[0, 3] = sine
            third[3, 0] = -sine
            third[3, 3] = cosine
            rotation = third @ rotation

        rotated = rotation @ raw
        residual_pair = rotated[1:3, 1]
        residual_radius = residual_pair.square().sum().sqrt()
        if float(residual_radius.item()) > 0.0:
            fourth = torch.eye(
                self.out_features,
                device=raw.device,
                dtype=raw.dtype,
            )
            cosine = residual_pair[0] / residual_radius
            sine = residual_pair[1] / residual_radius
            fourth[1, 1] = cosine
            fourth[1, 2] = sine
            fourth[2, 1] = -sine
            fourth[2, 2] = cosine
            rotation = fourth @ rotation
            rotated = fourth @ rotated

        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[4:, 0]))
        )
        self.second_column.copy_(
            torch.cat((rotated[:2, 1], rotated[3:, 1]))
        )
        self.weight_rest.copy_(rotated[:, 2:])
        self.initial_rotation = rotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(3),
                self.first_column[1:],
            )
        )
        second_column = torch.cat(
            (
                self.second_column[:2],
                self.second_column.new_zeros(1),
                self.second_column[2:],
            )
        )
        full_weight = torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                self.weight_rest,
            ),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.linear(x, full_weight)


class ThreeAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final three scales absorbed downstream."""

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = GaugeFixedValueProjection(
            d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = ThreeAbsorbedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.attn.proj.rotate_value_basis_(
                blk.attn.v_proj.initial_rotation
            )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
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
=======
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedValueProjection):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The final two ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            2, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
=======
    # The final three ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            3, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight_rest[:, -3:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -2:],
                    blk.attn.k_proj.weight.grad[:, -2:],
                    blk.attn.v_proj.weight.grad[:, -2:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            ).detach()
=======
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -3:],
                    blk.attn.k_proj.weight.grad[:, -3:],
                    blk.attn.v_proj.full_weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.full_weight[:, -3:],
                ),
                dim=0,
            ).detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight.grad[:, -3:].zero_()
            blk.attn.v_proj.weight_rest.grad[:, -3:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.attn.q_proj.weight[:, -2:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -2:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -2:].copy_(
                    effective_weight[k_end:]
                )
=======
                blk.attn.q_proj.weight[:, -3:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -3:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight_rest[:, -3:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE