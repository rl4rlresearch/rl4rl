MECHANISM: Orthonormal centered pre-MLP row gauge

HYPOTHESIS: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while an orthonormal mean-zero representation removes one exact LayerNorm-null `fc1` component without the poor conditioning of the failed last-coordinate-zero gauge.

INTENDED_EDIT: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then replace `fc1` with a centered first-row gauge and update that row through the existing full ambient scale-factorized AdamW path.

EVIDENCE: Reference Design 3 achieved 99.39% at 1,378 parameters. The last-coordinate-zero `fc1` gauge collapsed to 10.66%, while centering an attention gauge improved its result to 93.3%; the centered `fc1` trial was never verified, so an orthonormal implementation is the most informative unresolved reduction.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and eleven output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(11)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 11)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with every output-shift gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(in_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        self.weight_rest.copy_(raw_weight[:, 11:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.bias.new_empty(
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
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
=======
        weight = torch.stack(full_weight_prefix, dim=1)
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
>>>>>>> REPLACE

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
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and four weight-column output gauges removed."""

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
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
class MLP(nn.Module):
=======
class CenteredGaugeLinear(nn.Module):
    """Linear layer with the first weight row's mean-null gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first = nn.Parameter(torch.empty(in_features - 1))
        self.rest = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        basis = torch.zeros(in_features, in_features - 1)
        for column in range(in_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.first.copy_(self.basis.transpose(0, 1) @ raw_weight[0])
        self.rest.copy_(raw_weight[1:])
        self.bias.copy_(raw_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.basis @ self.first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        if torch.is_grad_enabled():
            weight.retain_grad()
            self.full_weight = weight
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = CenteredGaugeLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = ThreeAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                module.weight_rest.copy_(raw_weight[:, 11:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
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
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.bias.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                nn.init.zeros_(module.bias)
        elif isinstance(module, CenteredGaugeLinear):
            with torch.no_grad():
                raw_weight = module.rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                module.first.copy_(
                    module.basis.transpose(0, 1) @ raw_weight[0]
                )
                module.rest.copy_(raw_weight[1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
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
                    blk.attn.v_proj.weight[:, -3:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    absorbed_scales = [
        torch.ones(8, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -8:])
        for blk in model.blocks
    ]
=======
    absorbed_scales = [
        torch.ones(8, device=device, dtype=blk.mlp.fc1.rest.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros(
            blk.mlp.fc1.out_features,
            blk.mlp.fc1.in_features,
            device=device,
            dtype=blk.mlp.fc1.rest.dtype,
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
                    blk.attn.v_proj.weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
                ),
                dim=0,
            ).detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -8:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -8:].detach()
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
=======
        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.full_weight.grad.detach().clone()
            )
            effective_weight = blk.mlp.fc1.full_weight.detach()
            virtual_weight = (
                effective_weight / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            stored_grad_sq = (
                blk.mlp.fc1.first.grad.detach().float().square().sum()
                + blk.mlp.fc1.rest.grad.detach().float().square().sum()
            )
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                    stored_grad_sq,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            for (
                _,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
                stored_grad_sq,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - stored_grad_sq
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
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
            blk.attn.v_proj.weight.grad[:, -3:].zero_()
            blk.mlp.fc1.first.grad.zero_()
            blk.mlp.fc1.rest.grad.zero_()
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
                blk.attn.v_proj.weight[:, -3:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
=======
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
                _,
            ) = absorbed_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -8:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                effective_weight = (
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
                centered_first = (
                    effective_weight[0] - effective_weight[0].mean()
                )
                blk.mlp.fc1.first.copy_(
                    blk.mlp.fc1.basis.transpose(0, 1)
                    @ centered_first
                )
                blk.mlp.fc1.rest.copy_(effective_weight[1:])
>>>>>>> REPLACE