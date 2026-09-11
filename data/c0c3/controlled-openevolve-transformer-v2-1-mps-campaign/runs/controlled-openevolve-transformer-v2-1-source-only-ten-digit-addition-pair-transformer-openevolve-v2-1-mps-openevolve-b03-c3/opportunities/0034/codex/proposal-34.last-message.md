MECHANISM: Single-row ambient-Adam gauge on the folded MLP input weight

HYPOTHESIS: Folding `ln2` scales into `fc1` and removing one exact mean-zero row-shift coordinate will produce a 1,477-parameter model with at least 99% accuracy while avoiding the overhead that prevented the 12-row version from finishing.

INTENDED_EDIT: Port the qualified LayerNorm-to-MLP fold, compact only the first `fc1` row to seven differences, and preserve full-width initialization, gradients, clipping, AdamW moments, decay, and updates in ambient coordinates.

EVIDENCE: The full-width fold achieved 99.88% at 1,478 parameters. The 1,466-parameter extension timed out rather than failing accuracy, so testing one of its exact row gauges is the smallest informative reduction with substantially less compact-parameter overhead.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
=======
class OneRowGaugeLinear(nn.Module):
    """Linear map with one mean-zero-input row-shift coordinate removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_row = nn.Parameter(torch.empty(in_features - 1))
        self.weight = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "initial_ambient_weight",
            torch.empty(out_features, in_features),
            persistent=False,
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def _store_raw_weight(self, raw_weight: torch.Tensor) -> None:
        self.initial_ambient_weight.copy_(raw_weight)
        self.first_row.copy_(
            raw_weight[0, :-1] - raw_weight[0, -1]
        )
        self.weight.copy_(raw_weight[1:])

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        self._store_raw_weight(raw_weight)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def reset_normal_parameters(self, std: float) -> None:
        raw_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(raw_weight, mean=0.0, std=std)
        self._store_raw_weight(raw_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_row = torch.cat(
            (self.first_row, self.first_row.new_zeros(1))
        )
        full_weight = torch.cat(
            (first_row.unsqueeze(0), self.weight), dim=0
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
            self.full_weight = full_weight
        return F.linear(x, full_weight, self.bias)

    @torch.no_grad()
    def set_effective_weight(
        self, full_weight: torch.Tensor
    ) -> None:
        self.first_row.copy_(
            full_weight[0, :-1] - full_weight[0, -1]
        )
        self.weight.copy_(full_weight[1:])


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = OneRowGaugeLinear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
=======
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, OneRowGaugeLinear):
            module.reset_normal_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    value_attentions = [
        blk.attn for blk in model.blocks
    ]
    mlp_linears = [
        blk.mlp.fc1 for blk in model.blocks
    ]
    mlp_compact_params = [
        p
        for linear in mlp_linears
        for p in (linear.first_row, linear.weight)
    ]
    mlp_compact_ids = {
        id(p) for p in mlp_compact_params
    }
    special_ids = {
        id(p)
        for p in (
            gauge_params
            + [position_bias_param]
            + value_bias_params
            + projection_bias_params
            + mlp_compact_params
        )
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]
    gauge_step = 0
=======
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]

    # Optimizer state for the eliminated ln2 scales and the ambient
    # full-width fc1 weights. The model stores their canonical product.
    mlp_ambient_weights = [
        linear.initial_ambient_weight.detach().clone()
        for linear in mlp_linears
    ]
    mlp_ambient_scales = [
        torch.ones(
            linear.in_features, device=device, dtype=linear.weight.dtype
        )
        for linear in mlp_linears
    ]
    mlp_weight_m = [
        torch.zeros_like(weight) for weight in mlp_ambient_weights
    ]
    mlp_weight_v = [
        torch.zeros_like(moment) for moment in mlp_weight_m
    ]
    mlp_scale_m = [
        torch.zeros_like(scale) for scale in mlp_ambient_scales
    ]
    mlp_scale_v = [
        torch.zeros_like(moment) for moment in mlp_scale_m
    ]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()
=======
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        for linear in mlp_linears:
            linear.first_row.grad = None
            linear.weight.grad = None
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
        ]
        clip_scale = 1.0
=======
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
        ]
        effective_mlp_grads = [
            linear.full_weight.grad.detach()
            for linear in mlp_linears
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
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
=======
                for p in model.parameters()
                if (
                    p.grad is not None
                    and id(p) not in mlp_compact_ids
                )
            )
            for full_grad in full_gauge_grads:
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                projection_param.add_(
                    attention.proj.weight[:, -1]
                    * value_direction[-1],
                    alpha=-lr_now,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
                projection_param.add_(
                    attention.proj.weight[:, -1]
                    * value_direction[-1],
                    alpha=-lr_now,
                )

        for (
            stored_linear,
            ambient_weight,
            ambient_scale,
            weight_grad,
            scale_grad,
            weight_moment,
            weight_variance,
            scale_moment,
            scale_variance,
        ) in zip(
            mlp_linears,
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
            weight_moment.mul_(0.9).add_(
                weight_grad, alpha=0.1
            )
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(
                scale_grad, alpha=0.1
            )
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
                stored_linear.set_effective_weight(
                    ambient_weight * ambient_scale.unsqueeze(0)
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE