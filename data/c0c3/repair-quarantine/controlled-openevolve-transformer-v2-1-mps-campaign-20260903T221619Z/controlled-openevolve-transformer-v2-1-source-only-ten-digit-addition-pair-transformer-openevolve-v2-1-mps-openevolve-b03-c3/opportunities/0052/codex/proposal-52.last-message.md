MECHANISM: Coupled LayerNorm-scale absorption with mean-zero input-weight gauge

HYPOTHESIS: A 1,523-parameter model will retain at least 99% accuracy because five-scale absorption already achieved 99.95%, while each `fc1` row has one exactly redundant coefficient on LayerNorm’s mean-zero subspace; jointly updating the full ambient scales and weights preserves their AdamW dynamics before materializing the deduplicated model.

INTENDED_EDIT: Reproduce five absorbed `ln2` scales and remove the final input coefficient from every `fc1` row, training the corresponding full eight-scale and eight-column representation solely as optimizer-coordinate state.

EVIDENCE: The qualified five-scale factorization reached 99.95% at 1,535 parameters, whereas the sixth-scale extension failed; this patch keeps the verified five fixed scales and targets a distinct exact 12-parameter redundancy while avoiding the unstable quotient reconstruction by directly maintaining full ambient weights and scales.

<<<<<<< SEARCH
class OneFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed by the following linear."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedTerminalLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class FiveFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with five scales absorbed by the following linear."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )


class MeanGaugeLinear(nn.Module):
    """Linear layer with one mean-zero input coefficient removed per row."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "initial_full_weight",
            torch.empty(out_features, in_features),
            persistent=False,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.initial_full_weight.copy_(raw_weight)
        self.weight.copy_(
            raw_weight[:, :-1] - raw_weight[:, -1:]
        )
        self.bias.copy_(raw_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight.new_zeros(self.out_features, 1)),
            dim=1,
        )
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanGaugeLinear(d_model, d_ff)
        self.fc2 = GaugeFixedTerminalLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = OneFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        elif isinstance(module, MeanGaugeLinear):
            with torch.no_grad():
                raw_weight = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                module.initial_full_weight.copy_(raw_weight)
                module.weight.copy_(
                    raw_weight[:, :-1] - raw_weight[:, -1:]
                )
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    # Jointly preserve the full eight-scale/eight-column AdamW coordinates.
    # The submitted model stores five fewer scales and one fewer coefficient
    # per fc1 row, while these tensors exist only as training optimizer state.
    factor_params = []
    for blk in model.blocks:
        factor_params.append(blk.ln2.weight)
        factor_params.append(blk.mlp.fc1.weight)

    excluded_ids = {
        id(p) for p in gauge_params + factor_params
    }
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in excluded_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0

    ambient_weights = [
        blk.mlp.fc1.initial_full_weight.detach().clone()
        for blk in model.blocks
    ]
    ambient_scales = [
        torch.ones(
            model_cfg.d_model,
            device=device,
            dtype=blk.mlp.fc1.weight.dtype,
        )
        for blk in model.blocks
    ]
    ambient_weight_m = [
        torch.zeros_like(weight) for weight in ambient_weights
    ]
    ambient_weight_v = [
        torch.zeros_like(weight) for weight in ambient_weights
    ]
    ambient_scale_m = [
        torch.zeros_like(scale) for scale in ambient_scales
    ]
    ambient_scale_v = [
        torch.zeros_like(scale) for scale in ambient_scales
    ]
    factor_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        for factor_param in factor_params:
            factor_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        clip_scale = 1.0
=======
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        factor_grads = []
        for blk, ambient_weight, ambient_scale in zip(
            model.blocks, ambient_weights, ambient_scales
        ):
            model_scale = torch.cat(
                (
                    blk.ln2.weight.detach(),
                    blk.ln2.weight.new_ones(5),
                )
            )
            stored_grad = blk.mlp.fc1.weight.grad.detach()
            coefficient_prefix_grad = (
                stored_grad / model_scale[:-1].unsqueeze(0)
            )
            coefficient_grad = torch.cat(
                (
                    coefficient_prefix_grad,
                    -coefficient_prefix_grad.sum(
                        dim=1, keepdim=True
                    ),
                ),
                dim=1,
            )
            ambient_weight_grad = (
                coefficient_grad * ambient_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                coefficient_grad * ambient_weight
            ).sum(dim=0)
            factor_grads.append(
                (ambient_weight_grad, ambient_scale_grad)
            )

        # Only ambient gradients contribute to clipping and updates for this
        # coupled factorization.
        for factor_param in factor_params:
            factor_param.grad = None

        clip_scale = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
=======
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for ambient_weight_grad, ambient_scale_grad in factor_grads:
                grad_sq = (
                    grad_sq
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        gauge_step += 1
=======
        optimizer.step()

        factor_step += 1
        factor_bc1 = 1.0 - 0.9 ** factor_step
        factor_bc2 = 1.0 - 0.999 ** factor_step
        for i, (blk, factor_grad) in enumerate(
            zip(model.blocks, factor_grads)
        ):
            ambient_weight_grad, ambient_scale_grad = factor_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = ambient_weight_m[i]
            weight_variance = ambient_weight_v[i]
            scale_moment = ambient_scale_m[i]
            scale_variance = ambient_scale_v[i]
            ambient_weight = ambient_weights[i]
            ambient_scale = ambient_scales[i]

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / factor_bc1
            ) / (
                (weight_variance / factor_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / factor_bc1
            ) / (
                (scale_variance / factor_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                ambient_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                ambient_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )

                model_scale = torch.cat(
                    (
                        ambient_scale[:3],
                        ambient_scale.new_ones(5),
                    )
                )
                coefficients = (
                    ambient_weight * ambient_scale.unsqueeze(0)
                )
                blk.ln2.weight.copy_(ambient_scale[:3])
                blk.mlp.fc1.weight.copy_(
                    (
                        coefficients[:, :-1]
                        - coefficients[:, -1:]
                    )
                    / model_scale[:-1].unsqueeze(0)
                )

        gauge_step += 1
>>>>>>> REPLACE