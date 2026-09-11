MECHANISM: Virtual-AdamW LayerNorm-scale gauge projection

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 3 while restoring its omitted gradient and AdamW moment through the downstream `fc1` affine gauge will reduce the model to 1615 parameters and improve the prior 98.67% result to at least 99%.

INTENDED_EDIT: Remove LayerNorm scale coordinate 3, then jointly optimize the reduced scale and downstream `fc1` parameters using a virtual fifth scale coordinate, gauge-projected updates, and gauge-aware gradient clipping.

EVIDENCE: Fixing coordinate 3 previously reached 98.67%, the closest reported 1615-parameter attempt; coordinates 0, 1, and 4 are already fixed in the 99.93% design, so restoring optimizer dynamics for coordinate 3 is the most targeted follow-up.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 0, 1, and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 0, 1, 3, and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for two coupled embedding gauges."""
=======
class LayerNormScaleGaugeAdamW:
    """AdamW with a virtual LayerNorm scale absorbed into an affine map."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [
            parameter
            for scale, _, weight, bias in self.gauges
            for parameter in (scale, weight, bias)
        ]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.compact_index = 1
        self.feature_index = 3
        self.state = {
            scale: {
                "step": 0,
                "scale_exp_avg": torch.zeros(
                    scale.numel() + 1,
                    device=scale.device,
                    dtype=scale.dtype,
                ),
                "scale_exp_avg_sq": torch.zeros(
                    scale.numel() + 1,
                    device=scale.device,
                    dtype=scale.dtype,
                ),
                "weight_exp_avg": torch.zeros_like(weight),
                "weight_exp_avg_sq": torch.zeros_like(weight),
                "bias_exp_avg": torch.zeros_like(bias),
                "bias_exp_avg_sq": torch.zeros_like(bias),
            }
            for scale, _, weight, bias in self.gauges
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for scale, scale_bias, weight, bias in self.gauges:
            if (
                scale.grad is None
                or weight.grad is None
                or bias.grad is None
            ):
                continue

            keep = torch.ones(
                scale.numel() + 1,
                dtype=torch.bool,
                device=scale.device,
            )
            keep[self.compact_index] = False
            virtual_scale_grad = scale.grad.new_zeros(keep.numel())
            virtual_scale_grad[keep] = scale.grad.detach().reshape(-1)

            beta = scale_bias.detach()[self.feature_index]
            column = weight.detach()[:, self.feature_index].clone()
            virtual_scale_grad[self.compact_index] = (
                column
                * (
                    weight.grad.detach()[:, self.feature_index]
                    - beta * bias.grad.detach()
                )
            ).sum()
            weight_grad = weight.grad.detach()
            bias_grad = bias.grad.detach()

            state = self.state[scale]
            state["step"] += 1
            step = state["step"]

            scale_exp_avg = state["scale_exp_avg"]
            scale_exp_avg_sq = state["scale_exp_avg_sq"]
            scale_exp_avg.mul_(self.beta1).add_(
                virtual_scale_grad,
                alpha=1.0 - self.beta1,
            )
            scale_exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_scale_grad,
                virtual_scale_grad,
                value=1.0 - self.beta2,
            )

            weight_exp_avg = state["weight_exp_avg"]
            weight_exp_avg_sq = state["weight_exp_avg_sq"]
            weight_exp_avg.mul_(self.beta1).add_(
                weight_grad,
                alpha=1.0 - self.beta1,
            )
            weight_exp_avg_sq.mul_(self.beta2).addcmul_(
                weight_grad,
                weight_grad,
                value=1.0 - self.beta2,
            )

            bias_exp_avg = state["bias_exp_avg"]
            bias_exp_avg_sq = state["bias_exp_avg_sq"]
            bias_exp_avg.mul_(self.beta1).add_(
                bias_grad,
                alpha=1.0 - self.beta1,
            )
            bias_exp_avg_sq.mul_(self.beta2).addcmul_(
                bias_grad,
                bias_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            correction2 = math.sqrt(bias_correction2)
            scale_direction = scale_exp_avg / (
                scale_exp_avg_sq.sqrt().div(correction2).add(self.eps)
            )
            weight_direction = weight_exp_avg / (
                weight_exp_avg_sq.sqrt().div(correction2).add(self.eps)
            )
            bias_direction = bias_exp_avg / (
                bias_exp_avg_sq.sqrt().div(correction2).add(self.eps)
            )

            omitted_direction = scale_direction[self.compact_index]
            quotient_weight = weight_direction.clone()
            quotient_weight[:, self.feature_index].add_(
                column * omitted_direction
            )
            quotient_bias = bias_direction - (
                column * beta * omitted_direction
            )

            scale.mul_(1.0 - self.lr * self.weight_decay)
            weight.mul_(1.0 - self.lr * self.weight_decay)
            bias.mul_(1.0 - self.lr * self.weight_decay)
            scale.add_(
                scale_direction[keep].view_as(scale),
                alpha=-self.lr / bias_correction1,
            )
            weight.add_(
                quotient_weight,
                alpha=-self.lr / bias_correction1,
            )
            bias.add_(
                quotient_bias,
                alpha=-self.lr / bias_correction1,
            )


class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for two coupled embedding gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    token_position_gauge,
    key_gauges,
    max_norm: float,
) -> None:
=======
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    token_position_gauge,
    key_gauges,
    scale_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
=======
    # Recover the omitted LayerNorm-scale gradient through the affine gauge.
    for scale, scale_bias, weight, bias in scale_gauges:
        if (
            scale.grad is not None
            and weight.grad is not None
            and bias.grad is not None
        ):
            beta = scale_bias.detach().float()[3]
            column = weight.detach().float()[:, 3]
            omitted = (
                column
                * (
                    weight.grad.detach().float()[:, 3]
                    - beta * bias.grad.detach().float()
                )
            ).sum()
            total_sq.add_(omitted.pow(2))

    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            token_position_gauge[0],
            token_position_gauge[1],
            *gauge_parameters,
            *[parameter for parameter, _, _, _ in key_gauges],
        ]
    }
=======
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    scale_gauges = [
        (
            block.ln2.weight,
            block.ln2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            token_position_gauge[0],
            token_position_gauge[1],
            *gauge_parameters,
            *[parameter for parameter, _, _, _ in key_gauges],
            *[
                parameter
                for scale, _, weight, bias in scale_gauges
                for parameter in (scale, weight, bias)
            ],
        ]
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scale_optimizer = LayerNormScaleGaugeAdamW(
        scale_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        gauge_optimizer.lr = lr_now
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
        scale_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
=======
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                scale_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        scale_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE