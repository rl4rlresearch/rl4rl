MECHANISM: Virtual AdamW for an affine LayerNorm-bias gauge

HYPOTHESIS: Fixing LayerNorm-2 bias coordinate 6 while preserving its omitted gradient, AdamW moments, and downstream `fc1.bias` compensation will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 6, reconstruct it as zero, and optimize its virtual coordinate through the exact downstream-affine gauge.

EVIDENCE: Directly fixing coordinate 6 reduced accuracy to 73.2%, while the current four-coordinate quotient reaches 99.96%; restoring the removed coordinate’s optimization pathway directly tests whether optimizer dynamics caused that failure.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:1],
                self.bias.new_zeros(3),
                self.bias[1:],
            )
        )
=======
        full_bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:1],
                self.bias.new_zeros(3),
                self.bias[1:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


def clip_grad_norm_with_virtual_gauge(
=======
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


class AffineBiasGaugeAdamW:
    """AdamW with a virtual LayerNorm bias absorbed into a downstream bias."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + 1,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + 1,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter in self.parameters
        }
        self.pending_offsets = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.pending_offsets = []
        for (
            parameter,
            downstream_weight,
            downstream_bias,
            feature_index,
            omitted_position,
        ) in self.gauges:
            if parameter.grad is None or downstream_bias.grad is None:
                continue

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_empty(grad.numel() + 1)
            virtual_grad[:omitted_position] = grad[:omitted_position]
            virtual_grad[omitted_position + 1 :] = grad[omitted_position:]
            virtual_grad[omitted_position] = torch.dot(
                downstream_weight[:, feature_index].detach(),
                downstream_bias.grad.detach(),
            )

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            keep = torch.ones(
                direction.numel(),
                dtype=torch.bool,
                device=direction.device,
            )
            keep[omitted_position] = False

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                direction[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
            omitted_update = (
                -self.lr * direction[omitted_position] / bias_correction1
            )
            self.pending_offsets.append(
                (
                    downstream_weight,
                    downstream_bias,
                    feature_index,
                    omitted_update,
                )
            )

    @torch.no_grad()
    def project_biases(self) -> None:
        for weight, bias, feature_index, offset in self.pending_offsets:
            bias.add_(weight[:, feature_index] * offset)
        self.pending_offsets = []


def clip_grad_norm_with_virtual_gauge(
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
    affine_bias_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))

    coefficient = min(
=======
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))

    # Recover each omitted LayerNorm-bias gradient through the downstream affine.
    for (
        parameter,
        downstream_weight,
        downstream_bias,
        feature_index,
        _,
    ) in affine_bias_gauges:
        if parameter.grad is not None and downstream_bias.grad is not None:
            omitted = torch.dot(
                downstream_weight[:, feature_index].detach().float(),
                downstream_bias.grad.detach().float(),
            )
            total_sq.add_(omitted.pow(2))

    coefficient = min(
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
    affine_bias_gauges = [
        (
            block.ln2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
            6,
            2,
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
                for parameter, _, _, _, _ in affine_bias_gauges
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
    affine_bias_optimizer = AffineBiasGaugeAdamW(
        affine_bias_gauges,
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
=======
        gauge_optimizer.lr = lr_now
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
        affine_bias_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        affine_bias_optimizer.zero_grad(set_to_none=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        token_position_optimizer.step()
=======
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                affine_bias_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        affine_bias_optimizer.step()
        optimizer.step()
        affine_bias_optimizer.project_biases()
        gauge_optimizer.step()
        token_position_optimizer.step()
>>>>>>> REPLACE