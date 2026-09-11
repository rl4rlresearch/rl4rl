MECHANISM: LayerNorm-null query-weight/q-bias affine gauge

HYPOTHESIS: Fixing `qkv.weight[4, 7]` will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because the omitted query-weight direction is exactly absorbed by `q_bias[4]`, with virtual AdamW moments preserving optimization of the full affine map.

INTENDED_EDIT: Add query row 4 to the QKV quotient, reconstruct its omitted gradient using LayerNorm scale, LayerNorm bias, and query-bias gradients, and jointly project QKV and query-bias AdamW updates.

EVIDENCE: The current feature-4 embedding, LayerNorm-scale, LayerNorm-bias, and MLP-output gauges support 99.93% accuracy. Unlike the failed removal of an essential value bias or an additional key coordinate, this change retains the complete learned query affine map through an exact bias-coupled LayerNorm-null gauge.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with five softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with five key gauges and one query affine gauge fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.query_fixed_rows = (second_offset,)
        self.key_fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
        self.fixed_rows = self.query_fixed_rows + self.key_fixed_rows
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeyGaugeAdamW:
    """AdamW with a virtual coordinate for a LayerNorm-null key direction."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, fixed_rows in self.gauges
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for parameter, ln_scale, d_model, fixed_rows in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            full_numel = parameter.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[list(fixed_indices)] = False

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(full_numel)
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1)
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()

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
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            direction = exp_avg / denom

            quotient_full = direction.clone()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                quotient_full[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            quotient_direction = quotient_full[keep]

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
=======
class ProjectionGaugeAdamW:
    """AdamW for LayerNorm-null key and bias-coupled query gauges."""

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
            for weight, _, _, q_bias, _, _, _ in self.gauges
            for parameter in (weight, q_bias)
        ]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            weight: {
                "step": 0,
                "exp_avg": torch.zeros(
                    weight.numel() + len(key_rows) + len(query_rows),
                    device=weight.device,
                    dtype=weight.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    weight.numel() + len(key_rows) + len(query_rows),
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            }
            for (
                weight,
                _,
                _,
                _,
                _,
                key_rows,
                query_rows,
            ) in self.gauges
        }
        self.bias_state = {
            q_bias: {
                "exp_avg": torch.zeros_like(q_bias),
                "exp_avg_sq": torch.zeros_like(q_bias),
            }
            for _, _, _, q_bias, _, _, _ in self.gauges
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for (
            weight,
            ln_scale,
            ln_bias,
            q_bias,
            d_model,
            key_rows,
            query_rows,
        ) in self.gauges:
            if weight.grad is None:
                continue

            fixed_rows = tuple(query_rows) + tuple(key_rows)
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            full_numel = weight.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=weight.device,
            )
            keep[list(fixed_indices)] = False

            grad = weight.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(full_numel)
            virtual_grad[keep] = grad
            q_bias_grad = (
                q_bias.grad.detach()
                if q_bias.grad is not None
                else torch.zeros_like(q_bias)
            )

            scale = ln_scale.detach().reshape(-1)
            layer_bias = ln_bias.detach().reshape(-1)
            bias_coupling = layer_bias[-1] + scale[-1] * (
                layer_bias[:-1] / scale[:-1]
            ).sum()

            for row in query_rows:
                fixed_index = row * d_model + d_model - 1
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum() + bias_coupling * q_bias_grad[row]

            for row in key_rows:
                fixed_index = row * d_model + d_model - 1
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()

            state = self.state[weight]
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

            bias_state = self.bias_state[q_bias]
            bias_exp_avg = bias_state["exp_avg"]
            bias_exp_avg_sq = bias_state["exp_avg_sq"]
            bias_exp_avg.mul_(self.beta1).add_(
                q_bias_grad,
                alpha=1.0 - self.beta1,
            )
            bias_exp_avg_sq.mul_(self.beta2).addcmul_(
                q_bias_grad,
                q_bias_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            bias_direction = bias_exp_avg / (
                bias_exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )

            quotient_full = direction.clone()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                quotient_full[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            quotient_direction = quotient_full[keep]

            quotient_bias = bias_direction.clone()
            for row in query_rows:
                fixed_index = row * d_model + d_model - 1
                quotient_bias[row].add_(
                    direction[fixed_index] * bias_coupling
                )

            weight.mul_(1.0 - self.lr * self.weight_decay)
            q_bias.mul_(1.0 - self.lr * self.weight_decay)
            weight.add_(
                quotient_direction.view_as(weight),
                alpha=-self.lr / bias_correction1,
            )
            q_bias.add_(
                quotient_bias,
                alpha=-self.lr / bias_correction1,
            )
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
    projection_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            grad = parameter.grad.detach().reshape(-1).float()
            virtual_grad = grad.new_zeros(grad.numel() + len(fixed_indices))
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1).float()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))
=======
    # Recover omitted key and bias-coupled query gradients.
    for (
        parameter,
        ln_scale,
        ln_bias,
        q_bias,
        d_model,
        key_rows,
        query_rows,
    ) in projection_gauges:
        if parameter.grad is not None:
            fixed_rows = tuple(query_rows) + tuple(key_rows)
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            grad = parameter.grad.detach().reshape(-1).float()
            virtual_grad = grad.new_zeros(grad.numel() + len(fixed_indices))
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1).float()
            layer_bias = ln_bias.detach().reshape(-1).float()
            bias_coupling = layer_bias[-1] + scale[-1] * (
                layer_bias[:-1] / scale[:-1]
            ).sum()
            q_bias_grad = (
                q_bias.grad.detach().float()
                if q_bias.grad is not None
                else torch.zeros_like(q_bias, dtype=torch.float32)
            )

            for row in query_rows:
                fixed_index = row * d_model + d_model - 1
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum() + bias_coupling * q_bias_grad[row]
                total_sq.add_(omitted.pow(2))

            for row in key_rows:
                fixed_index = row * d_model + d_model - 1
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))
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
    projection_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            block.ln1.bias,
            block.attn.q_bias,
            model_cfg.d_model,
            block.attn.qkv.key_fixed_rows,
            block.attn.qkv.query_fixed_rows,
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            token_position_gauge[0],
            token_position_gauge[1],
            *gauge_parameters,
            *[gauge[0] for gauge in projection_gauges],
            *[gauge[3] for gauge in projection_gauges],
        ]
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    projection_optimizer = ProjectionGaugeAdamW(
        projection_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.lr = lr_now
=======
        projection_optimizer.lr = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
=======
        projection_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                token_position_gauge,
                projection_gauges,
                train_cfg.grad_clip,
            )
        projection_optimizer.step()
>>>>>>> REPLACE