MECHANISM: Dual virtual-AdamW LayerNorm-bias affine gauge

HYPOTHESIS: Virtually optimizing omitted LayerNorm-2 bias coordinates 1 and 6 will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because their learned updates remain represented through independent AdamW moments and compensation in `fc1.bias`.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 1, generalize the existing affine-gauge optimizer and gradient clipping to two omitted coordinates, and transfer both virtual updates through the downstream affine bias.

EVIDENCE: Virtual optimization rescued coordinate 6 from 73.2% to 99.86%; coordinate 1’s simple fixing reached 87.11%, while related coordinate-1 value-bias and LayerNorm-scale reductions reached 99.98% and 99.81%, making optimizer restoration the most informative next test.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 5))
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 1, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class AffineBiasGaugeAdamW:
    """AdamW with virtual LayerNorm biases absorbed into a downstream bias."""

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
                    parameter.numel() + len(feature_indices),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(feature_indices),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, feature_indices, _ in self.gauges
        }
        self.pending_offsets = []
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    @torch.no_grad()
    def step(self) -> None:
        self.pending_offsets = []
        for (
            parameter,
            downstream_weight,
            downstream_bias,
            feature_indices,
            omitted_positions,
        ) in self.gauges:
            if parameter.grad is None or downstream_bias.grad is None:
                continue

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(
                grad.numel() + len(omitted_positions)
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[list(omitted_positions)] = False
            virtual_grad[keep] = grad
            for feature_index, omitted_position in zip(
                feature_indices,
                omitted_positions,
            ):
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

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                direction[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
            for feature_index, omitted_position in zip(
                feature_indices,
                omitted_positions,
            ):
                omitted_update = (
                    -self.lr
                    * direction[omitted_position]
                    / bias_correction1
                )
                self.pending_offsets.append(
                    (
                        downstream_weight,
                        downstream_bias,
                        feature_index,
                        omitted_update,
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    for (
        parameter,
        downstream_weight,
        downstream_bias,
        feature_indices,
        _,
    ) in affine_bias_gauges:
        if parameter.grad is not None and downstream_bias.grad is not None:
            for feature_index in feature_indices:
                omitted = torch.dot(
                    downstream_weight[:, feature_index].detach().float(),
                    downstream_bias.grad.detach().float(),
                )
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    affine_bias_gauges = [
        (
            block.ln2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
            (1, 6),
            (0, 2),
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE