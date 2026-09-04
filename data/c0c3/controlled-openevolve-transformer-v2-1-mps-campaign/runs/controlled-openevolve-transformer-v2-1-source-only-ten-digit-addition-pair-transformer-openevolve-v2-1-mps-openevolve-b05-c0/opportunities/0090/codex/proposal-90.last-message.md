MECHANISM: Virtual-AdamW quotient of one LayerNorm-null MLP input weight

HYPOTHESIS: Gauge-fixing `fc1.weight[0, 7]` and absorbing its virtual updates into `fc1.bias` will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because this is the first application of the proven LayerNorm-null weight symmetry to the MLP input projection rather than an eighth quotient in the saturated key projection.

INTENDED_EDIT: Replace `fc1` with a one-coordinate gauge-fixed affine layer, reconstruct the omitted gradient, retain full-coordinate AdamW moments and clipping, and compensate each omitted-coordinate update through the affine bias.

EVIDENCE: Seven virtual-AdamW key-projection LayerNorm-null quotients reached 99.97% at 1607 parameters, while adding the eighth key quotient failed at 46.34%; applying the same exact symmetry to a distinct affine projection is the most informative next reduction.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            full_bias,
            self.eps,
        )


class GaugeFixedMLPOutput(nn.Module):
=======
    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )

    def full_bias(self) -> torch.Tensor:
        return torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.full_bias(),
            self.eps,
        )


class GaugeFixedMLPInput(nn.Module):
    """Affine projection with one LayerNorm-null input coordinate removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_row = 0
        self.fixed_column = in_features - 1
        self.fixed_index = self.fixed_row * in_features + self.fixed_column

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(self._reduce_weight(source.weight.detach()))
        self.bias = nn.Parameter(source.bias.detach().clone())

    def _weight_keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.out_features * self.in_features,
            dtype=torch.bool,
            device=device,
        )
        keep[self.fixed_index] = False
        return keep

    def _reduce_weight(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[self.fixed_row, self.fixed_column].clone()
        gauged[self.fixed_row].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._weight_keep_mask(flat.device)].clone()

    def full_weight(self) -> torch.Tensor:
        keep = self._weight_keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(self.out_features, self.in_features)

    @torch.no_grad()
    def reset_weight_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce_weight(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight(), self.bias)


class GaugeFixedMLPOutput(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = GaugeFixedMLPInput(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPOutput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
            nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedMLPInput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMLPOutput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
            nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPOutputWeightGaugeAdamW:
=======
class AffineInputWeightGaugeAdamW:
    """Virtual AdamW for an affine input weight removed through LayerNorm."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [module.weight for module, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            module.weight: {
                "step": 0,
                "exp_avg": torch.zeros(
                    module.weight.numel() + 1,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    module.weight.numel() + 1,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                ),
            }
            for module, _ in self.gauges
        }
        self.pending_directions = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def prepare_step(self) -> None:
        self.pending_directions = []
        for module, layer_norm in self.gauges:
            parameter = module.weight
            if parameter.grad is None or module.bias.grad is None:
                continue

            full_numel = parameter.numel() + 1
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[module.fixed_index] = False
            virtual_grad = parameter.grad.new_zeros(full_numel)
            virtual_grad[keep] = parameter.grad.detach().reshape(-1)

            scale = layer_norm.full_weight().detach()
            shift = layer_norm.full_bias().detach()
            virtual_matrix = virtual_grad.view(
                module.out_features,
                module.in_features,
            )
            virtual_grad[module.fixed_index] = (
                -scale[module.fixed_column]
                * (virtual_matrix[module.fixed_row] / scale).sum()
                + scale[module.fixed_column]
                * (shift / scale).sum()
                * module.bias.grad.detach()[module.fixed_row]
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
            self.pending_directions.append(
                (module, layer_norm, direction.clone(), bias_correction1)
            )

    @torch.no_grad()
    def step(self) -> None:
        for (
            module,
            layer_norm,
            direction,
            bias_correction1,
        ) in self.pending_directions:
            parameter = module.weight
            keep = torch.ones(
                parameter.numel() + 1,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[module.fixed_index] = False

            scale = layer_norm.full_weight().detach()
            shift = layer_norm.full_bias().detach()
            fixed_direction = direction[module.fixed_index].clone()
            quotient_full = direction.clone().view(
                module.out_features,
                module.in_features,
            )
            quotient_full[module.fixed_row].sub_(
                fixed_direction
                * scale[module.fixed_column]
                / scale
            )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_full.reshape(-1)[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )

            fixed_update = (
                -self.lr * fixed_direction / bias_correction1
            )
            module.bias[module.fixed_row].add_(
                fixed_update
                * scale[module.fixed_column]
                * (shift / scale).sum()
            )
        self.pending_directions = []


class MLPOutputWeightGaugeAdamW:
>>>>>>> REPLACE

<<<<<<< SEARCH
            parameter,
            downstream_weight,
            downstream_bias,
            feature_indices,
            omitted_positions,
        ) in self.gauges:
=======
            parameter,
            downstream_module,
            downstream_bias,
            feature_indices,
            omitted_positions,
        ) in self.gauges:
>>>>>>> REPLACE

<<<<<<< SEARCH
                virtual_grad[omitted_position] = torch.dot(
                    downstream_weight[:, feature_index].detach(),
                    downstream_bias.grad.detach(),
                )
=======
                virtual_grad[omitted_position] = torch.dot(
                    downstream_module.full_weight()[
                        :, feature_index
                    ].detach(),
                    downstream_bias.grad.detach(),
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                        downstream_weight,
                        downstream_bias,
                        feature_index,
                        omitted_update,
=======
                        downstream_module,
                        downstream_bias,
                        feature_index,
                        omitted_update,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def project_biases(self) -> None:
        for weight, bias, feature_index, offset in self.pending_offsets:
            bias.add_(weight[:, feature_index] * offset)
        self.pending_offsets = []


def clip_grad_norm_with_virtual_gauge(
=======
    def project_biases(self) -> None:
        for module, bias, feature_index, offset in self.pending_offsets:
            bias.add_(module.full_weight()[:, feature_index] * offset)
        self.pending_offsets = []


def clip_grad_norm_with_virtual_gauge(
>>>>>>> REPLACE

<<<<<<< SEARCH
    affine_bias_gauges,
    output_weight_gauges,
    value_bias_gauges,
=======
    affine_bias_gauges,
    input_weight_gauges,
    output_weight_gauges,
    value_bias_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
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

    # Recover omitted MLP output-weight gradients from common-shift invariance.
=======
        parameter,
        downstream_module,
        downstream_bias,
        feature_indices,
        _,
    ) in affine_bias_gauges:
        if parameter.grad is not None and downstream_bias.grad is not None:
            for feature_index in feature_indices:
                omitted = torch.dot(
                    downstream_module.full_weight()[
                        :, feature_index
                    ].detach().float(),
                    downstream_bias.grad.detach().float(),
                )
                total_sq.add_(omitted.pow(2))

    # Recover an omitted affine-input gradient from LayerNorm invariance.
    for module, layer_norm in input_weight_gauges:
        if module.weight.grad is not None and module.bias.grad is not None:
            virtual_grad = module.weight.grad.detach().float().new_zeros(
                module.weight.numel() + 1
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[module.fixed_index] = False
            virtual_grad[keep] = (
                module.weight.grad.detach().reshape(-1).float()
            )
            scale = layer_norm.full_weight().detach().float()
            shift = layer_norm.full_bias().detach().float()
            virtual_matrix = virtual_grad.view(
                module.out_features,
                module.in_features,
            )
            omitted = (
                -scale[module.fixed_column]
                * (virtual_matrix[module.fixed_row] / scale).sum()
                + scale[module.fixed_column]
                * (shift / scale).sum()
                * module.bias.grad.detach().float()[module.fixed_row]
            )
            total_sq.add_(omitted.pow(2))

    # Recover omitted MLP output-weight gradients from common-shift invariance.
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_weight_gauges = [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_column,
        )
        for block in model.blocks
    ]
=======
    input_weight_gauges = [
        (block.mlp.fc1, block.ln2)
        for block in model.blocks
    ]
    output_weight_gauges = [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_column,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.ln2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
=======
            block.ln2.bias,
            block.mlp.fc1,
            block.mlp.fc1.bias,
>>>>>>> REPLACE

<<<<<<< SEARCH
            *[parameter for parameter, _, _, _ in key_gauges],
            *[
                parameter
                for parameter, _, _, _, _ in output_weight_gauges
            ],
=======
            *[parameter for parameter, _, _, _ in key_gauges],
            *[module.weight for module, _ in input_weight_gauges],
            *[
                parameter
                for parameter, _, _, _, _ in output_weight_gauges
            ],
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_weight_optimizer = MLPOutputWeightGaugeAdamW(
        output_weight_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    input_weight_optimizer = AffineInputWeightGaugeAdamW(
        input_weight_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    output_weight_optimizer = MLPOutputWeightGaugeAdamW(
        output_weight_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.lr = lr_now
        output_weight_optimizer.lr = lr_now
=======
        key_optimizer.lr = lr_now
        input_weight_optimizer.lr = lr_now
        output_weight_optimizer.lr = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.zero_grad(set_to_none=True)
        output_weight_optimizer.zero_grad(set_to_none=True)
=======
        key_optimizer.zero_grad(set_to_none=True)
        input_weight_optimizer.zero_grad(set_to_none=True)
        output_weight_optimizer.zero_grad(set_to_none=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                key_gauges,
                affine_bias_gauges,
                output_weight_gauges,
=======
                key_gauges,
                affine_bias_gauges,
                input_weight_gauges,
                output_weight_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_optimizer.step()
        key_optimizer.step()
        output_weight_optimizer.step()
        affine_bias_optimizer.step()
        optimizer.step()
        affine_bias_optimizer.project_biases()
=======
        input_weight_optimizer.prepare_step()
        value_bias_optimizer.step()
        key_optimizer.step()
        output_weight_optimizer.step()
        affine_bias_optimizer.step()
        optimizer.step()
        input_weight_optimizer.step()
        affine_bias_optimizer.project_biases()
>>>>>>> REPLACE