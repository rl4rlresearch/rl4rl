MECHANISM: MLP output common-column gauge erased by final LayerNorm

HYPOTHESIS: Fixing `fc2.weight[4, 0]` and subtracting its value from every output row will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because it changes each residual by only a token-dependent uniform feature shift that the final LayerNorm removes, while a virtual AdamW coordinate preserves full-gradient optimization.

INTENDED_EDIT: Extend the coordinate-4 MLP-output gauge to one weight-column coordinate, reconstruct the canonical full weight during forward passes, and add virtual-gradient, clipping, and optimizer handling for the omitted weight.

EVIDENCE: The current 1616-parameter model achieves 99.93% while retaining the successful coordinate-4 MLP-output bias gauge and virtual AdamW treatment. This tests the corresponding activation-dependent output-shift quotient after additional token-position gauges proved accuracy-sensitive.

<<<<<<< SEARCH
class GaugeFixedMLPOutput(nn.Module):
    """MLP output projection with bias coordinate 4 fixed to zero."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fixed_index = 4

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(source.weight.detach().clone())
        full_bias = source.bias.detach()
        self.bias = nn.Parameter(
            torch.cat(
                (
                    full_bias[: self.fixed_index],
                    full_bias[self.fixed_index + 1 :],
                )
            ).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class GaugeFixedMLPOutput(nn.Module):
    """MLP output projection with weight-column and bias gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_index = 4
        self.fixed_weight_row = 4
        self.fixed_weight_col = 0
        self.fixed_weight_index = (
            self.fixed_weight_row * in_features + self.fixed_weight_col
        )

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            self._reduce_weight(source.weight.detach())
        )
        full_bias = source.bias.detach()
        self.bias = nn.Parameter(
            torch.cat(
                (
                    full_bias[: self.fixed_index],
                    full_bias[self.fixed_index + 1 :],
                )
            ).clone()
        )

    def _weight_keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.out_features * self.in_features,
            dtype=torch.bool,
            device=device,
        )
        keep[self.fixed_weight_index] = False
        return keep

    def _reduce_weight(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[
            self.fixed_weight_row,
            self.fixed_weight_col,
        ].clone()
        gauged[:, self.fixed_weight_col].sub_(anchor)
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
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
        return F.linear(x, self.full_weight(), full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPOutput):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
=======
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
class TokenPositionGaugeAdamW:
=======
class MLPWeightGaugeAdamW:
    """AdamW with a virtual coordinate for an MLP output-column gauge."""

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
            parameter for parameter, _, _, _, _ in self.gauges
        ]
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
            for parameter, _, _, _, _ in self.gauges
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
            parameter,
            out_features,
            in_features,
            fixed_row,
            fixed_col,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_index = fixed_row * in_features + fixed_col
            full_numel = parameter.numel() + 1
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[fixed_index] = False

            virtual_grad = parameter.grad.new_zeros(full_numel)
            virtual_grad[keep] = parameter.grad.detach().reshape(-1)
            virtual_matrix = virtual_grad.view(out_features, in_features)
            virtual_grad[fixed_index] = -virtual_matrix[:, fixed_col].sum()

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
            quotient_full.view(out_features, in_features)[:, fixed_col].sub_(
                direction[fixed_index]
            )
            quotient_direction = quotient_full[keep]

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


class TokenPositionGaugeAdamW:
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
    mlp_weight_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
=======
    # Recover the omitted MLP weight gradients from output-shift invariance.
    for (
        parameter,
        out_features,
        in_features,
        fixed_row,
        fixed_col,
    ) in mlp_weight_gauges:
        if parameter.grad is not None:
            fixed_index = fixed_row * in_features + fixed_col
            virtual_grad = parameter.grad.detach().float().new_zeros(
                parameter.numel() + 1
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[fixed_index] = False
            virtual_grad[keep] = parameter.grad.detach().reshape(-1).float()
            omitted = -virtual_grad.view(
                out_features,
                in_features,
            )[:, fixed_col].sum()
            total_sq.add_(omitted.pow(2))

    # Recover the omitted key gradients from the LayerNorm-null directions.
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
    mlp_weight_gauges = [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_col,
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
                for parameter, _, _, _, _ in mlp_weight_gauges
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
=======
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    mlp_weight_optimizer = MLPWeightGaugeAdamW(
        mlp_weight_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
=======
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
        mlp_weight_optimizer.lr = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        mlp_weight_optimizer.zero_grad(set_to_none=True)
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
                token_position_gauge,
                key_gauges,
                train_cfg.grad_clip,
=======
                token_position_gauge,
                key_gauges,
                mlp_weight_gauges,
                train_cfg.grad_clip,
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        token_position_optimizer.step()
=======
        key_optimizer.step()
        mlp_weight_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        token_position_optimizer.step()
>>>>>>> REPLACE