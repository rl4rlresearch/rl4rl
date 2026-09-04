MECHANISM: Quotient-aware LayerNorm key-projection gauge fixing

HYPOTHESIS: Removing one key-projection coordinate will reduce the model from 1633 to 1632 parameters while retaining at least 99% accuracy, because LayerNorm confines token variation to a mean-zero subspace, making one inverse-scale key direction position-constant and therefore exactly invisible to causal softmax; virtual full-coordinate AdamW preserves the omitted coordinate’s optimizer dynamics.

INTENDED_EDIT: Replace the QKV projection with a one-scalar gauge-fixed projection, preserve gauge-equivalent initialization and RNG ordering, and optimize its reconstructed full gradient with gauge-aware AdamW and clipping.

EVIDENCE: The 1633-parameter tied-embedding quotient achieved 99.72%, while positional and MLP-bias quotients failed despite optimizer corrections; this motivates testing a distinct exact redundancy located inside the key projection, whose removed direction is directly canceled by softmax.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
    """QKV projection with one softmax-invisible key coordinate removed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.fixed_index = d_model * d_model + d_model - 1

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(d_model, 3 * d_model, bias=False)
        full_weight = source.weight.detach().clone()
        self.weight = nn.Parameter(self._reduce(full_weight))

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[self.d_model, -1].clone()
        gauged[self.d_model].sub_(anchor)
        flat = gauged.reshape(-1)
        return torch.cat((flat[: self.fixed_index], flat[self.fixed_index + 1 :]))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_zeros(1),
                self.weight[self.fixed_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = GaugeFixedQKV(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedQKV):
            full_weight = torch.empty(
                3 * module.d_model,
                module.d_model,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    max_norm: float,
) -> None:
=======
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
        self.parameters = [parameter for parameter, _, _ in self.gauges]
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

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for parameter, ln_scale, d_model in self.gauges:
            if parameter.grad is None:
                continue

            fixed_index = d_model * d_model + d_model - 1
            row_start = d_model * d_model
            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = torch.cat(
                (
                    grad[:fixed_index],
                    grad.new_zeros(1),
                    grad[fixed_index:],
                )
            )

            scale = ln_scale.detach().reshape(-1)
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

            quotient_direction = torch.cat(
                (direction[:fixed_index], direction[fixed_index + 1 :])
            ).clone()
            quotient_direction[row_start:fixed_index].sub_(
                direction[fixed_index] * scale[-1] / scale[:-1]
            )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    key_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    coefficient = min(
=======
    # The omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    # Recover the omitted key gradient from the LayerNorm-null direction.
    for parameter, ln_scale, d_model in key_gauges:
        if parameter.grad is not None:
            fixed_index = d_model * d_model + d_model - 1
            row_start = d_model * d_model
            grad = parameter.grad.detach().reshape(-1).float()
            scale = ln_scale.detach().reshape(-1).float()
            omitted = -scale[-1] * (
                grad[row_start:fixed_index] / scale[:-1]
            ).sum()
            total_sq.add_(omitted.pow(2))

    coefficient = min(
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
    gauge_parameter_ids = {id(p) for p in gauge_parameters}
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
    key_gauges = [
        (block.attn.qkv.weight, block.ln1.weight, model_cfg.d_model)
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            *gauge_parameters,
            *[parameter for parameter, _, _ in key_gauges],
        ]
    }
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                train_cfg.grad_clip,
            )
        optimizer.step()
        gauge_optimizer.step()
=======
        gauge_optimizer.lr = lr_now
        key_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE