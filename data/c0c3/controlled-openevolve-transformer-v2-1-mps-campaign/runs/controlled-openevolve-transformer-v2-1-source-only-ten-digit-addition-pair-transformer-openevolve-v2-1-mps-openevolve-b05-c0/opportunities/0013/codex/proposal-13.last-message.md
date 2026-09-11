MECHANISM: Quotient-aware AdamW for a gauge-fixed attention output bias

HYPOTHESIS: Removing the LayerNorm-invariant common-shift coordinate from the attention output bias will produce a 1634-parameter model with at least 99% accuracy when virtual eight-coordinate AdamW preserves the successful optimizer dynamics.

INTENDED_EDIT: Store seven output-bias coordinates with the eighth fixed at zero, and optimize them using virtual eight-dimensional AdamW moments and gauge-aware gradient clipping.

EVIDENCE: The 1635-parameter model reached 99.32%, while ordinary fixed-coordinate and orthonormal output-bias gauges reached only 73.43% and 4.04%; this suggests the exact redundancy is removable but AdamW’s coordinate-dependent dynamics must be preserved.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = F.linear(y, self.proj.weight, F.pad(self.proj_bias, (0, 1)))
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
=======
    return min_lr + (base_lr - min_lr) * cosine


class GaugeFixedAdamW:
    """AdamW on a bias quotient while retaining virtual full-bias moments."""

    def __init__(
        self,
        parameters,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            p: {
                "step": 0,
                "exp_avg": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
                "exp_avg_sq": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
            }
            for p in self.parameters
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            grad = p.grad.detach().reshape(-1)
            virtual_grad = torch.cat((grad, -grad.sum().reshape(1)))
            state = self.state[p]
            state["step"] += 1
            step = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(virtual_grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad, virtual_grad, value=1.0 - self.beta2
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            direction = exp_avg / denom

            p.mul_(1.0 - self.lr * self.weight_decay)
            quotient_direction = direction[:-1] - direction[-1]
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    max_norm: float,
) -> None:
    active = [p for p in model.parameters() if p.grad is not None]
    if not active:
        return

    total_sq = torch.zeros(
        (),
        device=active[0].grad.device,
        dtype=torch.float32,
    )
    for p in active:
        total_sq.add_(p.grad.detach().float().pow(2).sum())

    # The omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    coefficient = min(
        1.0,
        max_norm / (float(total_sq.sqrt().item()) + 1e-6),
    )
    if coefficient < 1.0:
        for p in active:
            p.grad.mul_(coefficient)


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    gauge_parameters = [block.attn.proj_bias for block in model.blocks]
    gauge_parameter_ids = {id(p) for p in gauge_parameters}
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        ordinary_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
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
>>>>>>> REPLACE