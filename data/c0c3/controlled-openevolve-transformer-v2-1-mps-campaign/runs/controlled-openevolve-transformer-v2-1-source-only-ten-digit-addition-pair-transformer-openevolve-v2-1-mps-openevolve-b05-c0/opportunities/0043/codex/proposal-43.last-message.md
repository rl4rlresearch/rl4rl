MECHANISM: Virtual Adam coordinate for the absorbed value bias

HYPOTHESIS: Removing the final value-bias scalar while retaining its adaptive gradient moment as a virtual quotient direction will reduce the model to 1621 parameters and restore at least 99% accuracy.

INTENDED_EDIT: Eliminate `v_bias`; reconstruct its gradient from the projection column, maintain scalar Adam moments without a trainable parameter, and continuously fold each update into the gauge-fixed projection bias.

EVIDENCE: Directly removing the scalar reached 88.2% versus 99.92% with it present, despite its exact absorbability into projection bias, indicating lost optimization geometry rather than lost model capacity.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v_bias = torch.cat(
            (
                self.v_bias.new_zeros(5),
                self.v_bias,
                self.v_bias.new_zeros(2),
            )
        )
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class KeyGaugeAdamW:
=======
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class ValueBiasGaugeAdamW:
    """Virtual Adam coordinate for a value bias absorbed into output bias."""

    def __init__(
        self,
        gauges,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            bias: {
                "step": 0,
                "exp_avg": torch.zeros((), device=bias.device, dtype=bias.dtype),
                "exp_avg_sq": torch.zeros(
                    (), device=bias.device, dtype=bias.dtype
                ),
            }
            for bias, _, _ in self.gauges
        }
        self.pending = {}

    @torch.no_grad()
    def capture(self) -> None:
        self.pending.clear()
        for bias, projection, value_index in self.gauges:
            if bias.grad is None:
                continue

            grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat((grad, -grad.sum().reshape(1)))
            virtual_grad = torch.dot(
                full_grad,
                projection[:, value_index].detach(),
            )

            state = self.state[bias]
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
            self.pending[bias] = (
                (exp_avg / denom).clone(),
                bias_correction1,
            )

    @torch.no_grad()
    def step(self) -> None:
        for bias, projection, value_index in self.gauges:
            pending = self.pending.get(bias)
            if pending is None:
                continue

            direction, bias_correction1 = pending
            column = projection[:, value_index].detach()
            quotient_column = column[:-1] - column[-1]
            bias.add_(
                quotient_column * direction,
                alpha=-self.lr / bias_correction1,
            )
        self.pending.clear()


class KeyGaugeAdamW:
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    key_gauges,
    max_norm: float,
) -> None:
=======
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    key_gauges,
    value_bias_gauges,
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

    # Include the gradient of each eliminated value-bias coordinate.
    for bias, projection, value_index in value_bias_gauges:
        if bias.grad is not None:
            grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat((grad, -grad.sum().reshape(1)))
            virtual_grad = torch.dot(
                full_grad,
                projection[:, value_index].detach(),
            )
            total_sq.add_(virtual_grad.float().pow(2))

    coefficient = min(
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
    key_gauges = [
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
    value_bias_gauges = [
        (
            block.attn.proj_bias,
            block.attn.proj.weight,
            5,
        )
        for block in model.blocks
    ]
    key_gauges = [
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
    value_bias_optimizer = ValueBiasGaugeAdamW(
        value_bias_gauges,
        lr=train_cfg.lr,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now
        key_optimizer.lr = lr_now

        _, loss = model(x, y)
=======
        gauge_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
        value_bias_optimizer.lr = lr_now

        _, loss = model(x, y)
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
=======
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                key_gauges,
                value_bias_gauges,
                train_cfg.grad_clip,
            )
        value_bias_optimizer.capture()
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        value_bias_optimizer.step()
>>>>>>> REPLACE