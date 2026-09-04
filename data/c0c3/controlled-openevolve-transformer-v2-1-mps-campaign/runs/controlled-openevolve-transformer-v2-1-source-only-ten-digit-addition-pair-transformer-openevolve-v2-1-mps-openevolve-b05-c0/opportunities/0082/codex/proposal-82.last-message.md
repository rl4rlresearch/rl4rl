MECHANISM: Virtual-AdamW quotient of the remaining attention value bias

HYPOTHESIS: Removing the sole learned value-bias coordinate while preserving its virtual gradient, AdamW moments, gauge-aware clipping, and equivalent attention-projection bias updates will reduce the model from 1610 to 1609 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Eliminate `v_bias[5]`, reconstruct a zero value bias, and train its omitted coordinate virtually by absorbing each update into the gauge-fixed attention projection bias.

EVIDENCE: Feature 5 is the only retained and empirically essential value-bias pathway, so simple removal is risky; virtual optimization previously rescued an omitted LayerNorm-bias coordinate from 73.2% to 99.86%, supporting preservation of optimizer dynamics for this exact downstream-affine gauge.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.virtual_v_bias_feature = 5
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
class AffineBiasGaugeAdamW:
    """AdamW with virtual LayerNorm biases absorbed into a downstream bias."""
=======
class ValueBiasGaugeAdamW:
    """Virtual AdamW for a value bias absorbed into the projection bias."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            bias: {
                "step": 0,
                "exp_avg": torch.zeros(
                    (),
                    device=bias.device,
                    dtype=bias.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    (),
                    device=bias.device,
                    dtype=bias.dtype,
                ),
            }
            for _, bias, _ in self.gauges
        }
        self.pending_offsets = []

    @torch.no_grad()
    def step(self) -> None:
        self.pending_offsets = []
        for weight, bias, feature_index in self.gauges:
            if bias.grad is None:
                continue

            reduced_grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            virtual_grad = torch.dot(
                weight[:, feature_index].detach(),
                full_grad,
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
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            offset = -self.lr * direction / bias_correction1
            self.pending_offsets.append(
                (weight, bias, feature_index, offset)
            )

    @torch.no_grad()
    def project_biases(self) -> None:
        for weight, bias, feature_index, offset in self.pending_offsets:
            column = weight[:, feature_index]
            bias.add_((column[:-1] - column[-1]) * offset)
        self.pending_offsets = []


class AffineBiasGaugeAdamW:
    """AdamW with virtual LayerNorm biases absorbed into a downstream bias."""
>>>>>>> REPLACE

<<<<<<< SEARCH
    affine_bias_gauges,
    output_weight_gauges,
    max_norm: float,
=======
    affine_bias_gauges,
    output_weight_gauges,
    value_bias_gauges,
    max_norm: float,
>>>>>>> REPLACE

<<<<<<< SEARCH
            total_sq.add_(omitted.pow(2))

    coefficient = min(
        1.0,
=======
            total_sq.add_(omitted.pow(2))

    # Recover the omitted value-bias gradient through the output projection.
    for weight, bias, feature_index in value_bias_gauges:
        if bias.grad is not None:
            reduced_grad = bias.grad.detach().reshape(-1).float()
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            omitted = torch.dot(
                weight[:, feature_index].detach().float(),
                full_grad,
            )
            total_sq.add_(omitted.pow(2))

    coefficient = min(
        1.0,
>>>>>>> REPLACE

<<<<<<< SEARCH
    affine_bias_gauges = [
        (
            block.ln2.bias,
=======
    value_bias_gauges = [
        (
            block.attn.proj.weight,
            block.attn.proj_bias,
            block.attn.virtual_v_bias_feature,
        )
        for block in model.blocks
    ]
    affine_bias_gauges = [
        (
            block.ln2.bias,
>>>>>>> REPLACE

<<<<<<< SEARCH
    affine_bias_optimizer = AffineBiasGaugeAdamW(
        affine_bias_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    affine_bias_optimizer = AffineBiasGaugeAdamW(
        affine_bias_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    value_bias_optimizer = ValueBiasGaugeAdamW(
        value_bias_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output_weight_optimizer.lr = lr_now
        affine_bias_optimizer.lr = lr_now

        _, loss = model(x, y)
=======
        output_weight_optimizer.lr = lr_now
        affine_bias_optimizer.lr = lr_now
        value_bias_optimizer.lr = lr_now

        _, loss = model(x, y)
>>>>>>> REPLACE

<<<<<<< SEARCH
                affine_bias_gauges,
                output_weight_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
=======
                affine_bias_gauges,
                output_weight_gauges,
                value_bias_gauges,
                train_cfg.grad_clip,
            )
        value_bias_optimizer.step()
        key_optimizer.step()
>>>>>>> REPLACE

<<<<<<< SEARCH
        affine_bias_optimizer.project_biases()
        gauge_optimizer.step()
        token_position_optimizer.step()
=======
        affine_bias_optimizer.project_biases()
        gauge_optimizer.step()
        value_bias_optimizer.project_biases()
        token_position_optimizer.step()
>>>>>>> REPLACE