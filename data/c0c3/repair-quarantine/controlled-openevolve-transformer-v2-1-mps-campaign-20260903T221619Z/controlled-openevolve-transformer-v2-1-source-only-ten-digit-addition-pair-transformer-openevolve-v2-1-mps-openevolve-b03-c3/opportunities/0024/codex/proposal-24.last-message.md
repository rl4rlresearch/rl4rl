MECHANISM: Terminal MLP output-direction gauge fixing

HYPOTHESIS: Extending the qualified terminal-bias gauge to one `fc2` weight column will yield 1,625 parameters and retain at least 99% accuracy, because its omitted all-ones output component is erased by the final LayerNorm while full eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Gauge-fix the terminal MLP bias and first weight column, reconstruct both eight-dimensional tensors during forward passes, and optimize their stored differences using ambient AdamW moments and clipping.

EVIDENCE: The positional-plus-terminal-bias ambient gauge achieved 99.95% accuracy at 1,626 parameters. The new reduction applies the same proven gauge and optimizer treatment to one input-dependent scalar shift from the same terminal MLP, avoiding the earlier unsuccessful attention-output gauge.

<<<<<<< SEARCH
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)


class CausalSelfAttention(nn.Module):
=======
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)


class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with two all-ones output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_weight = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_first_weight = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.first_weight.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_first_weight = torch.cat(
            (self.first_weight, self.first_weight.new_zeros(1))
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_first_weight.retain_grad()
            full_bias.retain_grad()
            self.full_first_weight = full_first_weight
            self.full_bias = full_bias
        weight = torch.cat(
            (full_first_weight.unsqueeze(1), self.weight_rest), dim=1
        )
        return F.linear(x, weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedTerminalLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                module.first_weight.copy_(
                    raw_weight[:-1, 0] - raw_weight[-1, 0]
                )
                module.weight_rest.copy_(raw_weight[:, 1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The seven stored coordinates are differences from the omitted eighth
    # coordinate. Optimize all eight ambient coordinates, then apply their
    # update differences to the learned gauge-fixed parameter.
    gauge_param = model.pos_emb.first
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p is not gauge_param),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = torch.zeros(
        model_cfg.d_model, device=device, dtype=gauge_param.dtype
    )
    gauge_v = torch.zeros_like(gauge_m)
    gauge_step = 0
=======
    # Each stored seven-vector contains differences from an omitted eighth
    # coordinate. Preserve the full ambient AdamW dynamics for the positional
    # row, terminal bias, and one terminal weight column.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.extend(
            [blk.mlp.fc2.bias, blk.mlp.fc2.first_weight]
        )
    gauge_ids = {id(p) for p in gauge_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in gauge_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_param.grad = None
        loss.backward()

        full_gauge_grad = model.pos_emb.full_first.grad.detach()
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            # Parameter gradients contain the first seven coordinates; add
            # the omitted ambient coordinate to reproduce full-model clipping.
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            grad_sq = grad_sq + full_gauge_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()

        gauge_step += 1
        ambient_grad = full_gauge_grad * clip_scale
        gauge_m.mul_(0.9).add_(ambient_grad, alpha=0.1)
        gauge_v.mul_(0.999).addcmul_(
            ambient_grad, ambient_grad, value=0.001
        )
        m_hat = gauge_m / (1.0 - 0.9 ** gauge_step)
        v_hat = gauge_v / (1.0 - 0.999 ** gauge_step)
        direction = m_hat / (v_hat.sqrt() + 1e-8)
        with torch.no_grad():
            gauge_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
            gauge_param.add_(
                direction[-1] - direction[:-1], alpha=lr_now
            )
=======
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.extend(
                [
                    blk.mlp.fc2.full_bias.grad.detach(),
                    blk.mlp.fc2.full_first_weight.grad.detach(),
                ]
            )

        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            # Stored gradients contain the first seven coordinates; include
            # every omitted ambient coordinate in the clipping norm.
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()

        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            gauge_params, full_gauge_grads, gauge_m, gauge_v
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(ambient_grad, alpha=0.1)
            variance.mul_(0.999).addcmul_(
                ambient_grad, ambient_grad, value=0.001
            )
            m_hat = moment / (1.0 - 0.9 ** gauge_step)
            v_hat = variance / (1.0 - 0.999 ** gauge_step)
            direction = m_hat / (v_hat.sqrt() + 1e-8)
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )
>>>>>>> REPLACE