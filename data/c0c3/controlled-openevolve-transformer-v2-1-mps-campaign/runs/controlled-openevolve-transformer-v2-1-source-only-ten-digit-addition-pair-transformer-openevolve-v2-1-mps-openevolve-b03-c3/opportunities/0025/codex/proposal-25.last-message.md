MECHANISM: Dual ambient-Adam gauge fixing

HYPOTHESIS: Gauge-fixing the terminal MLP output bias alongside the existing positional gauge, while preserving both full eight-coordinate AdamW dynamics, will reproduce the qualified 1,626-parameter design with at least 99% accuracy.

INTENDED_EDIT: Replace the eight-parameter `fc2` bias with seven learned differences and manually optimize both gauge-fixed vectors using their ambient gradients, moments, weight decay, and clipping contributions.

EVIDENCE: Reference Design 3 achieved 99.95% accuracy with 1,626 parameters using exactly these two ambient-Adam gauges, while adding a third gauge caused accuracy to collapse.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedBiasLinear(nn.Module):
    """Linear layer whose output bias omits its all-ones gauge scalar."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
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
    # Optimize each seven-coordinate gauge parameter through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
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
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
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