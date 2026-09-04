MECHANISM: LayerNorm-null bias gauge with quotient-preserving AdamW

HYPOTHESIS: Representing the attention projection bias with seven parameters and updating them as differences of the original eight-coordinate AdamW trajectory will produce at least 99% accuracy with 1635 parameters.

INTENDED_EDIT: Fix the final projection-bias coordinate at zero, reconstruct the full bias during inference, and preserve the eliminated coordinate’s optimization effect through non-model AdamW moment state and gauge-aware gradient clipping.

EVIDENCE: The 1636-parameter model reached 99.98%, while ordinary seven-coordinate projection-bias training reached only 68.44%; because a common projection-bias shift is removed by downstream LayerNorm, the failure motivates preserving the original optimizer trajectory on the seven-dimensional quotient rather than deleting one coordinate’s update.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugeFixedBiasLinear(nn.Linear):
    """Linear layer whose bias is represented modulo a common output shift."""

    def __init__(self, in_features: int, out_features: int):
        # Let Linear consume the baseline constructor RNG before replacing the
        # functionally redundant bias coordinate.
        super().__init__(in_features, out_features, bias=True)
        self.bias = nn.Parameter(self.weight.new_zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = GaugeFixedBiasLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
@torch.no_grad()
def clip_grad_norm_with_gauge(
    model: TinyDecoderLM, gauge_biases: List[torch.nn.Parameter], max_norm: float
) -> None:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return

    norms = [torch.linalg.vector_norm(g, 2.0) for g in grads]
    for bias in gauge_biases:
        if bias.grad is not None:
            # Translation invariance makes the omitted full-bias gradient the
            # negative sum of the retained coordinate gradients.
            norms.append(torch.abs(bias.grad.sum()))

    total_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(scale)


@torch.no_grad()
def quotient_adamw_step(
    gauge_biases: List[torch.nn.Parameter],
    states: List[Dict],
    lr: float,
    weight_decay: float,
) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for bias, state in zip(gauge_biases, states):
        if bias.grad is None:
            continue

        full_grad = torch.cat((bias.grad, -bias.grad.sum().reshape(1)))
        state["step"] += 1
        state["exp_avg"].mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(
            full_grad, full_grad, value=1.0 - beta2
        )

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denom = state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
        denom = denom.add(eps)
        full_update = (lr / bias_correction1) * state["exp_avg"] / denom

        # Store b_i - b_last. This is the exact observable quotient of the
        # original eight-coordinate AdamW update.
        bias.mul_(1.0 - lr * weight_decay)
        bias.add_(full_update[-1] - full_update[:-1])


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

    gauge_biases = [block.attn.proj.bias for block in model.blocks]
    gauge_ids = {id(bias) for bias in gauge_biases}
    ordinary_params = [p for p in model.parameters() if id(p) not in gauge_ids]
    optimizer = torch.optim.AdamW(
        ordinary_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    gauge_states = [
        {
            "step": 0,
            "exp_avg": bias.new_zeros(bias.numel() + 1),
            "exp_avg_sq": bias.new_zeros(bias.numel() + 1),
        }
        for bias in gauge_biases
    ]

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauge(model, gauge_biases, train_cfg.grad_clip)
        optimizer.step()
        quotient_adamw_step(
            gauge_biases, gauge_states, lr_now, train_cfg.weight_decay
        )
>>>>>>> REPLACE