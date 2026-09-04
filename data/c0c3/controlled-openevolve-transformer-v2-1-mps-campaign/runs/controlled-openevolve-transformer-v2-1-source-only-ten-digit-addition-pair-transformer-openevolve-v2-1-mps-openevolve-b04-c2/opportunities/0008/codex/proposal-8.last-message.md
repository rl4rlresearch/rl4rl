MECHANISM: Quotient-space terminal bias with full-coordinate AdamW dynamics

HYPOTHESIS: Combining the verified key-bias elimination with a seven-parameter terminal MLP bias will reach at least 99% accuracy at 1,635 parameters if training reconstructs the omitted coordinate’s gradient and preserves the original eight-coordinate AdamW and gradient-clipping dynamics.

INTENDED_EDIT: Remove the redundant attention key bias and one common-mode terminal MLP bias parameter, while maintaining virtual optimizer state for the eliminated MLP coordinate.

EVIDENCE: The optimizer-order-preserving key-bias design achieved 99.71% at 1,636 parameters, while the direct seven-coordinate terminal-bias parameterization fell to 73.37%; this isolates altered optimization geometry as the most informative issue to address.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve construction RNG and parameter ordering while removing the
        # softmax-invariant key bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The following final LayerNorm cancels common shifts, so retain only
        # seven relative output-bias coordinates.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class GaugeFixedAdamW(torch.optim.AdamW):
    """AdamW that trains compact biases as quotients of full bias vectors."""

    def __init__(self, params, gauge_params, **kwargs):
        all_params = list(params)
        self.gauge_params = list(gauge_params)
        gauge_ids = {id(p) for p in self.gauge_params}
        super().__init__([p for p in all_params if id(p) not in gauge_ids], **kwargs)
        self.gauge_state = {
            p: {
                "step": 0,
                "full": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
                "exp_avg": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
                "exp_avg_sq": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
            }
            for p in self.gauge_params
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for p in self.gauge_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        for p in self.gauge_params:
            if p.grad is None:
                continue

            state = self.gauge_state[p]
            grad = torch.cat((p.grad, -p.grad.sum().reshape(1)))
            state["step"] += 1
            state["full"].mul_(1.0 - lr * weight_decay)
            state["exp_avg"].lerp_(grad, 1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denom = state["exp_avg_sq"].sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            state["full"].addcdiv_(
                state["exp_avg"],
                denom,
                value=-lr / bias_correction1,
            )
            p.copy_(state["full"][:-1] - state["full"][-1])

        return loss


@torch.no_grad()
def clip_grad_norm_gauge(parameters, gauge_params, max_norm: float) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    gauge_ids = {id(p) for p in gauge_params}
    norms = []
    for p in params:
        if id(p) in gauge_ids:
            grad = torch.cat((p.grad, -p.grad.sum().reshape(1)))
            norms.append(torch.linalg.vector_norm(grad))
        else:
            norms.append(torch.linalg.vector_norm(p.grad))

    if not norms:
        return torch.tensor(0.0)

    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    clip_scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for p in params:
        p.grad.mul_(clip_scale)
    return total_norm


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    gauge_params = [blk.mlp.fc2.bias for blk in model.blocks]
    optimizer = GaugeFixedAdamW(
        model.parameters(),
        gauge_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        if train_cfg.grad_clip > 0:
            clip_grad_norm_gauge(model.parameters(), gauge_params, train_cfg.grad_clip)
        optimizer.step()
>>>>>>> REPLACE