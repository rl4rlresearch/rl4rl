MECHANISM: Gauge-aware anchored attention-output bias

HYPOTHESIS: Representing the shift-invariant eight-dimensional attention projection bias with seven relative coordinates and updating them with an implicit eight-coordinate AdamW state will reduce the model from 1,608 to 1,607 parameters while retaining at least 99% accuracy after 9,000 steps.

INTENDED_EDIT: Anchor the final projection-bias coordinate at zero, reconstruct it during inference, and use a gauge-aware optimizer that preserves the relative updates of the original eight-coordinate AdamW optimization.

EVIDENCE: The current 1,608-parameter model reached 99.91%, while naive zero-sum projection-bias reparameterization fell to 46.64%; this tests whether optimizer-geometry distortion, rather than loss of model capacity, caused that exact-gauge reduction to fail.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class GaugeAdamW:
    """AdamW on relative coordinates with one implicit shift-gauge coordinate."""

    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.param = param
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = torch.zeros(param.numel() + 1, device=param.device, dtype=param.dtype)
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        if self.param.grad is None:
            return

        grad = self.param.grad
        full_grad = torch.cat((grad, -grad.sum().reshape(1)))
        self.step_count += 1

        self.exp_avg.mul_(self.beta1).add_(full_grad, alpha=1.0 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(
            full_grad, full_grad, value=1.0 - self.beta2
        )

        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        update = (self.exp_avg / bias_correction1) / (
            (self.exp_avg_sq / bias_correction2).sqrt() + self.eps
        )

        self.param.mul_(1.0 - self.lr * self.weight_decay)
        self.param.add_(update[:-1] - update[-1], alpha=-self.lr)


def clip_grad_norm_with_gauges(
    parameters: List[torch.nn.Parameter],
    gauge_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        total_sq.add_(grad.detach().float().square().sum())
    for param in gauge_parameters:
        if param.grad is not None:
            total_sq.add_(param.grad.detach().float().sum().square())

    scale = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale.to(dtype=grad.dtype))


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    gauge_parameters = [block.attn.proj.bias for block in model.blocks]
    gauge_ids = {id(param) for param in gauge_parameters}
    ordinary_parameters = [param for param in model.parameters() if id(param) not in gauge_ids]
    optimizer = torch.optim.AdamW(
        ordinary_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizers = [
        GaugeAdamW(param, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        for param in gauge_parameters
    ]

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
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.zero_grad()
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauges(
                list(model.parameters()),
                gauge_parameters,
                train_cfg.grad_clip,
            )
        optimizer.step()
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.step()
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=8000)
=======
    p.add_argument("--train-steps", type=int, default=9000)
>>>>>>> REPLACE