MECHANISM: Batched final MLP output-column shift gauge

HYPOTHESIS: Zero-anchoring the remaining full `fc2` column will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy, and batching all gauge updates into one optimizer will avoid the timeouts seen with prior final-column attempts.

INTENDED_EDIT: Replace MLP column 6 with seven learned relative coordinates and an implicit zero anchor, then update all gauge parameters through one foreach-based `GaugeAdamW` instance.

EVIDENCE: The verified 1,573-parameter model reached 99.98% accuracy while applying the identical symmetry to the other eleven `fc2` columns; both previous final-column attempts timed out without adverse accuracy evidence, motivating the same reduction with lower optimizer dispatch overhead.

<<<<<<< SEARCH
        self.register_parameter("fc2_col_4_abs", None)
        self.register_parameter("fc2_col_5_abs", None)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.register_parameter("fc2_col_4_abs", None)
        self.register_parameter("fc2_col_5_abs", None)
        self.register_parameter("fc2_col_6", None)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_col_5_abs = torch.cat(
            (self.fc2_col_5_abs, self.fc2_col_5_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5 = torch.cat(
            (self.fc2_col_5, self.fc2_col_5.new_zeros(1))
        ).unsqueeze(1)
=======
        fc2_col_5_abs = torch.cat(
            (self.fc2_col_5_abs, self.fc2_col_5_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_6 = torch.cat(
            (self.fc2_col_6, self.fc2_col_6.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5 = torch.cat(
            (self.fc2_col_5, self.fc2_col_5.new_zeros(1))
        ).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                fc2_col_4_abs,
                fc2_col_5_abs,
                self.fc2.weight,
                fc2_col_5,
=======
                fc2_col_4_abs,
                fc2_col_5_abs,
                fc2_col_6,
                fc2_col_5,
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_fc2_col_5_abs = block.mlp.fc2.weight[:, 5].detach().clone()
                full_fc2_col_5 = block.mlp.fc2.weight[:, -5].detach().clone()
                full_fc2_col_4 = block.mlp.fc2.weight[:, -4].detach().clone()
                full_fc2_col_3 = block.mlp.fc2.weight[:, -3].detach().clone()
                full_fc2_col_2 = block.mlp.fc2.weight[:, -2].detach().clone()
                base_fc2_weight = block.mlp.fc2.weight[:, 6:-5].detach().clone()
                block.mlp.fc2.weight = nn.Parameter(base_fc2_weight)
=======
                full_fc2_col_5_abs = block.mlp.fc2.weight[:, 5].detach().clone()
                full_fc2_col_6 = block.mlp.fc2.weight[:, 6].detach().clone()
                full_fc2_col_5 = block.mlp.fc2.weight[:, -5].detach().clone()
                full_fc2_col_4 = block.mlp.fc2.weight[:, -4].detach().clone()
                full_fc2_col_3 = block.mlp.fc2.weight[:, -3].detach().clone()
                full_fc2_col_2 = block.mlp.fc2.weight[:, -2].detach().clone()
                block.mlp.fc2.weight = None
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_fc2_col_5_abs.sub_(full_fc2_col_5_abs[-1].clone())
                block.mlp.fc2_col_5_abs = nn.Parameter(
                    full_fc2_col_5_abs[:-1].clone()
                )
                full_fc2_col_5.sub_(full_fc2_col_5[-1].clone())
=======
                full_fc2_col_5_abs.sub_(full_fc2_col_5_abs[-1].clone())
                block.mlp.fc2_col_5_abs = nn.Parameter(
                    full_fc2_col_5_abs[:-1].clone()
                )
                full_fc2_col_6.sub_(full_fc2_col_6[-1].clone())
                block.mlp.fc2_col_6 = nn.Parameter(
                    full_fc2_col_6[:-1].clone()
                )
                full_fc2_col_5.sub_(full_fc2_col_5[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class GaugeAdamW:
    """Batched AdamW for parameters with one implicit shift coordinate each."""

    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = [
            torch.zeros(param.numel() + 1, device=param.device, dtype=param.dtype)
            for param in self.parameters
        ]
        self.exp_avg_sq = [
            torch.zeros_like(exp_avg) for exp_avg in self.exp_avg
        ]

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        active = [
            (param, exp_avg, exp_avg_sq)
            for param, exp_avg, exp_avg_sq in zip(
                self.parameters, self.exp_avg, self.exp_avg_sq
            )
            if param.grad is not None
        ]
        if not active:
            return

        parameters, exp_avg, exp_avg_sq = map(list, zip(*active))
        full_grads = [
            torch.cat(
                (
                    param.grad.reshape(-1),
                    -param.grad.sum().reshape(1),
                )
            )
            for param in parameters
        ]
        self.step_count += 1

        torch._foreach_mul_(exp_avg, self.beta1)
        torch._foreach_add_(exp_avg, full_grads, alpha=1.0 - self.beta1)
        torch._foreach_mul_(exp_avg_sq, self.beta2)
        torch._foreach_addcmul_(
            exp_avg_sq,
            full_grads,
            full_grads,
            value=1.0 - self.beta2,
        )

        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        updates = [
            (avg / bias_correction1)
            / ((avg_sq / bias_correction2).sqrt() + self.eps)
            for avg, avg_sq in zip(exp_avg, exp_avg_sq)
        ]
        relative_updates = [
            update[:-1] - update[-1] for update in updates
        ]

        torch._foreach_mul_(
            parameters, 1.0 - self.lr * self.weight_decay
        )
        torch._foreach_add_(
            parameters, relative_updates, alpha=-self.lr
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.mlp.fc2_col_5_abs for block in model.blocks
    ]
=======
    ] + [
        block.mlp.fc2_col_5_abs for block in model.blocks
    ] + [
        block.mlp.fc2_col_6 for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_optimizers = [
        GaugeAdamW(param, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        for param in gauge_parameters
    ]
=======
    gauge_optimizer = GaugeAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.lr = lr_now
=======
        gauge_optimizer.lr = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.zero_grad()
        loss.backward()
=======
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad()
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.step()
=======
        optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE