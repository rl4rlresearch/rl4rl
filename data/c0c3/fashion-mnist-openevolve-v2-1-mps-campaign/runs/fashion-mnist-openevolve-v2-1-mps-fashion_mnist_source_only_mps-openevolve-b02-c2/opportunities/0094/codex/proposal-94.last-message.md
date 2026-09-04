MECHANISM: Final-valley checkpoint averaging

HYPOTHESIS: Averaging four sparsely sampled low-learning-rate states with the terminal state will strictly exceed the current 9320.419371 score by stabilizing late optimization without adding forward passes or learned parameters.

INTENDED_EDIT: Accumulate floating model parameters and BatchNorm state every 16 steps during the final 64 steps, then install their average with the terminal state when evaluation begins.

EVIDENCE: The hard-maximum model reliably completed in 75.3 seconds with 9,320 correct, whereas top-k-based improvements repeatedly timed out; this isolates low-overhead late-state averaging on the reliable architecture.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )
        self._checkpoint_sum: dict[str, torch.Tensor] | None = None
        self._checkpoint_count = 0

    def train(self, mode: bool = True) -> ImageClassifier:
        if not mode and self._checkpoint_sum is not None:
            with torch.no_grad():
                state = self.state_dict()
                divisor = float(self._checkpoint_count + 1)
                for name, total in self._checkpoint_sum.items():
                    state[name].mul_(1.0 / divisor)
                    state[name].add_(total, alpha=1.0 / divisor)
            self._checkpoint_sum = None
            self._checkpoint_count = 0
        return super().train(mode)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
    optimizer._averaging_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    model = optimizer._averaging_model
    if step >= max(total_steps - 64, 0) and step % 16 == 0:
        with torch.no_grad():
            floating_state = {
                name: tensor
                for name, tensor in model.state_dict().items()
                if tensor.is_floating_point()
            }
            if model._checkpoint_sum is None:
                model._checkpoint_sum = {
                    name: tensor.detach().clone()
                    for name, tensor in floating_state.items()
                }
            else:
                for name, tensor in floating_state.items():
                    model._checkpoint_sum[name].add_(tensor)
            model._checkpoint_count += 1

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE