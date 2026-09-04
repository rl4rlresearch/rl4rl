MECHANISM: Late-training exponential model averaging

HYPOTHESIS: A 0.98 EMA over the final 40% of training will exceed 9,262 correct predictions by reducing late optimizer noise while preserving the proven architecture and smoothing schedule.

INTENDED_EDIT: Replace AdamW with an AdamW subclass that averages learned parameters and floating-point BatchNorm buffers late in training, then installs the averaged state after the final optimizer step.

EVIDENCE: The 233,434-parameter design with 0.020→0.009 smoothing remains strongest at 9,262 correct, while capacity reallocations regressed and smaller-batch training timed out; averaging the existing trajectory tests optimization stability with little added computation.

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
=======
def build_model() -> nn.Module:
    return ImageClassifier()


class EMAAdamW(torch.optim.AdamW):
    def __init__(self, model: nn.Module, total_steps: int) -> None:
        super().__init__(
            model.parameters(),
            lr=3.0e-4,
            weight_decay=2.0e-4,
            betas=(0.9, 0.99),
        )
        self._ema_targets = list(model.parameters()) + [
            buffer
            for buffer in model.buffers()
            if buffer.is_floating_point()
        ]
        self._ema_values: list[torch.Tensor] | None = None
        self._step_count = 0
        self._total_steps = max(total_steps, 1)
        self._ema_start_step = max(1, math.ceil(0.6 * self._total_steps))
        self._ema_decay = 0.98

    def step(self, closure=None):
        loss = super().step(closure)
        self._step_count += 1
        if self._step_count >= self._ema_start_step:
            with torch.no_grad():
                if self._ema_values is None:
                    self._ema_values = [
                        tensor.detach().clone()
                        for tensor in self._ema_targets
                    ]
                else:
                    for average, tensor in zip(
                        self._ema_values, self._ema_targets
                    ):
                        average.lerp_(tensor.detach(), 1.0 - self._ema_decay)
                if self._step_count >= self._total_steps:
                    for tensor, average in zip(
                        self._ema_targets, self._ema_values
                    ):
                        tensor.copy_(average)
        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(model, total_steps)
>>>>>>> REPLACE