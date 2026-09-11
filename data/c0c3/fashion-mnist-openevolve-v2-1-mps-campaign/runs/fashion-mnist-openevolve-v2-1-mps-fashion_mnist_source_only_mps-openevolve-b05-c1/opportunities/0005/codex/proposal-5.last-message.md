MECHANISM: Exponential moving average of weights and BatchNorm state

HYPOTHESIS: Replacing the final iterate with a bias-controlled EMA of the 782 training iterates will exceed 9,159 correct predictions by reducing late-stage optimization noise without the timeout incurred by batch size 64.

INTENDED_EDIT: Add an AdamW subclass that tracks model parameters and floating-point buffers after every update, then installs the averaged state after the final optimizer step; retain batch size 128, training, and cardinal-view inference.

EVIDENCE: Cardinal translation ensembling improved correctness from 9,134 to 9,159, while additional diagonal views regressed and batch size 64 timed out; this motivates a low-overhead optimization improvement without expanding validation inference or optimizer-step count.

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return ImageClassifier()
=======
class EMAAdamW(torch.optim.AdamW):
    def __init__(self, model: nn.Module, **kwargs: object) -> None:
        super().__init__(model.parameters(), **kwargs)
        self._model = model
        self._ema_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update_ema(self, step: int, total_steps: int) -> None:
        decay = min(0.99, (step + 1) / (step + 10))
        for name, value in self._model.state_dict().items():
            averaged = self._ema_state[name]
            if torch.is_floating_point(averaged):
                averaged.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                averaged.copy_(value)
        if step + 1 == total_steps:
            self._model.load_state_dict(self._ema_state)


def build_model() -> nn.Module:
    return ImageClassifier()
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return EMAAdamW(
        model,
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
=======
) -> None:
    if isinstance(optimizer, EMAAdamW):
        optimizer.update_ema(step, total_steps)
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
>>>>>>> REPLACE