MECHANISM: Late-trajectory exponential weight averaging

HYPOTHESIS: Averaging model parameters over the final half of training with decay 0.99 will exceed 9,330 correct predictions by reducing optimizer noise while preserving the proven training loss, schedule, and equal-logit ensemble.

INTENDED_EDIT: Maintain a parameter EMA after the midpoint of training, use it during evaluation, and restore online parameters if training resumes.

EVIDENCE: Decision-time probability fusion reduced correctness to 9,326, canonical weighting reduced it to 9,319, and label-smoothing annealing reduced it to 9,312; this motivates retaining the best verified decision and target procedures while improving the final learned parameter estimate.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )
        self._ema_swapped = False

    def train(self, mode: bool = True) -> ImageClassifier:
        if mode and self._ema_swapped:
            with torch.no_grad():
                for parameter in self.parameters():
                    online_value = getattr(parameter, "_online_value", None)
                    if online_value is not None:
                        parameter.copy_(online_value)
                        delattr(parameter, "_online_value")
            self._ema_swapped = False

        result = super().train(mode)

        if not mode and not self._ema_swapped:
            swapped = False
            with torch.no_grad():
                for parameter in self.parameters():
                    ema_value = getattr(parameter, "_ema_value", None)
                    if ema_value is not None:
                        parameter._online_value = parameter.detach().clone()
                        parameter.copy_(ema_value)
                        swapped = True
            self._ema_swapped = swapped

        return result

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer._ema_started = False
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier

    if step >= total_steps // 2:
        ema_started = optimizer._ema_started
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if not parameter.requires_grad:
                        continue
                    if not ema_started:
                        parameter._ema_value = parameter.detach().clone()
                    else:
                        parameter._ema_value.lerp_(parameter.detach(), 0.01)
        optimizer._ema_started = True
>>>>>>> REPLACE