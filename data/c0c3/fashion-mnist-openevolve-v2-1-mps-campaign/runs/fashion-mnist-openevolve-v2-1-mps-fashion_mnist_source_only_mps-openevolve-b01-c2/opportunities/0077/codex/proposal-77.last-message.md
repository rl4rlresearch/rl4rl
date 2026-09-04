MECHANISM: Tail-weight exponential averaging

HYPOTHESIS: A 0.99-decay EMA of learned parameters will exceed 9,330 correct predictions by smoothing the late cosine-schedule trajectory without adding learned parameters or model forwards.

INTENDED_EDIT: Maintain a parameter EMA after each optimizer step and substitute the averaged weights when the model enters evaluation mode.

EVIDENCE: The verified 249,393-parameter design reaches 9,330 correct, while architecture and augmentation changes repeatedly timed out; EMA preserves that exact training procedure and adds only lightweight parameter averaging.

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
        self._ema_parameters: list[torch.Tensor] | None = None

    def train(self, mode: bool = True) -> ImageClassifier:
        if not mode and self._ema_parameters is not None:
            with torch.no_grad():
                for parameter, average in zip(
                    self.parameters(),
                    self._ema_parameters,
                ):
                    parameter.copy_(average)
        return super().train(mode)

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
    optimizer._ema_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
=======
) -> None:
    ema_model = optimizer._ema_model
    parameters = tuple(ema_model.parameters())
    with torch.no_grad():
        if ema_model._ema_parameters is None:
            ema_model._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            torch._foreach_lerp_(
                ema_model._ema_parameters,
                parameters,
                0.01,
            )

    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE