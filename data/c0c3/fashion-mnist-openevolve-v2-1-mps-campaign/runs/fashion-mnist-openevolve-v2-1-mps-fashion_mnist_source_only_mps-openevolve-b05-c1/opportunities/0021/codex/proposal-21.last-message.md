MECHANISM: Late-trajectory exponential weight averaging

HYPOTHESIS: Evaluating a 0.99-decay EMA of the trained weights will exceed 9,240 correct predictions by preserving the successful 3.3e-3 optimization trajectory while reducing endpoint noise.

INTENDED_EDIT: Maintain an exponential moving average of model parameters and floating-point BatchNorm buffers after every optimizer step, and use those averaged values for validation inference without adding learned parameters.

EVIDENCE: Raising the peak rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened cross-entropy from 0.21392 to 0.21460, suggesting useful optimization progress with a noisier final iterate; EMA directly stabilizes that iterate while retaining the verified schedule and batch size.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._ema_parameters: list[torch.Tensor] | None = None
        self._ema_buffers: list[torch.Tensor] | None = None
        self.features = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        logits = 2.0 * self._forward_once(views[0])
        logits = logits + 2.0 * self._forward_once(views[0].flip(-1))
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        return logits / (2.0 * (len(views) + 1))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        parameters = tuple(self.parameters())
        floating_buffers = tuple(
            buffer for buffer in self.buffers() if buffer.is_floating_point()
        )
        original_parameters: tuple[torch.Tensor, ...] = ()
        original_buffers: tuple[torch.Tensor, ...] = ()
        ema_parameters = self._ema_parameters
        ema_buffers = self._ema_buffers
        if ema_parameters is not None and ema_buffers is not None:
            original_parameters = tuple(
                parameter.data for parameter in parameters
            )
            original_buffers = tuple(
                buffer.data for buffer in floating_buffers
            )
            for parameter, average in zip(parameters, ema_parameters):
                parameter.data = average
            for buffer, average in zip(floating_buffers, ema_buffers):
                buffer.data = average

        try:
            padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
            views = (
                images,
                padded[:, :, 0:28, 1:29],
                padded[:, :, 2:30, 1:29],
                padded[:, :, 1:29, 0:28],
                padded[:, :, 1:29, 2:30],
            )
            logits = 2.0 * self._forward_once(views[0])
            logits = logits + 2.0 * self._forward_once(views[0].flip(-1))
            for view in views[1:]:
                logits = logits + self._forward_once(view)
                logits = logits + self._forward_once(view.flip(-1))
            return logits / (2.0 * (len(views) + 1))
        finally:
            for parameter, original in zip(
                parameters, original_parameters
            ):
                parameter.data = original
            for buffer, original in zip(
                floating_buffers, original_buffers
            ):
                buffer.data = original
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    optimizer._ema_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier

    model = optimizer._ema_model
    with torch.no_grad():
        parameters = tuple(model.parameters())
        floating_buffers = tuple(
            buffer for buffer in model.buffers()
            if buffer.is_floating_point()
        )
        if model._ema_parameters is None:
            model._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            model._ema_buffers = [
                buffer.detach().clone() for buffer in floating_buffers
            ]
        else:
            decay = 0.99
            torch._foreach_mul_(model._ema_parameters, decay)
            torch._foreach_add_(
                model._ema_parameters, parameters, alpha=1.0 - decay
            )
            torch._foreach_mul_(model._ema_buffers, decay)
            torch._foreach_add_(
                model._ema_buffers, floating_buffers, alpha=1.0 - decay
            )
>>>>>>> REPLACE