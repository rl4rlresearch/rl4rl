MECHANISM: Late-trajectory parameter averaging

HYPOTHESIS: Averaging the final 32 low-learning-rate parameter states while retaining the best vertical-biased TTA will exceed Reference Design 3’s validation_score by reducing boundary instability without changing model capacity.

INTENDED_EDIT: Add a tail-averaging AdamW optimizer that installs the mean of the final 32 parameter states before validation, and restore the verified 0.10 vertical/0.08 horizontal TTA weights.

EVIDENCE: Reference Design 3 preserved 9,268 correct with the best cross-entropy, while architecture, pooling, gating, and augmentation changes regressed; this motivates preserving its representation and improving the final solution through low-risk trajectory averaging.

<<<<<<< SEARCH
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class TailAveragedAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params,
        *,
        total_steps: int,
        average_steps: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self._total_steps = total_steps
        self._average_steps = min(average_steps, total_steps)
        self._updates = 0
        self._average_count = 0
        self._averaged_parameters = None
        self._using_average = False

    def step(self, closure=None):
        loss = super().step(closure)
        self._updates += 1
        if self._updates > self._total_steps - self._average_steps:
            parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ]
            self._average_count += 1
            if self._averaged_parameters is None:
                self._averaged_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                blend = 1.0 / self._average_count
                for average, parameter in zip(
                    self._averaged_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), blend)
        return loss

    @torch.no_grad()
    def use_averaged_parameters(self) -> None:
        if (
            self._using_average
            or self._updates < self._total_steps
            or self._averaged_parameters is None
        ):
            return
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        for parameter, average in zip(
            parameters, self._averaged_parameters
        ):
            parameter.copy_(average)
        self._using_average = True


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return TailAveragedAdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        total_steps=total_steps,
        average_steps=32,
        lr=2.0e-3,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
    if isinstance(optimizer, TailAveragedAdamW):
        optimizer.use_averaged_parameters()
>>>>>>> REPLACE