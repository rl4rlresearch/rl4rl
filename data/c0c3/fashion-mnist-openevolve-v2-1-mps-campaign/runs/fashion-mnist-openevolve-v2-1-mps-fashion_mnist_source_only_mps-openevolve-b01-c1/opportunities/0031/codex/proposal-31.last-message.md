MECHANISM: Late-training exponential weight averaging

HYPOTHESIS: Averaging the final half of the AdamW trajectory with decay 0.99 will exceed 9,236 correct predictions by reducing parameter noise while preserving the proven architecture and six-view objective.

INTENDED_EDIT: Replace AdamW with an interface-compatible optimizer that maintains an EMA after the midpoint of training and installs the averaged parameters on the final step.

EVIDENCE: The 0.75 ensemble-loss design achieved 9,236 correct, while nearby loss weights regressed and the larger architectural replacement timed out; inexpensive weight averaging tests an orthogonal generalization improvement without increasing learned parameters or training forwards.

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
    def __init__(
        self,
        params,
        total_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.ema_start = max(total_steps // 2, 1)
        self.ema_final_step = total_steps
        self.ema_step = 0
        self.ema_decay = 0.99
        self.ema_parameters: list[torch.Tensor] = []

    def step(self, closure=None):
        loss = super().step(closure)
        self.ema_step += 1
        if self.ema_step >= self.ema_start:
            parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            with torch.no_grad():
                if not self.ema_parameters:
                    self.ema_parameters = [
                        parameter.detach().clone()
                        for parameter in parameters
                    ]
                else:
                    for average, parameter in zip(
                        self.ema_parameters, parameters
                    ):
                        average.lerp_(
                            parameter.detach(),
                            1.0 - self.ema_decay,
                        )
                if self.ema_step == self.ema_final_step:
                    for parameter, average in zip(
                        parameters, self.ema_parameters
                    ):
                        parameter.copy_(average)
        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(
        model.parameters(),
        total_steps=total_steps,
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE