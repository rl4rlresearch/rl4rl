MECHANISM: Late-training exponential weight averaging

HYPOTHESIS: Averaging the final low-learning-rate trajectory will exceed 9,290 correct predictions by reducing optimizer noise without increasing the batch count or model size.

INTENDED_EDIT: Wrap the successful AdamW configuration with an EMA that begins after 70% of training and installs the averaged weights after the final optimizer step.

EVIDENCE: Batch size 64 improved correctness to 9,290 through finer-grained optimization, while batch size 32 exceeded the time limit; EMA seeks additional stability using the existing 1,564-step run.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )
=======
class EMAAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params,
        total_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.total_steps = total_steps
        self.ema_start = max(1, int(0.70 * total_steps))
        self.completed_steps = 0
        self.ema_decay = 0.98
        self.ema_parameters: list[torch.Tensor] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.completed_steps += 1

        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if self.completed_steps == self.ema_start:
            self.ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        elif self.completed_steps > self.ema_start:
            for average, parameter in zip(self.ema_parameters, parameters):
                average.lerp_(parameter.detach(), 1.0 - self.ema_decay)

        if (
            self.completed_steps >= self.total_steps
            and self.ema_parameters
        ):
            for parameter, average in zip(parameters, self.ema_parameters):
                parameter.copy_(average)

        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(
        model.parameters(),
        total_steps=total_steps,
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )
>>>>>>> REPLACE